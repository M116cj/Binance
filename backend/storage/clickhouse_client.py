"""
ClickHouse client for time series data storage and analytics.
Optimized for high-throughput market data ingestion and historical queries.
"""

import asyncio
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from contextlib import asynccontextmanager
import json

# ClickHouse client imports
try:
    from clickhouse_driver import Client as SyncClient
    from clickhouse_driver.errors import Error as ClickHouseError
    CLICKHOUSE_AVAILABLE = True
except ImportError:
    CLICKHOUSE_AVAILABLE = False
    SyncClient = None
    ClickHouseError = Exception

from backend.models.schemas import FeatureVector, TripleBarrierLabel, MarketRegimeState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClickHouseConfig:
    """ClickHouse configuration"""
    host: str = "localhost"
    port: int = 9000
    database: str = "crypto_surge"
    user: str = "default"
    password: str = ""
    secure: bool = False
    compression: str = "zstd"
    max_connections: int = 10
    max_block_size: int = 1000000
    settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.settings is None:
            self.settings = {
                'max_memory_usage': 10000000000,  # 10GB
                'max_threads': 8,
                'max_execution_time': 300,  # 5 minutes
                'send_progress_in_http_headers': 1,
                'log_queries': 1,
                'log_query_threads': 1,
                'allow_experimental_window_functions': 1
            }

class TableManager:
    """Manages ClickHouse table schemas and operations"""
    
    def __init__(self, client):
        self.client = client
        
        # Table schemas
        self.schemas = {
            'market_data': '''
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol LowCardinality(String),
                    stream_type LowCardinality(String),
                    timestamp DateTime64(3),
                    exchange_time DateTime64(3),
                    ingest_time DateTime64(3),
                    data String,
                    sequence_id UInt64,
                    quality_flags Array(LowCardinality(String)),
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, stream_type, timestamp)
                TTL date + INTERVAL 30 DAY
                SETTINGS index_granularity = 8192,
                         storage_policy = 'default'
            ''',
            
            'features': '''
                CREATE TABLE IF NOT EXISTS features (
                    symbol LowCardinality(String),
                    timestamp DateTime64(3),
                    window_start DateTime64(3),
                    window_end DateTime64(3),
                    
                    -- Order book features
                    queue_imbalance_1 Float64,
                    queue_imbalance_5 Float64,
                    microprice_deviation Float64,
                    depth_slope_bid Float64,
                    depth_slope_ask Float64,
                    near_touch_void Float64,
                    impact_lambda Float64,
                    
                    -- Trade flow features
                    order_flow_imbalance_10 Float64,
                    order_flow_imbalance_30 Float64,
                    cvd_slope Float64,
                    buy_cluster_intensity Float64,
                    follow_buy_ratio Float64,
                    
                    -- Volatility features
                    realized_volatility_1m Float64,
                    realized_volatility_5m Float64,
                    rv_ratio Float64,
                    bollinger_position Float64,
                    bollinger_squeeze Float64,
                    
                    -- Derivatives features
                    funding_delta Float64,
                    oi_pressure Float64,
                    long_short_ratio Float64,
                    lsr_divergence Float64,
                    
                    -- Liquidation features
                    liquidation_density_up Float64,
                    liquidation_density_down Float64,
                    post_liq_gap Float64,
                    
                    -- Cross-exchange features (optional)
                    lead_lag_spread Nullable(Float64),
                    arrival_rate_ratio Nullable(Float64),
                    
                    -- Metadata
                    feature_version LowCardinality(String),
                    quality_flags Array(LowCardinality(String)),
                    feature_count_valid UInt32,
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, timestamp)
                TTL date + INTERVAL 90 DAY
                SETTINGS index_granularity = 8192
            ''',
            
            'labels': '''
                CREATE TABLE IF NOT EXISTS labels (
                    symbol LowCardinality(String),
                    timestamp DateTime64(3),
                    label_timestamp DateTime64(3),
                    horizon_minutes UInt32,
                    theta_up Float64,
                    theta_dn Float64,
                    
                    -- Label results
                    label Int8,  -- -1, 0, 1
                    binary_up UInt8,  -- 0, 1
                    
                    -- Label details
                    breach_timestamp Nullable(DateTime64(3)),
                    breach_price Nullable(Float64),
                    max_favorable_excursion Float64,
                    max_adverse_excursion Float64,
                    
                    -- Regression targets
                    peak_return Float64,
                    time_to_peak Float64,
                    
                    -- Metadata
                    entry_price Float64,
                    exit_price Nullable(Float64),
                    holding_period Float64,
                    embargo_end DateTime64(3),
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, timestamp)
                TTL date + INTERVAL 180 DAY
                SETTINGS index_granularity = 8192
            ''',
            
            'predictions': '''
                CREATE TABLE IF NOT EXISTS predictions (
                    symbol LowCardinality(String),
                    timestamp DateTime64(3),
                    model_timestamp DateTime64(3),
                    
                    -- Predictions by horizon
                    horizon_5m_p_up Float64,
                    horizon_5m_ci_low Float64,
                    horizon_5m_ci_high Float64,
                    horizon_10m_p_up Float64,
                    horizon_10m_ci_low Float64,
                    horizon_10m_ci_high Float64,
                    horizon_30m_p_up Float64,
                    horizon_30m_ci_low Float64,
                    horizon_30m_ci_high Float64,
                    
                    -- Expected returns and utilities
                    expected_return_5m Float64,
                    expected_return_10m Float64,
                    expected_return_30m Float64,
                    estimated_cost_5m Float64,
                    estimated_cost_10m Float64,
                    estimated_cost_30m Float64,
                    utility_5m Float64,
                    utility_10m Float64,
                    utility_30m Float64,
                    
                    -- Decisions
                    decision_5m LowCardinality(String),
                    decision_10m LowCardinality(String),
                    decision_30m LowCardinality(String),
                    
                    -- Model metadata
                    model_version LowCardinality(String),
                    feature_version LowCardinality(String),
                    regime LowCardinality(String),
                    capacity_pct Float64,
                    prediction_confidence Float64,
                    
                    -- Quality and timing
                    quality_flags Array(LowCardinality(String)),
                    sla_latency_ms Float64,
                    cooldown_until Nullable(DateTime64(3)),
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, timestamp)
                TTL date + INTERVAL 30 DAY
                SETTINGS index_granularity = 8192
            ''',
            
            'regime_states': '''
                CREATE TABLE IF NOT EXISTS regime_states (
                    symbol LowCardinality(String),
                    timestamp DateTime64(3),
                    volatility_regime LowCardinality(String),
                    depth_regime LowCardinality(String),
                    funding_regime LowCardinality(String),
                    combined_regime LowCardinality(String),
                    regime_confidence Float64,
                    regime_stability Float64,
                    date Date MATERIALIZED toDate(timestamp)
                ) ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, timestamp)
                TTL date + INTERVAL 60 DAY
                SETTINGS index_granularity = 8192
            '''
        }
        
        # Materialized views for aggregations
        self.materialized_views = {
            'features_1s_agg': '''
                CREATE MATERIALIZED VIEW IF NOT EXISTS features_1s_agg
                ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, timestamp)
                TTL date + INTERVAL 7 DAY
                AS SELECT
                    symbol,
                    toStartOfSecond(timestamp) as timestamp,
                    avg(queue_imbalance_1) as avg_qi_1,
                    avg(order_flow_imbalance_10) as avg_ofi_10,
                    avg(microprice_deviation) as avg_microprice_dev,
                    avg(rv_ratio) as avg_rv_ratio,
                    avg(impact_lambda) as avg_impact_lambda,
                    count(*) as sample_count,
                    date
                FROM features
                GROUP BY symbol, timestamp, date
            ''',
            
            'features_1m_agg': '''
                CREATE MATERIALIZED VIEW IF NOT EXISTS features_1m_agg
                ENGINE = MergeTree()
                PARTITION BY date
                ORDER BY (symbol, timestamp)
                TTL date + INTERVAL 30 DAY
                AS SELECT
                    symbol,
                    toStartOfMinute(timestamp) as timestamp,
                    avg(queue_imbalance_1) as avg_qi_1,
                    stddevSamp(queue_imbalance_1) as std_qi_1,
                    avg(order_flow_imbalance_10) as avg_ofi_10,
                    stddevSamp(order_flow_imbalance_10) as std_ofi_10,
                    avg(microprice_deviation) as avg_microprice_dev,
                    avg(rv_ratio) as avg_rv_ratio,
                    avg(realized_volatility_1m) as avg_rv_1m,
                    avg(bollinger_position) as avg_bb_pos,
                    avg(funding_delta) as avg_funding_delta,
                    count(*) as sample_count,
                    date
                FROM features
                GROUP BY symbol, timestamp, date
            '''
        }
    
    def create_tables(self):
        """Create all tables"""
        logger.info("Creating ClickHouse tables...")
        
        for table_name, schema in self.schemas.items():
            try:
                self.client.execute(schema)
                logger.info(f"Created/verified table: {table_name}")
            except Exception as e:
                logger.error(f"Failed to create table {table_name}: {e}")
                raise
    
    def create_materialized_views(self):
        """Create materialized views"""
        logger.info("Creating materialized views...")
        
        for view_name, schema in self.materialized_views.items():
            try:
                self.client.execute(schema)
                logger.info(f"Created/verified materialized view: {view_name}")
            except Exception as e:
                logger.warning(f"Failed to create materialized view {view_name}: {e}")
    
    def optimize_tables(self):
        """Optimize tables for better performance"""
        logger.info("Optimizing tables...")
        
        for table_name in self.schemas.keys():
            try:
                # Force merge of parts
                self.client.execute(f"OPTIMIZE TABLE {table_name} FINAL")
                logger.info(f"Optimized table: {table_name}")
            except Exception as e:
                logger.warning(f"Failed to optimize table {table_name}: {e}")
    
    def get_table_sizes(self) -> Dict[str, Dict[str, Any]]:
        """Get table size information"""
        try:
            query = """
                SELECT 
                    table,
                    sum(rows) as total_rows,
                    formatReadableSize(sum(data_compressed_bytes)) as compressed_size,
                    formatReadableSize(sum(data_uncompressed_bytes)) as uncompressed_size,
                    round(sum(data_compressed_bytes) / sum(data_uncompressed_bytes), 3) as compression_ratio
                FROM system.parts 
                WHERE database = currentDatabase()
                AND active = 1
                GROUP BY table
                ORDER BY sum(data_compressed_bytes) DESC
            """
            
            result = self.client.execute(query)
            
            table_info = {}
            for row in result:
                table_info[row[0]] = {
                    'total_rows': row[1],
                    'compressed_size': row[2],
                    'uncompressed_size': row[3],
                    'compression_ratio': row[4]
                }
            
            return table_info
            
        except Exception as e:
            logger.error(f"Error getting table sizes: {e}")
            return {}

class ClickHouseManager:
    """High-performance ClickHouse client for time series data"""
    
    def __init__(self):
        # Load configuration from environment
        self.config = ClickHouseConfig(
            host=os.getenv("CLICKHOUSE_HOST", "localhost"),
            port=int(os.getenv("CLICKHOUSE_PORT", 9000)),
            database=os.getenv("CLICKHOUSE_DATABASE", "crypto_surge"),
            user=os.getenv("CLICKHOUSE_USER", "default"),
            password=os.getenv("CLICKHOUSE_PASSWORD", ""),
            secure=os.getenv("CLICKHOUSE_SECURE", "false").lower() == "true"
        )
        
        self.client = None
        self.table_manager = None
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()
        
        # Batch insertion
        self.batch_buffers = {
            'market_data': [],
            'features': [],
            'labels': [],
            'predictions': [],
            'regime_states': []
        }
        self.batch_sizes = {
            'market_data': 10000,
            'features': 1000,
            'labels': 1000,
            'predictions': 1000,
            'regime_states': 1000
        }
        self.batch_timeouts = {}  # Track last insert time for each table
        self.batch_timeout_seconds = 30
        
        # Performance tracking
        self.insert_counts = {table: 0 for table in self.batch_buffers.keys()}
        self.insert_errors = {table: 0 for table in self.batch_buffers.keys()}
        
    async def initialize(self):
        """Initialize ClickHouse connection and tables"""
        if not CLICKHOUSE_AVAILABLE:
            logger.error("ClickHouse driver not available. Install with: pip install clickhouse-driver")
            raise ImportError("ClickHouse driver not available")
        
        logger.info(f"Initializing ClickHouse connection to {self.config.host}:{self.config.port}")
        
        try:
            # Create client
            self.client = SyncClient(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password,
                secure=self.config.secure,
                compression=self.config.compression,
                settings=self.config.settings
            )
            
            # Test connection
            result = self.client.execute("SELECT version()")
            logger.info(f"Connected to ClickHouse version: {result[0][0]}")
            
            # Create database if not exists
            self.client.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.database}")
            
            # Initialize table manager
            self.table_manager = TableManager(self.client)
            
            # Create tables and views
            self.table_manager.create_tables()
            self.table_manager.create_materialized_views()
            
            # Initialize batch timeouts
            current_time = time.time()
            for table in self.batch_buffers.keys():
                self.batch_timeouts[table] = current_time
            
            logger.info("ClickHouse initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ClickHouse: {e}")
            raise
    
    def insert_market_data_batch(self, stream_type: str, market_data_list: List[Any]):
        """Insert batch of market data"""
        try:
            if not market_data_list:
                return
            
            rows = []
            for data in market_data_list:
                row = [
                    data.symbol,
                    stream_type,
                    datetime.fromtimestamp(data.ingest_time / 1000),
                    datetime.fromtimestamp(data.exchange_time / 1000),
                    datetime.fromtimestamp(data.ingest_time / 1000),
                    json.dumps(data.data) if isinstance(data.data, dict) else str(data.data),
                    data.sequence_id or 0,
                    data.quality_flags or []
                ]
                rows.append(row)
            
            with self.lock:
                self.batch_buffers['market_data'].extend(rows)
                
                if len(self.batch_buffers['market_data']) >= self.batch_sizes['market_data']:
                    self._flush_batch('market_data')
            
        except Exception as e:
            logger.error(f"Error inserting market data batch: {e}")
            self.insert_errors['market_data'] += 1
    
    def insert_features(self, feature_vector: FeatureVector):
        """Insert feature vector"""
        try:
            row = [
                feature_vector.symbol,
                datetime.fromtimestamp(feature_vector.timestamp / 1000),
                datetime.fromtimestamp(feature_vector.window_start / 1000),
                datetime.fromtimestamp(feature_vector.window_end / 1000),
                
                # Order book features
                feature_vector.queue_imbalance_1,
                feature_vector.queue_imbalance_5,
                feature_vector.microprice_deviation,
                feature_vector.depth_slope_bid,
                feature_vector.depth_slope_ask,
                feature_vector.near_touch_void,
                feature_vector.impact_lambda,
                
                # Trade flow features
                feature_vector.order_flow_imbalance_10,
                feature_vector.order_flow_imbalance_30,
                feature_vector.cvd_slope,
                feature_vector.buy_cluster_intensity,
                feature_vector.follow_buy_ratio,
                
                # Volatility features
                feature_vector.realized_volatility_1m,
                feature_vector.realized_volatility_5m,
                feature_vector.rv_ratio,
                feature_vector.bollinger_position,
                feature_vector.bollinger_squeeze,
                
                # Derivatives features
                feature_vector.funding_delta,
                feature_vector.oi_pressure,
                feature_vector.long_short_ratio,
                feature_vector.lsr_divergence,
                
                # Liquidation features
                feature_vector.liquidation_density_up,
                feature_vector.liquidation_density_down,
                feature_vector.post_liq_gap,
                
                # Cross-exchange features
                feature_vector.lead_lag_spread,
                feature_vector.arrival_rate_ratio,
                
                # Metadata
                feature_vector.feature_version,
                [flag.value if hasattr(flag, 'value') else str(flag) for flag in feature_vector.quality_flags],
                feature_vector.feature_count_valid
            ]
            
            with self.lock:
                self.batch_buffers['features'].append(row)
                
                if len(self.batch_buffers['features']) >= self.batch_sizes['features']:
                    self._flush_batch('features')
            
        except Exception as e:
            logger.error(f"Error inserting features: {e}")
            self.insert_errors['features'] += 1
    
    def insert_labels(self, labels: List[TripleBarrierLabel]):
        """Insert batch of labels"""
        try:
            rows = []
            for label in labels:
                row = [
                    label.symbol,
                    datetime.fromtimestamp(label.timestamp / 1000),
                    datetime.fromtimestamp(label.label_timestamp / 1000),
                    label.horizon_minutes,
                    label.theta_up,
                    label.theta_dn,
                    
                    # Label results
                    label.label,
                    1 if label.binary_up else 0,
                    
                    # Label details
                    datetime.fromtimestamp(label.breach_timestamp / 1000) if label.breach_timestamp else None,
                    label.breach_price,
                    label.max_favorable_excursion,
                    label.max_adverse_excursion,
                    
                    # Regression targets
                    label.peak_return,
                    label.time_to_peak,
                    
                    # Metadata
                    label.entry_price,
                    label.exit_price,
                    label.holding_period,
                    datetime.fromtimestamp(label.embargo_end / 1000)
                ]
                rows.append(row)
            
            with self.lock:
                self.batch_buffers['labels'].extend(rows)
                
                if len(self.batch_buffers['labels']) >= self.batch_sizes['labels']:
                    self._flush_batch('labels')
            
        except Exception as e:
            logger.error(f"Error inserting labels: {e}")
            self.insert_errors['labels'] += 1
    
    def insert_regime_state(self, regime_state: MarketRegimeState):
        """Insert market regime state"""
        try:
            row = [
                regime_state.symbol,
                datetime.fromtimestamp(regime_state.timestamp / 1000),
                regime_state.volatility_regime,
                regime_state.depth_regime,
                regime_state.funding_regime,
                regime_state.combined_regime,
                regime_state.regime_confidence,
                regime_state.regime_stability
            ]
            
            with self.lock:
                self.batch_buffers['regime_states'].append(row)
                
                if len(self.batch_buffers['regime_states']) >= self.batch_sizes['regime_states']:
                    self._flush_batch('regime_states')
            
        except Exception as e:
            logger.error(f"Error inserting regime state: {e}")
            self.insert_errors['regime_states'] += 1
    
    def _flush_batch(self, table_name: str):
        """Flush batch buffer to ClickHouse"""
        try:
            if not self.batch_buffers[table_name]:
                return
            
            rows = self.batch_buffers[table_name].copy()
            self.batch_buffers[table_name].clear()
            self.batch_timeouts[table_name] = time.time()
            
            # Execute insert in thread pool to avoid blocking
            self.executor.submit(self._execute_insert, table_name, rows)
            
        except Exception as e:
            logger.error(f"Error flushing batch for {table_name}: {e}")
    
    def _execute_insert(self, table_name: str, rows: List[List[Any]]):
        """Execute batch insert"""
        try:
            self.client.execute(f"INSERT INTO {table_name} VALUES", rows)
            self.insert_counts[table_name] += len(rows)
            logger.debug(f"Inserted {len(rows)} rows into {table_name}")
            
        except Exception as e:
            logger.error(f"Error executing insert into {table_name}: {e}")
            self.insert_errors[table_name] += len(rows)
    
    def flush_all_batches(self):
        """Flush all pending batches"""
        with self.lock:
            for table_name in self.batch_buffers.keys():
                if self.batch_buffers[table_name]:
                    self._flush_batch(table_name)
    
    def flush_expired_batches(self):
        """Flush batches that have exceeded timeout"""
        current_time = time.time()
        
        with self.lock:
            for table_name in self.batch_buffers.keys():
                if (self.batch_buffers[table_name] and 
                    current_time - self.batch_timeouts[table_name] > self.batch_timeout_seconds):
                    self._flush_batch(table_name)
    
    # Query methods
    def get_features_window(self, symbol: str, start_time: datetime, end_time: datetime,
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Get features within time window"""
        try:
            if columns:
                column_str = ", ".join(columns)
            else:
                column_str = "*"
            
            query = f"""
                SELECT {column_str}
                FROM features
                WHERE symbol = %(symbol)s
                AND timestamp BETWEEN %(start_time)s AND %(end_time)s
                ORDER BY timestamp
            """
            
            result = self.client.execute(query, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            })
            
            if columns:
                column_names = columns
            else:
                # Get column names from table schema
                column_names = self._get_table_columns('features')
            
            return pd.DataFrame(result, columns=column_names)
            
        except Exception as e:
            logger.error(f"Error getting features window: {e}")
            return pd.DataFrame()
    
    def get_labels_for_training(self, symbol: str, start_date: datetime, end_date: datetime,
                              theta_up: float, theta_dn: float, horizon_minutes: int) -> pd.DataFrame:
        """Get labels for training"""
        try:
            query = """
                SELECT 
                    symbol,
                    timestamp,
                    binary_up as label,
                    peak_return,
                    time_to_peak,
                    holding_period,
                    max_favorable_excursion,
                    max_adverse_excursion
                FROM labels
                WHERE symbol = %(symbol)s
                AND toDate(timestamp) BETWEEN %(start_date)s AND %(end_date)s
                AND theta_up = %(theta_up)s
                AND theta_dn = %(theta_dn)s
                AND horizon_minutes = %(horizon_minutes)s
                ORDER BY timestamp
            """
            
            result = self.client.execute(query, {
                'symbol': symbol,
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'theta_up': theta_up,
                'theta_dn': theta_dn,
                'horizon_minutes': horizon_minutes
            })
            
            columns = ['symbol', 'timestamp', 'label', 'peak_return', 'time_to_peak',
                      'holding_period', 'max_favorable_excursion', 'max_adverse_excursion']
            
            return pd.DataFrame(result, columns=columns)
            
        except Exception as e:
            logger.error(f"Error getting labels for training: {e}")
            return pd.DataFrame()
    
    def get_feature_label_dataset(self, symbol: str, start_date: datetime, end_date: datetime,
                                 theta_up: float = 0.006, theta_dn: float = 0.004,
                                 horizon_minutes: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get aligned feature and label dataset for training"""
        try:
            query = """
                SELECT 
                    f.*,
                    l.binary_up as label,
                    l.peak_return,
                    l.time_to_peak
                FROM features f
                LEFT JOIN labels l ON (
                    f.symbol = l.symbol 
                    AND f.timestamp = l.timestamp
                    AND l.theta_up = %(theta_up)s
                    AND l.theta_dn = %(theta_dn)s
                    AND l.horizon_minutes = %(horizon_minutes)s
                )
                WHERE f.symbol = %(symbol)s
                AND toDate(f.timestamp) BETWEEN %(start_date)s AND %(end_date)s
                AND l.label IS NOT NULL  -- Only include labeled samples
                ORDER BY f.timestamp
            """
            
            result = self.client.execute(query, {
                'symbol': symbol,
                'start_date': start_date.date(),
                'end_date': end_date.date(),
                'theta_up': theta_up,
                'theta_dn': theta_dn,
                'horizon_minutes': horizon_minutes
            })
            
            if not result:
                return pd.DataFrame(), pd.DataFrame()
            
            # Get column names
            feature_columns = self._get_table_columns('features')
            all_columns = feature_columns + ['label', 'peak_return', 'time_to_peak']
            
            df = pd.DataFrame(result, columns=all_columns)
            
            # Split features and labels
            feature_df = df[feature_columns]
            label_df = df[['label', 'peak_return', 'time_to_peak']]
            
            return feature_df, label_df
            
        except Exception as e:
            logger.error(f"Error getting feature-label dataset: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_market_data_ohlcv(self, symbol: str, start_time: datetime, end_time: datetime,
                             interval_seconds: int = 60) -> pd.DataFrame:
        """Get OHLCV data from market data"""
        try:
            query = f"""
                SELECT 
                    symbol,
                    toStartOfInterval(timestamp, INTERVAL {interval_seconds} SECOND) as timestamp,
                    argMin(JSONExtractFloat(data, 'price'), timestamp) as open,
                    max(JSONExtractFloat(data, 'price')) as high,
                    min(JSONExtractFloat(data, 'price')) as low,
                    argMax(JSONExtractFloat(data, 'price'), timestamp) as close,
                    sum(JSONExtractFloat(data, 'quantity')) as volume,
                    count(*) as trade_count
                FROM market_data
                WHERE symbol = %(symbol)s
                AND stream_type = 'aggTrade'
                AND timestamp BETWEEN %(start_time)s AND %(end_time)s
                GROUP BY symbol, timestamp
                ORDER BY timestamp
            """
            
            result = self.client.execute(query, {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            })
            
            columns = ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count']
            return pd.DataFrame(result, columns=columns)
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self, symbol: Optional[str] = None, 
                              days_back: int = 7) -> Dict[str, Any]:
        """Get performance metrics for the system"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            metrics = {}
            
            # Data ingestion metrics
            ingestion_query = """
                SELECT 
                    symbol,
                    stream_type,
                    count(*) as record_count,
                    countIf(length(quality_flags) > 0) as degraded_count,
                    min(timestamp) as first_timestamp,
                    max(timestamp) as last_timestamp
                FROM market_data
                WHERE toDate(timestamp) BETWEEN %(start_date)s AND %(end_date)s
            """
            
            if symbol:
                ingestion_query += " AND symbol = %(symbol)s"
            
            ingestion_query += " GROUP BY symbol, stream_type ORDER BY symbol, stream_type"
            
            params = {
                'start_date': start_date.date(),
                'end_date': end_date.date()
            }
            
            if symbol:
                params['symbol'] = symbol
            
            ingestion_result = self.client.execute(ingestion_query, params)
            
            metrics['ingestion'] = []
            for row in ingestion_result:
                metrics['ingestion'].append({
                    'symbol': row[0],
                    'stream_type': row[1],
                    'record_count': row[2],
                    'degraded_count': row[3],
                    'degraded_rate': row[3] / max(row[2], 1),
                    'first_timestamp': row[4],
                    'last_timestamp': row[5]
                })
            
            # Feature generation metrics
            feature_query = """
                SELECT 
                    symbol,
                    count(*) as feature_count,
                    countIf(length(quality_flags) > 0) as degraded_features,
                    avg(feature_count_valid) as avg_valid_features
                FROM features
                WHERE toDate(timestamp) BETWEEN %(start_date)s AND %(end_date)s
            """
            
            if symbol:
                feature_query += " AND symbol = %(symbol)s"
            
            feature_query += " GROUP BY symbol ORDER BY symbol"
            
            feature_result = self.client.execute(feature_query, params)
            
            metrics['features'] = []
            for row in feature_result:
                metrics['features'].append({
                    'symbol': row[0],
                    'feature_count': row[1],
                    'degraded_features': row[2],
                    'degraded_rate': row[2] / max(row[1], 1),
                    'avg_valid_features': row[3]
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get column names for a table"""
        try:
            query = f"DESCRIBE TABLE {table_name}"
            result = self.client.execute(query)
            return [row[0] for row in result]
            
        except Exception as e:
            logger.error(f"Error getting table columns for {table_name}: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # Table sizes
            stats['table_sizes'] = self.table_manager.get_table_sizes()
            
            # Insert counts
            stats['insert_counts'] = self.insert_counts.copy()
            stats['insert_errors'] = self.insert_errors.copy()
            
            # Batch buffer status
            stats['batch_buffers'] = {
                table: len(buffer) for table, buffer in self.batch_buffers.items()
            }
            
            # Query performance
            query_stats_query = """
                SELECT 
                    query_kind,
                    count() as query_count,
                    avg(query_duration_ms) as avg_duration_ms,
                    quantile(0.95)(query_duration_ms) as p95_duration_ms
                FROM system.query_log
                WHERE event_date >= today() - 1
                GROUP BY query_kind
                ORDER BY query_count DESC
                LIMIT 10
            """
            
            try:
                query_stats = self.client.execute(query_stats_query)
                stats['query_performance'] = [
                    {
                        'query_kind': row[0],
                        'query_count': row[1],
                        'avg_duration_ms': row[2],
                        'p95_duration_ms': row[3]
                    }
                    for row in query_stats
                ]
            except Exception:
                stats['query_performance'] = []
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for ClickHouse"""
        try:
            start_time = time.time()
            
            # Test query
            result = self.client.execute("SELECT count() FROM system.tables WHERE database = currentDatabase()")
            query_latency = (time.time() - start_time) * 1000
            
            table_count = result[0][0] if result else 0
            
            # Get database stats
            stats = self.get_database_stats()
            
            return {
                'status': 'healthy',
                'query_latency_ms': query_latency,
                'table_count': table_count,
                'database': self.config.database,
                'host': self.config.host,
                'insert_counts': self.insert_counts,
                'batch_buffer_sizes': {
                    table: len(buffer) for table, buffer in self.batch_buffers.items()
                },
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"ClickHouse health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': int(time.time() * 1000)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up ClickHouse connections...")
        
        try:
            # Flush all pending batches
            self.flush_all_batches()
            
            # Wait for pending inserts to complete
            self.executor.shutdown(wait=True)
            
            # Close client
            if self.client:
                self.client.disconnect()
            
            logger.info("ClickHouse cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during ClickHouse cleanup: {e}")
