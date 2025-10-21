import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
import orjson
from scipy import stats
from scipy.optimize import minimize_scalar
import numba
from numba import jit, njit
import os

from backend.storage.redis_client import RedisManager  
from backend.storage.clickhouse_client import ClickHouseManager
from backend.models.features import FeatureEngine
from backend.models.labeling import LabelGenerator
from backend.utils.monitoring import MetricsCollector
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureVector:
    """Feature vector with metadata"""
    symbol: str
    timestamp: int
    window_start: int
    window_end: int
    features: Dict[str, float]
    quality_flags: List[str]
    feature_version: str = "1.0.0"
    
    def __post_init__(self):
        if self.quality_flags is None:
            self.quality_flags = []

class RingBuffer:
    """High-performance ring buffer for streaming data"""
    
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.data = np.full(maxlen, np.nan)
        self.timestamps = np.zeros(maxlen, dtype=np.int64)
        self.head = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def append(self, value: float, timestamp: int):
        """Thread-safe append"""
        with self.lock:
            self.data[self.head] = value
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.maxlen
            self.size = min(self.size + 1, self.maxlen)
    
    def get_window(self, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get most recent window"""
        with self.lock:
            if self.size < window_size:
                return np.array([]), np.array([])
            
            # Get data in correct order
            if self.head >= window_size:
                data_slice = self.data[self.head - window_size:self.head]
                ts_slice = self.timestamps[self.head - window_size:self.head]
            else:
                # Wrap around
                data_slice = np.concatenate([
                    self.data[self.maxlen - (window_size - self.head):],
                    self.data[:self.head]
                ])
                ts_slice = np.concatenate([
                    self.timestamps[self.maxlen - (window_size - self.head):],
                    self.timestamps[:self.head]
                ])
            
            return data_slice, ts_slice
    
    def is_sufficient(self, min_pct: float = 0.5) -> bool:
        """Check if buffer has sufficient valid data"""
        with self.lock:
            if self.size == 0:
                return False
            valid_count = np.sum(~np.isnan(self.data[:min(self.size, self.maxlen)]))
            return valid_count / min(self.size, self.maxlen) >= min_pct

@njit
def welford_update(mean: float, m2: float, count: int, new_value: float) -> Tuple[float, float, int]:
    """Numba-optimized Welford online variance algorithm"""
    count += 1
    delta = new_value - mean
    mean += delta / count
    delta2 = new_value - mean
    m2 += delta * delta2
    return mean, m2, count

@njit
def calculate_queue_imbalance(bid_sizes: np.ndarray, ask_sizes: np.ndarray) -> float:
    """Numba-optimized queue imbalance calculation"""
    bid_sum = np.sum(bid_sizes)
    ask_sum = np.sum(ask_sizes)
    total = bid_sum + ask_sum
    if total == 0:
        return 0.0
    return (bid_sum - ask_sum) / total

@njit
def calculate_microprice_deviation(best_bid: float, best_ask: float, 
                                 bid_size: float, ask_size: float, mid: float) -> float:
    """Numba-optimized microprice deviation calculation"""
    total_size = bid_size + ask_size
    if total_size == 0 or mid == 0:
        return 0.0
    
    microprice = (best_ask * bid_size + best_bid * ask_size) / total_size
    return (microprice - mid) / mid

@njit
def calculate_ofi(buy_vol: float, sell_vol: float) -> float:
    """Numba-optimized order flow imbalance calculation"""
    total_vol = buy_vol + sell_vol
    if total_vol == 0:
        return 0.0
    return (buy_vol - sell_vol) / total_vol

class FeatureService:
    """High-performance feature engineering service"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_manager = RedisManager()
        self.clickhouse_manager = ClickHouseManager()
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGenerator()
        self.metrics_collector = MetricsCollector("features")
        
        # Ring buffers for each symbol
        self.buffers: Dict[str, Dict[str, RingBuffer]] = defaultdict(lambda: defaultdict(lambda: RingBuffer(1000)))
        
        # Online statistics storage
        self.online_stats: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(dict))
        
        # Feature computation scheduling
        self.feature_schedule = {
            'fast': ['qi', 'ofi', 'microprice_deviation', 'depth_slope'],  # Every update
            'medium': ['near_touch_void', 'impact_lambda', 'rv_ratio'],    # Every 1s
            'slow': ['hawkes_intensity', 'full_shap']                      # Every 5s
        }
        
        self.last_computation = defaultdict(lambda: defaultdict(float))
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=6)
        self.shutdown_flag = False
        
        # Feature cache
        self.feature_cache: Dict[str, FeatureVector] = {}
        self.cache_ttl = 1.0  # 1 second TTL
    
    async def initialize(self):
        """Initialize feature service"""
        logger.info("Initializing feature service...")
        
        await self.redis_manager.initialize()
        await self.clickhouse_manager.initialize()
        
        # Start feature computation thread
        self.feature_thread = threading.Thread(target=self._feature_computation_loop)
        self.feature_thread.start()
        
        logger.info("Feature service initialized")
    
    def _feature_computation_loop(self):
        """Main feature computation loop"""
        logger.info("Starting feature computation loop")
        
        while not self.shutdown_flag:
            try:
                current_time = time.time()
                
                # Get symbols to process
                symbols = self._get_active_symbols()
                
                for symbol in symbols:
                    # Update ring buffers from Redis
                    self._update_buffers(symbol)
                    
                    # Compute features based on schedule
                    self._compute_scheduled_features(symbol, current_time)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.1)  # 100ms
                
            except Exception as e:
                logger.error(f"Error in feature computation loop: {e}")
                time.sleep(1)
        
        logger.info("Feature computation loop stopped")
    
    def _get_active_symbols(self) -> List[str]:
        """Get list of active symbols from Redis"""
        try:
            # Get symbols that have recent data
            keys = self.redis_manager.client.keys("spot:*:*")
            symbols = set()
            for key in keys:
                parts = key.decode().split(':')
                if len(parts) >= 2:
                    symbols.add(parts[1])
            return list(symbols)
        except Exception as e:
            logger.error(f"Error getting active symbols: {e}")
            return ['BTCUSDT', 'ETHUSDT']  # Fallback
    
    def _update_buffers(self, symbol: str):
        """Update ring buffers from Redis data"""
        try:
            # Get latest market data for symbol
            streams = ['aggtrade', 'bookticker', 'depth', 'kline_1m', 'ticker']
            
            for stream in streams:
                key = f"spot:{symbol}:{stream}"
                data_json = self.redis_manager.client.get(key)
                
                if data_json:
                    data = orjson.loads(data_json)
                    timestamp = data['exchange_time']
                    
                    # Update appropriate buffers based on stream type
                    if stream == 'bookticker':
                        self._update_orderbook_buffers(symbol, data['data'], timestamp)
                    elif stream == 'aggtrade':
                        self._update_trade_buffers(symbol, data['data'], timestamp)
                    elif stream == 'depth':
                        self._update_depth_buffers(symbol, data['data'], timestamp)
                    elif stream == 'kline_1m':
                        self._update_kline_buffers(symbol, data['data'], timestamp)
        
        except Exception as e:
            logger.error(f"Error updating buffers for {symbol}: {e}")
    
    def _update_orderbook_buffers(self, symbol: str, data: dict, timestamp: int):
        """Update order book related buffers"""
        try:
            best_bid = float(data.get('b', 0))
            best_ask = float(data.get('a', 0))
            bid_size = float(data.get('B', 0))
            ask_size = float(data.get('A', 0))
            
            if best_bid > 0 and best_ask > 0:
                mid_price = (best_bid + best_ask) / 2
                spread = (best_ask - best_bid) / mid_price
                
                self.buffers[symbol]['mid_price'].append(mid_price, timestamp)
                self.buffers[symbol]['best_bid'].append(best_bid, timestamp)
                self.buffers[symbol]['best_ask'].append(best_ask, timestamp)
                self.buffers[symbol]['bid_size'].append(bid_size, timestamp)
                self.buffers[symbol]['ask_size'].append(ask_size, timestamp)
                self.buffers[symbol]['spread'].append(spread, timestamp)
        
        except Exception as e:
            logger.error(f"Error updating orderbook buffers for {symbol}: {e}")
    
    def _update_trade_buffers(self, symbol: str, data: dict, timestamp: int):
        """Update trade related buffers"""
        try:
            price = float(data.get('p', 0))
            quantity = float(data.get('q', 0))
            is_buyer_maker = data.get('m', False)
            
            # Classify as buy/sell
            if is_buyer_maker:
                # Buyer is market maker, so this is a sell
                self.buffers[symbol]['sell_volume'].append(quantity, timestamp)
                self.buffers[symbol]['buy_volume'].append(0, timestamp)
            else:
                # Seller is market maker, so this is a buy
                self.buffers[symbol]['buy_volume'].append(quantity, timestamp)
                self.buffers[symbol]['sell_volume'].append(0, timestamp)
            
            self.buffers[symbol]['trade_price'].append(price, timestamp)
            self.buffers[symbol]['trade_volume'].append(quantity, timestamp)
        
        except Exception as e:
            logger.error(f"Error updating trade buffers for {symbol}: {e}")
    
    def _update_depth_buffers(self, symbol: str, data: dict, timestamp: int):
        """Update depth buffers"""
        try:
            bids = data.get('b', [])
            asks = data.get('a', [])
            
            if bids and asks:
                # Calculate total liquidity at different levels
                bid_liq_1 = sum(float(bid[1]) for bid in bids[:1])
                ask_liq_1 = sum(float(ask[1]) for ask in asks[:1])
                bid_liq_5 = sum(float(bid[1]) for bid in bids[:5])
                ask_liq_5 = sum(float(ask[1]) for ask in asks[:5])
                
                self.buffers[symbol]['bid_liquidity_1'].append(bid_liq_1, timestamp)
                self.buffers[symbol]['ask_liquidity_1'].append(ask_liq_1, timestamp)
                self.buffers[symbol]['bid_liquidity_5'].append(bid_liq_5, timestamp)
                self.buffers[symbol]['ask_liquidity_5'].append(ask_liq_5, timestamp)
        
        except Exception as e:
            logger.error(f"Error updating depth buffers for {symbol}: {e}")
    
    def _update_kline_buffers(self, symbol: str, data: dict, timestamp: int):
        """Update kline buffers"""
        try:
            close_price = float(data.get('c', 0))
            volume = float(data.get('v', 0))
            
            if close_price > 0:
                self.buffers[symbol]['kline_close'].append(close_price, timestamp)
                self.buffers[symbol]['kline_volume'].append(volume, timestamp)
        
        except Exception as e:
            logger.error(f"Error updating kline buffers for {symbol}: {e}")
    
    def _compute_scheduled_features(self, symbol: str, current_time: float):
        """Compute features based on schedule"""
        try:
            features = {}
            quality_flags = []
            
            # Fast features (every update)
            if self._should_compute('fast', symbol, current_time, 0.1):
                fast_features = self._compute_fast_features(symbol)
                features.update(fast_features)
                self.last_computation[symbol]['fast'] = current_time
            
            # Medium features (every 1s)
            if self._should_compute('medium', symbol, current_time, 1.0):
                medium_features = self._compute_medium_features(symbol)
                features.update(medium_features)
                self.last_computation[symbol]['medium'] = current_time
            
            # Slow features (every 5s)
            if self._should_compute('slow', symbol, current_time, 5.0):
                slow_features = self._compute_slow_features(symbol)
                features.update(slow_features)
                self.last_computation[symbol]['slow'] = current_time
            
            # Check data quality
            if not self._check_data_quality(symbol):
                quality_flags.append('window_insufficient')
            
            # Store features if we have any
            if features:
                timestamp = int(current_time * 1000)
                feature_vector = FeatureVector(
                    symbol=symbol,
                    timestamp=timestamp,
                    window_start=timestamp - 30000,  # 30s window
                    window_end=timestamp,
                    features=features,
                    quality_flags=quality_flags
                )
                
                self._store_features(feature_vector)
                self.feature_cache[symbol] = feature_vector
        
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
    
    def _should_compute(self, schedule_type: str, symbol: str, current_time: float, interval: float) -> bool:
        """Check if features should be computed based on schedule"""
        last_time = self.last_computation[symbol].get(schedule_type, 0)
        return current_time - last_time >= interval
    
    def _compute_fast_features(self, symbol: str) -> Dict[str, float]:
        """Compute fast features (updated every batch)"""
        features = {}
        
        try:
            # Queue Imbalance (QI)
            bid_data, _ = self.buffers[symbol]['bid_size'].get_window(1)
            ask_data, _ = self.buffers[symbol]['ask_size'].get_window(1)
            
            if len(bid_data) > 0 and len(ask_data) > 0:
                features['qi_1'] = calculate_queue_imbalance(bid_data[-1:], ask_data[-1:])
            
            # Microprice deviation
            best_bid_data, _ = self.buffers[symbol]['best_bid'].get_window(1)
            best_ask_data, _ = self.buffers[symbol]['best_ask'].get_window(1)
            mid_price_data, _ = self.buffers[symbol]['mid_price'].get_window(1)
            
            if len(best_bid_data) > 0 and len(best_ask_data) > 0 and len(mid_price_data) > 0:
                features['microprice_dev'] = calculate_microprice_deviation(
                    best_bid_data[-1], best_ask_data[-1], 
                    bid_data[-1] if len(bid_data) > 0 else 0,
                    ask_data[-1] if len(ask_data) > 0 else 0,
                    mid_price_data[-1]
                )
            
            # Order Flow Imbalance (OFI)
            buy_vol_data, _ = self.buffers[symbol]['buy_volume'].get_window(10)
            sell_vol_data, _ = self.buffers[symbol]['sell_volume'].get_window(10)
            
            if len(buy_vol_data) > 0 and len(sell_vol_data) > 0:
                buy_sum = np.sum(buy_vol_data)
                sell_sum = np.sum(sell_vol_data)
                features['ofi_10'] = calculate_ofi(buy_sum, sell_sum)
        
        except Exception as e:
            logger.error(f"Error computing fast features for {symbol}: {e}")
        
        return features
    
    def _compute_medium_features(self, symbol: str) -> Dict[str, float]:
        """Compute medium frequency features (every 1s)"""
        features = {}
        
        try:
            # Realized volatility
            mid_prices, timestamps = self.buffers[symbol]['mid_price'].get_window(60)  # 1 minute
            if len(mid_prices) > 1:
                log_returns = np.diff(np.log(mid_prices[mid_prices > 0]))
                if len(log_returns) > 0:
                    features['rv_1m'] = np.sqrt(np.sum(log_returns ** 2))
            
            # RV ratio (short/long)
            short_prices, _ = self.buffers[symbol]['mid_price'].get_window(15)  # 15s
            if len(short_prices) > 1:
                short_returns = np.diff(np.log(short_prices[short_prices > 0]))
                if len(short_returns) > 0:
                    rv_short = np.sqrt(np.sum(short_returns ** 2))
                    rv_long = features.get('rv_1m', 1.0)
                    features['rv_ratio'] = rv_short / max(rv_long, 1e-8)
            
            # Impact lambda
            trade_volumes, _ = self.buffers[symbol]['trade_volume'].get_window(20)
            mid_changes, _ = self._get_mid_price_changes(symbol, 20)
            
            if len(trade_volumes) > 5 and len(mid_changes) > 5:
                total_volume = np.sum(trade_volumes)
                total_price_change = np.sum(np.abs(mid_changes))
                if total_volume > 0:
                    features['impact_lambda'] = total_price_change / total_volume
        
        except Exception as e:
            logger.error(f"Error computing medium features for {symbol}: {e}")
        
        return features
    
    def _compute_slow_features(self, symbol: str) -> Dict[str, float]:
        """Compute slow features (every 5s)"""
        features = {}
        
        try:
            # Depth slope (requires more computation)
            features.update(self._compute_depth_slope(symbol))
            
            # Near touch void
            features.update(self._compute_near_touch_void(symbol))
            
            # Bollinger squeeze indicator
            features.update(self._compute_bollinger_squeeze(symbol))
        
        except Exception as e:
            logger.error(f"Error computing slow features for {symbol}: {e}")
        
        return features
    
    def _get_mid_price_changes(self, symbol: str, window: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get mid price changes over window"""
        mid_prices, timestamps = self.buffers[symbol]['mid_price'].get_window(window + 1)
        if len(mid_prices) > 1:
            changes = np.diff(mid_prices)
            return changes, timestamps[1:]
        return np.array([]), np.array([])
    
    def _compute_depth_slope(self, symbol: str) -> Dict[str, float]:
        """Compute depth slope feature"""
        features = {}
        
        try:
            # Get liquidity at different levels
            bid_liq_data, _ = self.buffers[symbol]['bid_liquidity_5'].get_window(1)
            ask_liq_data, _ = self.buffers[symbol]['ask_liquidity_5'].get_window(1)
            
            if len(bid_liq_data) > 0 and len(ask_liq_data) > 0:
                # Simplified depth slope calculation
                bid_slope = np.log(max(bid_liq_data[-1], 1e-8))
                ask_slope = np.log(max(ask_liq_data[-1], 1e-8))
                features['depth_slope_bid'] = bid_slope
                features['depth_slope_ask'] = ask_slope
        
        except Exception as e:
            logger.error(f"Error computing depth slope for {symbol}: {e}")
        
        return features
    
    def _compute_near_touch_void(self, symbol: str) -> Dict[str, float]:
        """Compute near touch void feature"""
        features = {}
        
        try:
            # Ratio of near vs far liquidity
            bid_liq_1, _ = self.buffers[symbol]['bid_liquidity_1'].get_window(1)
            ask_liq_1, _ = self.buffers[symbol]['ask_liquidity_1'].get_window(1)
            bid_liq_5, _ = self.buffers[symbol]['bid_liquidity_5'].get_window(1)
            ask_liq_5, _ = self.buffers[symbol]['ask_liquidity_5'].get_window(1)
            
            if (len(bid_liq_1) > 0 and len(ask_liq_1) > 0 and 
                len(bid_liq_5) > 0 and len(ask_liq_5) > 0):
                
                near_total = bid_liq_1[-1] + ask_liq_1[-1]
                far_total = bid_liq_5[-1] + ask_liq_5[-1]
                
                if far_total > 0:
                    features['near_touch_ratio'] = near_total / far_total
        
        except Exception as e:
            logger.error(f"Error computing near touch void for {symbol}: {e}")
        
        return features
    
    def _compute_bollinger_squeeze(self, symbol: str) -> Dict[str, float]:
        """Compute Bollinger band squeeze indicator"""
        features = {}
        
        try:
            # Get price data
            prices, _ = self.buffers[symbol]['mid_price'].get_window(60)  # 1 minute
            
            if len(prices) > 20:
                # Simple moving average and standard deviation
                ma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                
                # Current price relative to bands
                current_price = prices[-1]
                if std > 0:
                    bb_position = (current_price - ma) / (2 * std)  # Position within 2-sigma bands
                    features['bb_position'] = bb_position
                    
                    # Squeeze indicator (low volatility)
                    features['bb_squeeze'] = 1.0 / max(std / ma, 1e-8)  # Higher when volatility is low
        
        except Exception as e:
            logger.error(f"Error computing Bollinger squeeze for {symbol}: {e}")
        
        return features
    
    def _check_data_quality(self, symbol: str) -> bool:
        """Check if data quality is sufficient for feature computation"""
        try:
            # Check key buffers have sufficient data
            required_buffers = ['mid_price', 'bid_size', 'ask_size']
            
            for buffer_name in required_buffers:
                buffer = self.buffers[symbol][buffer_name]
                if not buffer.is_sufficient(min_pct=0.5):
                    return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking data quality for {symbol}: {e}")
            return False
    
    def _store_features(self, feature_vector: FeatureVector):
        """Store feature vector in Redis and ClickHouse"""
        try:
            # Store in Redis for fast access
            key = f"features:{feature_vector.symbol}"
            data = orjson.dumps(asdict(feature_vector))
            self.redis_manager.client.setex(key, 30, data)  # 30 second TTL
            
            # Store in ClickHouse for historical analysis
            self.executor.submit(self._store_features_clickhouse, feature_vector)
            
            # Update metrics
            self.metrics_collector.increment_counter("features_computed", 
                                                   {"symbol": feature_vector.symbol})
            self.metrics_collector.observe_histogram("feature_count", len(feature_vector.features))
        
        except Exception as e:
            logger.error(f"Error storing features: {e}")
    
    def _store_features_clickhouse(self, feature_vector: FeatureVector):
        """Store features in ClickHouse (runs in thread pool)"""
        try:
            self.clickhouse_manager.insert_features(feature_vector)
        except Exception as e:
            logger.error(f"Error storing features in ClickHouse: {e}")
    
    async def get_features(self, symbol: str, timestamp: Optional[int] = None) -> Optional[FeatureVector]:
        """Get latest features for a symbol"""
        try:
            # Check cache first
            if symbol in self.feature_cache:
                cached_features = self.feature_cache[symbol]
                if time.time() - cached_features.timestamp / 1000 < self.cache_ttl:
                    return cached_features
            
            # Try Redis
            key = f"features:{symbol}"
            data_json = self.redis_manager.client.get(key)
            
            if data_json:
                data = orjson.loads(data_json)
                return FeatureVector(**data)
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up feature service...")
        
        self.shutdown_flag = True
        
        if hasattr(self, 'feature_thread'):
            self.feature_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        
        logger.info("Feature service cleanup complete")

async def main():
    """Main entry point for feature service"""
    feature_service = FeatureService()
    
    try:
        await feature_service.initialize()
        
        # Keep service running
        while True:
            await asyncio.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        await feature_service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
