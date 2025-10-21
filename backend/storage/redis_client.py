"""
High-performance Redis client for crypto surge prediction system.
Handles hot cache, real-time data, and cost lookup tables with pipelining.
"""

import asyncio
import redis
import redis.asyncio as aioredis
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import orjson
from dataclasses import asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from contextlib import asynccontextmanager
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RedisManager:
    """Redis manager with sync/async support and connection pooling"""
    
    def __init__(self):
        # Connection parameters
        self.host = os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", 6379))
        self.db = int(os.getenv("REDIS_DB", 0))
        self.password = os.getenv("REDIS_PASSWORD", None)
        self.max_connections = int(os.getenv("REDIS_MAX_CONNECTIONS", 50))
        
        # Connection pools
        self.sync_pool = None
        self.async_pool = None
        self.client = None
        self.async_client = None
        
        # Performance tracking
        self.operation_counts = {
            'get': 0,
            'set': 0,
            'pipeline': 0,
            'errors': 0
        }
        
        # Pipeline batching
        self.pipeline_batch_size = 100
        self.pipeline_timeout_ms = 50
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Symbol ID mapping for key compression
        self.symbol_id_map: Dict[str, int] = {}
        self.id_symbol_map: Dict[int, str] = {}
        self.next_symbol_id = 1
    
    async def initialize(self):
        """Initialize Redis connections"""
        logger.info(f"Initializing Redis connection to {self.host}:{self.port}")
        
        try:
            # Create connection pools
            self.sync_pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False  # Handle bytes directly for performance
            )
            
            self.async_pool = aioredis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections
            )
            
            # Create clients
            self.client = redis.Redis(connection_pool=self.sync_pool)
            self.async_client = aioredis.Redis(connection_pool=self.async_pool)
            
            # Test connections
            await self._test_connections()
            
            # Load symbol ID mappings
            await self._load_symbol_mappings()
            
            logger.info("Redis connections established successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _test_connections(self):
        """Test both sync and async connections"""
        # Test sync connection
        try:
            self.client.ping()
            logger.info("Sync Redis connection test: OK")
        except Exception as e:
            logger.error(f"Sync Redis connection test failed: {e}")
            raise
        
        # Test async connection
        try:
            await self.async_client.ping()
            logger.info("Async Redis connection test: OK")
        except Exception as e:
            logger.error(f"Async Redis connection test failed: {e}")
            raise
    
    async def _load_symbol_mappings(self):
        """Load symbol to ID mappings for key compression"""
        try:
            mappings = await self.async_client.hgetall("symbol_id_map")
            
            for symbol_bytes, id_bytes in mappings.items():
                symbol = symbol_bytes.decode('utf-8')
                symbol_id = int(id_bytes.decode('utf-8'))
                
                self.symbol_id_map[symbol] = symbol_id
                self.id_symbol_map[symbol_id] = symbol
                
                if symbol_id >= self.next_symbol_id:
                    self.next_symbol_id = symbol_id + 1
            
            logger.info(f"Loaded {len(self.symbol_id_map)} symbol mappings")
            
        except Exception as e:
            logger.warning(f"Could not load symbol mappings: {e}")
    
    def _get_symbol_id(self, symbol: str) -> int:
        """Get or create symbol ID for key compression"""
        if symbol not in self.symbol_id_map:
            with self.lock:
                if symbol not in self.symbol_id_map:
                    symbol_id = self.next_symbol_id
                    self.symbol_id_map[symbol] = symbol_id
                    self.id_symbol_map[symbol_id] = symbol
                    self.next_symbol_id += 1
                    
                    # Store in Redis
                    try:
                        self.client.hset("symbol_id_map", symbol, symbol_id)
                        self.client.hset("id_symbol_map", symbol_id, symbol)
                    except Exception as e:
                        logger.error(f"Failed to store symbol mapping: {e}")
        
        return self.symbol_id_map[symbol]
    
    def _get_compressed_key(self, key_template: str, symbol: str, **kwargs) -> str:
        """Generate compressed key using symbol ID"""
        symbol_id = self._get_symbol_id(symbol)
        
        # Replace symbol with ID in key template
        if 'symbol' in kwargs:
            kwargs['symbol'] = symbol_id
        elif '{symbol}' in key_template:
            key_template = key_template.replace('{symbol}', str(symbol_id))
        
        return key_template.format(**kwargs)
    
    # Synchronous operations
    def set_with_ttl(self, key: str, value: Union[str, bytes, dict], ttl_seconds: int) -> bool:
        """Set key with TTL"""
        try:
            if isinstance(value, dict):
                value = orjson.dumps(value)
            elif isinstance(value, str):
                value = value.encode('utf-8')
            
            result = self.client.setex(key, ttl_seconds, value)
            self.operation_counts['set'] += 1
            return result
            
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            self.operation_counts['errors'] += 1
            return False
    
    def get_value(self, key: str) -> Optional[bytes]:
        """Get value by key"""
        try:
            value = self.client.get(key)
            self.operation_counts['get'] += 1
            return value
            
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            self.operation_counts['errors'] += 1
            return None
    
    def get_json(self, key: str) -> Optional[dict]:
        """Get and decode JSON value"""
        try:
            value = self.get_value(key)
            if value:
                return orjson.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Error decoding JSON for key {key}: {e}")
            return None
    
    def pipeline_execute(self, operations: List[Tuple[str, str, Any, Optional[int]]]) -> List[Any]:
        """
        Execute operations in pipeline for better performance.
        operations: List of (operation, key, value, ttl) tuples
        """
        try:
            pipe = self.client.pipeline()
            
            for operation, key, value, ttl in operations:
                if operation == 'set':
                    if isinstance(value, dict):
                        value = orjson.dumps(value)
                    elif isinstance(value, str):
                        value = value.encode('utf-8')
                    
                    if ttl:
                        pipe.setex(key, ttl, value)
                    else:
                        pipe.set(key, value)
                
                elif operation == 'get':
                    pipe.get(key)
                
                elif operation == 'zadd':
                    pipe.zadd(key, value)
                    if ttl:
                        pipe.expire(key, ttl)
                
                elif operation == 'expire':
                    pipe.expire(key, ttl or 3600)
            
            results = pipe.execute()
            self.operation_counts['pipeline'] += 1
            return results
            
        except Exception as e:
            logger.error(f"Pipeline execution error: {e}")
            self.operation_counts['errors'] += 1
            return []
    
    # Market data specific operations
    def store_market_data(self, symbol: str, stream_type: str, data: dict, 
                         timestamp: int, ttl_seconds: int = 10) -> bool:
        """Store market data with compressed key"""
        try:
            key = self._get_compressed_key("md:{symbol}:{stream}", symbol, stream=stream_type)
            
            market_data = {
                'data': data,
                'timestamp': timestamp,
                'symbol': symbol,
                'stream_type': stream_type
            }
            
            return self.set_with_ttl(key, market_data, ttl_seconds)
            
        except Exception as e:
            logger.error(f"Error storing market data for {symbol}: {e}")
            return False
    
    def store_features(self, symbol: str, features: dict, timestamp: int, 
                      ttl_seconds: int = 30) -> bool:
        """Store feature vector"""
        try:
            key = self._get_compressed_key("features:{symbol}", symbol)
            
            feature_data = {
                'features': features,
                'timestamp': timestamp,
                'symbol': symbol
            }
            
            return self.set_with_ttl(key, feature_data, ttl_seconds)
            
        except Exception as e:
            logger.error(f"Error storing features for {symbol}: {e}")
            return False
    
    def get_features(self, symbol: str) -> Optional[dict]:
        """Get latest features for symbol"""
        try:
            key = self._get_compressed_key("features:{symbol}", symbol)
            return self.get_json(key)
            
        except Exception as e:
            logger.error(f"Error getting features for {symbol}: {e}")
            return None
    
    def store_cost_lookup(self, regime: str, symbol: str, timeframe: str, 
                         cost_data: dict, ttl_minutes: int = 15) -> bool:
        """Store cost lookup table entry"""
        try:
            key = f"cost:{regime}:{self._get_symbol_id(symbol)}:{timeframe}"
            
            cost_entry = {
                'cost_data': cost_data,
                'regime': regime,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': int(time.time() * 1000)
            }
            
            return self.set_with_ttl(key, cost_entry, ttl_minutes * 60)
            
        except Exception as e:
            logger.error(f"Error storing cost lookup for {symbol}: {e}")
            return False
    
    def get_cost_lookup(self, regime: str, symbol: str, timeframe: str) -> Optional[dict]:
        """Get cost lookup table entry"""
        try:
            key = f"cost:{regime}:{self._get_symbol_id(symbol)}:{timeframe}"
            return self.get_json(key)
            
        except Exception as e:
            logger.error(f"Error getting cost lookup for {symbol}: {e}")
            return None
    
    def store_time_series(self, symbol: str, stream_type: str, timestamp: int, 
                         data: dict, max_entries: int = 1000) -> bool:
        """Store time series data using sorted sets"""
        try:
            key = self._get_compressed_key("ts:{symbol}:{stream}", symbol, stream=stream_type)
            score = timestamp  # Use timestamp as score for sorting
            value = orjson.dumps(data)
            
            pipe = self.client.pipeline()
            pipe.zadd(key, {value: score})
            pipe.zremrangebyrank(key, 0, -(max_entries + 1))  # Keep only recent entries
            pipe.expire(key, 3600)  # 1 hour expiry
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing time series for {symbol}: {e}")
            return False
    
    def get_time_series_window(self, symbol: str, stream_type: str, 
                              start_time: int, end_time: int) -> List[dict]:
        """Get time series data within time window"""
        try:
            key = self._get_compressed_key("ts:{symbol}:{stream}", symbol, stream=stream_type)
            
            # Get data within score range (timestamp range)
            raw_data = self.client.zrangebyscore(key, start_time, end_time)
            
            result = []
            for item in raw_data:
                try:
                    data = orjson.loads(item)
                    result.append(data)
                except Exception as e:
                    logger.warning(f"Failed to decode time series item: {e}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting time series window for {symbol}: {e}")
            return []
    
    def get_recent_time_series(self, symbol: str, stream_type: str, 
                              count: int = 100) -> List[dict]:
        """Get most recent time series entries"""
        try:
            key = self._get_compressed_key("ts:{symbol}:{stream}", symbol, stream=stream_type)
            
            # Get most recent entries
            raw_data = self.client.zrevrange(key, 0, count - 1)
            
            result = []
            for item in raw_data:
                try:
                    data = orjson.loads(item)
                    result.append(data)
                except Exception as e:
                    logger.warning(f"Failed to decode time series item: {e}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting recent time series for {symbol}: {e}")
            return []
    
    # Async operations
    async def async_set_with_ttl(self, key: str, value: Union[str, bytes, dict], 
                               ttl_seconds: int) -> bool:
        """Async set key with TTL"""
        try:
            if isinstance(value, dict):
                value = orjson.dumps(value)
            elif isinstance(value, str):
                value = value.encode('utf-8')
            
            result = await self.async_client.setex(key, ttl_seconds, value)
            return result
            
        except Exception as e:
            logger.error(f"Error async setting key {key}: {e}")
            return False
    
    async def async_get_json(self, key: str) -> Optional[dict]:
        """Async get and decode JSON value"""
        try:
            value = await self.async_client.get(key)
            if value:
                return orjson.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Error async getting JSON for key {key}: {e}")
            return None
    
    async def async_pipeline_execute(self, operations: List[Tuple[str, str, Any, Optional[int]]]) -> List[Any]:
        """Async pipeline execution"""
        try:
            pipe = self.async_client.pipeline()
            
            for operation, key, value, ttl in operations:
                if operation == 'set':
                    if isinstance(value, dict):
                        value = orjson.dumps(value)
                    elif isinstance(value, str):
                        value = value.encode('utf-8')
                    
                    if ttl:
                        pipe.setex(key, ttl, value)
                    else:
                        pipe.set(key, value)
                
                elif operation == 'get':
                    pipe.get(key)
                
                elif operation == 'zadd':
                    pipe.zadd(key, value)
                    if ttl:
                        pipe.expire(key, ttl)
            
            results = await pipe.execute()
            return results
            
        except Exception as e:
            logger.error(f"Async pipeline execution error: {e}")
            return []
    
    # Monitoring and metrics
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get Redis performance statistics"""
        try:
            info = self.client.info()
            
            return {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'operations_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'operation_counts': self.operation_counts.copy(),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1)
                )
            }
            
        except Exception as e:
            logger.error(f"Error getting Redis stats: {e}")
            return {'error': str(e)}
    
    def cleanup_expired_keys(self, pattern: str = "*", max_scan_count: int = 1000):
        """Cleanup expired keys matching pattern"""
        try:
            cursor = 0
            cleaned_count = 0
            
            while True:
                cursor, keys = self.client.scan(cursor, match=pattern, count=max_scan_count)
                
                if keys:
                    # Check TTL and remove keys with no expiration
                    pipe = self.client.pipeline()
                    for key in keys:
                        pipe.ttl(key)
                    
                    ttls = pipe.execute()
                    
                    # Remove keys with TTL = -1 (no expiration) if they're old
                    for key, ttl in zip(keys, ttls):
                        if ttl == -1:  # No expiration set
                            # Add expiration to prevent memory leaks
                            self.client.expire(key, 3600)  # 1 hour default
                            cleaned_count += 1
                
                if cursor == 0:
                    break
            
            logger.info(f"Added expiration to {cleaned_count} keys")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired keys: {e}")
            return 0
    
    def get_memory_usage(self, sample_size: int = 100) -> Dict[str, Any]:
        """Analyze memory usage by key patterns"""
        try:
            cursor = 0
            pattern_memory = defaultdict(int)
            total_keys = 0
            
            while total_keys < sample_size:
                cursor, keys = self.client.scan(cursor, count=100)
                
                for key in keys:
                    try:
                        memory = self.client.memory_usage(key)
                        if memory:
                            # Extract pattern from key
                            if b':' in key:
                                pattern = key.split(b':')[0].decode('utf-8')
                            else:
                                pattern = 'other'
                            
                            pattern_memory[pattern] += memory
                            total_keys += 1
                            
                            if total_keys >= sample_size:
                                break
                    
                    except Exception:
                        continue
                
                if cursor == 0:
                    break
            
            return {
                'pattern_memory': dict(pattern_memory),
                'total_sampled_keys': total_keys,
                'sample_size': sample_size
            }
            
        except Exception as e:
            logger.error(f"Error analyzing memory usage: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            start_time = time.time()
            
            # Test ping
            await self.async_client.ping()
            ping_latency = (time.time() - start_time) * 1000
            
            # Get info
            info = await self.async_client.info()
            stats = self.get_performance_stats()
            
            # Check connection pool
            pool_status = {
                'sync_pool_created_connections': self.sync_pool.created_connections,
                'sync_pool_available_connections': len(self.sync_pool._available_connections),
                'sync_pool_in_use_connections': len(self.sync_pool._in_use_connections)
            }
            
            return {
                'status': 'healthy',
                'ping_latency_ms': ping_latency,
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'hit_rate': stats.get('hit_rate', 0),
                'operations_per_sec': info.get('instantaneous_ops_per_sec', 0),
                'pool_status': pool_status,
                'symbol_mappings': len(self.symbol_id_map)
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': int(time.time() * 1000)
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Redis connections...")
        
        try:
            if self.async_client:
                await self.async_client.close()
            
            if self.client:
                self.client.close()
            
            if self.sync_pool:
                self.sync_pool.disconnect()
            
            if self.async_pool:
                await self.async_pool.disconnect()
            
            logger.info("Redis cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during Redis cleanup: {e}")
