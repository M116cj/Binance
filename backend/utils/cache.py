"""
LRU Cache with TTL support for API response caching.

This module provides an async-compatible caching system with:
- LRU (Least Recently Used) eviction policy
- TTL (Time To Live) expiration
- Cache key generation utilities
- Thread-safe operations
"""

import asyncio
import time
import hashlib
import json
from collections import OrderedDict
from typing import Any, Optional, Callable, Dict
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 10.0):
        """
        Initialize TTL cache.
        
        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict = OrderedDict()
        self._lock = asyncio.Lock()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if not expired.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, expiry_time = self._cache[key]
            
            # Check if expired
            if time.time() > expiry_time:
                del self._cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None):
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        async with self._lock:
            ttl = ttl if ttl is not None else self.default_ttl
            expiry_time = time.time() + ttl
            
            # Remove oldest if at capacity
            if key not in self._cache and len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self.evictions += 1
            
            self._cache[key] = (value, expiry_time)
            self._cache.move_to_end(key)
    
    async def delete(self, key: str):
        """Delete a key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    async def clear(self):
        """Clear all cache entries"""
        async with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    async def cleanup_expired(self):
        """Remove expired entries from cache"""
        async with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time > expiry
            ]
            for key in expired_keys:
                del self._cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": round(hit_rate, 2),
            "total_requests": total_requests
        }


def generate_cache_key(prefix: str, **kwargs) -> str:
    """
    Generate a consistent cache key from function arguments.
    
    Args:
        prefix: Key prefix (usually function name)
        **kwargs: Key-value pairs to include in key
        
    Returns:
        MD5 hash-based cache key
    """
    # Sort kwargs for consistency
    sorted_items = sorted(kwargs.items())
    key_data = f"{prefix}:" + ":".join(f"{k}={v}" for k, v in sorted_items)
    
    # Hash for compact key
    key_hash = hashlib.md5(key_data.encode()).hexdigest()
    return f"{prefix}:{key_hash}"


def cache_response(cache: TTLCache, ttl: Optional[float] = None, key_prefix: Optional[str] = None):
    """
    Decorator to cache async function responses.
    
    Args:
        cache: TTLCache instance to use
        ttl: Time-to-live for this cached response
        key_prefix: Prefix for cache key (uses function name if None)
        
    Example:
        @cache_response(my_cache, ttl=10.0)
        async def get_data(symbol: str):
            return expensive_operation(symbol)
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            prefix = key_prefix or func.__name__
            
            # Include both args and kwargs in key
            key_dict = {}
            if args:
                key_dict['_args'] = str(args)
            key_dict.update(kwargs)
            
            cache_key = generate_cache_key(prefix, **key_dict)
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {prefix}: {cache_key}")
                return cached_value
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {prefix}: {cache_key}")
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance (10 seconds TTL, 1000 item capacity)
global_cache = TTLCache(max_size=1000, default_ttl=10.0)


async def start_cache_cleanup_task(cache: TTLCache, interval: float = 60.0):
    """
    Background task to periodically clean up expired cache entries.
    
    Args:
        cache: Cache instance to clean
        interval: Cleanup interval in seconds
    """
    while True:
        await asyncio.sleep(interval)
        await cache.cleanup_expired()
        stats = cache.get_stats()
        logger.info(f"Cache stats: {stats}")
