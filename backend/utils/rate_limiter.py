"""
Token bucket rate limiter with concurrent request control.

This module provides:
- Token bucket algorithm for rate limiting
- Per-client rate limiting
- Global concurrent request limiting
- FastAPI middleware integration
"""

import asyncio
import time
from typing import Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """Token bucket for rate limiting a single client"""
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.
        
        Args:
            rate: Tokens per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if rate limit exceeded
        """
        async with self._lock:
            now = time.time()
            
            # Add tokens based on time elapsed
            elapsed = now - self.last_update
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, timeout: float = 5.0) -> bool:
        """
        Wait for a token to become available.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if token acquired, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.consume():
                return True
            await asyncio.sleep(0.01)  # Small delay before retry
        
        return False


class RateLimiter:
    """Rate limiter with per-client tracking and global concurrency control"""
    
    def __init__(
        self,
        requests_per_minute: int = 300,
        max_concurrent: int = 100,
        burst_multiplier: float = 1.5
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per client
            max_concurrent: Maximum concurrent requests globally
            burst_multiplier: Allow burst up to this multiple of rate
        """
        self.requests_per_minute = requests_per_minute
        self.max_concurrent = max_concurrent
        
        # Convert to requests per second
        self.rate = requests_per_minute / 60.0
        self.capacity = int(self.rate * burst_multiplier)
        
        # Per-client buckets
        self._buckets: Dict[str, TokenBucket] = {}
        self._buckets_lock = asyncio.Lock()
        
        # Global concurrency tracking
        self._active_requests = 0
        self._concurrency_lock = asyncio.Lock()
        
        # Metrics
        self.total_requests = 0
        self.rate_limited = 0
        self.concurrency_limited = 0
    
    async def _get_bucket(self, client_id: str) -> TokenBucket:
        """Get or create token bucket for client"""
        async with self._buckets_lock:
            if client_id not in self._buckets:
                self._buckets[client_id] = TokenBucket(self.rate, self.capacity)
            return self._buckets[client_id]
    
    async def acquire(self, client_id: str = "default") -> bool:
        """
        Try to acquire permission to make a request.
        
        Args:
            client_id: Identifier for the client
            
        Returns:
            True if request allowed, False if rate limited
        """
        self.total_requests += 1
        
        # Check global concurrency limit
        async with self._concurrency_lock:
            if self._active_requests >= self.max_concurrent:
                self.concurrency_limited += 1
                logger.warning(
                    f"Concurrent request limit reached: {self._active_requests}/{self.max_concurrent}"
                )
                return False
        
        # Check per-client rate limit
        bucket = await self._get_bucket(client_id)
        allowed = await bucket.consume()
        
        if not allowed:
            self.rate_limited += 1
            logger.warning(f"Rate limit exceeded for client: {client_id}")
            return False
        
        # Increment active requests
        async with self._concurrency_lock:
            self._active_requests += 1
        
        return True
    
    async def release(self):
        """Release a request slot (call after request completes)"""
        async with self._concurrency_lock:
            self._active_requests = max(0, self._active_requests - 1)
    
    async def cleanup_stale_buckets(self, max_age_seconds: float = 3600.0):
        """Remove inactive client buckets"""
        async with self._buckets_lock:
            current_time = time.time()
            stale_clients = [
                client_id for client_id, bucket in self._buckets.items()
                if current_time - bucket.last_update > max_age_seconds
            ]
            
            for client_id in stale_clients:
                del self._buckets[client_id]
            
            if stale_clients:
                logger.info(f"Cleaned up {len(stale_clients)} stale rate limit buckets")
    
    def get_stats(self) -> Dict:
        """Get rate limiter statistics"""
        return {
            "active_requests": self._active_requests,
            "max_concurrent": self.max_concurrent,
            "total_requests": self.total_requests,
            "rate_limited": self.rate_limited,
            "concurrency_limited": self.concurrency_limited,
            "active_clients": len(self._buckets),
            "requests_per_minute": self.requests_per_minute
        }


# Global rate limiter instance
global_rate_limiter = RateLimiter(
    requests_per_minute=300,
    max_concurrent=100,
    burst_multiplier=1.5
)


class RateLimitContext:
    """Context manager for rate-limited operations"""
    
    def __init__(self, rate_limiter: RateLimiter, client_id: str = "default"):
        self.rate_limiter = rate_limiter
        self.client_id = client_id
        self.acquired = False
    
    async def __aenter__(self):
        self.acquired = await self.rate_limiter.acquire(self.client_id)
        if not self.acquired:
            raise RateLimitExceeded(
                f"Rate limit exceeded for client: {self.client_id}"
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            await self.rate_limiter.release()


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded"""
    pass


async def start_rate_limiter_cleanup_task(rate_limiter: RateLimiter, interval: float = 600.0):
    """
    Background task to clean up stale rate limiter data.
    
    Args:
        rate_limiter: RateLimiter instance
        interval: Cleanup interval in seconds
    """
    while True:
        await asyncio.sleep(interval)
        await rate_limiter.cleanup_stale_buckets()
        stats = rate_limiter.get_stats()
        logger.info(f"Rate limiter stats: {stats}")
