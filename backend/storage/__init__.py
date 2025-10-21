"""
Storage package for crypto surge prediction system.

This package provides storage clients for:
- Redis: Hot cache and real-time data
- ClickHouse: Time series data and analytics
"""

from .redis_client import RedisManager
from .clickhouse_client import ClickHouseManager

__all__ = [
    'RedisManager',
    'ClickHouseManager'
]
