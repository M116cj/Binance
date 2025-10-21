"""
Configuration settings for the crypto surge prediction system.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    ENV: str = Field(default="development", env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Redis configuration
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    REDIS_MAX_CONNECTIONS: int = Field(default=50, env="REDIS_MAX_CONNECTIONS")
    
    # ClickHouse configuration
    CLICKHOUSE_HOST: str = Field(default="localhost", env="CLICKHOUSE_HOST")
    CLICKHOUSE_PORT: int = Field(default=8123, env="CLICKHOUSE_PORT")
    CLICKHOUSE_USER: str = Field(default="default", env="CLICKHOUSE_USER")
    CLICKHOUSE_PASSWORD: Optional[str] = Field(default=None, env="CLICKHOUSE_PASSWORD")
    CLICKHOUSE_DATABASE: str = Field(default="crypto_surge", env="CLICKHOUSE_DATABASE")
    
    # Binance configuration
    BINANCE_API_KEY: Optional[str] = Field(default=None, env="BINANCE_API_KEY")
    BINANCE_API_SECRET: Optional[str] = Field(default=None, env="BINANCE_API_SECRET")
    BINANCE_TESTNET: bool = Field(default=False, env="BINANCE_TESTNET")
    
    # Model paths
    MODEL_PATH: str = Field(default="./models/lightgbm_model.onnx", env="MODEL_PATH")
    CALIBRATOR_PATH: str = Field(default="./models/isotonic_calibrator.pkl", env="CALIBRATOR_PATH")
    
    # Feature engineering
    FEATURE_VERSION: str = Field(default="1.0.0", env="FEATURE_VERSION")
    MAX_FEATURE_AGE_MS: int = Field(default=5000, env="MAX_FEATURE_AGE_MS")
    
    # Inference configuration
    BATCH_SIZE: int = Field(default=32, env="BATCH_SIZE")
    MAX_BATCH_WAIT_MS: int = Field(default=25, env="MAX_BATCH_WAIT_MS")
    INFERENCE_TIMEOUT_MS: int = Field(default=100, env="INFERENCE_TIMEOUT_MS")
    USE_FP16: bool = Field(default=True, env="USE_FP16")
    
    # Latency SLOs
    P99_LATENCY_MS: int = Field(default=800, env="P99_LATENCY_MS")
    P95_LATENCY_MS: int = Field(default=500, env="P95_LATENCY_MS")
    
    # Throughput targets
    TARGET_RPS: int = Field(default=300, env="TARGET_RPS")
    
    # Backtest configuration
    BACKTEST_DAYS: int = Field(default=30, env="BACKTEST_DAYS")
    BACKTEST_RESOLUTION: str = Field(default="1m", env="BACKTEST_RESOLUTION")
    
    # Monitoring
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Auto-degradation
    ENABLE_AUTO_DEGRADATION: bool = Field(default=True, env="ENABLE_AUTO_DEGRADATION")
    QUEUE_BACKPRESSURE_THRESHOLD: int = Field(default=1000, env="QUEUE_BACKPRESSURE_THRESHOLD")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
