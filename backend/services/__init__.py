"""
Unified services package for crypto surge prediction system.

This package provides a clean interface to core services:
- InferenceService: ONNX-based model inference
- FeatureService: Real-time feature computation
- BacktestService: Strategy backtesting

Supports dual-mode operation:
- Demo mode: Uses mock data for development
- Production mode: Real ONNX inference with live features
"""

from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded service instances
_inference_service: Optional[object] = None
_feature_service: Optional[object] = None
_backtest_service: Optional[object] = None

def get_mode() -> str:
    """Get current operating mode from environment"""
    return os.getenv("API_MODE", "demo").lower()

def is_production_mode() -> bool:
    """Check if running in production mode"""
    return get_mode() == "production"

def get_inference_service():
    """Get or create inference service instance (singleton)"""
    global _inference_service
    
    if not is_production_mode():
        logger.warning("Inference service requested in demo mode - returning None")
        return None
    
    if _inference_service is None:
        from backend.inference_service import InferenceService
        _inference_service = InferenceService()
        logger.info("Initialized InferenceService")
    
    return _inference_service

def get_feature_service():
    """Get or create feature service instance (singleton)"""
    global _feature_service
    
    if not is_production_mode():
        logger.warning("Feature service requested in demo mode - returning None")
        return None
    
    if _feature_service is None:
        from backend.feature_service import FeatureService
        _feature_service = FeatureService()
        logger.info("Initialized FeatureService")
    
    return _feature_service

def get_backtest_service():
    """Get or create backtest service instance (singleton)"""
    global _backtest_service
    
    if _backtest_service is None:
        from backend.backtest_service import BacktestService
        _backtest_service = BacktestService()
        logger.info("Initialized BacktestService")
    
    return _backtest_service

# Export core classes for direct import if needed
__all__ = [
    'get_inference_service',
    'get_feature_service',
    'get_backtest_service',
    'get_mode',
    'is_production_mode'
]
