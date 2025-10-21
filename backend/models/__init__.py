"""
Machine learning models package for crypto surge prediction system.

This package contains:
- Feature engineering components
- Labeling algorithms (triple-barrier, etc.)
- Cost modeling for execution
- Model schemas and data structures
"""

from .features import FeatureEngine
from .labeling import LabelGenerator
from .cost_model import CostModel
from .schemas import *

__all__ = [
    'FeatureEngine',
    'LabelGenerator', 
    'CostModel'
]
