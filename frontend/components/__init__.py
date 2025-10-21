"""
Frontend components package for Crypto Surge Prediction System.

This package contains specialized Streamlit components for rendering
the 7 different report types with optimized visualizations and interactions.
"""

from .signal_card import SignalCard
from .regime_state import RegimeState
from .probability_window import ProbabilityWindow
from .cost_capacity import CostCapacity
from .backtest_performance import BacktestPerformance
from .calibration_analysis import CalibrationAnalysis
from .attribution_comparison import AttributionComparison

__all__ = [
    'SignalCard',
    'RegimeState',
    'ProbabilityWindow', 
    'CostCapacity',
    'BacktestPerformance',
    'CalibrationAnalysis',
    'AttributionComparison'
]

