"""
Data schemas and structures for the crypto surge prediction system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum
import numpy as np
from datetime import datetime

class StreamType(Enum):
    """Market data stream types"""
    AGG_TRADE = "aggTrade"
    BOOK_TICKER = "bookTicker" 
    DEPTH = "depth"
    KLINE = "kline_1m"
    TICKER = "ticker"
    FUNDING = "markPrice"
    OPEN_INTEREST = "openInterest"
    LIQUIDATION = "forceOrder"

class QualityFlag(Enum):
    """Data quality flags"""
    DEGRADED_TIME_SYNC = "degraded_time_sync"
    DEPTH_GAP = "depth_gap"
    WINDOW_INSUFFICIENT = "window_insufficient"
    MISSING_FEATURES = "missing_features"
    HIGH_LATENCY = "high_latency"
    SEQUENCE_ERROR = "sequence_error"

@dataclass
class TimeSeriesPoint:
    """Basic time series data point"""
    timestamp: int
    value: float
    quality_flags: List[QualityFlag] = field(default_factory=list)

@dataclass
class OrderBookLevel:
    """Order book level (bid or ask)"""
    price: float
    size: float

@dataclass
class OrderBookSnapshot:
    """Order book snapshot"""
    symbol: str
    timestamp: int
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_update_id: int
    sequence_start: Optional[int] = None
    sequence_end: Optional[int] = None

@dataclass 
class Trade:
    """Individual trade"""
    symbol: str
    timestamp: int
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: int

@dataclass
class Kline:
    """Kline/candlestick data"""
    symbol: str
    timestamp: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    number_of_trades: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

@dataclass
class TickerStats:
    """24hr ticker statistics"""
    symbol: str
    timestamp: int
    price_change: float
    price_change_percent: float
    weighted_avg_price: float
    last_price: float
    last_quantity: float
    best_bid_price: float
    best_bid_quantity: float
    best_ask_price: float
    best_ask_quantity: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    quote_volume: float
    open_time: int
    close_time: int
    first_id: int
    last_id: int
    count: int

@dataclass
class FundingRate:
    """Funding rate data"""
    symbol: str
    timestamp: int
    mark_price: float
    index_price: float
    estimated_settle_price: float
    last_funding_rate: float
    next_funding_time: int

@dataclass
class OpenInterest:
    """Open interest data"""
    symbol: str
    timestamp: int
    open_interest: float
    sum_open_interest: float
    sum_open_interest_value: float

@dataclass
class Liquidation:
    """Liquidation order"""
    symbol: str
    timestamp: int
    side: str  # "BUY" or "SELL"
    order_type: str
    time_in_force: str
    quantity: float
    price: float
    avg_price: float
    order_status: str
    last_filled_quantity: float
    filled_accumulated_quantity: float

@dataclass
class MarketRegimeState:
    """Market regime classification"""
    symbol: str
    timestamp: int
    volatility_regime: str  # "low", "medium", "high"
    depth_regime: str       # "thin", "medium", "thick" 
    funding_regime: str     # "negative", "neutral", "positive"
    combined_regime: str
    regime_confidence: float
    regime_stability: float

@dataclass
class FeatureVector:
    """Complete feature vector for ML model"""
    symbol: str
    timestamp: int
    window_start: int
    window_end: int
    
    # Order book features
    queue_imbalance_1: float
    queue_imbalance_5: float
    microprice_deviation: float
    depth_slope_bid: float
    depth_slope_ask: float
    near_touch_void: float
    impact_lambda: float
    
    # Trade flow features
    order_flow_imbalance_10: float
    order_flow_imbalance_30: float
    cvd_slope: float
    buy_cluster_intensity: float
    follow_buy_ratio: float
    
    # Volatility features  
    realized_volatility_1m: float
    realized_volatility_5m: float
    rv_ratio: float
    bollinger_position: float
    bollinger_squeeze: float
    
    # Derivatives features
    funding_delta: float
    oi_pressure: float
    long_short_ratio: float
    lsr_divergence: float
    
    # Liquidation features
    liquidation_density_up: float
    liquidation_density_down: float
    post_liq_gap: float
    
    # Cross-exchange features (optional)
    lead_lag_spread: Optional[float] = None
    arrival_rate_ratio: Optional[float] = None
    
    # Feature metadata
    feature_version: str = "1.0.0"
    quality_flags: List[QualityFlag] = field(default_factory=list)
    feature_count_valid: int = 0

@dataclass
class TripleBarrierLabel:
    """Triple barrier labeling result"""
    symbol: str
    timestamp: int
    label_timestamp: int
    horizon_minutes: int
    theta_up: float
    theta_dn: float
    
    # Label result
    label: int  # -1, 0, 1 for down, timeout, up
    binary_up: bool  # True if up breach occurred first
    
    # Label details
    breach_timestamp: Optional[int] = None
    breach_price: Optional[float] = None
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    
    # Regression targets
    peak_return: float = 0.0
    time_to_peak: float = 0.0
    
    # Metadata
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    holding_period: float = 0.0
    embargo_end: int = 0

@dataclass
class CostEstimate:
    """Execution cost estimate"""
    symbol: str
    timestamp: int
    position_size_usd: float
    horizon_minutes: int
    
    # Cost components
    maker_fee: float
    taker_fee: float
    expected_slippage: float
    slippage_p50: float
    slippage_p95: float
    slippage_p99: float
    funding_cost: float
    opportunity_cost: float
    
    # Total costs
    total_cost_estimate: float
    cost_per_unit: float
    
    # Market impact factors
    impact_lambda: float
    available_liquidity: float
    liquidity_ratio: float
    
    # Model metadata
    cost_model_version: str = "v1.2.0"
    confidence: float = 0.85

@dataclass
class Prediction:
    """ML model prediction"""
    symbol: str
    timestamp: int
    model_timestamp: int
    
    # Core predictions by horizon
    predictions_by_horizon: Dict[int, Dict[str, float]]  # horizon -> {p_up, ci_low, ci_high}
    
    # Expected returns and utilities
    expected_returns: Dict[int, float]  # horizon -> expected_return
    estimated_costs: Dict[int, float]   # horizon -> estimated_cost
    utilities: Dict[int, float]         # horizon -> utility (E[R]/C)
    
    # Decision outputs
    decisions: Dict[int, str]           # horizon -> decision ("A", "B", "none")
    
    # Model explanations
    feature_importance: Dict[str, float]
    shap_values_top5: Dict[str, float]
    prediction_confidence: float
    
    # Metadata
    model_version: str
    feature_version: str
    cost_model_version: str
    regime: str
    capacity_pct: float
    
    # Quality and timing
    quality_flags: List[QualityFlag] = field(default_factory=list)
    sla_latency_ms: float = 0.0
    cooldown_until: Optional[int] = None

@dataclass 
class Signal:
    """Trading signal output"""
    id: str
    symbol: str
    timestamp: int
    
    # Signal properties
    tier: str  # "A", "B"
    horizon_minutes: int
    probability: float
    expected_return: float
    utility: float
    confidence: float
    
    # Signal context
    regime: str
    entry_reason: str
    exit_policy: str
    position_size_recommendation: float
    capacity_utilization: float
    
    # Risk management
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_holding_period: int = 30  # minutes
    
    # Metadata
    model_versions: Dict[str, str] = field(default_factory=dict)
    data_lineage: List[str] = field(default_factory=list)
    quality_score: float = 1.0

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    symbol: str
    start_date: datetime
    end_date: datetime
    config_hash: str
    
    # Performance metrics
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    
    # Signal statistics
    total_signals: int
    signals_taken: int
    signals_profitable: int
    avg_signal_return: float
    median_signal_return: float
    
    # Risk metrics
    value_at_risk_95: float
    expected_shortfall_95: float
    volatility_annual: float
    skewness: float
    kurtosis: float
    
    # Trade analysis
    total_trades: int
    avg_trade_duration_minutes: float
    largest_winner: float
    largest_loser: float
    
    # Time series results
    equity_curve: List[Tuple[int, float]]  # (timestamp, equity)
    drawdown_curve: List[Tuple[int, float]]
    
    # Model performance by regime
    performance_by_regime: Dict[str, Dict[str, float]]
    
    # Execution analysis
    execution_summary: Dict[str, Any]
    slippage_analysis: Dict[str, float]
    
    # Sensitivity analysis
    parameter_sensitivity: Dict[str, Dict[str, float]]

@dataclass
class ModelValidationResult:
    """Model validation and calibration results"""
    model_version: str
    validation_period_start: datetime
    validation_period_end: datetime
    
    # Calibration metrics
    brier_score: float
    expected_calibration_error: float
    reliability_score: float
    resolution: float
    
    # Classification metrics
    precision_at_k: Dict[int, float]  # k -> precision
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    auc_pr: float
    auc_roc: float
    
    # Regression metrics (for return/timing prediction)
    mae_return: float
    rmse_return: float
    mae_timing: float
    rmse_timing: float
    
    # Stability metrics
    feature_stability: Dict[str, float]
    prediction_stability: float
    temporal_consistency: float
    
    # Error analysis
    false_positive_patterns: Dict[str, Any]
    false_negative_patterns: Dict[str, Any]
    error_distribution: Dict[str, float]

# Type aliases for common data structures
TimeSeriesData = List[TimeSeriesPoint]
OrderBookData = List[OrderBookSnapshot]
TradeData = List[Trade]
FeatureMatrix = np.ndarray
LabelVector = np.ndarray
PredictionBatch = List[Prediction]
