"""
Database models for crypto surge prediction system.
Stores signal history, predictions, and model metadata.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class ModelVersion(Base):
    """Model version tracking and metadata"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'lightgbm', 'tcn', etc.
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    is_canary = Column(Boolean, default=False, nullable=False)
    canary_percentage = Column(Float, default=0.0)
    
    # Model configuration
    config = Column(JSON)  # Model hyperparameters and settings
    metrics = Column(JSON)  # Training metrics: PR-AUC, Hit@TopK, etc.
    
    # Calibration info
    calibration_method = Column(String(50))  # 'isotonic', 'platt', etc.
    calibration_ece = Column(Float)  # Expected calibration error
    
    # Deployment info
    deployed_at = Column(DateTime)
    deployed_by = Column(String(100))
    rollback_version = Column(String(50))
    
    # Relationships
    predictions = relationship("Prediction", back_populates="model")
    
    __table_args__ = (
        Index('idx_model_active', 'is_active', 'version'),
    )

class Signal(Base):
    """Generated trading signals with full context"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    exchange_time = Column(DateTime, nullable=False)
    ingest_time = Column(DateTime)
    infer_time = Column(DateTime)
    
    # Symbol and horizon
    symbol = Column(String(20), nullable=False, index=True)
    horizon_min = Column(Integer, nullable=False)  # 5, 10, 30
    
    # Decision
    decision = Column(String(20), nullable=False)  # LONG, SHORT, WAIT
    tier = Column(String(10), nullable=False)  # A, B, none
    
    # Probabilities
    p_up = Column(Float, nullable=False)
    p_up_ci_low = Column(Float)
    p_up_ci_high = Column(Float)
    
    # Utility
    expected_return = Column(Float)
    estimated_cost = Column(Float)
    net_utility = Column(Float)
    
    # Thresholds used
    tau_threshold = Column(Float)
    kappa_threshold = Column(Float)
    theta_up = Column(Float)
    theta_dn = Column(Float)
    
    # Market regime
    regime = Column(String(50))
    volatility = Column(Float)
    
    # Top features (JSON)
    features_top5 = Column(JSON)
    
    # Quality
    quality_flags = Column(JSON)
    sla_latency_ms = Column(Float)
    
    # Model info
    model_version = Column(String(50), ForeignKey('model_versions.version'))
    feature_version = Column(String(50))
    cost_model_version = Column(String(50))
    
    # Cooldown
    cooldown_until = Column(DateTime)
    
    # Outcome (filled later)
    actual_outcome = Column(String(20))  # WIN, LOSS, NEUTRAL
    actual_return = Column(Float)
    actual_peak_time = Column(Integer)  # minutes to peak
    
    # Relationships
    prediction = relationship("Prediction", back_populates="signal", uselist=False)
    
    __table_args__ = (
        Index('idx_signal_symbol_time', 'symbol', 'created_at'),
        Index('idx_signal_decision', 'decision', 'tier'),
    )

class Prediction(Base):
    """Detailed prediction data with feature attribution"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), ForeignKey('signals.signal_id'), unique=True, nullable=False)
    model_version = Column(String(50), ForeignKey('model_versions.version'), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # All horizon predictions
    predictions_5m = Column(JSON)  # {'p_up': ..., 'confidence': ...}
    predictions_10m = Column(JSON)
    predictions_30m = Column(JSON)
    
    # Raw model output (pre-calibration)
    raw_score = Column(Float)
    calibrated_score = Column(Float)
    
    # Feature values at prediction time
    features = Column(JSON)  # Full feature vector
    
    # SHAP attributions
    shap_values = Column(JSON)
    shap_base_value = Column(Float)
    
    # Cost breakdown
    cost_breakdown = Column(JSON)  # {'spread': ..., 'impact': ..., 'funding': ...}
    
    # Data lineage
    data_window_start = Column(DateTime)
    data_window_end = Column(DateTime)
    data_quality_score = Column(Float)
    
    # Relationships
    signal = relationship("Signal", back_populates="prediction")
    model = relationship("ModelVersion", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_model_time', 'model_version', 'created_at'),
    )

class PerformanceMetric(Base):
    """Time-series performance metrics for monitoring"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    model_version = Column(String(50), ForeignKey('model_versions.version'))
    
    # Windowed metrics (e.g., last hour, last day)
    window_size = Column(String(20), nullable=False)  # '1h', '24h', '7d'
    
    # Classification metrics
    pr_auc = Column(Float)
    hit_at_top_k = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # Financial metrics
    avg_utility = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Operational metrics
    false_positive_rate = Column(Float)
    calibration_error = Column(Float)
    brier_score = Column(Float)
    
    # Latency metrics
    p50_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    
    # System health
    signal_count = Column(Integer)
    quality_flag_rate = Column(Float)
    
    __table_args__ = (
        Index('idx_metrics_model_window', 'model_version', 'window_size', 'timestamp'),
    )

class ABTest(Base):
    """A/B testing configuration and results"""
    __tablename__ = 'ab_tests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    test_name = Column(String(100), unique=True, nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    
    is_active = Column(Boolean, default=False, nullable=False)
    
    # Test configuration
    control_version = Column(String(50), ForeignKey('model_versions.version'))
    treatment_version = Column(String(50), ForeignKey('model_versions.version'))
    traffic_split = Column(Float, default=0.5)  # 0.0 to 1.0
    
    # Test parameters
    test_config = Column(JSON)  # Threshold variations, etc.
    
    # Results
    control_metrics = Column(JSON)
    treatment_metrics = Column(JSON)
    statistical_significance = Column(Float)
    winner = Column(String(20))  # 'control', 'treatment', 'inconclusive'
    
    # Decision
    decision_made_at = Column(DateTime)
    decision = Column(Text)  # Explanation of decision

class AuditLog(Base):
    """Audit trail for system changes and decisions"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    event_type = Column(String(50), nullable=False, index=True)
    user = Column(String(100))
    
    # What changed
    entity_type = Column(String(50))  # 'model', 'config', 'threshold', etc.
    entity_id = Column(String(100))
    
    # Change details
    action = Column(String(50))  # 'deploy', 'rollback', 'update', etc.
    old_value = Column(JSON)
    new_value = Column(JSON)
    
    # Context
    reason = Column(Text)
    event_metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_audit_type_time', 'event_type', 'timestamp'),
    )
