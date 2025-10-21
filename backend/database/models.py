"""
加密货币涨跌预测系统的数据库模型
存储信号历史、预测和模型元数据
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, Index, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class ModelVersion(Base):
    """模型版本跟踪和元数据"""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(50), unique=True, nullable=False, index=True)
    model_type = Column(String(50), nullable=False)  # 'lightgbm', 'tcn'等
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=False, nullable=False)
    is_canary = Column(Boolean, default=False, nullable=False)
    canary_percentage = Column(Float, default=0.0)
    
    # 模型配置
    config = Column(JSON)  # 模型超参数和设置
    metrics = Column(JSON)  # 训练指标：PR-AUC, Hit@TopK等
    
    # 校准信息
    calibration_method = Column(String(50))  # 'isotonic', 'platt'等
    calibration_ece = Column(Float)  # 期望校准误差
    
    # 部署信息
    deployed_at = Column(DateTime)
    deployed_by = Column(String(100))
    rollback_version = Column(String(50))
    
    # 关联关系
    predictions = relationship("Prediction", back_populates="model")
    
    __table_args__ = (
        Index('idx_model_active', 'is_active', 'version'),
    )

class Signal(Base):
    """生成的交易信号及完整上下文"""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # 时间相关
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    exchange_time = Column(DateTime, nullable=False)
    ingest_time = Column(DateTime)
    infer_time = Column(DateTime)
    
    # 交易对和时间窗口
    symbol = Column(String(20), nullable=False, index=True)
    horizon_min = Column(Integer, nullable=False)  # 5, 10, 30
    
    # 决策
    decision = Column(String(20), nullable=False)  # LONG, SHORT, WAIT
    tier = Column(String(10), nullable=False)  # A, B, none
    
    # 概率
    p_up = Column(Float, nullable=False)
    p_up_ci_low = Column(Float)
    p_up_ci_high = Column(Float)
    
    # 效用
    expected_return = Column(Float)
    estimated_cost = Column(Float)
    net_utility = Column(Float)
    
    # 使用的阈值
    tau_threshold = Column(Float)
    kappa_threshold = Column(Float)
    theta_up = Column(Float)
    theta_dn = Column(Float)
    
    # 市场状态
    regime = Column(String(50))
    volatility = Column(Float)
    
    # 顶部特征（JSON）
    features_top5 = Column(JSON)
    
    # 质量
    quality_flags = Column(JSON)
    sla_latency_ms = Column(Float)
    
    # 模型信息
    model_version = Column(String(50), ForeignKey('model_versions.version'))
    feature_version = Column(String(50))
    cost_model_version = Column(String(50))
    
    # 冷却期
    cooldown_until = Column(DateTime)
    
    # 结果（稍后填充）
    actual_outcome = Column(String(20))  # WIN, LOSS, NEUTRAL
    actual_return = Column(Float)
    actual_peak_time = Column(Integer)  # 到达峰值的分钟数
    
    # 关联关系
    prediction = relationship("Prediction", back_populates="signal", uselist=False)
    
    __table_args__ = (
        Index('idx_signal_symbol_time', 'symbol', 'created_at'),
        Index('idx_signal_decision', 'decision', 'tier'),
    )

class Prediction(Base):
    """详细的预测数据及特征归因"""
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String(100), ForeignKey('signals.signal_id'), unique=True, nullable=False)
    model_version = Column(String(50), ForeignKey('model_versions.version'), nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 所有时间窗口的预测
    predictions_5m = Column(JSON)  # {'p_up': ..., 'confidence': ...}
    predictions_10m = Column(JSON)
    predictions_30m = Column(JSON)
    
    # 原始模型输出（校准前）
    raw_score = Column(Float)
    calibrated_score = Column(Float)
    
    # 预测时的特征值
    features = Column(JSON)  # 完整特征向量
    
    # SHAP归因
    shap_values = Column(JSON)
    shap_base_value = Column(Float)
    
    # 成本分解
    cost_breakdown = Column(JSON)  # {'spread': ..., 'impact': ..., 'funding': ...}
    
    # 数据溯源
    data_window_start = Column(DateTime)
    data_window_end = Column(DateTime)
    data_quality_score = Column(Float)
    
    # 关联关系
    signal = relationship("Signal", back_populates="prediction")
    model = relationship("ModelVersion", back_populates="predictions")
    
    __table_args__ = (
        Index('idx_prediction_model_time', 'model_version', 'created_at'),
    )

class PerformanceMetric(Base):
    """用于监控的时间序列性能指标"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    model_version = Column(String(50), ForeignKey('model_versions.version'))
    
    # 窗口化指标（例如：最近一小时、最近一天）
    window_size = Column(String(20), nullable=False)  # '1h', '24h', '7d'
    
    # 分类指标
    pr_auc = Column(Float)
    hit_at_top_k = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    
    # 财务指标
    avg_utility = Column(Float)
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # 运营指标
    false_positive_rate = Column(Float)
    calibration_error = Column(Float)
    brier_score = Column(Float)
    
    # 延迟指标
    p50_latency_ms = Column(Float)
    p95_latency_ms = Column(Float)
    p99_latency_ms = Column(Float)
    
    # 系统健康状态
    signal_count = Column(Integer)
    quality_flag_rate = Column(Float)
    
    __table_args__ = (
        Index('idx_metrics_model_window', 'model_version', 'window_size', 'timestamp'),
    )

class ABTest(Base):
    """A/B测试配置和结果"""
    __tablename__ = 'ab_tests'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    test_name = Column(String(100), unique=True, nullable=False, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    
    is_active = Column(Boolean, default=False, nullable=False)
    
    # 测试配置
    control_version = Column(String(50), ForeignKey('model_versions.version'))
    treatment_version = Column(String(50), ForeignKey('model_versions.version'))
    traffic_split = Column(Float, default=0.5)  # 0.0到1.0
    
    # 测试参数
    test_config = Column(JSON)  # 阈值变化等
    
    # 结果
    control_metrics = Column(JSON)
    treatment_metrics = Column(JSON)
    statistical_significance = Column(Float)
    winner = Column(String(20))  # 'control', 'treatment', 'inconclusive'
    
    # 决策
    decision_made_at = Column(DateTime)
    decision = Column(Text)  # 决策说明

class AuditLog(Base):
    """系统变更和决策的审计跟踪"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    event_type = Column(String(50), nullable=False, index=True)
    user = Column(String(100))
    
    # 变更内容
    entity_type = Column(String(50))  # 'model', 'config', 'threshold'等
    entity_id = Column(String(100))
    
    # 变更详情
    action = Column(String(50))  # 'deploy', 'rollback', 'update'等
    old_value = Column(JSON)
    new_value = Column(JSON)
    
    # 上下文
    reason = Column(Text)
    event_metadata = Column(JSON)
    
    __table_args__ = (
        Index('idx_audit_type_time', 'event_type', 'timestamp'),
    )
