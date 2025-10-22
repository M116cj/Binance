"""
统一配置管理系统 - 使用 Pydantic Settings
支持多环境配置（dev/prod）、类型验证和环境变量覆盖
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal, Optional
from pathlib import Path


class IngestionSettings(BaseSettings):
    """数据摄取配置"""
    
    # WebSocket配置
    symbols_per_connection: int = Field(25, description="每个连接的交易对数量")
    micro_batch_ms: int = Field(20, description="微批处理时间（毫秒）")
    heartbeat_interval_s: int = Field(5, description="心跳间隔（秒）")
    
    # 重连策略
    initial_backoff_s: float = Field(0.5, description="初始退避时间（秒）")
    max_backoff_s: float = Field(8.0, description="最大退避时间（秒）")
    backoff_multiplier: float = Field(2.0, description="退避倍数")
    max_retries: int = Field(5, description="最大重试次数")
    
    # 质量控制
    max_clock_drift_ms: float = Field(100.0, description="最大时钟漂移（毫秒）")
    max_gap_ratio: float = Field(0.002, description="最大丢包率 (0.2%)")
    snapshot_rebuild_threshold: int = Field(10, description="快照重建阈值")
    
    model_config = SettingsConfigDict(env_prefix='INGEST_')


class FeatureSettings(BaseSettings):
    """特征工程配置"""
    
    # 时间窗口（毫秒）
    window_lengths_ms: List[int] = Field([50, 250, 1000], description="多时间窗口特征")
    horizon_minutes: List[int] = Field([5, 10, 30], description="预测时间窗口（分钟）")
    
    # 缓冲区配置
    ring_buffer_size: int = Field(10000, description="环形缓冲区大小")
    
    # 标准化方法
    normalization_method: Literal["median_mad", "rank", "zscore"] = Field(
        "median_mad", 
        description="标准化方法"
    )
    lookback_window: int = Field(1000, description="标准化回看窗口")
    
    model_config = SettingsConfigDict(env_prefix='FEATURE_')


class ModelSettings(BaseSettings):
    """模型训练和推理配置"""
    
    # LightGBM超参数
    num_leaves: int = Field(128, ge=31, le=256, description="叶子节点数")
    max_depth: int = Field(8, ge=3, le=15, description="最大深度")
    learning_rate: float = Field(0.01, gt=0.0, lt=1.0, description="学习率")
    n_estimators: int = Field(500, ge=100, le=2000, description="估计器数量")
    
    # Focal Loss
    focal_gamma: float = Field(1.5, ge=0.0, le=5.0, description="Focal Loss gamma")
    
    # 校准
    calibration_method: Literal["isotonic", "sigmoid", "beta"] = Field(
        "isotonic", 
        description="校准方法"
    )
    calibration_bins: int = Field(20, ge=10, le=50, description="校准分箱数")
    
    # ONNX推理
    onnx_intra_op_threads: int = Field(4, description="ONNX内部线程数")
    onnx_inter_op_threads: int = Field(2, description="ONNX外部线程数")
    inference_batch_size: int = Field(32, description="推理批大小")
    
    # 模型版本
    model_version: str = Field("1.0.0", description="模型版本")
    feature_version: str = Field("1.0.0", description="特征版本")
    
    model_config = SettingsConfigDict(env_prefix='MODEL_')


class LabelingSettings(BaseSettings):
    """标记和训练配置"""
    
    # Triple Barrier参数
    theta_up: float = Field(0.006, gt=0.0, lt=0.1, description="上涨阈值")
    theta_dn: float = Field(0.004, gt=0.0, lt=0.1, description="下跌阈值")
    max_hold_minutes: int = Field(60, ge=5, le=480, description="最大持有时间（分钟）")
    
    # 时间隔离
    cooldown_minutes: int = Field(30, ge=10, le=120, description="冷却期（分钟）")
    embargo_pct: float = Field(0.01, ge=0.0, le=0.1, description="禁入期百分比")
    n_splits: int = Field(5, ge=3, le=10, description="K折数量")
    
    # 训练质量
    min_samples_per_fold: int = Field(1000, description="每折最小样本数")
    class_balance_ratio: float = Field(0.3, description="类别平衡比率")
    
    model_config = SettingsConfigDict(env_prefix='LABEL_')


class RiskSettings(BaseSettings):
    """风险控制和成本配置"""
    
    # 交易成本
    maker_fee: float = Field(0.0002, description="挂单手续费")
    taker_fee: float = Field(0.0004, description="吃单手续费")
    slippage_bps: float = Field(2.0, description="滑点（基点）")
    
    # 杠杆和仓位
    max_leverage: float = Field(3.0, ge=1.0, le=20.0, description="最大杠杆")
    max_position_pct: float = Field(0.3, gt=0.0, le=1.0, description="最大仓位百分比")
    
    # 止损规则
    max_consecutive_losses: int = Field(5, description="最大连续亏损次数")
    max_drawdown_pct: float = Field(0.15, description="最大回撤百分比")
    
    # 决策阈值（策略层级）
    tau_conservative: float = Field(0.75, description="保守策略-概率阈值")
    kappa_conservative: float = Field(1.20, description="保守策略-效用阈值")
    
    tau_balanced: float = Field(0.65, description="平衡策略-概率阈值")
    kappa_balanced: float = Field(1.00, description="平衡策略-效用阈值")
    
    tau_aggressive: float = Field(0.55, description="激进策略-概率阈值")
    kappa_aggressive: float = Field(0.80, description="激进策略-效用阈值")
    
    model_config = SettingsConfigDict(env_prefix='RISK_')


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    
    # PostgreSQL
    postgres_url: str = Field(default="", description="PostgreSQL连接URL（从DATABASE_URL环境变量加载）")
    postgres_pool_size: int = Field(10, description="连接池大小")
    postgres_max_overflow: int = Field(20, description="连接池最大溢出")
    
    # Redis
    redis_host: str = Field("localhost", description="Redis主机")
    redis_port: int = Field(6379, description="Redis端口")
    redis_db: int = Field(0, description="Redis数据库")
    redis_ttl_ms: int = Field(200, description="Redis TTL（毫秒）")
    
    # ClickHouse（可选）
    clickhouse_host: Optional[str] = Field(None, description="ClickHouse主机")
    clickhouse_port: Optional[int] = Field(9000, description="ClickHouse端口")
    clickhouse_database: str = Field("crypto_data", description="ClickHouse数据库")
    
    model_config = SettingsConfigDict(env_prefix='DB_')


class APISettings(BaseSettings):
    """API服务配置"""
    
    # FastAPI
    api_host: str = Field("0.0.0.0", description="API主机")
    api_port: int = Field(8000, description="API端口")
    api_workers: int = Field(1, description="工作进程数")
    
    # 性能
    enable_uvloop: bool = Field(True, description="启用uvloop")
    enable_orjson: bool = Field(True, description="启用orjson")
    
    # 限流
    rate_limit_per_minute: int = Field(300, description="每分钟请求限制")
    max_concurrent_requests: int = Field(100, description="最大并发请求")
    
    # 缓存
    enable_response_cache: bool = Field(True, description="启用响应缓存")
    cache_ttl_seconds: int = Field(10, description="缓存TTL（秒）")
    
    # CORS
    allow_origins: List[str] = Field(["*"], description="允许的来源")
    
    model_config = SettingsConfigDict(env_prefix='API_')


class MonitoringSettings(BaseSettings):
    """监控和日志配置"""
    
    # Prometheus
    enable_metrics: bool = Field(True, description="启用Prometheus指标")
    metrics_port: int = Field(9090, description="指标端口")
    
    # 日志
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", 
        description="日志级别"
    )
    enable_structured_logging: bool = Field(True, description="启用结构化日志")
    
    # 追踪
    enable_tracing: bool = Field(False, description="启用OpenTelemetry追踪")
    jaeger_endpoint: Optional[str] = Field(None, description="Jaeger端点")
    
    # 告警阈值
    alert_latency_p95_ms: float = Field(800.0, description="P95延迟告警阈值（毫秒）")
    alert_error_rate_pct: float = Field(1.0, description="错误率告警阈值（百分比）")
    alert_gap_ratio: float = Field(0.002, description="丢包率告警阈值")
    
    model_config = SettingsConfigDict(env_prefix='MONITOR_')


class BacktestSettings(BaseSettings):
    """回测配置"""
    
    days_back: int = Field(30, ge=1, le=365, description="回测天数")
    initial_capital: float = Field(10000.0, description="初始资金")
    
    # 撮合引擎
    enable_latency_injection: bool = Field(True, description="启用延迟注入")
    min_latency_ms: float = Field(10.0, description="最小延迟（毫秒）")
    max_latency_ms: float = Field(100.0, description="最大延迟（毫秒）")
    
    # 执行模式
    execution_mode: Literal["conservative", "neutral", "aggressive"] = Field(
        "conservative",
        description="执行模式"
    )
    
    model_config = SettingsConfigDict(env_prefix='BACKTEST_')


class AppSettings(BaseSettings):
    """应用全局配置"""
    
    # 环境
    environment: Literal["dev", "staging", "prod"] = Field("dev", description="运行环境")
    debug: bool = Field(False, description="调试模式")
    
    # 项目元数据
    project_name: str = Field("Crypto Surge Prediction", description="项目名称")
    version: str = Field("2.0.0", description="系统版本")
    
    # 组件配置  
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)  # type: ignore
    feature: FeatureSettings = Field(default_factory=FeatureSettings)  # type: ignore
    model: ModelSettings = Field(default_factory=ModelSettings)  # type: ignore
    labeling: LabelingSettings = Field(default_factory=LabelingSettings)  # type: ignore
    risk: RiskSettings = Field(default_factory=RiskSettings)  # type: ignore
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)  # type: ignore
    api: APISettings = Field(default_factory=APISettings)  # type: ignore
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)  # type: ignore
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)  # type: ignore
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """验证环境设置"""
        if v not in ['dev', 'staging', 'prod']:
            raise ValueError(f"Invalid environment: {v}")
        return v
    
    def is_production(self) -> bool:
        """检查是否为生产环境"""
        return self.environment == 'prod'
    
    def is_development(self) -> bool:
        """检查是否为开发环境"""
        return self.environment == 'dev'


# 全局配置实例
# 延迟初始化，在导入时不自动创建
_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    """获取全局配置实例"""
    global _settings
    if _settings is None:
        _settings = AppSettings(
            environment="dev",
            debug=False,
            project_name="Crypto Surge Prediction",
            version="2.0.0"
        )
    return _settings

settings = get_settings()


# 环境特定配置加载器
def load_config_for_env(env: Literal["dev", "staging", "prod"]) -> AppSettings:
    """
    加载特定环境的配置
    
    Args:
        env: 环境名称 ('dev', 'staging', 'prod')
        
    Returns:
        配置对象
    """
    config_file = Path(f"backend/config/{env}.env")
    
    # 使用环境变量或默认配置
    return AppSettings(
        environment=env,
        debug=(env == 'dev'),
        project_name="Crypto Surge Prediction",
        version="2.0.0"
    )


# 导出配置验证函数
def validate_config() -> None:
    """验证配置完整性和合理性"""
    
    # 检查数据库连接
    if not settings.database.postgres_url:
        raise ValueError("PostgreSQL URL must be set")
    
    # 检查关键路径存在性
    # ... 可以添加更多验证逻辑
    
    print(f"✅ Configuration validated for environment: {settings.environment}")
    print(f"   - Model Version: {settings.model.model_version}")
    print(f"   - Feature Version: {settings.model.feature_version}")
    print(f"   - API Port: {settings.api.api_port}")
    print(f"   - Inference Batch Size: {settings.model.inference_batch_size}")
