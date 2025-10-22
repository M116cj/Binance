# 📋 系统参数完整参考手册

本文档列出所有模型、报表和系统配置参数，按功能模块分类。

---

## 📊 1. 报表API参数

### 1.1 实时信号卡片 (`/reports/realtime`)
| 参数 | 类型 | 默认值（优化后） | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对（如 BTCUSDT） |
| `theta_up` | float | **0.008** | 上涨判定线（**0.8%**，优化扩大覆盖范围） |
| `theta_dn` | float | **0.0056** | 下跌判定线（**0.56%**，70% of theta_up） |
| `tau` | float | 0.75 | 概率阈值（75%置信度，A级信号） |
| `kappa` | float | 1.20 | 效用阈值（收益/成本比，A级信号） |

**用途**：生成当前最新的交易信号，包含方向、概率、预期收益等核心指标。

---

### 1.2 市场状态 (`/reports/regime`)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对 |

**用途**：识别当前市场状态（趋势/震荡/高波动），无需参数调整。

---

### 1.3 概率窗口分析 (`/reports/window`)
| 参数 | 类型 | 默认值（优化后） | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对 |
| `theta_up` | float | **0.008** | 上涨判定线（优化后） |
| `theta_dn` | float | **0.0056** | 下跌判定线（优化后） |

**用途**：显示不同时间窗口（**10分钟/20分钟**，优化为2窗口）的预测概率分布。

---

### 1.4 回测性能 (`/reports/backtest`)
| 参数 | 类型 | 默认值（优化后） | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对 |
| `theta_up` | float | **0.008** | 上涨判定线（优化后） |
| `theta_dn` | float | **0.0056** | 下跌判定线（优化后） |
| `tau` | float | 0.75 | 概率阈值（A级） |
| `kappa` | float | 1.20 | 效用阈值（A级） |
| `days_back` | int | **20** | 回测天数（**优化为20天**，提升速度） |

**用途**：模拟历史交易表现，计算胜率、夏普比率、最大回撤等指标。

---

### 1.5 模型校准 (`/reports/calibration`)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对 |
| `theta_up` | float | 0.006 | 上涨判定线 |
| `theta_dn` | float | 0.004 | 下跌判定线 |

**用途**：评估预测概率的准确性，检测过拟合/欠拟合。

---

### 1.6 事件归因 (`/reports/attribution`)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对 |
| `theta_up` | float | 0.006 | 上涨判定线 |
| `theta_dn` | float | 0.004 | 下跌判定线 |
| `tau` | float | 0.75 | 概率阈值 |
| `kappa` | float | 1.20 | 效用阈值 |

**用途**：分析收益来源，对比不同策略参数下的表现。

---

## 🎯 2. 信号查询参数

### 2.1 获取最近信号 (`/signals`)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbol` | string | 可选 | 筛选特定交易对 |
| `tier` | string | 可选 | 筛选等级（A/B） |
| `limit` | int | 20 | 返回数量（最多1000） |

---

### 2.2 信号历史 (`/signals/history`)
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbol` | string | 必填 | 交易对 |
| `hours` | int | 24 | 历史时长（小时） |

---

## 🧠 3. 模型训练参数 (`ModelSettings`)

### 3.1 LightGBM超参数
| 参数 | 类型 | 默认值（优化后） | 范围 | 说明 |
|------|------|--------|------|------|
| `num_leaves` | int | 128 | 31-256 | 叶子节点数量 |
| `max_depth` | int | 8 | 3-15 | 树的最大深度 |
| `learning_rate` | float | 0.01 | 0.0-1.0 | 学习率 |
| `n_estimators` | int | 500 | 100-2000 | 决策树数量 |
| `focal_gamma` | float | **1.8** | 0.0-5.0 | Focal Loss聚焦参数（**优化提升至1.8**） |

**环境变量前缀**：`MODEL_`
**示例**：`MODEL_NUM_LEAVES=256`

---

### 3.2 模型校准
| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `calibration_method` | string | isotonic | isotonic/sigmoid/beta | 校准方法 |
| `calibration_bins` | int | 20 | 10-50 | 校准分箱数 |

---

### 3.3 ONNX推理
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `onnx_intra_op_threads` | int | 4 | ONNX内部线程数 |
| `onnx_inter_op_threads` | int | 2 | ONNX外部线程数 |
| `inference_batch_size` | int | 32 | 推理批大小 |

---

## 🏷️ 4. 标记参数 (`LabelingSettings`)

### 4.1 Triple Barrier设定（优化后）
| 参数 | 类型 | 默认值（优化后） | 范围 | 说明 |
|------|------|--------|------|------|
| `theta_up` | float | **0.008** | 0.0-0.1 | 上涨阈值（**0.8%**，优化扩大覆盖） |
| `theta_dn` | float | **0.0056** | 0.0-0.1 | 下跌阈值（**0.56%**，70% of theta_up） |
| `max_hold_minutes` | int | 60 | 5-480 | 最大持有时间（分钟） |

---

### 4.2 时间隔离（优化后）
| 参数 | 类型 | 默认值（优化后） | 范围 | 说明 |
|------|------|--------|------|------|
| `cooldown_minutes` | int | **15** | 10-120 | 冷却期（**优化为15分钟**，提升响应速度） |
| `embargo_pct` | float | 0.01 | 0.0-0.1 | 禁入期百分比（保持启用） |
| `n_splits` | int | 5 | 3-10 | K折交叉验证折数 |

**环境变量前缀**：`LABEL_`

---

## ⚠️ 5. 风险控制参数 (`RiskSettings`)

### 5.1 交易成本
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `maker_fee` | float | 0.0002 | 挂单手续费（0.02%） |
| `taker_fee` | float | 0.0004 | 吃单手续费（0.04%） |
| `slippage_bps` | float | 2.0 | 滑点（基点） |

---

### 5.2 杠杆与仓位
| 参数 | 类型 | 默认值 | 范围 | 说明 |
|------|------|--------|------|------|
| `max_leverage` | float | 3.0 | 1.0-20.0 | 最大杠杆倍数 |
| `max_position_pct` | float | 0.3 | 0.0-1.0 | 最大仓位百分比 |

---

### 5.3 策略预设阈值（优化后两档模型）
| 策略类型 | tau（概率阈值） | kappa（效用阈值） | theta_up | theta_dn | 特点 |
|----------|-----------------|-------------------|----------|----------|------|
| **⭐ A级信号** | 0.75 | 1.20 | 0.008 | 0.0056 | 高质量信号，严格筛选 |
| **🎯 B级信号** | 0.70 | 1.10 | 0.008 | 0.0056 | 标准质量，平衡频率 |

**结构性简化**：从3档简化为2档，减少决策复杂度，提升执行速度。

**环境变量示例**：
- `RISK_TAU_TIER_A=0.75` - A级概率阈值
- `RISK_KAPPA_TIER_B=1.10` - B级效用阈值

---

### 5.4 止损规则
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_consecutive_losses` | int | 5 | 最大连续亏损次数 |
| `max_drawdown_pct` | float | 0.15 | 最大回撤百分比（15%） |

---

## 🔧 6. 特征工程参数 (`FeatureSettings`)

### 6.1 时间窗口（优化后）
| 参数 | 类型 | 默认值（优化后） | 说明 |
|------|------|--------|------|
| `window_lengths_ms` | list[int] | [50, 250, 1000] | 多时间窗口（毫秒） |
| `horizon_minutes` | list[int] | **[10, 20]** | 预测时间窗口（**优化为2窗口**，专注短期精度） |

---

### 6.2 标准化
| 参数 | 类型 | 默认值 | 可选值 | 说明 |
|------|------|--------|--------|------|
| `normalization_method` | string | median_mad | median_mad/rank/zscore | 标准化方法 |
| `lookback_window` | int | 1000 | - | 标准化回看窗口 |
| `ring_buffer_size` | int | 10000 | - | 环形缓冲区大小 |

**环境变量前缀**：`FEATURE_`

---

## 📈 7. 回测参数 (`BacktestSettings`)

| 参数 | 类型 | 默认值（优化后） | 范围 | 说明 |
|------|------|--------|------|------|
| `days_back` | int | **20** | 1-365 | 回测天数（**优化为20天**，减少计算时间） |
| `initial_capital` | float | 10000.0 | - | 初始资金 |
| `enable_latency_injection` | bool | true | - | 启用延迟注入 |
| `min_latency_ms` | float | 10.0 | - | 最小延迟（毫秒） |
| `max_latency_ms` | float | 100.0 | - | 最大延迟（毫秒） |
| `execution_mode` | string | conservative | conservative/neutral/aggressive | 执行模式 |

**环境变量前缀**：`BACKTEST_`

---

## 🌐 8. API性能参数 (`APISettings`)

### 8.1 限流与并发
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rate_limit_per_minute` | int | 300 | 每分钟最大请求数 |
| `max_concurrent_requests` | int | 100 | 最大并发请求数 |

---

### 8.2 缓存
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_response_cache` | bool | true | 启用响应缓存 |
| `cache_ttl_seconds` | int | 10 | 缓存生存时间（秒） |

**环境变量前缀**：`API_`

---

## 📡 9. 数据摄取参数 (`IngestionSettings`)

### 9.1 WebSocket配置
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `symbols_per_connection` | int | 25 | 每个连接的交易对数量 |
| `micro_batch_ms` | int | 20 | 微批处理时间（毫秒） |
| `heartbeat_interval_s` | int | 5 | 心跳间隔（秒） |

---

### 9.2 质量控制
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_clock_drift_ms` | float | 100.0 | 最大时钟漂移（毫秒） |
| `max_gap_ratio` | float | 0.002 | 最大丢包率（0.2%） |
| `snapshot_rebuild_threshold` | int | 10 | 快照重建阈值 |

**环境变量前缀**：`INGEST_`

---

## 📊 10. 监控告警参数 (`MonitoringSettings`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alert_latency_p95_ms` | float | 800.0 | P95延迟告警阈值（毫秒） |
| `alert_error_rate_pct` | float | 1.0 | 错误率告警阈值（百分比） |
| `alert_gap_ratio` | float | 0.002 | 丢包率告警阈值 |
| `log_level` | string | INFO | 日志级别（DEBUG/INFO/WARNING/ERROR） |

**环境变量前缀**：`MONITOR_`

---

## 🗄️ 11. 数据库参数 (`DatabaseSettings`)

### 11.1 PostgreSQL
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `postgres_url` | string | 从环境变量 | PostgreSQL连接URL |
| `postgres_pool_size` | int | 10 | 连接池大小 |
| `postgres_max_overflow` | int | 20 | 连接池最大溢出 |

---

### 11.2 Redis
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `redis_host` | string | localhost | Redis主机 |
| `redis_port` | int | 6379 | Redis端口 |
| `redis_ttl_ms` | int | 200 | Redis缓存TTL（毫秒） |

**环境变量前缀**：`DB_`

---

## 🎛️ 使用示例

### 示例1：使用优化后的A级信号策略
```python
# 方法1：通过环境变量（优化后参数）
export LABEL_THETA_UP=0.008
export LABEL_THETA_DN=0.0056
export RISK_TAU_TIER_A=0.75
export RISK_KAPPA_TIER_A=1.20

# 方法2：通过API参数
GET /reports/backtest?symbol=BTCUSDT&theta_up=0.008&theta_dn=0.0056&tau=0.75&kappa=1.20&days_back=20
```

### 示例1b：切换到B级信号策略
```python
# 获取更多信号，略微放宽质量要求
GET /reports/backtest?symbol=BTCUSDT&theta_up=0.008&theta_dn=0.0056&tau=0.70&kappa=1.10&days_back=20
```

---

### 示例2：优化回测配置
```python
# 延长回测周期，启用更真实的执行模拟
export BACKTEST_DAYS_BACK=90
export BACKTEST_EXECUTION_MODE=neutral
export BACKTEST_ENABLE_LATENCY_INJECTION=true
export BACKTEST_MAX_LATENCY_MS=150.0
```

---

### 示例3：提升模型复杂度
```python
# 增加树复杂度，适用于更多数据场景
export MODEL_NUM_LEAVES=256
export MODEL_MAX_DEPTH=12
export MODEL_N_ESTIMATORS=1000
export MODEL_LEARNING_RATE=0.005
```

---

## ⚡ 快速参考

### 核心策略参数对照表（优化后两档模型）
| 参数组合 | theta_up | theta_dn | tau | kappa | 适用场景 |
|----------|----------|----------|-----|-------|---------|
| **⭐ A级信号** | 0.008 | 0.0056 | 0.75 | 1.20 | 高质量信号，严格筛选，追求确定性 |
| **🎯 B级信号** | 0.008 | 0.0056 | 0.70 | 1.10 | 标准质量，平衡频率与质量 |

**优化要点**（2025-10-22更新）：
- ✅ **theta扩大至0.8%/0.56%**：覆盖更广波动范围，提升趋势检测
- ✅ **简化为两档**：减少决策复杂度，提升执行速度
- ✅ **horizon缩减为2窗口**：专注短期精度（10/20分钟）
- ✅ **cooldown降至15分钟**：提升响应速度
- ✅ **focal_gamma提升至1.8**：更好聚焦困难样本
- ✅ **days_back缩短为20天**：加快回测计算

---

## 📝 注意事项

1. **参数优先级**：API查询参数 > 环境变量 > 默认配置
2. **类型安全**：所有参数都经过Pydantic验证，超出范围会报错
3. **性能影响**：
   - 降低`theta_up/theta_dn`会显著增加信号频率
   - 增加`days_back`会增加回测计算时间
   - 提高`num_leaves`和`n_estimators`会增加训练时间
4. **风险提示**：激进参数可能导致过度交易和高滑点成本

---

---

## 🎯 优化历程

### 2025-10-22 参数优化（v2.1）
**核心目标**：提升响应速度、简化决策流程、增强模型聚焦能力

| 参数 | 优化前 | 优化后 | 改进原因 |
|------|--------|--------|---------|
| `theta_up` | 0.006 (0.6%) | **0.008 (0.8%)** | 扩大波动覆盖，提升趋势检测 |
| `theta_dn` | 0.004 (0.4%) | **0.0056 (0.56%)** | 保持70%比例，增强平衡性 |
| `horizon_minutes` | [5, 10, 30] | **[10, 20]** | 减少窗口，专注短期精度 |
| `focal_gamma` | 1.5 | **1.8** | 更好聚焦困难样本 |
| `cooldown_minutes` | 30 | **15** | 提升响应速度，一致交易间隔 |
| `days_back` | 30 | **20** | 减少计算时间，保持相关性 |
| **策略层级** | 3档（保守/平衡/激进） | **2档（A/B级）** | 简化决策，降低复杂度 |

**性能目标提升**：
- PR-AUC: 提升至 **0.75**
- Hit@Top-K: 提升至 **0.70**
- Calibration ECE: 收紧至 **0.03**

---

**最后更新**：2025-10-22  
**版本**：v2.1.0（优化版）
