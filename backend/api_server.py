"""
加密货币涨跌预测仪表板的独立API服务器
演示模式使用模拟数据 - 不需要Redis/ClickHouse

性能优化:
- 响应缓存 (10秒TTL)
- 请求限流 (300/分钟)
- 并发控制 (最多100个)
- 批量查询优化
"""

import time
import logging
import numpy as np
import uuid
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
import uvicorn

from backend.database import db_manager, get_db, ModelVersion, Signal, Prediction
from backend.export_utils import signals_to_protobuf_batch, signals_to_jsonl
from backend.symbol_service import symbol_service
from backend.utils.cache import global_cache, cache_response, start_cache_cleanup_task
from backend.utils.rate_limiter import global_rate_limiter, RateLimitExceeded, start_rate_limiter_cleanup_task
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Surge Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware - 300 requests/minute per client, max 100 concurrent"""
    
    # Skip rate limiting for health check
    if request.url.path == "/health":
        return await call_next(request)
    
    # Get client identifier (IP address)
    client_id = request.client.host if request.client else "unknown"
    
    # Try to acquire rate limit slot
    if not await global_rate_limiter.acquire(client_id):
        # Rate limit exceeded
        stats = global_rate_limiter.get_stats()
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "limit": global_rate_limiter.requests_per_minute,
                "retry_after_seconds": 60,
                "stats": stats
            },
            headers={"Retry-After": "60"}
        )
    
    try:
        # Process request
        response = await call_next(request)
        return response
    finally:
        # Release slot
        await global_rate_limiter.release()


# Background cleanup tasks
_cleanup_tasks = []


@app.on_event("startup")
async def startup_event():
    """启动时初始化数据库和后台任务"""
    logger.info("Initializing database...")
    db_manager.initialize()
    
    # 创建默认模型版本（如果不存在）
    with db_manager.get_session() as db:
        existing_model = db.query(ModelVersion).filter(ModelVersion.version == "1.0.0").first()
        if not existing_model:
            default_model = ModelVersion(
                version="1.0.0",
                model_type="lightgbm_demo",
                is_active=True,
                config={"demo": True, "focal_gamma": 1.5},
                metrics={"pr_auc": 0.72, "hit_at_top_k": 0.65},
                calibration_method="isotonic",
                calibration_ece=0.04,
                deployed_at=datetime.utcnow(),
                deployed_by="system"
            )
            db.add(default_model)
            db.commit()
            logger.info("Created default model version 1.0.0")
    
    # Start background cleanup tasks
    logger.info("Starting background cleanup tasks...")
    _cleanup_tasks.append(asyncio.create_task(start_cache_cleanup_task(global_cache, interval=60.0)))
    _cleanup_tasks.append(asyncio.create_task(start_rate_limiter_cleanup_task(global_rate_limiter, interval=600.0)))
    logger.info("API server startup complete with caching and rate limiting enabled")


@app.on_event("shutdown")
async def shutdown_event():
    """关闭数据库连接和后台任务"""
    logger.info("Shutting down background tasks...")
    for task in _cleanup_tasks:
        task.cancel()
    
    logger.info("Closing database connections...")
    db_manager.close()
    
    # Log final stats
    cache_stats = global_cache.get_stats()
    rate_limit_stats = global_rate_limiter.get_stats()
    logger.info(f"Final cache stats: {cache_stats}")
    logger.info(f"Final rate limiter stats: {rate_limit_stats}")

def generate_mock_time_series(days: int, start_value: float = 0.0) -> Dict:
    """生成模拟时间序列数据"""
    dates = [(datetime.now() - timedelta(days=days-i)).isoformat() for i in range(days)]
    cumulative_returns = np.cumsum(np.random.normal(0.002, 0.02, days)).tolist()
    return {
        'dates': dates,
        'cumulative_returns': cumulative_returns
    }

def save_signal_to_db(
    db: Session,
    symbol: str,
    horizon_min: int,
    decision: str,
    tier: str,
    p_up: float,
    expected_return: float,
    estimated_cost: float,
    net_utility: float,
    tau: float,
    kappa: float,
    theta_up: float,
    theta_dn: float,
    features_top5: Dict,
    quality_flags: List,
    sla_latency_ms: float,
    regime: str = "normal",
    volatility: float = 0.02
) -> Signal:
    """将生成的信号保存到数据库"""
    signal_id = f"{symbol}_{horizon_min}m_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    now = datetime.utcnow()
    
    signal = Signal(
        signal_id=signal_id,
        exchange_time=now - timedelta(milliseconds=np.random.uniform(50, 200)),
        ingest_time=now - timedelta(milliseconds=np.random.uniform(20, 100)),
        infer_time=now,
        symbol=symbol,
        horizon_min=horizon_min,
        decision=decision,
        tier=tier,
        p_up=p_up,
        p_up_ci_low=p_up - 0.05,
        p_up_ci_high=p_up + 0.05,
        expected_return=expected_return,
        estimated_cost=estimated_cost,
        net_utility=net_utility,
        tau_threshold=tau,
        kappa_threshold=kappa,
        theta_up=theta_up,
        theta_dn=theta_dn,
        regime=regime,
        volatility=volatility,
        features_top5=features_top5,
        quality_flags=quality_flags,
        sla_latency_ms=sla_latency_ms,
        model_version="1.0.0",
        feature_version="1.0.0",
        cost_model_version="v1.2.0"
    )
    
    db.add(signal)
    db.commit()
    db.refresh(signal)
    
    return signal

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "exchange_lag_s": np.random.uniform(0.1, 1.5),
        "mode": "demo"
    }

@app.get("/reports/realtime")
async def get_realtime_signal_card(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold"),
    db: Session = Depends(get_db)
):
    """报告1：实时信号卡片"""
    p_up_5 = np.random.uniform(0.55, 0.85)
    p_up_10 = np.random.uniform(0.50, 0.80)
    p_up_30 = np.random.uniform(0.45, 0.75)
    
    expected_return = np.random.uniform(0.005, 0.015)
    estimated_cost = np.random.uniform(0.001, 0.003)
    net_utility = (expected_return / estimated_cost) if estimated_cost > 0 else 0
    
    decision_action = "LONG" if p_up_5 > tau and net_utility > kappa else "WAIT"
    signal_tier = "A" if p_up_5 > 0.75 and net_utility > 1.2 else "B" if p_up_5 > 0.65 else "none"
    
    features_top5 = {
        'qi_1': np.random.uniform(-0.2, 0.3),
        'ofi_10': np.random.uniform(-0.15, 0.25),
        'microprice_dev': np.random.uniform(-0.1, 0.2),
        'rv_ratio': np.random.uniform(-0.15, 0.2),
        'depth_slope_bid': np.random.uniform(-0.1, 0.15)
    }
    
    sla_latency = np.random.uniform(50, 200)
    
    # 保存信号到数据库（5分钟时间窗口）
    if decision_action != "WAIT":
        try:
            save_signal_to_db(
                db=db,
                symbol=symbol,
                horizon_min=5,
                decision=decision_action,
                tier=signal_tier,
                p_up=p_up_5,
                expected_return=expected_return,
                estimated_cost=estimated_cost,
                net_utility=net_utility,
                tau=tau,
                kappa=kappa,
                theta_up=theta_up,
                theta_dn=theta_dn,
                features_top5=features_top5,
                quality_flags=[],
                sla_latency_ms=sla_latency
            )
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'current_price': np.random.uniform(40000, 45000) if symbol == 'BTCUSDT' else np.random.uniform(2000, 3000),
        'probabilities': {
            '5m': {'value': p_up_5, 'ci_low': p_up_5 - 0.05, 'ci_high': p_up_5 + 0.05},
            '10m': {'value': p_up_10, 'ci_low': p_up_10 - 0.05, 'ci_high': p_up_10 + 0.05},
            '30m': {'value': p_up_30, 'ci_low': p_up_30 - 0.05, 'ci_high': p_up_30 + 0.05}
        },
        'expected_return': expected_return,
        'estimated_cost': estimated_cost,
        'utility': net_utility,
        'decision': decision_action,
        'tier': signal_tier,
        'thresholds': {
            'tau': tau,
            'kappa': kappa,
            'theta_up': theta_up,
            'theta_dn': theta_dn
        },
        'features_top5': features_top5,
        'quality_flags': [],
        'sla_latency_ms': sla_latency,
        'model_version': '1.0.0',
        'feature_version': '1.0.0',
        'cost_model': 'v1.2.0'
    }

@cache_response(global_cache, ttl=10.0, key_prefix="regime_state")
async def _get_regime_state_cached(symbol: str):
    """缓存的市场状态计算"""
    regimes = ['calm', 'choppy', 'trending', 'volatile']
    regime = np.random.choice(regimes, p=[0.3, 0.3, 0.2, 0.2])
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'regime': {
            'state': regime,
            'confidence': np.random.uniform(0.6, 0.9),
            'rv_ratio': np.random.uniform(0.5, 2.0),
            'volatility': np.random.uniform(0.01, 0.05)
        },
        'liquidity': {
            'bid_depth': np.random.uniform(100000, 500000),
            'ask_depth': np.random.uniform(100000, 500000),
            'depth_imbalance': np.random.uniform(-0.3, 0.3),
            'spread_bps': np.random.uniform(1, 10),
            'near_touch_void': np.random.uniform(0.1, 0.4)
        },
        'microstructure': {
            'arrival_rate': np.random.uniform(50, 200),
            'ofi_slope': np.random.uniform(-0.5, 0.5),
            'funding_delta': np.random.uniform(-0.01, 0.01),
            'oi_pressure': np.random.uniform(-0.2, 0.2)
        }
    }


@app.get("/reports/regime")
async def get_regime_state(symbol: str = Query(..., description="Trading symbol")):
    """报告2：市场状态与流动性（缓存10秒）"""
    return await _get_regime_state_cached(symbol)

@cache_response(global_cache, ttl=10.0, key_prefix="probability_window")
async def _get_probability_window_cached(symbol: str, theta_up: float, theta_dn: float):
    """缓存的概率窗口计算"""
    horizons = [5, 10, 15, 30, 60]
    probabilities = [np.random.uniform(0.45, 0.85) for _ in horizons]
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'probability_curve': {
            'horizons_min': horizons,
            'p_up_values': probabilities,
            'p_dn_values': [1 - p for p in probabilities],
            'confidence_intervals': [[p - 0.1, p + 0.1] for p in probabilities]
        },
        'optimal_window': {
            'horizon_min': horizons[np.argmax(probabilities)],
            'max_probability': max(probabilities),
            'expected_return': np.random.uniform(0.008, 0.015),
            'confidence': np.random.uniform(0.7, 0.9)
        },
        'decay_analysis': {
            'half_life_min': np.random.uniform(15, 45),
            'decay_rate': np.random.uniform(0.01, 0.05),
            'persistence_score': np.random.uniform(0.5, 0.9)
        }
    }


@app.get("/reports/window")
async def get_probability_window(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold")
):
    """报告3：预测概率与时间窗口（缓存10秒）"""
    return await _get_probability_window_cached(symbol, theta_up, theta_dn)

@cache_response(global_cache, ttl=10.0, key_prefix="cost_capacity")
async def _get_cost_capacity_cached(symbol: str):
    """缓存的成本容量计算"""
    sizes = [1000, 5000, 10000, 50000, 100000]
    costs = [size * np.random.uniform(0.0001, 0.0005) for size in sizes]
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'cost_curve': {
            'sizes_usd': sizes,
            'estimated_costs_bps': costs,
            'spread_cost': [c * 0.3 for c in costs],
            'impact_cost': [c * 0.7 for c in costs]
        },
        'capacity': {
            'max_size_usd': np.random.uniform(50000, 150000),
            'optimal_size_usd': np.random.uniform(10000, 50000),
            'cost_at_optimal': np.random.uniform(2, 8),
            'throughput_rps': np.random.uniform(250, 400)
        },
        'current_conditions': {
            'spread_bps': np.random.uniform(2, 10),
            'depth_ratio': np.random.uniform(0.8, 1.5),
            'liquidity_score': np.random.uniform(0.5, 0.9)
        }
    }


@app.get("/reports/cost")
async def get_cost_capacity(symbol: str = Query(..., description="Trading symbol")):
    """报告4：执行成本与容量分析（缓存10秒）"""
    return await _get_cost_capacity_cached(symbol)

@app.get("/reports/backtest")
async def get_backtest_performance(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold"),
    days_back: int = Query(30, description="Days to backtest")
):
    """报告5：历史回测性能"""
    time_series = generate_mock_time_series(days_back)
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'period': f'{days_back}_days',
        'performance_summary': {
            'total_return': np.random.uniform(-0.05, 0.25),
            'sharpe_ratio_post_cost': np.random.uniform(0.5, 2.5),
            'max_drawdown': np.random.uniform(-0.15, -0.02),
            'hit_rate': np.random.uniform(0.45, 0.65),
            'total_signals': np.random.randint(50, 200),
            'win_count': np.random.randint(25, 120),
            'avg_hold_time_min': np.random.uniform(15, 45)
        },
        'time_series': time_series,
        'monthly_breakdown': {
            f'month_{i}': {
                'return': np.random.uniform(-0.05, 0.15),
                'signals': np.random.randint(10, 50),
                'hit_rate': np.random.uniform(0.4, 0.7)
            } for i in range(1, min(4, days_back//7 + 1))
        },
        'pr_curve': {
            'precision': [np.random.uniform(0.5, 0.9) for _ in range(10)],
            'recall': [i/10 for i in range(10)],
            'auc': np.random.uniform(0.65, 0.85)
        }
    }

@app.get("/reports/calibration")
async def get_calibration_analysis(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold")
):
    """报告6：模型校准与误差分析"""
    n_bins = 10
    pred_probs = [i/n_bins + np.random.uniform(-0.02, 0.02) for i in range(n_bins)]
    obs_freqs = [p + np.random.uniform(-0.05, 0.05) for p in pred_probs]
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'calibration_metrics': {
            'brier_score': np.random.uniform(0.15, 0.25),
            'calibration_error': np.random.uniform(0.02, 0.08),
            'log_loss': np.random.uniform(0.3, 0.6)
        },
        'calibration_curve': {
            'predicted_probabilities': pred_probs,
            'observed_frequencies': obs_freqs
        },
        'brier_decomposition': {
            'reliability': np.random.uniform(0.02, 0.08),
            'resolution': np.random.uniform(0.05, 0.15),
            'uncertainty': np.random.uniform(0.15, 0.25)
        },
        'residuals': {
            'values': np.random.normal(0, 0.05, 100).tolist()
        },
        'predictions': np.random.beta(2, 2, 100).tolist(),
        'isotonic_mapping': {
            'raw_probabilities': [i/20 for i in range(20)],
            'calibrated_probabilities': [i/20 + np.random.uniform(-0.05, 0.05) for i in range(20)]
        },
        'error_metrics': {
            'mae': np.random.uniform(0.03, 0.08),
            'mse': np.random.uniform(0.005, 0.015),
            'rmse': np.random.uniform(0.07, 0.12),
            'mape': np.random.uniform(8, 15)
        }
    }

@app.get("/reports/attribution")
async def get_attribution_comparison(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold")
):
    """报告7：事件归因与策略对比"""
    features = ['qi_1', 'ofi_10', 'microprice_dev', 'rv_ratio', 'depth_slope', 'funding_delta', 'oi_pressure', 'arrival_rate']
    importance = [np.random.uniform(-0.3, 0.3) for _ in features]
    
    return {
        'symbol': symbol,
        'timestamp': int(time.time() * 1000),
        'current_tau': tau,
        'comparison_metrics': {
            'current_strategy_return': np.random.uniform(-0.02, 0.18),
            'benchmark_return': np.random.uniform(-0.05, 0.12),
            'alpha': np.random.uniform(-0.03, 0.08)
        },
        'feature_attribution': {
            'features': features,
            'importance': importance
        },
        'shap_waterfall': {
            'base_value': 0.5,
            'shap_values': [np.random.uniform(-0.1, 0.1) for _ in range(5)],
            'features': features[:5]
        },
        'strategy_comparison': {
            'Current Strategy': {
                'return': np.random.uniform(0.05, 0.20),
                'sharpe': np.random.uniform(1.0, 2.0),
                'hit_rate': np.random.uniform(0.55, 0.68)
            },
            'Buy & Hold': {
                'return': np.random.uniform(0.02, 0.15),
                'sharpe': np.random.uniform(0.5, 1.2),
                'hit_rate': 0.50
            },
            'Lower Threshold (τ=0.65)': {
                'return': np.random.uniform(0.03, 0.16),
                'sharpe': np.random.uniform(0.8, 1.6),
                'hit_rate': np.random.uniform(0.50, 0.62)
            }
        },
        'threshold_sensitivity': {
            'tau_thresholds': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
            'returns': [np.random.uniform(0.02, 0.18) for _ in range(7)],
            'signals_count': [np.random.randint(30, 150) for _ in range(7)]
        },
        'recent_events': [
            {
                'timestamp': (datetime.now() - timedelta(hours=i)).isoformat(),
                'type': np.random.choice(['Signal', 'Entry', 'Exit']),
                'probability': np.random.uniform(0.5, 0.9),
                'decision': np.random.choice(['LONG', 'WAIT', 'EXIT']),
                'outcome': np.random.choice(['Win', 'Loss', 'Pending'])
            } for i in range(10)
        ],
        'decision_matrix': {
            'true_positives': np.random.randint(30, 80),
            'false_positives': np.random.randint(10, 40),
            'true_negatives': np.random.randint(100, 200),
            'false_negatives': np.random.randint(15, 50)
        }
    }

@app.get("/signals")
async def get_recent_signals(
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    tier: Optional[str] = Query(None, description="Filter by tier (A, B)"),
    limit: int = Query(20, description="Number of signals to return"),
    db: Session = Depends(get_db)
):
    """获取最近的交易信号"""
    query = db.query(Signal).order_by(Signal.created_at.desc())
    
    if symbol:
        query = query.filter(Signal.symbol == symbol)
    if tier and tier != "none":
        query = query.filter(Signal.tier == tier)
    
    signals = query.limit(limit).all()
    
    return {
        "count": len(signals),
        "signals": [
            {
                "signal_id": s.signal_id,
                "created_at": s.created_at.isoformat(),
                "symbol": s.symbol,
                "horizon_min": s.horizon_min,
                "decision": s.decision,
                "tier": s.tier,
                "p_up": s.p_up,
                "expected_return": s.expected_return,
                "net_utility": s.net_utility,
                "sla_latency_ms": s.sla_latency_ms,
                "model_version": s.model_version
            }
            for s in signals
        ]
    }

@app.get("/signals/history")
async def get_signal_history(
    symbol: str = Query(..., description="Trading symbol"),
    hours: int = Query(24, description="Hours of history to retrieve"),
    db: Session = Depends(get_db)
):
    """获取特定交易对的信号历史"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    signals = db.query(Signal).filter(
        Signal.symbol == symbol,
        Signal.created_at >= cutoff_time
    ).order_by(Signal.created_at.asc()).all()
    
    # 计算统计数据
    total_signals = len(signals)
    a_tier_count = sum(1 for s in signals if s.tier == 'A')
    b_tier_count = sum(1 for s in signals if s.tier == 'B')
    long_count = sum(1 for s in signals if s.decision == 'LONG')
    
    avg_utility = np.mean([s.net_utility for s in signals]) if signals else 0
    avg_latency = np.mean([s.sla_latency_ms for s in signals]) if signals else 0
    
    return {
        "symbol": symbol,
        "period_hours": hours,
        "summary": {
            "total_signals": total_signals,
            "a_tier_signals": a_tier_count,
            "b_tier_signals": b_tier_count,
            "long_signals": long_count,
            "avg_utility": float(avg_utility),
            "avg_latency_ms": float(avg_latency)
        },
        "signals": [
            {
                "time": s.created_at.isoformat(),
                "decision": s.decision,
                "tier": s.tier,
                "p_up": s.p_up,
                "utility": s.net_utility,
                "latency_ms": s.sla_latency_ms
            }
            for s in signals
        ]
    }

@cache_response(global_cache, ttl=10.0, key_prefix="signal_stats")
async def _get_signal_stats_cached():
    """缓存的信号统计计算（使用SQL聚合优化）"""
    # This would be called with db parameter in the endpoint
    return None


@app.get("/signals/stats")
async def get_signal_stats(
    db: Session = Depends(get_db)
):
    """获取总体信号统计数据（优化批量查询，缓存10秒）"""
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    
    # 使用SQL聚合而不是加载所有数据到内存
    # 批量查询优化：使用group by进行聚合
    from sqlalchemy import case
    
    stats_query = db.query(
        Signal.symbol,
        func.count(Signal.id).label('total'),
        func.sum(case((Signal.tier == 'A', 1), else_=0)).label('a_tier'),
        func.sum(case((Signal.tier == 'B', 1), else_=0)).label('b_tier'),
        func.sum(case((Signal.decision == 'LONG', 1), else_=0)).label('long_count'),
        func.avg(Signal.net_utility).label('avg_utility'),
        func.avg(Signal.sla_latency_ms).label('avg_latency')
    ).filter(
        Signal.created_at >= cutoff_time
    ).group_by(Signal.symbol).all()
    
    # 获取总信号数（单个查询）
    total_signals = db.query(func.count(Signal.id)).filter(
        Signal.created_at >= cutoff_time
    ).scalar()
    
    # 格式化结果
    results = {}
    for row in stats_query:
        results[row.symbol] = {
            'total_signals': row.total,
            'a_tier_count': row.a_tier,
            'b_tier_count': row.b_tier,
            'long_count': row.long_count,
            'avg_utility': float(row.avg_utility) if row.avg_utility else 0,
            'avg_latency_ms': float(row.avg_latency) if row.avg_latency else 0
        }
    
    return {
        "period": "last_24_hours",
        "total_signals": total_signals or 0,
        "by_symbol": results
    }

@app.get("/models")
async def get_model_versions(
    db: Session = Depends(get_db)
):
    """获取所有模型版本"""
    models = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    
    return {
        "models": [
            {
                "version": m.version,
                "model_type": m.model_type,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat(),
                "metrics": m.metrics,
                "calibration_ece": m.calibration_ece
            }
            for m in models
        ]
    }

@app.get("/symbols")
async def get_symbols():
    """获取所有可用的币安USDT交易对
    
    返回:
        交易对列表，每个包含symbol、baseAsset、name和displayName
    """
    symbols = await symbol_service.get_usdt_symbols()
    return {
        "symbols": symbols,
        "count": len(symbols),
        "updated_at": datetime.utcnow().isoformat()
    }


@app.get("/stats/performance")
async def get_performance_stats():
    """获取API性能统计信息
    
    返回:
        缓存命中率、速率限制统计、并发请求等性能指标
    """
    cache_stats = global_cache.get_stats()
    rate_limit_stats = global_rate_limiter.get_stats()
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "cache": {
            "hit_rate_percent": cache_stats["hit_rate_percent"],
            "total_requests": cache_stats["total_requests"],
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"],
            "size": cache_stats["size"],
            "max_size": cache_stats["max_size"],
            "evictions": cache_stats["evictions"]
        },
        "rate_limiter": {
            "active_requests": rate_limit_stats["active_requests"],
            "max_concurrent": rate_limit_stats["max_concurrent"],
            "total_requests": rate_limit_stats["total_requests"],
            "rate_limited": rate_limit_stats["rate_limited"],
            "concurrency_limited": rate_limit_stats["concurrency_limited"],
            "active_clients": rate_limit_stats["active_clients"],
            "requests_per_minute_limit": rate_limit_stats["requests_per_minute"]
        },
        "performance_summary": {
            "cache_enabled": True,
            "cache_ttl_seconds": 10,
            "rate_limiting_enabled": True,
            "optimization_status": "active"
        }
    }

@app.get("/export/protobuf")
async def export_signals_protobuf(
    symbol: Optional[str] = None,
    decision: Optional[str] = None,
    tier: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(1000, le=10000),
    db: Session = Depends(get_db)
):
    """导出信号为二进制Protobuf格式，供下游交易机器人使用
    
    参数:
        symbol: 按交易对过滤 (例如: BTCUSDT)
        decision: 按决策过滤 (LONG, SHORT, WAIT)
        tier: 按等级过滤 (A, B, C)
        start_time: ISO格式开始时间
        end_time: ISO格式结束时间
        limit: 最大信号数量 (最多10000)
    
    返回:
        二进制Protobuf序列化的SignalBatch
    """
    query = db.query(Signal)
    
    # 应用过滤器
    if symbol:
        query = query.filter(Signal.symbol == symbol)
    if decision:
        query = query.filter(Signal.decision == decision)
    if tier:
        query = query.filter(Signal.tier == tier)
    if start_time:
        query = query.filter(Signal.created_at >= datetime.fromisoformat(start_time))
    if end_time:
        query = query.filter(Signal.created_at <= datetime.fromisoformat(end_time))
    
    # 按时间戳排序并限制数量
    signals = query.order_by(Signal.created_at.desc()).limit(limit).all()
    
    # 转换为Protobuf批次
    export_id = f"pb_{int(datetime.utcnow().timestamp() * 1000)}"
    pb_batch = signals_to_protobuf_batch(signals, export_id=export_id)
    
    # 序列化为二进制
    binary_data = pb_batch.SerializeToString()
    
    return Response(
        content=binary_data,
        media_type="application/x-protobuf",
        headers={
            "Content-Disposition": f"attachment; filename=signals_{export_id}.pb",
            "X-Export-Count": str(len(signals)),
            "X-Export-ID": export_id
        }
    )

@app.get("/export/jsonl")
async def export_signals_jsonl(
    symbol: Optional[str] = None,
    decision: Optional[str] = None,
    tier: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = Query(1000, le=10000),
    db: Session = Depends(get_db)
):
    """导出信号为JSONL格式（换行分隔的JSON），供下游系统使用
    
    参数:
        symbol: 按交易对过滤 (例如: BTCUSDT)
        decision: 按决策过滤 (LONG, SHORT, WAIT)
        tier: 按等级过滤 (A, B, C)
        start_time: ISO格式开始时间
        end_time: ISO格式结束时间
        limit: 最大信号数量 (最多10000)
    
    返回:
        JSONL文本文件（每行一个JSON对象）
    """
    query = db.query(Signal)
    
    # 应用过滤器
    if symbol:
        query = query.filter(Signal.symbol == symbol)
    if decision:
        query = query.filter(Signal.decision == decision)
    if tier:
        query = query.filter(Signal.tier == tier)
    if start_time:
        query = query.filter(Signal.created_at >= datetime.fromisoformat(start_time))
    if end_time:
        query = query.filter(Signal.created_at <= datetime.fromisoformat(end_time))
    
    # 按时间戳排序并限制数量
    signals = query.order_by(Signal.created_at.desc()).limit(limit).all()
    
    # 转换为JSONL
    export_id = f"jsonl_{int(datetime.utcnow().timestamp() * 1000)}"
    jsonl_data = signals_to_jsonl(signals)
    
    return Response(
        content=jsonl_data,
        media_type="application/x-ndjson",
        headers={
            "Content-Disposition": f"attachment; filename=signals_{export_id}.jsonl",
            "X-Export-Count": str(len(signals)),
            "X-Export-ID": export_id
        }
    )

if __name__ == "__main__":
    logger.info("Starting Crypto Surge Prediction API Server (Demo Mode)")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
