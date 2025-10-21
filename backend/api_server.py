"""
Standalone API server for crypto surge prediction dashboard.
Demo mode with mock data - doesn't require Redis/ClickHouse.
"""

import time
import logging
import numpy as np
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from fastapi import FastAPI, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn

from backend.database import db_manager, get_db, ModelVersion, Signal, Prediction

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

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    logger.info("Initializing database...")
    db_manager.initialize()
    
    # Create default model version if not exists
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

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections"""
    db_manager.close()

def generate_mock_time_series(days: int, start_value: float = 0.0) -> Dict:
    """Generate mock time series data"""
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
    """Save a generated signal to the database"""
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
    """Health check endpoint"""
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
    """Report 1: Real-time Signal Card"""
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
    
    # Save signal to database (5min horizon)
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

@app.get("/reports/regime")
async def get_regime_state(symbol: str = Query(..., description="Trading symbol")):
    """Report 2: Market Regime & Liquidity State"""
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

@app.get("/reports/window")
async def get_probability_window(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold")
):
    """Report 3: Pre-Surge Probability & Time Window"""
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

@app.get("/reports/cost")
async def get_cost_capacity(symbol: str = Query(..., description="Trading symbol")):
    """Report 4: Execution Cost & Capacity Analysis"""
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

@app.get("/reports/backtest")
async def get_backtest_performance(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold"),
    days_back: int = Query(30, description="Days to backtest")
):
    """Report 5: Historical Backtest Performance"""
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
    """Report 6: Model Calibration & Error Analysis"""
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
    """Report 7: Event Attribution & Strategy Comparison"""
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
    """Get recent trading signals"""
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
    """Get signal history for a specific symbol"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    signals = db.query(Signal).filter(
        Signal.symbol == symbol,
        Signal.created_at >= cutoff_time
    ).order_by(Signal.created_at.asc()).all()
    
    # Calculate stats
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

@app.get("/signals/stats")
async def get_signal_stats(
    db: Session = Depends(get_db)
):
    """Get overall signal statistics"""
    # Get signals from last 24 hours
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    recent_signals = db.query(Signal).filter(Signal.created_at >= cutoff_time).all()
    
    # Calculate statistics by symbol
    symbols_stats = {}
    for signal in recent_signals:
        if signal.symbol not in symbols_stats:
            symbols_stats[signal.symbol] = {
                'total': 0,
                'a_tier': 0,
                'b_tier': 0,
                'long': 0,
                'utilities': [],
                'latencies': []
            }
        
        stats = symbols_stats[signal.symbol]
        stats['total'] += 1
        if signal.tier == 'A':
            stats['a_tier'] += 1
        elif signal.tier == 'B':
            stats['b_tier'] += 1
        if signal.decision == 'LONG':
            stats['long'] += 1
        stats['utilities'].append(signal.net_utility)
        stats['latencies'].append(signal.sla_latency_ms)
    
    # Format results
    results = {}
    for symbol, stats in symbols_stats.items():
        results[symbol] = {
            'total_signals': stats['total'],
            'a_tier_count': stats['a_tier'],
            'b_tier_count': stats['b_tier'],
            'long_count': stats['long'],
            'avg_utility': float(np.mean(stats['utilities'])) if stats['utilities'] else 0,
            'avg_latency_ms': float(np.mean(stats['latencies'])) if stats['latencies'] else 0
        }
    
    return {
        "period": "last_24_hours",
        "total_signals": len(recent_signals),
        "by_symbol": results
    }

@app.get("/models")
async def get_model_versions(
    db: Session = Depends(get_db)
):
    """Get all model versions"""
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

if __name__ == "__main__":
    logger.info("Starting Crypto Surge Prediction API Server (Demo Mode)")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
