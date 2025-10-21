"""
Standalone API server for crypto surge prediction dashboard.
Demo mode with mock data - doesn't require Redis/ClickHouse.
"""

import time
import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

def generate_mock_time_series(days: int, start_value: float = 0.0) -> Dict:
    """Generate mock time series data"""
    dates = [(datetime.now() - timedelta(days=days-i)).isoformat() for i in range(days)]
    cumulative_returns = np.cumsum(np.random.normal(0.002, 0.02, days)).tolist()
    return {
        'dates': dates,
        'cumulative_returns': cumulative_returns
    }

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
    kappa: float = Query(1.20, description="Utility threshold")
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
        'features_top5': {
            'qi_1': np.random.uniform(-0.2, 0.3),
            'ofi_10': np.random.uniform(-0.15, 0.25),
            'microprice_dev': np.random.uniform(-0.1, 0.2),
            'rv_ratio': np.random.uniform(-0.15, 0.2),
            'depth_slope_bid': np.random.uniform(-0.1, 0.15)
        },
        'quality_flags': [],
        'sla_latency_ms': np.random.uniform(50, 200),
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

if __name__ == "__main__":
    logger.info("Starting Crypto Surge Prediction API Server (Demo Mode)")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
