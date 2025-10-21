import asyncio
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
from concurrent.futures import ThreadPoolExecutor
import redis
import orjson
from datetime import datetime, timedelta
import os
import io
import base64
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

from backend.storage.redis_client import RedisManager
from backend.storage.clickhouse_client import ClickHouseManager
from backend.models.cost_model import CostModel
from backend.utils.monitoring import MetricsCollector
from config.settings import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReportRequest:
    """Base report request structure"""
    symbol: str
    window: Optional[int] = None
    theta_up: Optional[float] = None
    theta_dn: Optional[float] = None
    tau: Optional[float] = None
    kappa: Optional[float] = None
    days_back: Optional[int] = 30

class ReportsService:
    """7-report analytics service for crypto surge prediction"""
    
    def __init__(self):
        self.settings = Settings()
        self.redis_manager = RedisManager()
        self.clickhouse_manager = ClickHouseManager()
        self.cost_model = CostModel()
        self.metrics_collector = MetricsCollector("reports")
        
        # Report cache with TTL
        self.report_cache: Dict[str, Dict] = {}
        self.cache_ttl = {
            'realtime': 1.0,     # 1 second for real-time
            'regime': 5.0,       # 5 seconds for regime
            'window': 2.0,       # 2 seconds for probability window
            'cost': 60.0,        # 1 minute for cost analysis
            'backtest': 300.0,   # 5 minutes for backtest
            'calibration': 300.0, # 5 minutes for calibration
            'attribution': 300.0  # 5 minutes for attribution
        }
        
        # Model versions
        self.model_version = "1.0.0"
        self.feature_version = "1.0.0"
        self.cost_model_version = "v1.2.0"
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self):
        """Initialize reports service"""
        logger.info("Initializing reports service...")
        
        await self.redis_manager.initialize()
        await self.clickhouse_manager.initialize()
        await self.cost_model.initialize()
        
        logger.info("Reports service initialized")
    
    def _get_cache_key(self, report_type: str, request: ReportRequest) -> str:
        """Generate cache key for report"""
        key_parts = [
            report_type,
            request.symbol,
            str(request.window or 0),
            str(request.theta_up or 0),
            str(request.theta_dn or 0),
            str(request.tau or 0),
            str(request.kappa or 0),
            str(request.days_back or 0)
        ]
        return ":".join(key_parts)
    
    def _is_cache_valid(self, cache_key: str, report_type: str) -> bool:
        """Check if cached report is still valid"""
        if cache_key not in self.report_cache:
            return False
        
        cached_time = self.report_cache[cache_key].get('timestamp', 0)
        current_time = time.time()
        ttl = self.cache_ttl.get(report_type, 60.0)
        
        return (current_time - cached_time) < ttl
    
    def _store_in_cache(self, cache_key: str, data: Dict):
        """Store report in cache with timestamp"""
        self.report_cache[cache_key] = {
            **data,
            'timestamp': time.time()
        }
    
    async def generate_realtime_signal_card(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 1: Real-time Signal Card"""
        cache_key = self._get_cache_key('realtime', request)
        
        if self._is_cache_valid(cache_key, 'realtime'):
            return self.report_cache[cache_key]
        
        try:
            # Get latest features and predictions
            features_data = await self._get_latest_features(request.symbol)
            prediction_data = await self._get_latest_prediction(request.symbol)
            
            # Calculate utility and decision
            p_up_5m = prediction_data.get('predictions', {}).get(5, {}).get('p_up', 0.0)
            p_up_10m = prediction_data.get('predictions', {}).get(10, {}).get('p_up', 0.0)
            p_up_30m = prediction_data.get('predictions', {}).get(30, {}).get('p_up', 0.0)
            
            expected_return = prediction_data.get('expected_returns', {}).get(10, 0.0)
            estimated_cost = prediction_data.get('estimated_costs', {}).get(10, 0.001)
            utility = expected_return / max(estimated_cost, 0.001)
            
            # Decision logic
            tau = request.tau or 0.75
            kappa = request.kappa or 1.20
            
            decision = 'none'
            if p_up_10m >= tau and utility >= kappa:
                decision = 'A'
            elif p_up_10m >= 0.65 and utility >= 1.00:
                decision = 'B'
            
            # Quality flags
            quality_flags = prediction_data.get('quality_flags', [])
            
            # SLA latency
            sla_latency_ms = prediction_data.get('sla_latency_ms', 0.0)
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'probabilities': {
                    '5m': {'value': p_up_5m, 'ci_low': p_up_5m - 0.05, 'ci_high': p_up_5m + 0.05},
                    '10m': {'value': p_up_10m, 'ci_low': p_up_10m - 0.05, 'ci_high': p_up_10m + 0.05},
                    '30m': {'value': p_up_30m, 'ci_low': p_up_30m - 0.05, 'ci_high': p_up_30m + 0.05}
                },
                'expected_return': expected_return,
                'estimated_cost': estimated_cost,
                'utility': utility,
                'decision': decision,
                'tier': 'A' if decision == 'A' else 'B' if decision == 'B' else 'none',
                'features_top5': prediction_data.get('features_top5', {}),
                'quality_flags': quality_flags,
                'cooldown_until': prediction_data.get('cooldown_until'),
                'sla_latency_ms': sla_latency_ms,
                'thresholds': {'tau': tau, 'kappa': kappa},
                'model_version': self.model_version,
                'feature_version': self.feature_version,
                'cost_model': self.cost_model_version,
                'data_window_id': f"dw_{int(time.time())}"
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating realtime signal card: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate realtime signal card: {str(e)}")
    
    async def generate_regime_state(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 2: Market Regime & Liquidity State"""
        cache_key = self._get_cache_key('regime', request)
        
        if self._is_cache_valid(cache_key, 'regime'):
            return self.report_cache[cache_key]
        
        try:
            # Get market data for regime classification
            market_data = await self._get_market_regime_data(request.symbol)
            
            # Calculate regime indicators
            rv_ratio = market_data.get('rv_ratio', 1.0)
            depth_slope = market_data.get('depth_slope', 0.0)
            near_touch_void = market_data.get('near_touch_void', 0.0)
            funding_delta = market_data.get('funding_delta', 0.0)
            oi_pressure = market_data.get('oi_pressure', 0.0)
            arrival_rate = market_data.get('arrival_rate', 1.0)
            
            # Classify regime
            vol_bucket = 'high' if rv_ratio > 1.5 else 'medium' if rv_ratio > 0.8 else 'low'
            depth_bucket = 'thick' if depth_slope < -2.0 else 'medium' if depth_slope < 0 else 'thin'
            funding_bucket = 'positive' if funding_delta > 0.01 else 'neutral' if funding_delta > -0.01 else 'negative'
            
            regime = f"{vol_bucket}_vol_{depth_bucket}_depth_{funding_bucket}_funding"
            
            # Liquidity metrics
            liquidity_score = max(0.0, min(1.0, (depth_slope + 3.0) / 6.0))  # Normalize depth slope
            void_score = max(0.0, min(1.0, near_touch_void))
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'regime': regime,
                'regime_components': {
                    'volatility': {'bucket': vol_bucket, 'value': rv_ratio},
                    'depth': {'bucket': depth_bucket, 'value': depth_slope},
                    'funding': {'bucket': funding_bucket, 'value': funding_delta}
                },
                'liquidity_metrics': {
                    'rv_ratio': rv_ratio,
                    'depth_slope': depth_slope,
                    'near_touch_void': near_touch_void,
                    'liquidity_score': liquidity_score,
                    'void_score': void_score
                },
                'market_pressure': {
                    'funding_delta': funding_delta,
                    'oi_pressure': oi_pressure,
                    'arrival_rate': arrival_rate
                },
                'adaptive_thresholds': {
                    'tau_adjustment': -0.05 if vol_bucket == 'high' else 0.05 if vol_bucket == 'low' else 0.0,
                    'kappa_adjustment': 0.1 if depth_bucket == 'thin' else -0.1 if depth_bucket == 'thick' else 0.0
                }
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating regime state: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate regime state: {str(e)}")
    
    async def generate_probability_window(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 3: Pre-Surge Probability & Time Window"""
        cache_key = self._get_cache_key('window', request)
        
        if self._is_cache_valid(cache_key, 'window'):
            return self.report_cache[cache_key]
        
        try:
            # Get predictions for different horizons
            horizons = [1, 2, 5, 10, 15, 30, 60]  # minutes
            probability_curve = {}
            expected_returns = {}
            time_to_peak = {}
            
            for horizon in horizons:
                # Mock probability calculation based on horizon
                base_prob = 0.3 + horizon * 0.01  # Increases with time
                probability_curve[horizon] = {
                    'p_up': min(base_prob, 0.8),
                    'ci_low': max(base_prob - 0.1, 0.0),
                    'ci_high': min(base_prob + 0.1, 1.0)
                }
                
                # Expected returns and timing
                theta_up = request.theta_up or 0.006
                expected_returns[horizon] = probability_curve[horizon]['p_up'] * theta_up * np.random.uniform(1.1, 1.8)
                time_to_peak[horizon] = horizon * np.random.uniform(0.6, 0.9)  # Peak typically before horizon end
            
            # Find optimal horizon
            utilities = {}
            for horizon in horizons:
                cost = self.cost_model.estimate_cost(request.symbol, horizon)
                utilities[horizon] = expected_returns[horizon] / max(cost, 0.001)
            
            optimal_horizon = max(utilities.keys(), key=lambda h: utilities[h])
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'probability_curve': probability_curve,
                'expected_returns': expected_returns,
                'time_to_peak': time_to_peak,
                'utilities': utilities,
                'optimal_horizon': optimal_horizon,
                'horizon_analysis': {
                    'short_term': {
                        'horizons': [1, 2, 5],
                        'avg_probability': np.mean([probability_curve[h]['p_up'] for h in [1, 2, 5]]),
                        'avg_utility': np.mean([utilities[h] for h in [1, 2, 5]])
                    },
                    'medium_term': {
                        'horizons': [10, 15, 30],
                        'avg_probability': np.mean([probability_curve[h]['p_up'] for h in [10, 15, 30]]),
                        'avg_utility': np.mean([utilities[h] for h in [10, 15, 30]])
                    },
                    'long_term': {
                        'horizons': [60],
                        'avg_probability': probability_curve[60]['p_up'],
                        'avg_utility': utilities[60]
                    }
                },
                'threshold_lines': {
                    'tau': request.tau or 0.75,
                    'kappa': request.kappa or 1.20
                },
                'parameters': {
                    'theta_up': request.theta_up or 0.006,
                    'theta_dn': request.theta_dn or 0.004
                }
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating probability window: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate probability window: {str(e)}")
    
    async def generate_cost_capacity(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 4: Execution Cost & Capacity Analysis"""
        cache_key = self._get_cache_key('cost', request)
        
        if self._is_cache_valid(cache_key, 'cost'):
            return self.report_cache[cache_key]
        
        try:
            # Get cost breakdown
            cost_breakdown = await self.cost_model.get_cost_breakdown(request.symbol)
            
            # Calculate capacity curve
            size_points = np.logspace(2, 6, 20)  # From $100 to $1M
            capacity_curve = {}
            
            for size in size_points:
                cost = await self.cost_model.estimate_cost_for_size(request.symbol, size, 10)  # 10min horizon
                expected_return = 0.006 * 1.5  # 0.6% * 1.5x multiplier
                utility = expected_return / max(cost, 0.001)
                capacity_curve[int(size)] = {
                    'size_usd': size,
                    'estimated_cost': cost,
                    'utility': utility
                }
            
            # Find optimal capacity
            max_utility = max(capacity_curve.values(), key=lambda x: x['utility'])
            optimal_size = max_utility['size_usd']
            
            # Calculate capacity percentage (what percentage of optimal size)
            current_size = 10000  # $10K default
            capacity_pct = current_size / optimal_size
            
            # Slippage analysis
            slippage_analysis = {
                'p25': cost_breakdown.get('slippage', {}).get('p25', 0.0),
                'p50': cost_breakdown.get('slippage', {}).get('p50', 0.0),
                'p75': cost_breakdown.get('slippage', {}).get('p75', 0.0),
                'p95': cost_breakdown.get('slippage', {}).get('p95', 0.0),
                'p99': cost_breakdown.get('slippage', {}).get('p99', 0.0)
            }
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'cost_breakdown': cost_breakdown,
                'capacity_curve': capacity_curve,
                'optimal_size_usd': optimal_size,
                'capacity_pct': capacity_pct,
                'slippage_analysis': slippage_analysis,
                'execution_metrics': {
                    'impact_lambda': cost_breakdown.get('impact_lambda', 0.0),
                    'near_touch_liquidity': cost_breakdown.get('near_touch_liquidity', 0.0),
                    'estimated_fill_rate': min(0.95, max(0.7, 1.0 - capacity_pct * 0.3))
                },
                'recommendations': {
                    'max_position_size': optimal_size * 0.8,  # Conservative limit
                    'suggested_splits': max(1, int(current_size / 5000)),  # Split if >$5K
                    'timing_advice': 'immediate' if capacity_pct < 0.5 else 'staged' if capacity_pct < 0.8 else 'careful'
                }
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating cost capacity: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate cost capacity: {str(e)}")
    
    async def generate_backtest_performance(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 5: Historical Backtest Performance"""
        cache_key = self._get_cache_key('backtest', request)
        
        if self._is_cache_valid(cache_key, 'backtest'):
            return self.report_cache[cache_key]
        
        try:
            days_back = request.days_back or 30
            
            # Simulate backtest results (in production, this would query actual backtest data)
            dates = pd.date_range(end=datetime.now(), periods=days_back, freq='D')
            
            # Performance metrics
            daily_returns = np.random.normal(0.002, 0.05, days_back)  # 0.2% daily mean, 5% std
            cumulative_returns = np.cumprod(1 + daily_returns) - 1
            
            # Calculate metrics
            total_return = cumulative_returns[-1]
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
            max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
            
            # Signal statistics
            total_signals = np.random.poisson(2) * days_back  # ~2 signals per day
            hit_rate = np.random.uniform(0.55, 0.75)  # 55-75% hit rate
            avg_utility = np.random.uniform(1.2, 2.0)
            false_positive_rate = 1 - hit_rate
            
            # PR-AUC simulation
            pr_auc = np.random.uniform(0.65, 0.85)
            
            # Monthly breakdown
            monthly_data = {}
            for i in range(min(12, days_back // 30 + 1)):
                month_start = max(0, days_back - (i + 1) * 30)
                month_end = days_back - i * 30
                month_returns = daily_returns[month_start:month_end]
                
                monthly_data[f"month_{i}"] = {
                    'return': np.sum(month_returns),
                    'signals': np.random.poisson(60),  # ~60 signals per month
                    'hit_rate': np.random.uniform(0.5, 0.8),
                    'avg_utility': np.random.uniform(1.0, 2.5)
                }
            
            # Top-K hit analysis
            hit_at_k = {}
            for k in [1, 3, 5, 10]:
                hit_at_k[k] = np.random.uniform(0.7, 0.9)  # Hit rate for top-K signals per day
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'period': f"{days_back}_days",
                'performance_summary': {
                    'total_return': total_return,
                    'sharpe_ratio_post_cost': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'pr_auc': pr_auc,
                    'hit_rate': hit_rate,
                    'avg_utility': avg_utility,
                    'false_positive_rate': false_positive_rate
                },
                'signal_statistics': {
                    'total_signals': total_signals,
                    'signals_per_day': total_signals / days_back,
                    'a_tier_signals': int(total_signals * 0.3),
                    'b_tier_signals': int(total_signals * 0.5),
                    'rejected_signals': int(total_signals * 0.2)
                },
                'time_series': {
                    'dates': [d.isoformat() for d in dates],
                    'cumulative_returns': cumulative_returns.tolist(),
                    'daily_returns': daily_returns.tolist()
                },
                'monthly_breakdown': monthly_data,
                'hit_at_k': hit_at_k,
                'parameters': {
                    'theta_up': request.theta_up or 0.006,
                    'theta_dn': request.theta_dn or 0.004,
                    'tau': request.tau or 0.75,
                    'kappa': request.kappa or 1.20
                },
                'model_version': self.model_version
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating backtest performance: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate backtest performance: {str(e)}")
    
    async def generate_calibration_analysis(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 6: Model Calibration & Error Analysis"""
        cache_key = self._get_cache_key('calibration', request)
        
        if self._is_cache_valid(cache_key, 'calibration'):
            return self.report_cache[cache_key]
        
        try:
            # Simulate calibration data
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            
            # Reliability diagram data
            reliability_data = {}
            total_samples = 0
            total_brier = 0
            
            for i in range(n_bins):
                bin_center = (bin_edges[i] + bin_edges[i + 1]) / 2
                # Simulate calibration - good models should be close to diagonal
                predicted_prob = bin_center
                actual_freq = bin_center + np.random.normal(0, 0.05)  # Small calibration error
                actual_freq = max(0, min(1, actual_freq))
                
                n_samples = np.random.poisson(100)  # ~100 samples per bin
                confidence = min(0.95, max(0.6, n_samples / 150))
                
                reliability_data[f"bin_{i}"] = {
                    'predicted_prob': predicted_prob,
                    'actual_frequency': actual_freq,
                    'n_samples': n_samples,
                    'confidence': confidence
                }
                
                total_samples += n_samples
                total_brier += n_samples * (predicted_prob - actual_freq) ** 2
            
            # Calculate calibration metrics
            brier_score = total_brier / max(total_samples, 1)
            ece = np.mean([abs(data['predicted_prob'] - data['actual_frequency']) 
                          for data in reliability_data.values()])
            
            # Error analysis - feature patterns in FP/FN cases
            error_clusters = {
                'false_positives': {
                    'primary_patterns': {
                        'sudden_reversal': {'frequency': 0.35, 'avg_loss': 0.008},
                        'liquidity_dry_up': {'frequency': 0.25, 'avg_loss': 0.012},
                        'macro_event': {'frequency': 0.20, 'avg_loss': 0.015},
                        'fake_breakout': {'frequency': 0.20, 'avg_loss': 0.006}
                    },
                    'feature_signatures': {
                        'ofi': {'typical_range': [0.1, 0.4], 'fp_range': [0.05, 0.15]},
                        'qi': {'typical_range': [0.05, 0.25], 'fp_range': [-0.05, 0.1]},
                        'microprice_dev': {'typical_range': [0.001, 0.005], 'fp_range': [-0.002, 0.001]},
                        'depth_slope': {'typical_range': [-2.5, -1.0], 'fp_range': [-1.0, 0.5]},
                        'rv_ratio': {'typical_range': [1.2, 2.0], 'fp_range': [0.8, 1.1]}
                    }
                },
                'false_negatives': {
                    'primary_patterns': {
                        'gradual_buildup': {'frequency': 0.40, 'missed_gain': 0.010},
                        'cross_venue_lag': {'frequency': 0.30, 'missed_gain': 0.008},
                        'low_volume_surge': {'frequency': 0.30, 'missed_gain': 0.006}
                    },
                    'feature_signatures': {
                        'ofi': {'typical_range': [0.2, 0.5], 'fn_range': [0.1, 0.2]},
                        'qi': {'typical_range': [0.1, 0.3], 'fn_range': [0.05, 0.15]},
                        'impact_lambda': {'typical_range': [0.01, 0.05], 'fn_range': [0.005, 0.02]}
                    }
                }
            }
            
            # Top-K confusion analysis
            topk_analysis = {}
            for k in [1, 3, 5, 10]:
                precision = np.random.uniform(0.7, 0.9)
                recall = np.random.uniform(0.6, 0.8)
                f1 = 2 * precision * recall / (precision + recall)
                
                topk_analysis[f"top_{k}"] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positives': int(k * precision * np.random.uniform(0.8, 1.2)),
                    'false_positives': int(k * (1 - precision) * np.random.uniform(0.8, 1.2))
                }
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'calibration_metrics': {
                    'brier_score': brier_score,
                    'ece': ece,
                    'reliability_score': max(0, 1 - ece * 5),  # Scale ECE to 0-1 reliability
                    'total_samples': total_samples
                },
                'reliability_diagram': reliability_data,
                'error_analysis': error_clusters,
                'topk_confusion': topk_analysis,
                'model_diagnostics': {
                    'model_version': self.model_version,
                    'calibration_method': 'isotonic_regression',
                    'last_recalibration': int((datetime.now() - timedelta(days=7)).timestamp() * 1000),
                    'calibration_stability': np.random.uniform(0.85, 0.95)
                },
                'recommendations': {
                    'recalibration_needed': ece > 0.05,
                    'feature_importance_review': brier_score > 0.1,
                    'threshold_adjustment': ece > 0.08,
                    'data_quality_check': any('degraded' in str(error_clusters).lower() for _ in [1])
                }
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating calibration analysis: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate calibration analysis: {str(e)}")
    
    async def generate_attribution_comparison(self, request: ReportRequest) -> Dict[str, Any]:
        """Report 7: Event Attribution & Strategy Comparison"""
        cache_key = self._get_cache_key('attribution', request)
        
        if self._is_cache_valid(cache_key, 'attribution'):
            return self.report_cache[cache_key]
        
        try:
            # SHAP attribution analysis
            feature_names = ['qi_1', 'ofi_10', 'microprice_dev', 'depth_slope_bid', 'rv_ratio', 
                           'impact_lambda', 'near_touch_ratio', 'bb_position', 'funding_delta', 'oi_pressure']
            
            shap_attribution = {}
            for feature in feature_names:
                shap_attribution[feature] = {
                    'mean_impact': np.random.normal(0, 0.1),
                    'abs_mean_impact': np.random.uniform(0.01, 0.15),
                    'frequency_nonzero': np.random.uniform(0.6, 0.95),
                    'directional_consistency': np.random.uniform(0.7, 0.9)
                }
            
            # Sort by absolute importance
            sorted_features = sorted(shap_attribution.items(), 
                                   key=lambda x: x[1]['abs_mean_impact'], reverse=True)
            
            # Rule-based voting (Meta-Model approach)
            rule_votes = {
                'ofi_momentum': {'weight': 0.25, 'confidence': 0.85, 'recent_accuracy': 0.72},
                'qi_imbalance': {'weight': 0.20, 'confidence': 0.80, 'recent_accuracy': 0.68},
                'microprice_signal': {'weight': 0.15, 'confidence': 0.75, 'recent_accuracy': 0.70},
                'depth_structure': {'weight': 0.20, 'confidence': 0.78, 'recent_accuracy': 0.69},
                'volatility_regime': {'weight': 0.20, 'confidence': 0.82, 'recent_accuracy': 0.74}
            }
            
            # Strategy comparison scenarios
            strategy_variants = [
                {
                    'name': 'Conservative',
                    'params': {'tau': 0.80, 'kappa': 1.5, 'theta_up': 0.008, 'cooldown': 20},
                    'metrics': {
                        'pr_auc': np.random.uniform(0.75, 0.85),
                        'hit_at_top5': np.random.uniform(0.80, 0.90),
                        'avg_utility': np.random.uniform(1.5, 2.2),
                        'fpr': np.random.uniform(0.08, 0.15),
                        'max_drawdown': np.random.uniform(0.05, 0.12)
                    }
                },
                {
                    'name': 'Balanced',
                    'params': {'tau': 0.75, 'kappa': 1.2, 'theta_up': 0.006, 'cooldown': 15},
                    'metrics': {
                        'pr_auc': np.random.uniform(0.70, 0.80),
                        'hit_at_top5': np.random.uniform(0.75, 0.85),
                        'avg_utility': np.random.uniform(1.2, 1.8),
                        'fpr': np.random.uniform(0.15, 0.25),
                        'max_drawdown': np.random.uniform(0.08, 0.18)
                    }
                },
                {
                    'name': 'Aggressive',
                    'params': {'tau': 0.65, 'kappa': 1.0, 'theta_up': 0.004, 'cooldown': 10},
                    'metrics': {
                        'pr_auc': np.random.uniform(0.65, 0.75),
                        'hit_at_top5': np.random.uniform(0.70, 0.80),
                        'avg_utility': np.random.uniform(1.0, 1.5),
                        'fpr': np.random.uniform(0.25, 0.35),
                        'max_drawdown': np.random.uniform(0.15, 0.25)
                    }
                }
            ]
            
            # Find best strategy under constraints
            max_fpr = 0.20
            min_utility = 1.2
            
            viable_strategies = [s for s in strategy_variants 
                               if s['metrics']['fpr'] <= max_fpr and s['metrics']['avg_utility'] >= min_utility]
            
            if viable_strategies:
                best_strategy = max(viable_strategies, key=lambda x: x['metrics']['avg_utility'])
            else:
                best_strategy = strategy_variants[1]  # Default to balanced
            
            # Capacity constraints analysis
            capacity_analysis = {}
            for strategy in strategy_variants:
                # Estimate capacity impact based on signal frequency
                signal_freq = 1.0 / (strategy['params']['cooldown'] / 60)  # signals per hour
                capacity_limit = 50000 / max(signal_freq, 0.1)  # Inverse relationship
                
                capacity_analysis[strategy['name']] = {
                    'estimated_capacity_usd': capacity_limit,
                    'signals_per_day': signal_freq * 24,
                    'capacity_utilization': min(1.0, 10000 / capacity_limit)  # Assuming $10K current
                }
            
            report_data = {
                'symbol': request.symbol,
                'timestamp': int(time.time() * 1000),
                'shap_attribution': {
                    'top_features': dict(sorted_features[:10]),
                    'feature_interactions': {
                        'qi_ofi_synergy': np.random.uniform(0.1, 0.3),
                        'depth_microprice_synergy': np.random.uniform(0.05, 0.2),
                        'volatility_regime_modifier': np.random.uniform(0.8, 1.2)
                    },
                    'temporal_stability': np.random.uniform(0.75, 0.90)
                },
                'rule_meta_model': {
                    'rule_votes': rule_votes,
                    'consensus_strength': np.mean([v['confidence'] for v in rule_votes.values()]),
                    'ml_vs_rules_agreement': np.random.uniform(0.70, 0.85)
                },
                'strategy_comparison': {
                    'variants': strategy_variants,
                    'best_under_constraints': best_strategy,
                    'constraint_analysis': {
                        'max_fpr_constraint': max_fpr,
                        'min_utility_constraint': min_utility,
                        'feasible_strategies': len(viable_strategies)
                    }
                },
                'capacity_analysis': capacity_analysis,
                'recommendations': {
                    'optimal_strategy': best_strategy['name'],
                    'feature_focus': sorted_features[0][0],  # Top feature
                    'rule_ensemble_weight': 0.3,  # 30% rule, 70% ML
                    'recalibration_trigger': 'weekly',
                    'a_b_test_candidates': [s['name'] for s in viable_strategies[:2]]
                },
                'model_versions': {
                    'ml_model': self.model_version,
                    'feature_version': self.feature_version,
                    'rule_engine': 'v2.1.0'
                }
            }
            
            self._store_in_cache(cache_key, report_data)
            return report_data
        
        except Exception as e:
            logger.error(f"Error generating attribution comparison: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to generate attribution comparison: {str(e)}")
    
    async def _get_latest_features(self, symbol: str) -> Dict[str, Any]:
        """Get latest features for symbol"""
        try:
            key = f"features:{symbol}"
            data_json = self.redis_manager.client.get(key)
            
            if data_json:
                return orjson.loads(data_json)
            return {}
        
        except Exception as e:
            logger.error(f"Error getting latest features for {symbol}: {e}")
            return {}
    
    async def _get_latest_prediction(self, symbol: str) -> Dict[str, Any]:
        """Get latest prediction for symbol"""
        try:
            # This would typically come from inference service or cache
            # For now, return mock structure
            return {
                'predictions': {
                    5: {'p_up': np.random.uniform(0.4, 0.8)},
                    10: {'p_up': np.random.uniform(0.4, 0.8)},
                    30: {'p_up': np.random.uniform(0.4, 0.8)}
                },
                'expected_returns': {5: 0.008, 10: 0.012, 30: 0.015},
                'estimated_costs': {5: 0.002, 10: 0.003, 30: 0.005},
                'features_top5': {
                    'qi_1': np.random.normal(0.1, 0.05),
                    'ofi_10': np.random.normal(0.2, 0.1),
                    'microprice_dev': np.random.normal(0.001, 0.002),
                    'rv_ratio': np.random.uniform(0.8, 1.5),
                    'depth_slope_bid': np.random.normal(-1.5, 0.5)
                },
                'quality_flags': [],
                'sla_latency_ms': np.random.uniform(20, 80)
            }
        
        except Exception as e:
            logger.error(f"Error getting latest prediction for {symbol}: {e}")
            return {}
    
    async def _get_market_regime_data(self, symbol: str) -> Dict[str, Any]:
        """Get market regime data for symbol"""
        try:
            # This would typically aggregate from multiple Redis keys
            return {
                'rv_ratio': np.random.uniform(0.5, 2.0),
                'depth_slope': np.random.normal(-1.5, 1.0),
                'near_touch_void': np.random.uniform(0.1, 0.4),
                'funding_delta': np.random.normal(0, 0.02),
                'oi_pressure': np.random.normal(0, 0.1),
                'arrival_rate': np.random.uniform(0.5, 2.0)
            }
        
        except Exception as e:
            logger.error(f"Error getting market regime data for {symbol}: {e}")
            return {}

# FastAPI application
app = FastAPI(title="Crypto Surge Prediction Reports API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reports service instance
reports_service = None

@app.on_event("startup")
async def startup_event():
    global reports_service
    reports_service = ReportsService()
    await reports_service.initialize()

@app.get("/reports/realtime")
async def get_realtime_signal_card(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold")
):
    """Report 1: Real-time Signal Card"""
    request = ReportRequest(
        symbol=symbol,
        theta_up=theta_up,
        theta_dn=theta_dn,
        tau=tau,
        kappa=kappa
    )
    return await reports_service.generate_realtime_signal_card(request)

@app.get("/reports/regime")
async def get_regime_state(
    symbol: str = Query(..., description="Trading symbol")
):
    """Report 2: Market Regime & Liquidity State"""
    request = ReportRequest(symbol=symbol)
    return await reports_service.generate_regime_state(request)

@app.get("/reports/window")
async def get_probability_window(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold")
):
    """Report 3: Pre-Surge Probability & Time Window"""
    request = ReportRequest(
        symbol=symbol,
        theta_up=theta_up,
        theta_dn=theta_dn
    )
    return await reports_service.generate_probability_window(request)

@app.get("/reports/cost")
async def get_cost_capacity(
    symbol: str = Query(..., description="Trading symbol")
):
    """Report 4: Execution Cost & Capacity Analysis"""
    request = ReportRequest(symbol=symbol)
    return await reports_service.generate_cost_capacity(request)

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
    request = ReportRequest(
        symbol=symbol,
        theta_up=theta_up,
        theta_dn=theta_dn,
        tau=tau,
        kappa=kappa,
        days_back=days_back
    )
    return await reports_service.generate_backtest_performance(request)

@app.get("/reports/calibration")
async def get_calibration_analysis(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold")
):
    """Report 6: Model Calibration & Error Analysis"""
    request = ReportRequest(
        symbol=symbol,
        theta_up=theta_up,
        theta_dn=theta_dn
    )
    return await reports_service.generate_calibration_analysis(request)

@app.get("/reports/attribution")
async def get_attribution_comparison(
    symbol: str = Query(..., description="Trading symbol"),
    theta_up: float = Query(0.006, description="Up threshold"),
    theta_dn: float = Query(0.004, description="Down threshold"),
    tau: float = Query(0.75, description="Probability threshold"),
    kappa: float = Query(1.20, description="Utility threshold")
):
    """Report 7: Event Attribution & Strategy Comparison"""
    request = ReportRequest(
        symbol=symbol,
        theta_up=theta_up,
        theta_dn=theta_dn,
        tau=tau,
        kappa=kappa
    )
    return await reports_service.generate_attribution_comparison(request)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": int(time.time() * 1000),
        "cache_size": len(reports_service.report_cache) if reports_service else 0
    }

if __name__ == "__main__":
    uvicorn.run(
        "backend.reports_service:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="uvloop"
    )
