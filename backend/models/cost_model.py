"""
Sophisticated execution cost modeling for crypto trading.
Implements multi-component cost estimation with market regime awareness.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from collections import defaultdict, deque
from scipy import stats, optimize
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import redis
import orjson
from numba import jit, njit

from .schemas import CostEstimate, QualityFlag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit
def calculate_power_law_impact(volume: float, lambda_param: float, psi: float = 0.5) -> float:
    """
    Numba-optimized power law market impact calculation.
    Impact = λ * Volume^ψ
    """
    if volume <= 0 or lambda_param <= 0:
        return 0.0
    
    return lambda_param * (volume ** psi)

@njit
def calculate_linear_impact(volume: float, lambda_param: float) -> float:
    """
    Numba-optimized linear market impact calculation.
    Impact = λ * Volume
    """
    if volume <= 0 or lambda_param <= 0:
        return 0.0
    
    return lambda_param * volume

@njit
def calculate_sqrt_impact(volume: float, lambda_param: float) -> float:
    """
    Numba-optimized square root market impact calculation.
    Impact = λ * sqrt(Volume)
    """
    if volume <= 0 or lambda_param <= 0:
        return 0.0
    
    return lambda_param * np.sqrt(volume)

class MarketImpactModel:
    """Market impact model with multiple functional forms"""
    
    def __init__(self):
        self.models = {
            'linear': calculate_linear_impact,
            'sqrt': calculate_sqrt_impact,
            'power': calculate_power_law_impact
        }
        self.default_model = 'sqrt'
        
        # Model parameters by regime
        self.parameters = {
            'high_vol_thin_depth': {'lambda': 0.0008, 'psi': 0.6},
            'high_vol_medium_depth': {'lambda': 0.0006, 'psi': 0.55},
            'high_vol_thick_depth': {'lambda': 0.0004, 'psi': 0.5},
            'medium_vol_thin_depth': {'lambda': 0.0006, 'psi': 0.55},
            'medium_vol_medium_depth': {'lambda': 0.0004, 'psi': 0.5},
            'medium_vol_thick_depth': {'lambda': 0.0003, 'psi': 0.45},
            'low_vol_thin_depth': {'lambda': 0.0004, 'psi': 0.5},
            'low_vol_medium_depth': {'lambda': 0.0003, 'psi': 0.45},
            'low_vol_thick_depth': {'lambda': 0.0002, 'psi': 0.4}
        }
    
    def estimate_impact(self, volume_usd: float, regime: str, 
                       model_type: Optional[str] = None) -> float:
        """Estimate market impact for given volume and regime"""
        if model_type is None:
            model_type = self.default_model
        
        if regime not in self.parameters:
            regime = 'medium_vol_medium_depth'  # Default regime
        
        params = self.parameters[regime]
        lambda_param = params['lambda']
        
        if model_type == 'power':
            psi = params.get('psi', 0.5)
            return calculate_power_law_impact(volume_usd, lambda_param, psi)
        elif model_type == 'sqrt':
            return calculate_sqrt_impact(volume_usd, lambda_param)
        else:  # linear
            return calculate_linear_impact(volume_usd, lambda_param)
    
    def calibrate_model(self, volume_data: np.ndarray, impact_data: np.ndarray, 
                       regime: str, model_type: str = 'sqrt') -> Dict[str, float]:
        """Calibrate model parameters using historical data"""
        try:
            if model_type == 'linear':
                # Linear regression: impact = λ * volume
                slope, _, _, _, _ = stats.linregress(volume_data, impact_data)
                return {'lambda': max(0, slope)}
            
            elif model_type == 'sqrt':
                # Fit impact = λ * sqrt(volume)
                sqrt_volumes = np.sqrt(volume_data)
                slope, _, _, _, _ = stats.linregress(sqrt_volumes, impact_data)
                return {'lambda': max(0, slope)}
            
            elif model_type == 'power':
                # Fit log(impact) = log(λ) + ψ * log(volume)
                log_volumes = np.log(volume_data + 1e-10)
                log_impacts = np.log(impact_data + 1e-10)
                
                # Filter valid data
                valid_mask = np.isfinite(log_volumes) & np.isfinite(log_impacts)
                if np.sum(valid_mask) < 10:
                    return self.parameters.get(regime, {'lambda': 0.0005, 'psi': 0.5})
                
                slope, intercept, _, _, _ = stats.linregress(
                    log_volumes[valid_mask], log_impacts[valid_mask]
                )
                
                return {
                    'lambda': max(0, np.exp(intercept)),
                    'psi': max(0.1, min(1.0, slope))  # Constrain psi
                }
            
        except Exception as e:
            logger.error(f"Error calibrating impact model for regime {regime}: {e}")
            return self.parameters.get(regime, {'lambda': 0.0005, 'psi': 0.5})
    
    def get_confidence_interval(self, volume_usd: float, regime: str, 
                              confidence: float = 0.95) -> Tuple[float, float]:
        """Get confidence interval for impact estimate"""
        base_impact = self.estimate_impact(volume_usd, regime)
        
        # Simple confidence interval based on regime uncertainty
        uncertainty_multiplier = {
            'high_vol_thin_depth': 1.5,
            'high_vol_medium_depth': 1.3,
            'high_vol_thick_depth': 1.2,
            'medium_vol_thin_depth': 1.3,
            'medium_vol_medium_depth': 1.1,
            'medium_vol_thick_depth': 1.0,
            'low_vol_thin_depth': 1.2,
            'low_vol_medium_depth': 1.0,
            'low_vol_thick_depth': 0.9
        }.get(regime, 1.2)
        
        # Calculate z-score for confidence level
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std_multiplier = uncertainty_multiplier * 0.3  # 30% base std
        
        lower_bound = max(0, base_impact * (1 - z_score * std_multiplier))
        upper_bound = base_impact * (1 + z_score * std_multiplier)
        
        return lower_bound, upper_bound

class SlippageModel:
    """Advanced slippage modeling with percentile estimates"""
    
    def __init__(self):
        self.regime_parameters = {
            'high_vol_thin_depth': {
                'base_slippage': 0.0012,
                'volume_sensitivity': 0.8,
                'percentiles': {
                    'p25': 0.7,
                    'p50': 1.0,
                    'p75': 1.4,
                    'p95': 2.2,
                    'p99': 3.5
                }
            },
            'medium_vol_medium_depth': {
                'base_slippage': 0.0008,
                'volume_sensitivity': 0.6,
                'percentiles': {
                    'p25': 0.8,
                    'p50': 1.0,
                    'p75': 1.3,
                    'p95': 1.8,
                    'p99': 2.5
                }
            },
            'low_vol_thick_depth': {
                'base_slippage': 0.0005,
                'volume_sensitivity': 0.4,
                'percentiles': {
                    'p25': 0.9,
                    'p50': 1.0,
                    'p75': 1.2,
                    'p95': 1.5,
                    'p99': 2.0
                }
            }
        }
        
        # Historical slippage data storage
        self.historical_slippage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.lock = threading.Lock()
    
    def estimate_slippage(self, volume_usd: float, available_liquidity: float, 
                         regime: str, percentile: str = 'p50') -> float:
        """Estimate slippage for given volume and market conditions"""
        if regime not in self.regime_parameters:
            regime = 'medium_vol_medium_depth'
        
        params = self.regime_parameters[regime]
        base_slippage = params['base_slippage']
        volume_sensitivity = params['volume_sensitivity']
        percentile_multiplier = params['percentiles'].get(percentile, 1.0)
        
        # Calculate liquidity ratio impact
        if available_liquidity > 0:
            liquidity_ratio = volume_usd / available_liquidity
            liquidity_impact = np.power(liquidity_ratio, volume_sensitivity)
        else:
            liquidity_impact = 2.0  # High impact if no liquidity data
        
        # Base slippage calculation
        estimated_slippage = base_slippage * liquidity_impact * percentile_multiplier
        
        # Cap maximum slippage
        max_slippage = 0.05  # 5% maximum
        estimated_slippage = min(estimated_slippage, max_slippage)
        
        return estimated_slippage
    
    def get_all_percentiles(self, volume_usd: float, available_liquidity: float, 
                           regime: str) -> Dict[str, float]:
        """Get slippage estimates for all percentiles"""
        percentiles = {}
        for p in ['p25', 'p50', 'p75', 'p95', 'p99']:
            percentiles[p] = self.estimate_slippage(
                volume_usd, available_liquidity, regime, p
            )
        return percentiles
    
    def record_actual_slippage(self, symbol: str, volume_usd: float, 
                             actual_slippage: float, regime: str):
        """Record actual slippage for model improvement"""
        with self.lock:
            record = {
                'timestamp': int(time.time() * 1000),
                'volume_usd': volume_usd,
                'slippage': actual_slippage,
                'regime': regime
            }
            self.historical_slippage[symbol].append(record)
    
    def update_model_parameters(self, symbol: str) -> bool:
        """Update model parameters based on historical data"""
        with self.lock:
            if symbol not in self.historical_slippage or len(self.historical_slippage[symbol]) < 50:
                return False
            
            try:
                data = list(self.historical_slippage[symbol])
                df = pd.DataFrame(data)
                
                # Group by regime and update parameters
                for regime in df['regime'].unique():
                    regime_data = df[df['regime'] == regime]
                    
                    if len(regime_data) >= 20:
                        # Calculate new percentiles
                        slippage_values = regime_data['slippage'].values
                        new_percentiles = {}
                        
                        for p_name, p_value in [('p25', 25), ('p50', 50), ('p75', 75), 
                                              ('p95', 95), ('p99', 99)]:
                            percentile_value = np.percentile(slippage_values, p_value)
                            median_value = np.percentile(slippage_values, 50)
                            
                            if median_value > 0:
                                new_percentiles[p_name] = percentile_value / median_value
                            else:
                                new_percentiles[p_name] = self.regime_parameters[regime]['percentiles'][p_name]
                        
                        # Update parameters with smoothing
                        if regime in self.regime_parameters:
                            old_percentiles = self.regime_parameters[regime]['percentiles']
                            smoothing_factor = 0.1  # 10% of new data
                            
                            for p_name in new_percentiles:
                                old_value = old_percentiles.get(p_name, 1.0)
                                new_value = new_percentiles[p_name]
                                smoothed_value = old_value * (1 - smoothing_factor) + new_value * smoothing_factor
                                self.regime_parameters[regime]['percentiles'][p_name] = smoothed_value
                        
                        logger.info(f"Updated slippage parameters for {symbol} regime {regime}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error updating slippage model for {symbol}: {e}")
                return False

class FundingCostModel:
    """Funding cost model for perpetual contracts"""
    
    def __init__(self):
        self.typical_funding_rates = {
            'BTCUSDT': 0.0001,   # 0.01% per 8 hours
            'ETHUSDT': 0.0001,
            'BNBUSDT': 0.0002,
            'ADAUSDT': 0.0003,
            'default': 0.0002
        }
        
        # Funding rate predictions (simplified)
        self.rate_predictions: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def estimate_funding_cost(self, symbol: str, holding_period_minutes: int, 
                            position_size_usd: float, current_funding_rate: Optional[float] = None) -> float:
        """Estimate funding cost for holding period"""
        try:
            # Get current funding rate or use typical
            if current_funding_rate is not None:
                base_rate = current_funding_rate
            else:
                base_rate = self.typical_funding_rates.get(symbol, self.typical_funding_rates['default'])
            
            # Funding is typically paid every 8 hours
            funding_interval_minutes = 8 * 60  # 480 minutes
            
            # Calculate number of funding payments during holding period
            num_payments = holding_period_minutes / funding_interval_minutes
            
            # Account for partial payments
            total_funding_cost = base_rate * num_payments * position_size_usd
            
            return abs(total_funding_cost)  # Always positive cost
            
        except Exception as e:
            logger.error(f"Error estimating funding cost for {symbol}: {e}")
            return position_size_usd * 0.0002 * (holding_period_minutes / 480)  # Fallback
    
    def predict_funding_rate(self, symbol: str, hours_ahead: int) -> float:
        """Predict funding rate (simplified implementation)"""
        with self.lock:
            if symbol in self.rate_predictions and self.rate_predictions[symbol]:
                recent_rates = self.rate_predictions[symbol][-24:]  # Last 24 hours
                if recent_rates:
                    # Simple trend-following prediction
                    mean_rate = np.mean(recent_rates)
                    trend = np.mean(np.diff(recent_rates)) if len(recent_rates) > 1 else 0
                    predicted_rate = mean_rate + trend * hours_ahead
                    
                    # Clamp to reasonable bounds
                    return max(-0.002, min(0.002, predicted_rate))
            
            return self.typical_funding_rates.get(symbol, self.typical_funding_rates['default'])
    
    def update_funding_rate(self, symbol: str, funding_rate: float):
        """Update funding rate data for predictions"""
        with self.lock:
            self.rate_predictions[symbol].append(funding_rate)
            # Keep only recent data
            if len(self.rate_predictions[symbol]) > 168:  # 1 week of hourly data
                self.rate_predictions[symbol] = self.rate_predictions[symbol][-168:]

class CostModel:
    """Comprehensive execution cost model"""
    
    def __init__(self):
        self.impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel()
        self.funding_model = FundingCostModel()
        
        # Fee schedules (maker/taker)
        self.fee_schedules = {
            'binance_spot': {'maker': 0.001, 'taker': 0.001},
            'binance_futures': {'maker': 0.0002, 'taker': 0.0004},
            'default': {'maker': 0.001, 'taker': 0.001}
        }
        
        # Current market state cache
        self.market_state_cache: Dict[str, Dict] = {}
        self.cache_timestamps: Dict[str, int] = {}
        self.cache_ttl_ms = 5000  # 5 second cache
        
        # Redis for cost lookup tables
        self.redis_client = None
        self.cost_lookup_cache: Dict[str, Dict] = {}
        
        # Model performance tracking
        self.prediction_errors: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=2)
    
    async def initialize(self):
        """Initialize cost model"""
        logger.info("Initializing cost model...")
        
        try:
            # Initialize Redis client (would be injected in production)
            # self.redis_client = redis.Redis(...)
            
            # Load pre-computed cost lookup tables
            await self._load_cost_lookup_tables()
            
            logger.info("Cost model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing cost model: {e}")
    
    async def _load_cost_lookup_tables(self):
        """Load pre-computed cost lookup tables from Redis"""
        # In production, this would load from Redis with keys like:
        # cost_lookup:{regime}:{symbol}:{timeframe}
        
        # For now, create default lookup tables
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        regimes = ['high_vol_thin_depth', 'medium_vol_medium_depth', 'low_vol_thick_depth']
        
        for symbol in symbols:
            for regime in regimes:
                key = f"{regime}:{symbol}:1m"
                self.cost_lookup_cache[key] = {
                    'base_slippage': self.slippage_model.regime_parameters.get(
                        regime, self.slippage_model.regime_parameters['medium_vol_medium_depth']
                    )['base_slippage'],
                    'impact_lambda': self.impact_model.parameters.get(
                        regime, self.impact_model.parameters['medium_vol_medium_depth']
                    )['lambda'],
                    'typical_funding': self.funding_model.typical_funding_rates.get(
                        symbol, self.funding_model.typical_funding_rates['default']
                    ),
                    'last_updated': int(time.time() * 1000)
                }
    
    def estimate_cost(self, symbol: str, horizon_minutes: int, 
                     position_size_usd: float = 10000, regime: Optional[str] = None) -> float:
        """Estimate total execution cost"""
        try:
            # Use cached regime or default
            if regime is None:
                regime = self._get_current_regime(symbol)
            
            # Get market state
            market_state = self._get_market_state(symbol)
            
            # Calculate individual cost components
            cost_breakdown = self._calculate_cost_breakdown(
                symbol, position_size_usd, horizon_minutes, regime, market_state
            )
            
            return cost_breakdown['total_cost_estimate']
            
        except Exception as e:
            logger.error(f"Error estimating cost for {symbol}: {e}")
            # Fallback cost estimate
            return position_size_usd * 0.003  # 0.3% fallback
    
    async def estimate_cost_for_size(self, symbol: str, position_size_usd: float, 
                                   horizon_minutes: int) -> float:
        """Estimate cost for specific position size"""
        return self.estimate_cost(symbol, horizon_minutes, position_size_usd)
    
    async def get_cost_breakdown(self, symbol: str, position_size_usd: float = 10000,
                               horizon_minutes: int = 10, regime: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        try:
            if regime is None:
                regime = self._get_current_regime(symbol)
            
            market_state = self._get_market_state(symbol)
            
            return self._calculate_cost_breakdown(
                symbol, position_size_usd, horizon_minutes, regime, market_state
            )
            
        except Exception as e:
            logger.error(f"Error getting cost breakdown for {symbol}: {e}")
            return self._get_fallback_cost_breakdown(symbol, position_size_usd)
    
    def _calculate_cost_breakdown(self, symbol: str, position_size_usd: float,
                                horizon_minutes: int, regime: str, 
                                market_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed cost breakdown"""
        breakdown = {}
        
        # Trading fees
        fee_schedule = self.fee_schedules.get('binance_futures', self.fee_schedules['default'])
        breakdown['maker_fee'] = position_size_usd * fee_schedule['maker']
        breakdown['taker_fee'] = position_size_usd * fee_schedule['taker']
        
        # Market impact
        impact_cost = self.impact_model.estimate_impact(position_size_usd, regime)
        breakdown['impact_cost'] = impact_cost * position_size_usd
        
        # Slippage estimates
        available_liquidity = market_state.get('available_liquidity', position_size_usd * 2)
        slippage_percentiles = self.slippage_model.get_all_percentiles(
            position_size_usd, available_liquidity, regime
        )
        
        base_slippage = slippage_percentiles['p50']
        breakdown['slippage'] = {
            'expected': base_slippage * position_size_usd,
            'p25': slippage_percentiles['p25'] * position_size_usd,
            'p50': slippage_percentiles['p50'] * position_size_usd,
            'p75': slippage_percentiles['p75'] * position_size_usd,
            'p95': slippage_percentiles['p95'] * position_size_usd,
            'p99': slippage_percentiles['p99'] * position_size_usd
        }
        
        # Funding cost
        current_funding_rate = market_state.get('funding_rate', None)
        funding_cost = self.funding_model.estimate_funding_cost(
            symbol, horizon_minutes, position_size_usd, current_funding_rate
        )
        breakdown['funding_cost'] = funding_cost
        
        # Opportunity cost (simplified)
        opportunity_cost = position_size_usd * 0.0001 * (horizon_minutes / 60)  # 0.01% per hour
        breakdown['opportunity_cost'] = opportunity_cost
        
        # Market data
        breakdown['impact_lambda'] = self.impact_model.parameters.get(
            regime, self.impact_model.parameters['medium_vol_medium_depth']
        )['lambda']
        breakdown['available_liquidity'] = available_liquidity
        breakdown['liquidity_ratio'] = position_size_usd / available_liquidity
        
        # Total cost
        breakdown['total_cost_estimate'] = (
            breakdown['taker_fee'] +  # Assume taker for conservative estimate
            breakdown['impact_cost'] +
            breakdown['slippage']['expected'] +
            breakdown['funding_cost'] +
            breakdown['opportunity_cost']
        )
        
        # Cost per unit
        breakdown['cost_per_unit'] = breakdown['total_cost_estimate'] / position_size_usd
        
        # Confidence
        breakdown['confidence'] = self._calculate_cost_confidence(regime, market_state)
        
        # Model metadata
        breakdown['cost_model_version'] = "v1.2.0"
        breakdown['regime'] = regime
        breakdown['timestamp'] = int(time.time() * 1000)
        
        return breakdown
    
    def _get_current_regime(self, symbol: str) -> str:
        """Get current market regime for symbol"""
        # Check cache
        cache_key = f"regime:{symbol}"
        current_time = int(time.time() * 1000)
        
        if (cache_key in self.market_state_cache and 
            current_time - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl_ms):
            return self.market_state_cache[cache_key].get('regime', 'medium_vol_medium_depth')
        
        # In production, this would query the regime classification service
        # For now, return a reasonable default based on symbol
        regime_mapping = {
            'BTCUSDT': 'medium_vol_thick_depth',
            'ETHUSDT': 'medium_vol_medium_depth',
            'BNBUSDT': 'low_vol_medium_depth'
        }
        
        regime = regime_mapping.get(symbol, 'medium_vol_medium_depth')
        
        # Cache result
        self.market_state_cache[cache_key] = {'regime': regime}
        self.cache_timestamps[cache_key] = current_time
        
        return regime
    
    def _get_market_state(self, symbol: str) -> Dict[str, Any]:
        """Get current market state for cost calculations"""
        # Check cache
        cache_key = f"market_state:{symbol}"
        current_time = int(time.time() * 1000)
        
        if (cache_key in self.market_state_cache and 
            current_time - self.cache_timestamps.get(cache_key, 0) < self.cache_ttl_ms):
            return self.market_state_cache[cache_key]
        
        # In production, this would aggregate real market data
        # For now, provide reasonable defaults
        market_state = {
            'available_liquidity': 50000,  # $50K default liquidity
            'funding_rate': self.funding_model.typical_funding_rates.get(
                symbol, self.funding_model.typical_funding_rates['default']
            ),
            'volatility': 0.02,  # 2% daily volatility
            'spread': 0.0001,    # 1 bp spread
            'last_updated': current_time
        }
        
        # Cache result
        self.market_state_cache[cache_key] = market_state
        self.cache_timestamps[cache_key] = current_time
        
        return market_state
    
    def _calculate_cost_confidence(self, regime: str, market_state: Dict[str, Any]) -> float:
        """Calculate confidence score for cost estimate"""
        base_confidence = 0.85
        
        # Reduce confidence for extreme regimes
        if 'high_vol' in regime and 'thin' in regime:
            base_confidence *= 0.8
        elif 'low_vol' in regime and 'thick' in regime:
            base_confidence *= 0.95
        
        # Reduce confidence if market data is stale
        data_age_ms = int(time.time() * 1000) - market_state.get('last_updated', 0)
        if data_age_ms > 10000:  # More than 10 seconds old
            base_confidence *= 0.9
        
        return max(0.5, min(1.0, base_confidence))
    
    def _get_fallback_cost_breakdown(self, symbol: str, position_size_usd: float) -> Dict[str, Any]:
        """Get fallback cost breakdown when calculation fails"""
        return {
            'maker_fee': position_size_usd * 0.0002,
            'taker_fee': position_size_usd * 0.0004,
            'impact_cost': position_size_usd * 0.0005,
            'slippage': {
                'expected': position_size_usd * 0.0008,
                'p25': position_size_usd * 0.0006,
                'p50': position_size_usd * 0.0008,
                'p75': position_size_usd * 0.0012,
                'p95': position_size_usd * 0.0020,
                'p99': position_size_usd * 0.0035
            },
            'funding_cost': position_size_usd * 0.0001,
            'opportunity_cost': position_size_usd * 0.0001,
            'total_cost_estimate': position_size_usd * 0.0020,
            'cost_per_unit': 0.0020,
            'confidence': 0.6,
            'cost_model_version': "v1.2.0",
            'regime': 'medium_vol_medium_depth',
            'timestamp': int(time.time() * 1000)
        }
    
    def record_actual_cost(self, symbol: str, position_size_usd: float, 
                          actual_cost: float, regime: str):
        """Record actual execution cost for model improvement"""
        try:
            with self.lock:
                predicted_cost = self.estimate_cost(symbol, 10, position_size_usd, regime)
                error = abs(actual_cost - predicted_cost) / max(predicted_cost, 0.001)
                
                self.prediction_errors[symbol].append({
                    'timestamp': int(time.time() * 1000),
                    'predicted': predicted_cost,
                    'actual': actual_cost,
                    'error': error,
                    'regime': regime,
                    'position_size': position_size_usd
                })
            
            # Update slippage model with actual data
            actual_slippage = actual_cost / position_size_usd - 0.0006  # Remove base fees/costs
            if actual_slippage > 0:
                self.slippage_model.record_actual_slippage(
                    symbol, position_size_usd, actual_slippage, regime
                )
            
        except Exception as e:
            logger.error(f"Error recording actual cost for {symbol}: {e}")
    
    def get_model_performance(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get cost model performance metrics"""
        with self.lock:
            if symbol:
                symbols = [symbol] if symbol in self.prediction_errors else []
            else:
                symbols = list(self.prediction_errors.keys())
            
            performance = {}
            
            for sym in symbols:
                if not self.prediction_errors[sym]:
                    continue
                
                errors = [record['error'] for record in self.prediction_errors[sym]]
                predictions = [record['predicted'] for record in self.prediction_errors[sym]]
                actuals = [record['actual'] for record in self.prediction_errors[sym]]
                
                if errors:
                    performance[sym] = {
                        'mean_absolute_percentage_error': np.mean(errors),
                        'median_error': np.median(errors),
                        'prediction_count': len(errors),
                        'correlation': np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0.0,
                        'recent_performance': np.mean(errors[-20:]) if len(errors) >= 20 else np.mean(errors)
                    }
            
            return performance
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
