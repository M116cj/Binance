"""
Advanced feature engineering for crypto surge prediction.
Implements high-performance vectorized feature computation with temporal validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from numba import jit, njit
import warnings
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

from .schemas import FeatureVector, QualityFlag, MarketRegimeState, TimeSeriesPoint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit
def welford_mean_var(values: np.ndarray) -> Tuple[float, float]:
    """Numba-optimized online mean and variance calculation"""
    if len(values) == 0:
        return 0.0, 0.0
    
    mean = 0.0
    m2 = 0.0
    count = 0
    
    for value in values:
        if not np.isnan(value):
            count += 1
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2
    
    if count < 2:
        return mean, 0.0
    
    variance = m2 / (count - 1)
    return mean, variance

@njit
def calculate_linear_regression_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Numba-optimized linear regression slope calculation"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    # Remove NaN pairs
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    
    if len(x_clean) < 2:
        return 0.0
    
    n = len(x_clean)
    sum_x = np.sum(x_clean)
    sum_y = np.sum(y_clean)
    sum_xy = np.sum(x_clean * y_clean)
    sum_x2 = np.sum(x_clean * x_clean)
    
    denominator = n * sum_x2 - sum_x * sum_x
    if abs(denominator) < 1e-10:
        return 0.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

@njit
def calculate_hawkes_intensity(timestamps: np.ndarray, current_time: float, 
                              mu: float = 0.1, alpha: float = 0.5, beta: float = 1.0) -> float:
    """Numba-optimized Hawkes process intensity calculation"""
    if len(timestamps) == 0:
        return mu
    
    # Filter recent events (within decay window)
    recent_events = timestamps[timestamps > current_time - 5000]  # 5 second window
    
    if len(recent_events) == 0:
        return mu
    
    intensity = mu
    for t in recent_events:
        decay = np.exp(-beta * (current_time - t) / 1000.0)  # Convert ms to seconds
        intensity += alpha * decay
    
    return intensity

class RingBuffer:
    """Thread-safe ring buffer for efficient streaming data storage"""
    
    def __init__(self, maxlen: int):
        self.maxlen = maxlen
        self.data = np.full(maxlen, np.nan, dtype=np.float64)
        self.timestamps = np.zeros(maxlen, dtype=np.int64)
        self.head = 0
        self.size = 0
        self.lock = threading.RLock()
    
    def append(self, value: float, timestamp: int):
        """Thread-safe append operation"""
        with self.lock:
            self.data[self.head] = value
            self.timestamps[self.head] = timestamp
            self.head = (self.head + 1) % self.maxlen
            self.size = min(self.size + 1, self.maxlen)
    
    def get_recent(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get n most recent values and timestamps"""
        with self.lock:
            if self.size == 0 or n <= 0:
                return np.array([]), np.array([])
            
            n = min(n, self.size)
            
            if self.head >= n:
                # No wrap-around
                data_slice = self.data[self.head - n:self.head].copy()
                ts_slice = self.timestamps[self.head - n:self.head].copy()
            else:
                # Wrap-around case
                data_slice = np.concatenate([
                    self.data[self.maxlen - (n - self.head):],
                    self.data[:self.head]
                ])
                ts_slice = np.concatenate([
                    self.timestamps[self.maxlen - (n - self.head):],
                    self.timestamps[:self.head]
                ])
            
            return data_slice, ts_slice
    
    def get_window_by_time(self, window_ms: int, current_time: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get data within time window"""
        with self.lock:
            if self.size == 0:
                return np.array([]), np.array([])
            
            # Get all data
            if self.size < self.maxlen:
                data_slice = self.data[:self.size].copy()
                ts_slice = self.timestamps[:self.size].copy()
            else:
                if self.head == 0:
                    data_slice = self.data.copy()
                    ts_slice = self.timestamps.copy()
                else:
                    data_slice = np.concatenate([
                        self.data[self.head:],
                        self.data[:self.head]
                    ])
                    ts_slice = np.concatenate([
                        self.timestamps[self.head:],
                        self.timestamps[:self.head]
                    ])
            
            # Filter by time window
            cutoff_time = current_time - window_ms
            mask = ts_slice >= cutoff_time
            
            return data_slice[mask], ts_slice[mask]
    
    def is_sufficient(self, min_pct: float = 0.5) -> bool:
        """Check if buffer has sufficient valid data"""
        with self.lock:
            if self.size == 0:
                return False
            
            valid_count = np.sum(~np.isnan(self.data[:min(self.size, self.maxlen)]))
            return valid_count / min(self.size, self.maxlen) >= min_pct

class OnlineStatistics:
    """Online statistics computation with Welford's algorithm"""
    
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.lock = threading.Lock()
    
    def update(self, value: float):
        """Update statistics with new value"""
        if np.isnan(value) or np.isinf(value):
            return
        
        with self.lock:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2
            
            self.min_val = min(self.min_val, value)
            self.max_val = max(self.max_val, value)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics"""
        with self.lock:
            if self.count < 2:
                return {
                    'mean': self.mean,
                    'std': 0.0,
                    'var': 0.0,
                    'min': self.min_val if self.count > 0 else 0.0,
                    'max': self.max_val if self.count > 0 else 0.0,
                    'count': self.count
                }
            
            variance = self.m2 / (self.count - 1)
            std = np.sqrt(variance)
            
            return {
                'mean': self.mean,
                'std': std,
                'var': variance,
                'min': self.min_val,
                'max': self.max_val,
                'count': self.count
            }

class RegimeClassifier:
    """Market regime classification based on multiple indicators"""
    
    def __init__(self):
        self.volatility_thresholds = [0.8, 1.5]  # low/medium, medium/high
        self.depth_thresholds = [-2.0, 0.0]     # thin/medium, medium/thick
        self.funding_thresholds = [-0.01, 0.01] # negative/neutral, neutral/positive
    
    def classify_regime(self, features: Dict[str, float]) -> MarketRegimeState:
        """Classify current market regime"""
        timestamp = int(time.time() * 1000)
        
        # Extract key indicators
        rv_ratio = features.get('rv_ratio', 1.0)
        depth_slope = features.get('depth_slope_bid', 0.0)
        funding_delta = features.get('funding_delta', 0.0)
        
        # Classify volatility regime
        if rv_ratio <= self.volatility_thresholds[0]:
            vol_regime = "low"
        elif rv_ratio <= self.volatility_thresholds[1]:
            vol_regime = "medium"
        else:
            vol_regime = "high"
        
        # Classify depth regime
        if depth_slope <= self.depth_thresholds[0]:
            depth_regime = "thin"
        elif depth_slope <= self.depth_thresholds[1]:
            depth_regime = "medium"
        else:
            depth_regime = "thick"
        
        # Classify funding regime
        if funding_delta <= self.funding_thresholds[0]:
            funding_regime = "negative"
        elif funding_delta <= self.funding_thresholds[1]:
            funding_regime = "neutral"
        else:
            funding_regime = "positive"
        
        # Combined regime
        combined_regime = f"{vol_regime}_vol_{depth_regime}_depth_{funding_regime}_funding"
        
        # Calculate confidence based on distance from thresholds
        vol_confidence = min(
            abs(rv_ratio - self.volatility_thresholds[0]) / 0.3,
            abs(rv_ratio - self.volatility_thresholds[1]) / 0.5
        )
        depth_confidence = min(
            abs(depth_slope - self.depth_thresholds[0]) / 1.0,
            abs(depth_slope - self.depth_thresholds[1]) / 1.0
        )
        funding_confidence = min(
            abs(funding_delta - self.funding_thresholds[0]) / 0.005,
            abs(funding_delta - self.funding_thresholds[1]) / 0.005
        )
        
        regime_confidence = np.mean([vol_confidence, depth_confidence, funding_confidence])
        regime_confidence = max(0.5, min(1.0, regime_confidence))  # Clamp to [0.5, 1.0]
        
        return MarketRegimeState(
            symbol="",  # Will be set by caller
            timestamp=timestamp,
            volatility_regime=vol_regime,
            depth_regime=depth_regime,
            funding_regime=funding_regime,
            combined_regime=combined_regime,
            regime_confidence=regime_confidence,
            regime_stability=0.8  # Would be calculated from historical regime changes
        )

class FeatureEngine:
    """High-performance feature engineering with vectorized operations"""
    
    def __init__(self, max_window_size: int = 1800):  # 30 minutes at 1s resolution
        self.max_window_size = max_window_size
        
        # Feature buffers for each symbol
        self.buffers: Dict[str, Dict[str, RingBuffer]] = defaultdict(
            lambda: self._create_symbol_buffers()
        )
        
        # Online statistics for normalization
        self.online_stats: Dict[str, Dict[str, OnlineStatistics]] = defaultdict(
            lambda: defaultdict(OnlineStatistics)
        )
        
        # Regime classifier
        self.regime_classifier = RegimeClassifier()
        
        # Feature computation cache
        self.feature_cache: Dict[str, Dict] = {}
        self.cache_timestamps: Dict[str, int] = {}
        self.cache_ttl_ms = 1000  # 1 second TTL
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        # Feature definitions
        self.feature_definitions = self._initialize_feature_definitions()
        
    def _create_symbol_buffers(self) -> Dict[str, RingBuffer]:
        """Create buffer dictionary for a symbol"""
        return {
            'mid_price': RingBuffer(self.max_window_size),
            'best_bid': RingBuffer(self.max_window_size),
            'best_ask': RingBuffer(self.max_window_size),
            'bid_size_1': RingBuffer(self.max_window_size),
            'ask_size_1': RingBuffer(self.max_window_size),
            'bid_size_5': RingBuffer(self.max_window_size),
            'ask_size_5': RingBuffer(self.max_window_size),
            'spread': RingBuffer(self.max_window_size),
            'trade_price': RingBuffer(self.max_window_size),
            'trade_volume': RingBuffer(self.max_window_size),
            'buy_volume': RingBuffer(self.max_window_size),
            'sell_volume': RingBuffer(self.max_window_size),
            'trade_timestamps': RingBuffer(self.max_window_size),
            'funding_rate': RingBuffer(self.max_window_size),
            'open_interest': RingBuffer(self.max_window_size),
            'liquidation_volume': RingBuffer(self.max_window_size)
        }
    
    def _initialize_feature_definitions(self) -> Dict[str, Dict]:
        """Initialize feature computation definitions"""
        return {
            'fast': {
                'names': [
                    'queue_imbalance_1', 'queue_imbalance_5', 'microprice_deviation',
                    'order_flow_imbalance_10', 'order_flow_imbalance_30'
                ],
                'interval_ms': 100,  # Update every 100ms
                'window_sizes': {'short': 10, 'medium': 30, 'long': 60}
            },
            'medium': {
                'names': [
                    'depth_slope_bid', 'depth_slope_ask', 'near_touch_void',
                    'impact_lambda', 'realized_volatility_1m', 'rv_ratio'
                ],
                'interval_ms': 1000,  # Update every 1s
                'window_sizes': {'short': 60, 'medium': 300, 'long': 900}
            },
            'slow': {
                'names': [
                    'buy_cluster_intensity', 'bollinger_position', 'bollinger_squeeze',
                    'funding_delta', 'oi_pressure', 'liquidation_density_up'
                ],
                'interval_ms': 5000,  # Update every 5s
                'window_sizes': {'short': 300, 'medium': 900, 'long': 1800}
            }
        }
    
    def update_market_data(self, symbol: str, stream_type: str, data: Dict[str, Any], timestamp: int):
        """Update market data for feature computation"""
        try:
            buffers = self.buffers[symbol]
            
            if stream_type == 'bookTicker':
                self._update_book_ticker_buffers(buffers, data, timestamp)
            elif stream_type == 'aggTrade':
                self._update_trade_buffers(buffers, data, timestamp)
            elif stream_type == 'depth':
                self._update_depth_buffers(buffers, data, timestamp)
            elif stream_type == 'markPrice':
                self._update_funding_buffers(buffers, data, timestamp)
            elif stream_type == 'openInterest':
                self._update_oi_buffers(buffers, data, timestamp)
            elif stream_type == 'forceOrder':
                self._update_liquidation_buffers(buffers, data, timestamp)
            
            # Invalidate cache for this symbol
            if symbol in self.feature_cache:
                del self.feature_cache[symbol]
                del self.cache_timestamps[symbol]
                
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
    
    def _update_book_ticker_buffers(self, buffers: Dict[str, RingBuffer], data: Dict, timestamp: int):
        """Update order book ticker buffers"""
        best_bid = float(data.get('b', 0))
        best_ask = float(data.get('a', 0))
        bid_size = float(data.get('B', 0))
        ask_size = float(data.get('A', 0))
        
        if best_bid > 0 and best_ask > 0:
            mid_price = (best_bid + best_ask) / 2
            spread = (best_ask - best_bid) / mid_price
            
            buffers['mid_price'].append(mid_price, timestamp)
            buffers['best_bid'].append(best_bid, timestamp)
            buffers['best_ask'].append(best_ask, timestamp)
            buffers['bid_size_1'].append(bid_size, timestamp)
            buffers['ask_size_1'].append(ask_size, timestamp)
            buffers['spread'].append(spread, timestamp)
    
    def _update_trade_buffers(self, buffers: Dict[str, RingBuffer], data: Dict, timestamp: int):
        """Update trade buffers"""
        price = float(data.get('p', 0))
        quantity = float(data.get('q', 0))
        is_buyer_maker = data.get('m', False)
        
        buffers['trade_price'].append(price, timestamp)
        buffers['trade_volume'].append(quantity, timestamp)
        buffers['trade_timestamps'].append(timestamp, timestamp)
        
        # Classify buy/sell volume
        if is_buyer_maker:
            # Buyer is maker, so trade is a sell
            buffers['buy_volume'].append(0, timestamp)
            buffers['sell_volume'].append(quantity, timestamp)
        else:
            # Seller is maker, so trade is a buy
            buffers['buy_volume'].append(quantity, timestamp)
            buffers['sell_volume'].append(0, timestamp)
    
    def _update_depth_buffers(self, buffers: Dict[str, RingBuffer], data: Dict, timestamp: int):
        """Update depth buffers"""
        bids = data.get('b', [])
        asks = data.get('a', [])
        
        if len(bids) >= 5 and len(asks) >= 5:
            # Calculate aggregated sizes
            bid_size_5 = sum(float(bid[1]) for bid in bids[:5])
            ask_size_5 = sum(float(ask[1]) for ask in asks[:5])
            
            buffers['bid_size_5'].append(bid_size_5, timestamp)
            buffers['ask_size_5'].append(ask_size_5, timestamp)
    
    def _update_funding_buffers(self, buffers: Dict[str, RingBuffer], data: Dict, timestamp: int):
        """Update funding rate buffers"""
        funding_rate = float(data.get('r', 0))
        buffers['funding_rate'].append(funding_rate, timestamp)
    
    def _update_oi_buffers(self, buffers: Dict[str, RingBuffer], data: Dict, timestamp: int):
        """Update open interest buffers"""
        oi = float(data.get('openInterest', 0))
        buffers['open_interest'].append(oi, timestamp)
    
    def _update_liquidation_buffers(self, buffers: Dict[str, RingBuffer], data: Dict, timestamp: int):
        """Update liquidation buffers"""
        quantity = float(data.get('q', 0))
        buffers['liquidation_volume'].append(quantity, timestamp)
    
    def compute_features(self, symbol: str, timestamp: int, 
                        feature_types: Optional[List[str]] = None) -> Optional[FeatureVector]:
        """Compute feature vector for symbol at timestamp"""
        try:
            # Check cache
            if (symbol in self.feature_cache and 
                timestamp - self.cache_timestamps.get(symbol, 0) < self.cache_ttl_ms):
                return self.feature_cache[symbol]
            
            # Determine which feature types to compute
            if feature_types is None:
                feature_types = ['fast', 'medium', 'slow']
            
            # Initialize feature dictionary
            features = {}
            quality_flags = []
            
            # Check data sufficiency
            buffers = self.buffers[symbol]
            if not self._check_data_sufficiency(buffers):
                quality_flags.append(QualityFlag.WINDOW_INSUFFICIENT)
            
            # Compute features by type
            for feature_type in feature_types:
                if feature_type == 'fast':
                    features.update(self._compute_fast_features(symbol, timestamp))
                elif feature_type == 'medium':
                    features.update(self._compute_medium_features(symbol, timestamp))
                elif feature_type == 'slow':
                    features.update(self._compute_slow_features(symbol, timestamp))
            
            # Update online statistics
            self._update_online_statistics(symbol, features)
            
            # Create feature vector
            feature_vector = self._create_feature_vector(
                symbol, timestamp, features, quality_flags
            )
            
            # Cache result
            self.feature_cache[symbol] = feature_vector
            self.cache_timestamps[symbol] = timestamp
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            return None
    
    def _check_data_sufficiency(self, buffers: Dict[str, RingBuffer]) -> bool:
        """Check if buffers have sufficient data"""
        required_buffers = ['mid_price', 'bid_size_1', 'ask_size_1']
        
        for buffer_name in required_buffers:
            if buffer_name in buffers and not buffers[buffer_name].is_sufficient(0.5):
                return False
        
        return True
    
    def _compute_fast_features(self, symbol: str, timestamp: int) -> Dict[str, float]:
        """Compute fast features (order book, flow)"""
        features = {}
        buffers = self.buffers[symbol]
        
        try:
            # Queue Imbalance features
            bid_1, _ = buffers['bid_size_1'].get_recent(1)
            ask_1, _ = buffers['ask_size_1'].get_recent(1)
            
            if len(bid_1) > 0 and len(ask_1) > 0:
                total_1 = bid_1[0] + ask_1[0]
                if total_1 > 0:
                    features['queue_imbalance_1'] = (bid_1[0] - ask_1[0]) / total_1
                else:
                    features['queue_imbalance_1'] = 0.0
            else:
                features['queue_imbalance_1'] = 0.0
            
            # 5-level queue imbalance
            bid_5, _ = buffers['bid_size_5'].get_recent(1)
            ask_5, _ = buffers['ask_size_5'].get_recent(1)
            
            if len(bid_5) > 0 and len(ask_5) > 0:
                total_5 = bid_5[0] + ask_5[0]
                if total_5 > 0:
                    features['queue_imbalance_5'] = (bid_5[0] - ask_5[0]) / total_5
                else:
                    features['queue_imbalance_5'] = 0.0
            else:
                features['queue_imbalance_5'] = 0.0
            
            # Microprice deviation
            best_bid, _ = buffers['best_bid'].get_recent(1)
            best_ask, _ = buffers['best_ask'].get_recent(1)
            mid_price, _ = buffers['mid_price'].get_recent(1)
            
            if len(best_bid) > 0 and len(best_ask) > 0 and len(mid_price) > 0:
                if len(bid_1) > 0 and len(ask_1) > 0:
                    total_size = bid_1[0] + ask_1[0]
                    if total_size > 0 and mid_price[0] > 0:
                        microprice = (best_ask[0] * bid_1[0] + best_bid[0] * ask_1[0]) / total_size
                        features['microprice_deviation'] = (microprice - mid_price[0]) / mid_price[0]
                    else:
                        features['microprice_deviation'] = 0.0
                else:
                    features['microprice_deviation'] = 0.0
            else:
                features['microprice_deviation'] = 0.0
            
            # Order Flow Imbalance
            buy_vol_10, _ = buffers['buy_volume'].get_recent(10)
            sell_vol_10, _ = buffers['sell_volume'].get_recent(10)
            
            if len(buy_vol_10) > 0 and len(sell_vol_10) > 0:
                buy_sum = np.sum(buy_vol_10)
                sell_sum = np.sum(sell_vol_10)
                total_vol = buy_sum + sell_sum
                if total_vol > 0:
                    features['order_flow_imbalance_10'] = (buy_sum - sell_sum) / total_vol
                else:
                    features['order_flow_imbalance_10'] = 0.0
            else:
                features['order_flow_imbalance_10'] = 0.0
            
            # 30-period OFI
            buy_vol_30, _ = buffers['buy_volume'].get_recent(30)
            sell_vol_30, _ = buffers['sell_volume'].get_recent(30)
            
            if len(buy_vol_30) > 0 and len(sell_vol_30) > 0:
                buy_sum = np.sum(buy_vol_30)
                sell_sum = np.sum(sell_vol_30)
                total_vol = buy_sum + sell_sum
                if total_vol > 0:
                    features['order_flow_imbalance_30'] = (buy_sum - sell_sum) / total_vol
                else:
                    features['order_flow_imbalance_30'] = 0.0
            else:
                features['order_flow_imbalance_30'] = 0.0
            
        except Exception as e:
            logger.error(f"Error computing fast features for {symbol}: {e}")
        
        return features
    
    def _compute_medium_features(self, symbol: str, timestamp: int) -> Dict[str, float]:
        """Compute medium frequency features (volatility, depth)"""
        features = {}
        buffers = self.buffers[symbol]
        
        try:
            # Realized Volatility
            mid_prices, _ = buffers['mid_price'].get_recent(60)  # 1 minute
            if len(mid_prices) > 1:
                valid_prices = mid_prices[~np.isnan(mid_prices)]
                if len(valid_prices) > 1:
                    log_returns = np.diff(np.log(valid_prices))
                    features['realized_volatility_1m'] = np.sqrt(np.sum(log_returns ** 2))
                else:
                    features['realized_volatility_1m'] = 0.0
            else:
                features['realized_volatility_1m'] = 0.0
            
            # 5-minute realized volatility for ratio
            mid_prices_5m, _ = buffers['mid_price'].get_recent(300)  # 5 minutes
            if len(mid_prices_5m) > 1:
                valid_prices_5m = mid_prices_5m[~np.isnan(mid_prices_5m)]
                if len(valid_prices_5m) > 1:
                    log_returns_5m = np.diff(np.log(valid_prices_5m))
                    rv_5m = np.sqrt(np.sum(log_returns_5m ** 2))
                    if features['realized_volatility_1m'] > 0 and rv_5m > 0:
                        features['rv_ratio'] = features['realized_volatility_1m'] / rv_5m
                    else:
                        features['rv_ratio'] = 1.0
                else:
                    features['rv_ratio'] = 1.0
            else:
                features['rv_ratio'] = 1.0
            
            # Depth slope (simplified)
            bid_liq, _ = buffers['bid_size_5'].get_recent(1)
            ask_liq, _ = buffers['ask_size_5'].get_recent(1)
            
            if len(bid_liq) > 0 and len(ask_liq) > 0:
                features['depth_slope_bid'] = np.log(max(bid_liq[0], 1e-8))
                features['depth_slope_ask'] = np.log(max(ask_liq[0], 1e-8))
            else:
                features['depth_slope_bid'] = 0.0
                features['depth_slope_ask'] = 0.0
            
            # Near touch void (ratio of near vs total liquidity)
            bid_1, _ = buffers['bid_size_1'].get_recent(1)
            ask_1, _ = buffers['ask_size_1'].get_recent(1)
            
            if (len(bid_1) > 0 and len(ask_1) > 0 and 
                len(bid_liq) > 0 and len(ask_liq) > 0):
                near_total = bid_1[0] + ask_1[0]
                far_total = bid_liq[0] + ask_liq[0]
                if far_total > 0:
                    features['near_touch_void'] = near_total / far_total
                else:
                    features['near_touch_void'] = 0.0
            else:
                features['near_touch_void'] = 0.0
            
            # Impact lambda (price change per unit volume)
            trade_vols, _ = buffers['trade_volume'].get_recent(20)
            mid_price_changes = self._calculate_price_changes(symbol, 20)
            
            if len(trade_vols) > 5 and len(mid_price_changes) > 5:
                total_volume = np.sum(trade_vols)
                total_price_change = np.sum(np.abs(mid_price_changes))
                if total_volume > 0:
                    features['impact_lambda'] = total_price_change / total_volume
                else:
                    features['impact_lambda'] = 0.0
            else:
                features['impact_lambda'] = 0.0
            
        except Exception as e:
            logger.error(f"Error computing medium features for {symbol}: {e}")
        
        return features
    
    def _compute_slow_features(self, symbol: str, timestamp: int) -> Dict[str, float]:
        """Compute slow features (Hawkes, funding, regime)"""
        features = {}
        buffers = self.buffers[symbol]
        
        try:
            # Hawkes buy cluster intensity
            trade_timestamps, _ = buffers['trade_timestamps'].get_window_by_time(5000, timestamp)
            buy_volumes, buy_ts = buffers['buy_volume'].get_window_by_time(5000, timestamp)
            
            # Filter for actual buy trades (non-zero volume)
            buy_trade_timestamps = buy_ts[buy_volumes > 0]
            
            if len(buy_trade_timestamps) > 0:
                features['buy_cluster_intensity'] = calculate_hawkes_intensity(
                    buy_trade_timestamps, timestamp
                )
            else:
                features['buy_cluster_intensity'] = 0.1  # Base intensity
            
            # Bollinger Band features
            mid_prices, _ = buffers['mid_price'].get_recent(60)
            if len(mid_prices) > 20:
                valid_prices = mid_prices[~np.isnan(mid_prices)]
                if len(valid_prices) > 20:
                    ma = np.mean(valid_prices[-20:])
                    std = np.std(valid_prices[-20:])
                    current_price = valid_prices[-1]
                    
                    if std > 0:
                        features['bollinger_position'] = (current_price - ma) / (2 * std)
                        features['bollinger_squeeze'] = 1.0 / max(std / ma, 1e-8)
                    else:
                        features['bollinger_position'] = 0.0
                        features['bollinger_squeeze'] = 1.0
                else:
                    features['bollinger_position'] = 0.0
                    features['bollinger_squeeze'] = 1.0
            else:
                features['bollinger_position'] = 0.0
                features['bollinger_squeeze'] = 1.0
            
            # Funding delta
            funding_rates, _ = buffers['funding_rate'].get_recent(2)
            if len(funding_rates) >= 2:
                features['funding_delta'] = funding_rates[-1] - funding_rates[-2]
            else:
                features['funding_delta'] = 0.0
            
            # Open Interest pressure
            oi_values, _ = buffers['open_interest'].get_recent(2)
            if len(oi_values) >= 2:
                features['oi_pressure'] = (oi_values[-1] - oi_values[-2]) / max(oi_values[-2], 1)
            else:
                features['oi_pressure'] = 0.0
            
            # Liquidation density
            liq_volumes, _ = buffers['liquidation_volume'].get_window_by_time(60000, timestamp)  # 1 minute
            if len(liq_volumes) > 0:
                features['liquidation_density_up'] = np.sum(liq_volumes[liq_volumes > 0])
                features['liquidation_density_down'] = np.sum(np.abs(liq_volumes[liq_volumes < 0]))
            else:
                features['liquidation_density_up'] = 0.0
                features['liquidation_density_down'] = 0.0
            
            # CVD slope (cumulative volume delta slope)
            cvd_values = self._calculate_cvd_slope(symbol, 30)
            features['cvd_slope'] = cvd_values
            
            # Additional derivatives features
            features['long_short_ratio'] = 1.0  # Would require additional data
            features['lsr_divergence'] = 0.0    # Would require price correlation analysis
            features['follow_buy_ratio'] = 0.5  # Would require order clustering analysis
            features['post_liq_gap'] = 0.0      # Would require post-liquidation analysis
            features['lead_lag_spread'] = 0.0   # Would require cross-exchange data
            features['arrival_rate_ratio'] = 1.0 # Would require arrival rate calculation
            
        except Exception as e:
            logger.error(f"Error computing slow features for {symbol}: {e}")
        
        return features
    
    def _calculate_price_changes(self, symbol: str, window: int) -> np.ndarray:
        """Calculate mid price changes over window"""
        buffers = self.buffers[symbol]
        mid_prices, _ = buffers['mid_price'].get_recent(window + 1)
        
        if len(mid_prices) > 1:
            valid_prices = mid_prices[~np.isnan(mid_prices)]
            if len(valid_prices) > 1:
                return np.diff(valid_prices)
        
        return np.array([])
    
    def _calculate_cvd_slope(self, symbol: str, window: int) -> float:
        """Calculate cumulative volume delta slope"""
        buffers = self.buffers[symbol]
        
        buy_vols, buy_ts = buffers['buy_volume'].get_recent(window)
        sell_vols, sell_ts = buffers['sell_volume'].get_recent(window)
        
        if len(buy_vols) == len(sell_vols) and len(buy_vols) > 1:
            cvd = np.cumsum(buy_vols - sell_vols)
            time_indices = np.arange(len(cvd))
            
            # Calculate linear regression slope
            slope = calculate_linear_regression_slope(
                time_indices.astype(np.float64), cvd.astype(np.float64)
            )
            return slope
        
        return 0.0
    
    def _update_online_statistics(self, symbol: str, features: Dict[str, float]):
        """Update online statistics for feature normalization"""
        for feature_name, value in features.items():
            if not np.isnan(value) and not np.isinf(value):
                self.online_stats[symbol][feature_name].update(value)
    
    def _create_feature_vector(self, symbol: str, timestamp: int, 
                             features: Dict[str, float], quality_flags: List[QualityFlag]) -> FeatureVector:
        """Create standardized feature vector"""
        # Fill missing features with default values
        default_features = {
            'queue_imbalance_1': 0.0,
            'queue_imbalance_5': 0.0,
            'microprice_deviation': 0.0,
            'depth_slope_bid': 0.0,
            'depth_slope_ask': 0.0,
            'near_touch_void': 0.0,
            'impact_lambda': 0.0,
            'order_flow_imbalance_10': 0.0,
            'order_flow_imbalance_30': 0.0,
            'cvd_slope': 0.0,
            'buy_cluster_intensity': 0.1,
            'follow_buy_ratio': 0.5,
            'realized_volatility_1m': 0.0,
            'realized_volatility_5m': 0.0,
            'rv_ratio': 1.0,
            'bollinger_position': 0.0,
            'bollinger_squeeze': 1.0,
            'funding_delta': 0.0,
            'oi_pressure': 0.0,
            'long_short_ratio': 1.0,
            'lsr_divergence': 0.0,
            'liquidation_density_up': 0.0,
            'liquidation_density_down': 0.0,
            'post_liq_gap': 0.0,
            'lead_lag_spread': 0.0,
            'arrival_rate_ratio': 1.0
        }
        
        # Merge with computed features
        final_features = {**default_features, **features}
        
        # Count valid features
        valid_count = sum(1 for v in final_features.values() 
                         if not np.isnan(v) and not np.isinf(v))
        
        return FeatureVector(
            symbol=symbol,
            timestamp=timestamp,
            window_start=timestamp - 30000,  # 30 second window
            window_end=timestamp,
            **final_features,
            feature_version="1.0.0",
            quality_flags=quality_flags,
            feature_count_valid=valid_count
        )
    
    def get_regime_state(self, symbol: str, timestamp: int) -> Optional[MarketRegimeState]:
        """Get market regime classification"""
        try:
            # Get latest features for regime classification
            feature_vector = self.compute_features(symbol, timestamp, ['medium', 'slow'])
            
            if feature_vector is None:
                return None
            
            # Extract regime-relevant features
            regime_features = {
                'rv_ratio': feature_vector.rv_ratio,
                'depth_slope_bid': feature_vector.depth_slope_bid,
                'funding_delta': feature_vector.funding_delta
            }
            
            regime_state = self.regime_classifier.classify_regime(regime_features)
            regime_state.symbol = symbol
            
            return regime_state
            
        except Exception as e:
            logger.error(f"Error getting regime state for {symbol}: {e}")
            return None
    
    def normalize_features(self, features: Dict[str, float], symbol: str) -> Dict[str, float]:
        """Normalize features using online statistics"""
        normalized = {}
        
        for feature_name, value in features.items():
            stats = self.online_stats[symbol][feature_name].get_stats()
            
            if stats['count'] > 1 and stats['std'] > 0:
                # Z-score normalization
                normalized[feature_name] = (value - stats['mean']) / stats['std']
            else:
                # No normalization if insufficient data
                normalized[feature_name] = value
        
        return normalized
    
    def get_feature_importance_weights(self) -> Dict[str, float]:
        """Get feature importance weights for model interpretation"""
        # These would typically be loaded from trained model
        return {
            'queue_imbalance_1': 0.15,
            'order_flow_imbalance_10': 0.18,
            'microprice_deviation': 0.12,
            'rv_ratio': 0.14,
            'buy_cluster_intensity': 0.10,
            'depth_slope_bid': 0.09,
            'bollinger_position': 0.08,
            'impact_lambda': 0.07,
            'funding_delta': 0.04,
            'oi_pressure': 0.03
        }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
