"""
Advanced labeling algorithms for crypto surge prediction.
Implements triple-barrier method with embargo and purged cross-validation support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict
from numba import jit, njit
from scipy import stats
import warnings

from .schemas import TripleBarrierLabel, TimeSeriesPoint, QualityFlag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@njit
def find_first_barrier_touch(prices: np.ndarray, timestamps: np.ndarray, 
                            entry_price: float, upper_barrier: float, 
                            lower_barrier: float, max_horizon_ms: int, 
                            entry_time: int) -> Tuple[int, float, float, float]:
    """
    Numba-optimized function to find first barrier touch.
    
    Returns:
        label: -1 (down), 0 (timeout), 1 (up)
        breach_timestamp: timestamp of breach
        breach_price: price at breach  
        max_favorable_excursion: maximum favorable price movement
    """
    if len(prices) == 0:
        return 0, entry_time + max_horizon_ms, entry_price, 0.0
    
    max_favorable = 0.0
    max_adverse = 0.0
    
    for i in range(len(prices)):
        if timestamps[i] <= entry_time:
            continue
            
        if timestamps[i] > entry_time + max_horizon_ms:
            break
        
        price = prices[i]
        return_pct = (price - entry_price) / entry_price
        
        # Update excursions
        if return_pct > max_favorable:
            max_favorable = return_pct
        if return_pct < max_adverse:
            max_adverse = return_pct
        
        # Check barriers
        if price >= upper_barrier:
            return 1, timestamps[i], price, max_favorable
        elif price <= lower_barrier:
            return -1, timestamps[i], price, max_favorable
    
    # Timeout - no barrier touched
    return 0, entry_time + max_horizon_ms, entry_price, max_favorable

@njit
def calculate_peak_metrics(prices: np.ndarray, timestamps: np.ndarray,
                          entry_price: float, entry_time: int, 
                          max_horizon_ms: int) -> Tuple[float, float]:
    """
    Calculate peak return and time to peak within horizon.
    
    Returns:
        peak_return: maximum return achieved
        time_to_peak: time from entry to peak (in seconds)
    """
    if len(prices) == 0:
        return 0.0, 0.0
    
    max_return = 0.0
    peak_time = entry_time
    
    for i in range(len(prices)):
        if timestamps[i] <= entry_time:
            continue
            
        if timestamps[i] > entry_time + max_horizon_ms:
            break
        
        price = prices[i]
        return_pct = (price - entry_price) / entry_price
        
        if return_pct > max_return:
            max_return = return_pct
            peak_time = timestamps[i]
    
    time_to_peak_sec = (peak_time - entry_time) / 1000.0
    return max_return, time_to_peak_sec

class CooldownManager:
    """Manages cooldown periods to prevent overlapping labels"""
    
    def __init__(self):
        self.cooldowns: Dict[str, Dict[Tuple[float, float, int], int]] = defaultdict(dict)
        self.lock = threading.RLock()
    
    def is_in_cooldown(self, symbol: str, timestamp: int, 
                      theta_up: float, theta_dn: float, horizon_minutes: int) -> bool:
        """Check if symbol is in cooldown period"""
        key = (theta_up, theta_dn, horizon_minutes)
        
        with self.lock:
            if key in self.cooldowns[symbol]:
                cooldown_end = self.cooldowns[symbol][key]
                return timestamp < cooldown_end
        
        return False
    
    def set_cooldown(self, symbol: str, timestamp: int, theta_up: float, 
                    theta_dn: float, horizon_minutes: int, cooldown_minutes: int):
        """Set cooldown period for symbol and parameters"""
        key = (theta_up, theta_dn, horizon_minutes)
        cooldown_end = timestamp + cooldown_minutes * 60 * 1000  # Convert to milliseconds
        
        with self.lock:
            self.cooldowns[symbol][key] = cooldown_end
    
    def cleanup_expired(self, current_time: int):
        """Remove expired cooldowns"""
        with self.lock:
            for symbol in self.cooldowns:
                expired_keys = [
                    key for key, end_time in self.cooldowns[symbol].items()
                    if current_time >= end_time
                ]
                for key in expired_keys:
                    del self.cooldowns[symbol][key]

class EmbargoManager:
    """Manages embargo periods for purged cross-validation"""
    
    def __init__(self):
        self.embargo_periods: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.lock = threading.RLock()
    
    def add_embargo(self, symbol: str, start_time: int, end_time: int):
        """Add embargo period"""
        with self.lock:
            self.embargo_periods[symbol].append((start_time, end_time))
            # Keep sorted by start time
            self.embargo_periods[symbol].sort(key=lambda x: x[0])
    
    def is_embargoed(self, symbol: str, timestamp: int) -> bool:
        """Check if timestamp is in any embargo period"""
        with self.lock:
            for start_time, end_time in self.embargo_periods[symbol]:
                if start_time <= timestamp <= end_time:
                    return True
        return False
    
    def get_valid_timestamps(self, symbol: str, timestamps: np.ndarray) -> np.ndarray:
        """Filter out embargoed timestamps"""
        if symbol not in self.embargo_periods:
            return timestamps
        
        with self.lock:
            valid_mask = np.ones(len(timestamps), dtype=bool)
            
            for start_time, end_time in self.embargo_periods[symbol]:
                embargo_mask = (timestamps >= start_time) & (timestamps <= end_time)
                valid_mask &= ~embargo_mask
            
            return timestamps[valid_mask]

class ClassImbalanceHandler:
    """Handles class imbalance in labeling"""
    
    def __init__(self, target_ratio: float = 0.2, max_negatives_per_positive: int = 10):
        self.target_ratio = target_ratio  # Target ratio of positive labels
        self.max_negatives_per_positive = max_negatives_per_positive
        self.positive_count = 0
        self.negative_count = 0
        self.lock = threading.Lock()
    
    def should_keep_negative_sample(self) -> bool:
        """Decide whether to keep a negative sample based on current ratio"""
        with self.lock:
            if self.positive_count == 0:
                return True  # Keep all negatives if no positives yet
            
            current_ratio = self.positive_count / max(self.positive_count + self.negative_count, 1)
            
            if current_ratio < self.target_ratio:
                return False  # Skip negatives to increase positive ratio
            
            # Check if we have too many negatives per positive
            negatives_per_positive = self.negative_count / max(self.positive_count, 1)
            return negatives_per_positive < self.max_negatives_per_positive
    
    def record_sample(self, is_positive: bool):
        """Record a sample for ratio tracking"""
        with self.lock:
            if is_positive:
                self.positive_count += 1
            else:
                self.negative_count += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current labeling statistics"""
        with self.lock:
            total = self.positive_count + self.negative_count
            return {
                'positive_count': self.positive_count,
                'negative_count': self.negative_count,
                'total_count': total,
                'positive_ratio': self.positive_count / max(total, 1),
                'negative_ratio': self.negative_count / max(total, 1)
            }

class LabelGenerator:
    """Advanced label generator with triple-barrier method"""
    
    def __init__(self):
        self.cooldown_manager = CooldownManager()
        self.embargo_manager = EmbargoManager()
        self.imbalance_handler = ClassImbalanceHandler()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.lock = threading.RLock()
        
        # Configuration
        self.default_config = {
            'theta_up_range': (0.003, 0.010),     # 0.3% to 1.0%
            'theta_dn_range': (0.002, 0.006),     # 0.2% to 0.6%
            'horizon_minutes': [5, 10, 30],
            'cooldown_minutes_range': (10, 30),
            'enable_class_balancing': True,
            'enable_embargo': True,
            'min_samples_per_label': 50
        }
        
        # Label cache
        self.label_cache: Dict[str, List[TripleBarrierLabel]] = defaultdict(list)
        self.cache_lock = threading.Lock()
    
    def generate_labels_batch(self, symbol: str, price_data: pd.DataFrame,
                            timestamps: Optional[np.ndarray] = None,
                            config: Optional[Dict] = None) -> List[TripleBarrierLabel]:
        """
        Generate batch of triple-barrier labels for time series.
        
        Args:
            symbol: Trading symbol
            price_data: DataFrame with columns ['timestamp', 'price', ...]
            timestamps: Optional specific timestamps to label
            config: Optional configuration overrides
            
        Returns:
            List of TripleBarrierLabel objects
        """
        if config is None:
            config = self.default_config.copy()
        
        logger.info(f"Generating labels for {symbol} with {len(price_data)} price points")
        
        try:
            # Prepare data
            if not isinstance(price_data, pd.DataFrame):
                raise ValueError("price_data must be a pandas DataFrame")
            
            required_columns = ['timestamp', 'price']
            if not all(col in price_data.columns for col in required_columns):
                raise ValueError(f"price_data must contain columns: {required_columns}")
            
            # Sort by timestamp
            price_data = price_data.sort_values('timestamp').reset_index(drop=True)
            
            # Convert to numpy arrays for performance
            timestamps_array = price_data['timestamp'].values
            prices_array = price_data['price'].values
            
            # Determine labeling timestamps
            if timestamps is None:
                # Use all timestamps with proper spacing
                labeling_timestamps = self._select_labeling_timestamps(
                    timestamps_array, config.get('min_spacing_minutes', 5)
                )
            else:
                labeling_timestamps = timestamps
            
            # Filter out embargoed timestamps if enabled
            if config.get('enable_embargo', True):
                labeling_timestamps = self.embargo_manager.get_valid_timestamps(
                    symbol, labeling_timestamps
                )
            
            logger.info(f"Labeling {len(labeling_timestamps)} timestamps")
            
            # Generate labels
            labels = []
            for params in self._generate_parameter_combinations(config):
                theta_up, theta_dn, horizon_minutes = params
                
                batch_labels = self._generate_labels_for_parameters(
                    symbol, timestamps_array, prices_array, labeling_timestamps,
                    theta_up, theta_dn, horizon_minutes, config
                )
                
                labels.extend(batch_labels)
            
            # Apply class balancing if enabled
            if config.get('enable_class_balancing', True):
                labels = self._apply_class_balancing(labels)
            
            # Cache results
            with self.cache_lock:
                self.label_cache[symbol].extend(labels)
                # Keep only recent labels (memory management)
                max_cache_size = 10000
                if len(self.label_cache[symbol]) > max_cache_size:
                    self.label_cache[symbol] = self.label_cache[symbol][-max_cache_size:]
            
            logger.info(f"Generated {len(labels)} labels for {symbol}")
            return labels
            
        except Exception as e:
            logger.error(f"Error generating labels for {symbol}: {e}")
            return []
    
    def _select_labeling_timestamps(self, timestamps: np.ndarray, 
                                  min_spacing_minutes: int) -> np.ndarray:
        """Select timestamps for labeling with minimum spacing"""
        if len(timestamps) == 0:
            return timestamps
        
        min_spacing_ms = min_spacing_minutes * 60 * 1000
        selected = [timestamps[0]]
        
        for ts in timestamps[1:]:
            if ts - selected[-1] >= min_spacing_ms:
                selected.append(ts)
        
        return np.array(selected)
    
    def _generate_parameter_combinations(self, config: Dict) -> List[Tuple[float, float, int]]:
        """Generate parameter combinations for labeling"""
        combinations = []
        
        # Theta up values
        theta_up_min, theta_up_max = config.get('theta_up_range', (0.003, 0.010))
        theta_up_values = np.linspace(theta_up_min, theta_up_max, 3)
        
        # Theta down values
        theta_dn_min, theta_dn_max = config.get('theta_dn_range', (0.002, 0.006))
        theta_dn_values = np.linspace(theta_dn_min, theta_dn_max, 3)
        
        # Horizons
        horizons = config.get('horizon_minutes', [5, 10, 30])
        
        # Generate all combinations
        for theta_up in theta_up_values:
            for theta_dn in theta_dn_values:
                for horizon in horizons:
                    # Ensure theta_up > theta_dn
                    if theta_up > theta_dn:
                        combinations.append((theta_up, theta_dn, horizon))
        
        return combinations
    
    def _generate_labels_for_parameters(self, symbol: str, timestamps_array: np.ndarray,
                                      prices_array: np.ndarray, labeling_timestamps: np.ndarray,
                                      theta_up: float, theta_dn: float, horizon_minutes: int,
                                      config: Dict) -> List[TripleBarrierLabel]:
        """Generate labels for specific parameter combination"""
        labels = []
        cooldown_minutes = config.get('cooldown_minutes_range', (10, 30))[0]
        
        for entry_time in labeling_timestamps:
            # Check cooldown
            if self.cooldown_manager.is_in_cooldown(
                symbol, entry_time, theta_up, theta_dn, horizon_minutes
            ):
                continue
            
            # Find entry price
            entry_idx = np.searchsorted(timestamps_array, entry_time)
            if entry_idx >= len(prices_array):
                continue
                
            entry_price = prices_array[entry_idx]
            if entry_price <= 0:
                continue
            
            # Calculate barriers
            upper_barrier = entry_price * (1 + theta_up)
            lower_barrier = entry_price * (1 - theta_dn)
            
            # Find future data for analysis
            max_horizon_ms = horizon_minutes * 60 * 1000
            future_mask = (timestamps_array > entry_time) & (
                timestamps_array <= entry_time + max_horizon_ms
            )
            
            if not np.any(future_mask):
                continue
            
            future_timestamps = timestamps_array[future_mask]
            future_prices = prices_array[future_mask]
            
            # Find barrier touch
            label, breach_timestamp, breach_price, max_favorable = find_first_barrier_touch(
                future_prices, future_timestamps, entry_price,
                upper_barrier, lower_barrier, max_horizon_ms, entry_time
            )
            
            # Calculate regression targets
            peak_return, time_to_peak = calculate_peak_metrics(
                future_prices, future_timestamps, entry_price, entry_time, max_horizon_ms
            )
            
            # Calculate additional metrics
            if len(future_prices) > 0:
                min_price = np.min(future_prices)
                max_adverse_excursion = (min_price - entry_price) / entry_price
            else:
                max_adverse_excursion = 0.0
            
            # Determine exit price and holding period
            if label != 0:  # Barrier touched
                exit_price = breach_price
                holding_period = (breach_timestamp - entry_time) / 1000.0  # seconds
            else:  # Timeout
                exit_price = future_prices[-1] if len(future_prices) > 0 else entry_price
                holding_period = horizon_minutes * 60.0  # full horizon
            
            # Create label
            triple_label = TripleBarrierLabel(
                symbol=symbol,
                timestamp=entry_time,
                label_timestamp=int(time.time() * 1000),
                horizon_minutes=horizon_minutes,
                theta_up=theta_up,
                theta_dn=theta_dn,
                label=label,
                binary_up=label == 1,
                breach_timestamp=breach_timestamp if label != 0 else None,
                breach_price=breach_price if label != 0 else None,
                max_favorable_excursion=max_favorable,
                max_adverse_excursion=max_adverse_excursion,
                peak_return=peak_return,
                time_to_peak=time_to_peak,
                entry_price=entry_price,
                exit_price=exit_price,
                holding_period=holding_period,
                embargo_end=entry_time + max_horizon_ms
            )
            
            labels.append(triple_label)
            
            # Set cooldown
            self.cooldown_manager.set_cooldown(
                symbol, entry_time, theta_up, theta_dn, horizon_minutes, cooldown_minutes
            )
            
            # Add embargo if enabled
            if config.get('enable_embargo', True):
                self.embargo_manager.add_embargo(
                    symbol, entry_time, entry_time + max_horizon_ms
                )
        
        return labels
    
    def _apply_class_balancing(self, labels: List[TripleBarrierLabel]) -> List[TripleBarrierLabel]:
        """Apply class balancing to reduce negative sample dominance"""
        if not labels:
            return labels
        
        balanced_labels = []
        
        # Separate by class
        positive_labels = [label for label in labels if label.binary_up]
        negative_labels = [label for label in labels if not label.binary_up]
        
        # Always keep all positive labels
        balanced_labels.extend(positive_labels)
        
        # Subsample negative labels if needed
        if len(negative_labels) > 0:
            max_negatives = min(
                len(negative_labels),
                len(positive_labels) * self.imbalance_handler.max_negatives_per_positive
            )
            
            if max_negatives < len(negative_labels):
                # Use stratified sampling to maintain temporal distribution
                indices = np.linspace(0, len(negative_labels) - 1, max_negatives, dtype=int)
                selected_negatives = [negative_labels[i] for i in indices]
                balanced_labels.extend(selected_negatives)
            else:
                balanced_labels.extend(negative_labels)
        
        # Update statistics
        for label in balanced_labels:
            self.imbalance_handler.record_sample(label.binary_up)
        
        return balanced_labels
    
    def generate_single_label(self, symbol: str, entry_timestamp: int, entry_price: float,
                            future_prices: np.ndarray, future_timestamps: np.ndarray,
                            theta_up: float = 0.006, theta_dn: float = 0.004,
                            horizon_minutes: int = 10) -> Optional[TripleBarrierLabel]:
        """Generate single triple-barrier label"""
        try:
            # Check cooldown
            if self.cooldown_manager.is_in_cooldown(
                symbol, entry_timestamp, theta_up, theta_dn, horizon_minutes
            ):
                return None
            
            # Calculate barriers
            upper_barrier = entry_price * (1 + theta_up)
            lower_barrier = entry_price * (1 - theta_dn)
            max_horizon_ms = horizon_minutes * 60 * 1000
            
            # Find barrier touch
            label, breach_timestamp, breach_price, max_favorable = find_first_barrier_touch(
                future_prices, future_timestamps, entry_price,
                upper_barrier, lower_barrier, max_horizon_ms, entry_timestamp
            )
            
            # Calculate regression targets
            peak_return, time_to_peak = calculate_peak_metrics(
                future_prices, future_timestamps, entry_price, 
                entry_timestamp, max_horizon_ms
            )
            
            # Additional metrics
            max_adverse_excursion = 0.0
            if len(future_prices) > 0:
                min_price = np.min(future_prices)
                max_adverse_excursion = (min_price - entry_price) / entry_price
            
            # Exit metrics
            if label != 0:
                exit_price = breach_price
                holding_period = (breach_timestamp - entry_timestamp) / 1000.0
            else:
                exit_price = future_prices[-1] if len(future_prices) > 0 else entry_price
                holding_period = horizon_minutes * 60.0
            
            triple_label = TripleBarrierLabel(
                symbol=symbol,
                timestamp=entry_timestamp,
                label_timestamp=int(time.time() * 1000),
                horizon_minutes=horizon_minutes,
                theta_up=theta_up,
                theta_dn=theta_dn,
                label=label,
                binary_up=label == 1,
                breach_timestamp=breach_timestamp if label != 0 else None,
                breach_price=breach_price if label != 0 else None,
                max_favorable_excursion=max_favorable,
                max_adverse_excursion=max_adverse_excursion,
                peak_return=peak_return,
                time_to_peak=time_to_peak,
                entry_price=entry_price,
                exit_price=exit_price,
                holding_period=holding_period,
                embargo_end=entry_timestamp + max_horizon_ms
            )
            
            # Set cooldown
            self.cooldown_manager.set_cooldown(
                symbol, entry_timestamp, theta_up, theta_dn, horizon_minutes, 15
            )
            
            return triple_label
            
        except Exception as e:
            logger.error(f"Error generating single label for {symbol}: {e}")
            return None
    
    def get_label_statistics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get labeling statistics"""
        stats = {
            'imbalance_stats': self.imbalance_handler.get_statistics(),
            'cooldown_stats': {},
            'embargo_stats': {}
        }
        
        # Cooldown statistics
        with self.cooldown_manager.lock:
            if symbol:
                symbols = [symbol] if symbol in self.cooldown_manager.cooldowns else []
            else:
                symbols = list(self.cooldown_manager.cooldowns.keys())
            
            for sym in symbols:
                active_cooldowns = len([
                    end_time for end_time in self.cooldown_manager.cooldowns[sym].values()
                    if end_time > time.time() * 1000
                ])
                stats['cooldown_stats'][sym] = {
                    'total_cooldowns': len(self.cooldown_manager.cooldowns[sym]),
                    'active_cooldowns': active_cooldowns
                }
        
        # Embargo statistics
        with self.embargo_manager.lock:
            if symbol:
                symbols = [symbol] if symbol in self.embargo_manager.embargo_periods else []
            else:
                symbols = list(self.embargo_manager.embargo_periods.keys())
            
            for sym in symbols:
                current_time = int(time.time() * 1000)
                active_embargos = len([
                    (start, end) for start, end in self.embargo_manager.embargo_periods[sym]
                    if start <= current_time <= end
                ])
                stats['embargo_stats'][sym] = {
                    'total_embargos': len(self.embargo_manager.embargo_periods[sym]),
                    'active_embargos': active_embargos
                }
        
        # Cache statistics
        with self.cache_lock:
            total_cached_labels = sum(len(labels) for labels in self.label_cache.values())
            stats['cache_stats'] = {
                'symbols_cached': len(self.label_cache),
                'total_labels_cached': total_cached_labels,
                'average_labels_per_symbol': (
                    total_cached_labels / max(len(self.label_cache), 1)
                )
            }
        
        return stats
    
    def create_purged_splits(self, labels: List[TripleBarrierLabel], 
                           n_splits: int = 5, test_size: float = 0.2,
                           embargo_pct: float = 0.01) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create purged time series splits for cross-validation.
        
        Args:
            labels: List of labels to split
            n_splits: Number of cross-validation splits
            test_size: Fraction of data for test set
            embargo_pct: Fraction of data to embargo between train/test
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        if not labels:
            return []
        
        # Sort labels by timestamp
        sorted_labels = sorted(labels, key=lambda x: x.timestamp)
        n_samples = len(sorted_labels)
        
        splits = []
        test_size_samples = int(n_samples * test_size)
        embargo_size = int(n_samples * embargo_pct)
        
        for i in range(n_splits):
            # Calculate test set boundaries
            test_start = i * (n_samples // n_splits)
            test_end = min(test_start + test_size_samples, n_samples)
            
            # Create test indices
            test_indices = np.arange(test_start, test_end)
            
            # Create train indices with embargo
            train_indices = []
            
            # Train data before test set (with embargo)
            if test_start > embargo_size:
                train_indices.extend(range(0, test_start - embargo_size))
            
            # Train data after test set (with embargo)
            if test_end + embargo_size < n_samples:
                train_indices.extend(range(test_end + embargo_size, n_samples))
            
            train_indices = np.array(train_indices)
            
            # Ensure we have both train and test data
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        logger.info(f"Created {len(splits)} purged splits with average train size: "
                   f"{np.mean([len(train) for train, _ in splits]):.0f}, "
                   f"test size: {np.mean([len(test) for _, test in splits]):.0f}")
        
        return splits
    
    def validate_labels(self, labels: List[TripleBarrierLabel]) -> Dict[str, Any]:
        """Validate label quality and consistency"""
        if not labels:
            return {'valid': False, 'errors': ['No labels provided']}
        
        errors = []
        warnings = []
        
        # Check for basic consistency
        for label in labels[:100]:  # Sample check
            if label.entry_price <= 0:
                errors.append(f"Invalid entry price: {label.entry_price}")
            
            if label.theta_up <= label.theta_dn:
                errors.append(f"theta_up ({label.theta_up}) must be > theta_dn ({label.theta_dn})")
            
            if label.horizon_minutes <= 0:
                errors.append(f"Invalid horizon: {label.horizon_minutes}")
            
            if label.holding_period < 0:
                errors.append(f"Negative holding period: {label.holding_period}")
        
        # Check class distribution
        positive_count = sum(1 for label in labels if label.binary_up)
        negative_count = len(labels) - positive_count
        
        if positive_count == 0:
            warnings.append("No positive labels found")
        elif negative_count == 0:
            warnings.append("No negative labels found")
        else:
            imbalance_ratio = negative_count / positive_count
            if imbalance_ratio > 20:
                warnings.append(f"High class imbalance ratio: {imbalance_ratio:.1f}")
        
        # Check temporal coverage
        timestamps = [label.timestamp for label in labels]
        if timestamps:
            time_span_hours = (max(timestamps) - min(timestamps)) / (1000 * 3600)
            if time_span_hours < 24:
                warnings.append(f"Short temporal coverage: {time_span_hours:.1f} hours")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'statistics': {
                'total_labels': len(labels),
                'positive_labels': positive_count,
                'negative_labels': negative_count,
                'positive_ratio': positive_count / len(labels) if labels else 0,
                'unique_symbols': len(set(label.symbol for label in labels)),
                'time_span_hours': (max(timestamps) - min(timestamps)) / (1000 * 3600) if timestamps else 0
            }
        }
    
    def cleanup_expired_cooldowns(self):
        """Cleanup expired cooldowns and embargos"""
        current_time = int(time.time() * 1000)
        self.cooldown_manager.cleanup_expired(current_time)
        
        # Cleanup old embargos (keep only recent ones)
        cutoff_time = current_time - 24 * 3600 * 1000  # 24 hours
        with self.embargo_manager.lock:
            for symbol in self.embargo_manager.embargo_periods:
                self.embargo_manager.embargo_periods[symbol] = [
                    (start, end) for start, end in self.embargo_manager.embargo_periods[symbol]
                    if end >= cutoff_time
                ]
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        self.cleanup_expired_cooldowns()
