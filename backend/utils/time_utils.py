"""
Time synchronization and management utilities for crypto surge prediction system.
Handles NTP synchronization, clock drift detection, and precise timing operations.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import os
from collections import deque
import statistics

# NTP client imports (optional)
try:
    import ntplib
    NTP_AVAILABLE = True
except ImportError:
    NTP_AVAILABLE = False
    ntplib = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeMetrics:
    """Time synchronization metrics"""
    local_time: float
    exchange_time: float
    ntp_time: Optional[float]
    clock_drift_ms: float
    round_trip_time_ms: float
    precision_ms: float
    sync_quality: str  # 'good', 'degraded', 'poor'

class ClockDriftDetector:
    """Detects and tracks clock drift using EWMA"""
    
    def __init__(self, alpha: float = 0.1, degraded_threshold_ms: float = 100.0):
        self.alpha = alpha  # EWMA smoothing factor
        self.degraded_threshold_ms = degraded_threshold_ms
        self.ewma_drift_ms = 0.0
        self.sample_count = 0
        self.last_update_time = 0.0
        self.drift_history: deque = deque(maxlen=100)
        self.lock = threading.Lock()
    
    def update_drift(self, exchange_time_ms: int, local_time_ms: int):
        """Update drift calculation with new timestamp pair"""
        with self.lock:
            drift_ms = exchange_time_ms - local_time_ms
            
            if self.sample_count == 0:
                self.ewma_drift_ms = drift_ms
            else:
                self.ewma_drift_ms = (self.alpha * drift_ms + 
                                     (1 - self.alpha) * self.ewma_drift_ms)
            
            self.sample_count += 1
            self.last_update_time = time.time()
            self.drift_history.append(drift_ms)
    
    def get_current_drift_ms(self) -> float:
        """Get current EWMA drift in milliseconds"""
        with self.lock:
            return self.ewma_drift_ms
    
    def is_degraded(self) -> bool:
        """Check if clock sync is degraded"""
        with self.lock:
            return abs(self.ewma_drift_ms) > self.degraded_threshold_ms
    
    def get_drift_statistics(self) -> Dict[str, float]:
        """Get drift statistics"""
        with self.lock:
            if not self.drift_history:
                return {}
            
            drift_values = list(self.drift_history)
            return {
                'current_ewma_ms': self.ewma_drift_ms,
                'min_drift_ms': min(drift_values),
                'max_drift_ms': max(drift_values),
                'mean_drift_ms': statistics.mean(drift_values),
                'std_drift_ms': statistics.stdev(drift_values) if len(drift_values) > 1 else 0.0,
                'sample_count': len(drift_values),
                'is_degraded': self.is_degraded()
            }

class NTPSynchronizer:
    """NTP-based time synchronization"""
    
    def __init__(self, ntp_servers: Optional[List[str]] = None):
        self.ntp_servers = ntp_servers or [
            'pool.ntp.org',
            'time.google.com',
            'time.cloudflare.com',
            'time.apple.com'
        ]
        self.last_sync_time = 0.0
        self.ntp_offset_ms = 0.0
        self.sync_precision_ms = 0.0
        self.sync_interval_s = 300  # 5 minutes
        self.sync_timeout_s = 5.0
        self.lock = threading.Lock()
        self.sync_history: deque = deque(maxlen=50)
    
    async def synchronize(self) -> Optional[TimeMetrics]:
        """Perform NTP synchronization"""
        if not NTP_AVAILABLE:
            logger.warning("NTP client not available, skipping synchronization")
            return None
        
        best_sync = None
        best_precision = float('inf')
        
        for server in self.ntp_servers:
            try:
                sync_result = await self._sync_with_server(server)
                if sync_result and sync_result.precision_ms < best_precision:
                    best_sync = sync_result
                    best_precision = sync_result.precision_ms
            except Exception as e:
                logger.debug(f"NTP sync failed with {server}: {e}")
                continue
        
        if best_sync:
            with self.lock:
                self.ntp_offset_ms = best_sync.local_time - best_sync.ntp_time
                self.sync_precision_ms = best_sync.precision_ms
                self.last_sync_time = time.time()
                self.sync_history.append(best_sync)
            
            logger.info(f"NTP sync successful: offset={self.ntp_offset_ms:.1f}ms, "
                       f"precision={self.sync_precision_ms:.1f}ms")
        
        return best_sync
    
    async def _sync_with_server(self, server: str) -> Optional[TimeMetrics]:
        """Sync with a specific NTP server"""
        try:
            # Use asyncio to run NTP request in thread pool
            loop = asyncio.get_event_loop()
            
            def ntp_request():
                client = ntplib.NTPClient()
                return client.request(server, timeout=self.sync_timeout_s)
            
            response = await asyncio.wait_for(
                loop.run_in_executor(None, ntp_request),
                timeout=self.sync_timeout_s
            )
            
            local_time = time.time()
            ntp_time = response.tx_time
            
            # Calculate metrics
            round_trip_time_ms = response.delay * 1000
            precision_ms = response.precision * 1000
            
            # Determine sync quality
            if precision_ms < 10:
                sync_quality = 'good'
            elif precision_ms < 50:
                sync_quality = 'degraded'  
            else:
                sync_quality = 'poor'
            
            return TimeMetrics(
                local_time=local_time,
                exchange_time=0,  # Not applicable for NTP
                ntp_time=ntp_time,
                clock_drift_ms=(local_time - ntp_time) * 1000,
                round_trip_time_ms=round_trip_time_ms,
                precision_ms=precision_ms,
                sync_quality=sync_quality
            )
            
        except Exception as e:
            logger.debug(f"NTP sync failed with {server}: {e}")
            return None
    
    def get_adjusted_time(self) -> float:
        """Get NTP-adjusted current time"""
        with self.lock:
            if self.last_sync_time == 0:
                return time.time()
            
            # Check if sync is stale
            time_since_sync = time.time() - self.last_sync_time
            if time_since_sync > self.sync_interval_s * 2:
                return time.time()  # Fallback to local time
            
            return time.time() - (self.ntp_offset_ms / 1000.0)
    
    def is_sync_valid(self) -> bool:
        """Check if NTP sync is valid and recent"""
        with self.lock:
            if self.last_sync_time == 0:
                return False
            
            time_since_sync = time.time() - self.last_sync_time
            return time_since_sync < self.sync_interval_s * 2
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get NTP synchronization status"""
        with self.lock:
            return {
                'is_available': NTP_AVAILABLE,
                'is_valid': self.is_sync_valid(),
                'last_sync_time': self.last_sync_time,
                'offset_ms': self.ntp_offset_ms,
                'precision_ms': self.sync_precision_ms,
                'servers': self.ntp_servers,
                'sync_count': len(self.sync_history)
            }

class LatencyTracker:
    """Tracks and analyzes latency measurements"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record_latency(self, latency_ms: float):
        """Record a latency measurement"""
        with self.lock:
            self.latencies.append(latency_ms)
    
    def get_percentiles(self, percentiles: List[float] = [0.5, 0.95, 0.99]) -> Dict[str, float]:
        """Get latency percentiles"""
        with self.lock:
            if not self.latencies:
                return {}
            
            sorted_latencies = sorted(self.latencies)
            n = len(sorted_latencies)
            
            result = {}
            for p in percentiles:
                index = int(n * p)
                index = min(index, n - 1)
                result[f'p{int(p*100)}'] = sorted_latencies[index]
            
            return result
    
    def get_sla_compliance(self, sla_targets: Dict[str, float]) -> Dict[str, Dict]:
        """Check SLA compliance for latency targets"""
        percentiles = self.get_percentiles([0.95, 0.99])
        compliance = {}
        
        for target_name, target_ms in sla_targets.items():
            if target_name in percentiles:
                actual_ms = percentiles[target_name]
                is_compliant = actual_ms <= target_ms
                compliance[target_name] = {
                    'target_ms': target_ms,
                    'actual_ms': actual_ms,
                    'is_compliant': is_compliant,
                    'violation_ms': max(0, actual_ms - target_ms)
                }
        
        return compliance

class TimeManager:
    """Comprehensive time management for the trading system"""
    
    def __init__(self):
        self.clock_drift_detector = ClockDriftDetector()
        self.ntp_synchronizer = NTPSynchronizer()
        self.latency_trackers: Dict[str, LatencyTracker] = {}
        
        # SLA targets
        self.sla_targets = {
            'p95': 500.0,   # 500ms p95
            'p99': 800.0    # 800ms p99
        }
        
        # Background tasks
        self.sync_task = None
        self.monitoring_task = None
        self.shutdown_flag = False
        
        # Performance tracking
        self.performance_metrics: Dict[str, Any] = {}
        self.metrics_lock = threading.Lock()
    
    async def initialize(self):
        """Initialize time manager"""
        logger.info("Initializing time manager...")
        
        # Perform initial NTP sync
        await self.ntp_synchronizer.synchronize()
        
        # Start background tasks
        self.sync_task = asyncio.create_task(self._ntp_sync_loop())
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Time manager initialized")
    
    async def _ntp_sync_loop(self):
        """Background NTP synchronization loop"""
        while not self.shutdown_flag:
            try:
                await asyncio.sleep(self.ntp_synchronizer.sync_interval_s)
                await self.ntp_synchronizer.synchronize()
            except Exception as e:
                logger.error(f"Error in NTP sync loop: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while not self.shutdown_flag:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                self._update_performance_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        with self.metrics_lock:
            # Clock drift metrics
            drift_stats = self.clock_drift_detector.get_drift_statistics()
            
            # NTP sync metrics
            ntp_status = self.ntp_synchronizer.get_sync_status()
            
            # Latency metrics
            latency_metrics = {}
            for name, tracker in self.latency_trackers.items():
                percentiles = tracker.get_percentiles()
                sla_compliance = tracker.get_sla_compliance(self.sla_targets)
                latency_metrics[name] = {
                    'percentiles': percentiles,
                    'sla_compliance': sla_compliance
                }
            
            self.performance_metrics = {
                'timestamp': time.time(),
                'clock_drift': drift_stats,
                'ntp_sync': ntp_status,
                'latency_metrics': latency_metrics
            }
    
    def record_exchange_timestamp(self, exchange_time_ms: int):
        """Record exchange timestamp for drift calculation"""
        local_time_ms = int(time.time() * 1000)
        self.clock_drift_detector.update_drift(exchange_time_ms, local_time_ms)
    
    def record_latency(self, operation: str, latency_ms: float):
        """Record latency for a specific operation"""
        if operation not in self.latency_trackers:
            self.latency_trackers[operation] = LatencyTracker()
        
        self.latency_trackers[operation].record_latency(latency_ms)
    
    def get_current_time_ms(self) -> int:
        """Get current time in milliseconds (NTP-adjusted if available)"""
        if self.ntp_synchronizer.is_sync_valid():
            return int(self.ntp_synchronizer.get_adjusted_time() * 1000)
        else:
            return int(time.time() * 1000)
    
    def get_time_quality_flags(self) -> List[str]:
        """Get time quality flags"""
        flags = []
        
        # Check clock drift
        if self.clock_drift_detector.is_degraded():
            flags.append('degraded_time_sync')
        
        # Check NTP sync
        if not self.ntp_synchronizer.is_sync_valid():
            flags.append('ntp_sync_stale')
        
        return flags
    
    def calculate_end_to_end_latency(self, exchange_time_ms: int, 
                                   ingest_time_ms: int, infer_time_ms: int) -> Dict[str, float]:
        """Calculate end-to-end latency breakdown"""
        current_time_ms = self.get_current_time_ms()
        
        latency_breakdown = {
            'exchange_to_ingest_ms': ingest_time_ms - exchange_time_ms,
            'ingest_to_infer_ms': infer_time_ms - ingest_time_ms,
            'infer_to_now_ms': current_time_ms - infer_time_ms,
            'total_e2e_ms': current_time_ms - exchange_time_ms
        }
        
        # Record total E2E latency
        self.record_latency('e2e_total', latency_breakdown['total_e2e_ms'])
        
        return latency_breakdown
    
    def get_sla_status(self) -> Dict[str, Any]:
        """Get SLA compliance status"""
        sla_status = {
            'targets': self.sla_targets,
            'compliance': {},
            'overall_compliant': True
        }
        
        for name, tracker in self.latency_trackers.items():
            compliance = tracker.get_sla_compliance(self.sla_targets)
            sla_status['compliance'][name] = compliance
            
            # Check if any SLA is violated
            for target_compliance in compliance.values():
                if not target_compliance['is_compliant']:
                    sla_status['overall_compliant'] = False
        
        return sla_status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.metrics_lock:
            return self.performance_metrics.copy()
    
    def create_timing_context(self, operation_name: str):
        """Create timing context manager"""
        return TimingContext(self, operation_name)
    
    async def cleanup(self):
        """Cleanup time manager resources"""
        logger.info("Cleaning up time manager...")
        
        self.shutdown_flag = True
        
        if self.sync_task:
            self.sync_task.cancel()
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        logger.info("Time manager cleanup completed")

class TimingContext:
    """Context manager for measuring operation timing"""
    
    def __init__(self, time_manager: TimeManager, operation_name: str):
        self.time_manager = time_manager
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        latency_ms = (self.end_time - self.start_time) * 1000
        self.time_manager.record_latency(self.operation_name, latency_ms)
        
        # Log if operation took too long
        if latency_ms > 100:  # 100ms threshold
            logger.warning(f"Slow operation {self.operation_name}: {latency_ms:.1f}ms")
    
    def get_elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.start_time is None:
            return 0.0
        
        end_time = self.end_time or time.time()
        return (end_time - self.start_time) * 1000

# Utility functions
def create_three_timestamp_record(exchange_time_ms: int, time_manager: TimeManager) -> Dict[str, int]:
    """Create standard three-timestamp record"""
    current_time_ms = time_manager.get_current_time_ms()
    
    return {
        'exchange_time': exchange_time_ms,
        'ingest_time': current_time_ms,
        'infer_time': current_time_ms  # Will be updated during inference
    }

def validate_timestamp_sequence(timestamps: Dict[str, int], 
                              max_drift_ms: float = 1000.0) -> List[str]:
    """Validate timestamp sequence and return quality flags"""
    flags = []
    
    exchange_time = timestamps.get('exchange_time', 0)
    ingest_time = timestamps.get('ingest_time', 0)
    infer_time = timestamps.get('infer_time', 0)
    
    # Check sequence order
    if exchange_time > ingest_time:
        flags.append('invalid_timestamp_sequence')
    
    if ingest_time > infer_time:
        flags.append('invalid_timestamp_sequence')
    
    # Check for excessive drift
    if abs(exchange_time - ingest_time) > max_drift_ms:
        flags.append('high_latency')
    
    # Check for clock drift
    current_time_ms = int(time.time() * 1000)
    if abs(infer_time - current_time_ms) > max_drift_ms:
        flags.append('clock_drift')
    
    return flags

def format_latency(latency_ms: float) -> str:
    """Format latency for display"""
    if latency_ms < 1:
        return f"{latency_ms:.2f}ms"
    elif latency_ms < 1000:
        return f"{latency_ms:.1f}ms"
    else:
        return f"{latency_ms/1000:.2f}s"

