"""
Comprehensive monitoring and metrics collection for crypto surge prediction system.
Implements Prometheus metrics, distributed tracing, and performance monitoring.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import os
import json
from contextlib import contextmanager
import uuid
import hashlib

# Prometheus client imports
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, Info, start_http_server
    from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = Info = None
    CollectorRegistry = None

# OpenTelemetry imports (optional)
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: float
    labels: Dict[str, str]
    timestamp: float
    metric_type: str  # counter, gauge, histogram

@dataclass
class TraceSpan:
    """Distributed trace span"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float]
    tags: Dict[str, str]
    logs: List[Dict[str, Any]]
    status: str

class PerformanceTracker:
    """Tracks performance metrics with sliding windows"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.data_points: deque = deque(maxlen=window_size)
        self.lock = threading.Lock()
    
    def record(self, value: float, timestamp: Optional[float] = None):
        """Record a performance data point"""
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:
            self.data_points.append((timestamp, value))
    
    def get_stats(self, lookback_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get performance statistics"""
        with self.lock:
            if not self.data_points:
                return {}
            
            # Filter by lookback period if specified
            if lookback_seconds:
                cutoff_time = time.time() - lookback_seconds
                valid_points = [(ts, val) for ts, val in self.data_points if ts >= cutoff_time]
            else:
                valid_points = list(self.data_points)
            
            if not valid_points:
                return {}
            
            values = [val for _, val in valid_points]
            values.sort()
            n = len(values)
            
            stats = {
                'count': n,
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / n,
                'median': values[n // 2],
                'p95': values[int(n * 0.95)] if n > 0 else 0,
                'p99': values[int(n * 0.99)] if n > 0 else 0
            }
            
            return stats

class AlertManager:
    """Manages alerts based on metric thresholds"""
    
    def __init__(self):
        self.alert_rules: Dict[str, Dict] = {}
        self.active_alerts: Dict[str, Dict] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.lock = threading.Lock()
    
    def add_alert_rule(self, name: str, metric_name: str, threshold: float, 
                      comparison: str = 'greater', duration_seconds: int = 60):
        """Add alert rule"""
        with self.lock:
            self.alert_rules[name] = {
                'metric_name': metric_name,
                'threshold': threshold,
                'comparison': comparison,  # 'greater', 'less', 'equal'
                'duration_seconds': duration_seconds,
                'created_at': time.time()
            }
    
    def check_alerts(self, metrics: Dict[str, float]) -> List[Dict]:
        """Check metrics against alert rules"""
        triggered_alerts = []
        current_time = time.time()
        
        with self.lock:
            for alert_name, rule in self.alert_rules.items():
                metric_name = rule['metric_name']
                
                if metric_name not in metrics:
                    continue
                
                metric_value = metrics[metric_name]
                threshold = rule['threshold']
                comparison = rule['comparison']
                
                # Check if alert condition is met
                condition_met = False
                if comparison == 'greater' and metric_value > threshold:
                    condition_met = True
                elif comparison == 'less' and metric_value < threshold:
                    condition_met = True
                elif comparison == 'equal' and abs(metric_value - threshold) < 0.001:
                    condition_met = True
                
                if condition_met:
                    # Check if this is a new alert or existing one
                    if alert_name not in self.active_alerts:
                        # New alert
                        self.active_alerts[alert_name] = {
                            'started_at': current_time,
                            'rule': rule,
                            'current_value': metric_value
                        }
                    else:
                        # Update existing alert
                        self.active_alerts[alert_name]['current_value'] = metric_value
                    
                    # Check if alert has been active long enough
                    alert_duration = current_time - self.active_alerts[alert_name]['started_at']
                    if alert_duration >= rule['duration_seconds']:
                        triggered_alert = {
                            'name': alert_name,
                            'metric_name': metric_name,
                            'current_value': metric_value,
                            'threshold': threshold,
                            'comparison': comparison,
                            'duration_seconds': alert_duration,
                            'timestamp': current_time
                        }
                        triggered_alerts.append(triggered_alert)
                        
                        # Add to history
                        self.alert_history.append(triggered_alert.copy())
                
                else:
                    # Condition not met, remove from active alerts
                    if alert_name in self.active_alerts:
                        del self.active_alerts[alert_name]
        
        return triggered_alerts
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts"""
        with self.lock:
            return [
                {
                    'name': name,
                    'started_at': alert['started_at'],
                    'current_value': alert['current_value'],
                    'rule': alert['rule']
                }
                for name, alert in self.active_alerts.items()
            ]

class MetricsCollector:
    """Comprehensive metrics collector with Prometheus and OTEL support"""
    
    def __init__(self, service_name: str, enable_prometheus: bool = True, 
                 enable_otel: bool = True, metrics_port: Optional[int] = None):
        self.service_name = service_name
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.enable_otel = enable_otel and OTEL_AVAILABLE
        
        # Prometheus setup
        if self.enable_prometheus:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()
            
            # Start metrics HTTP server
            if metrics_port:
                start_http_server(metrics_port, registry=self.registry)
                logger.info(f"Prometheus metrics server started on port {metrics_port}")
        
        # OpenTelemetry setup
        if self.enable_otel:
            self.tracer = trace.get_tracer(__name__)
            self.trace_propagator = TraceContextTextMapPropagator()
        else:
            self.tracer = None
        
        # Custom metrics storage
        self.custom_metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        self.alert_manager = AlertManager()
        
        # Locks for thread safety
        self.metrics_lock = threading.Lock()
        self.trackers_lock = threading.Lock()
        
        # Active traces
        self.active_traces: Dict[str, TraceSpan] = {}
        
        # Setup default alert rules
        self._setup_default_alerts()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Counters
        self.counters = {
            'ws_messages_received': Counter(
                'ws_messages_received_total',
                'Total WebSocket messages received',
                ['connection', 'symbol', 'stream_type'],
                registry=self.registry
            ),
            'ws_message_errors': Counter(
                'ws_message_errors_total', 
                'WebSocket message processing errors',
                ['connection', 'error_type'],
                registry=self.registry
            ),
            'features_computed': Counter(
                'features_computed_total',
                'Total features computed', 
                ['symbol'],
                registry=self.registry
            ),
            'predictions_made': Counter(
                'predictions_made_total',
                'Total predictions made',
                ['symbol'],
                registry=self.registry
            ),
            'predict_errors': Counter(
                'predict_errors_total',
                'Prediction errors',
                ['symbol', 'error_type'],
                registry=self.registry
            )
        }
        
        # Histograms
        self.histograms = {
            'ws_e2e_latency_ms': Histogram(
                'ws_e2e_latency_ms',
                'End-to-end WebSocket processing latency',
                ['connection'],
                buckets=[10, 25, 50, 100, 200, 500, 1000, 2000, 5000],
                registry=self.registry
            ),
            'batch_processing_latency_ms': Histogram(
                'batch_processing_latency_ms', 
                'Batch processing latency',
                buckets=[1, 5, 10, 25, 50, 100, 200, 500],
                registry=self.registry
            ),
            'feature_compute_ms': Histogram(
                'feature_compute_ms',
                'Feature computation time',
                ['feature_name'],
                buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50],
                registry=self.registry
            ),
            'predict_latency_ms': Histogram(
                'predict_latency_ms',
                'Prediction latency',
                ['model_name'],
                buckets=[1, 5, 10, 25, 50, 100, 200, 500, 1000],
                registry=self.registry
            )
        }
        
        # Gauges
        self.gauges = {
            'ws_packet_loss_rate': Gauge(
                'ws_packet_loss_rate',
                'WebSocket packet loss rate',
                ['connection'],
                registry=self.registry
            ),
            'ws_depth_gap_rate': Gauge(
                'ws_depth_gap_rate', 
                'Depth gap rate per minute',
                ['symbol'],
                registry=self.registry
            ),
            'feature_nan_rate': Gauge(
                'feature_nan_rate',
                'Feature NaN rate',
                ['symbol', 'feature_name'],
                registry=self.registry
            ),
            'calibration_ece_pp': Gauge(
                'calibration_ece_pp',
                'Expected Calibration Error in percentage points',
                ['model_name'],
                registry=self.registry
            ),
            'rolling_pr_auc_7d': Gauge(
                'rolling_pr_auc_7d',
                '7-day rolling PR-AUC',
                ['symbol', 'model_name'],
                registry=self.registry
            )
        }
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High latency alerts
        self.alert_manager.add_alert_rule(
            'high_ws_latency',
            'ws_e2e_latency_ms_p95',
            300,  # 300ms
            'greater',
            600   # 10 minutes
        )
        
        self.alert_manager.add_alert_rule(
            'high_prediction_latency', 
            'predict_latency_ms_p95',
            50,   # 50ms
            'greater', 
            300   # 5 minutes
        )
        
        # Quality degradation alerts
        self.alert_manager.add_alert_rule(
            'high_packet_loss',
            'ws_packet_loss_rate',
            0.01,  # 1%
            'greater',
            300    # 5 minutes
        )
        
        self.alert_manager.add_alert_rule(
            'calibration_drift',
            'calibration_ece_pp',
            5.0,   # 5pp
            'greater',
            1800   # 30 minutes
        )
    
    # Counter methods
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0):
        """Increment counter metric"""
        if self.enable_prometheus and name in self.counters:
            if labels:
                self.counters[name].labels(**labels).inc(value)
            else:
                self.counters[name].inc(value)
        
        # Store in custom metrics
        self._store_custom_metric(name, value, labels or {}, 'counter')
    
    # Histogram methods  
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe histogram metric"""
        if self.enable_prometheus and name in self.histograms:
            if labels:
                self.histograms[name].labels(**labels).observe(value)
            else:
                self.histograms[name].observe(value)
        
        # Store in custom metrics
        self._store_custom_metric(name, value, labels or {}, 'histogram')
        
        # Update performance tracker
        self._update_performance_tracker(name, value)
    
    # Gauge methods
    def observe_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set gauge metric"""
        if self.enable_prometheus and name in self.gauges:
            if labels:
                self.gauges[name].labels(**labels).set(value)
            else:
                self.gauges[name].set(value)
        
        # Store in custom metrics
        self._store_custom_metric(name, value, labels or {}, 'gauge')
    
    def _store_custom_metric(self, name: str, value: float, labels: Dict[str, str], metric_type: str):
        """Store custom metric"""
        metric_point = MetricPoint(
            name=name,
            value=value,
            labels=labels,
            timestamp=time.time(),
            metric_type=metric_type
        )
        
        with self.metrics_lock:
            self.custom_metrics[name].append(metric_point)
            
            # Keep only recent metrics (last 1000 points)
            if len(self.custom_metrics[name]) > 1000:
                self.custom_metrics[name] = self.custom_metrics[name][-1000:]
    
    def _update_performance_tracker(self, name: str, value: float):
        """Update performance tracker"""
        with self.trackers_lock:
            if name not in self.performance_trackers:
                self.performance_trackers[name] = PerformanceTracker()
            
            self.performance_trackers[name].record(value)
    
    # Performance tracking methods
    def get_histogram_percentile(self, name: str, percentile: float, 
                                lookback_seconds: Optional[float] = None) -> Optional[float]:
        """Get histogram percentile"""
        with self.trackers_lock:
            if name not in self.performance_trackers:
                return None
            
            stats = self.performance_trackers[name].get_stats(lookback_seconds)
            
            if percentile == 0.5:
                return stats.get('median')
            elif percentile == 0.95:
                return stats.get('p95')
            elif percentile == 0.99:
                return stats.get('p99')
            else:
                # Calculate custom percentile
                with self.performance_trackers[name].lock:
                    points = list(self.performance_trackers[name].data_points)
                    if lookback_seconds:
                        cutoff_time = time.time() - lookback_seconds
                        points = [(ts, val) for ts, val in points if ts >= cutoff_time]
                    
                    if not points:
                        return None
                    
                    values = sorted([val for _, val in points])
                    index = int(len(values) * percentile)
                    return values[min(index, len(values) - 1)]
    
    # Distributed tracing methods
    @contextmanager
    def trace_span(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for tracing spans"""
        if not self.enable_otel or not self.tracer:
            yield None
            return
        
        span_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            start_time=time.time(),
            end_time=None,
            tags=tags or {},
            logs=[],
            status='started'
        )
        
        self.active_traces[span_id] = span
        
        try:
            yield span
            span.status = 'completed'
        except Exception as e:
            span.status = 'error'
            span.tags['error'] = str(e)
            raise
        finally:
            span.end_time = time.time()
            if span_id in self.active_traces:
                del self.active_traces[span_id]
    
    def add_span_log(self, span: TraceSpan, message: str, level: str = 'info'):
        """Add log to span"""
        if span:
            span.logs.append({
                'timestamp': time.time(),
                'level': level,
                'message': message
            })
    
    # Alert methods
    def check_and_trigger_alerts(self) -> List[Dict]:
        """Check metrics and trigger alerts"""
        # Collect current metric values
        current_metrics = {}
        
        # Get performance tracker stats
        with self.trackers_lock:
            for name, tracker in self.performance_trackers.items():
                stats = tracker.get_stats(3600)  # Last hour
                if stats:
                    current_metrics[f"{name}_p95"] = stats.get('p95', 0)
                    current_metrics[f"{name}_p99"] = stats.get('p99', 0)
                    current_metrics[f"{name}_mean"] = stats.get('mean', 0)
        
        # Add custom metrics (latest values)
        with self.metrics_lock:
            for name, points in self.custom_metrics.items():
                if points:
                    latest_point = points[-1]
                    current_metrics[name] = latest_point.value
        
        return self.alert_manager.check_alerts(current_metrics)
    
    # Health and status methods
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        health = {
            'service_name': self.service_name,
            'timestamp': time.time(),
            'status': 'healthy',
            'prometheus_enabled': self.enable_prometheus,
            'otel_enabled': self.enable_otel
        }
        
        # Performance metrics
        performance_stats = {}
        with self.trackers_lock:
            for name, tracker in self.performance_trackers.items():
                stats = tracker.get_stats(300)  # Last 5 minutes
                if stats:
                    performance_stats[name] = stats
        
        health['performance_stats'] = performance_stats
        
        # Active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        health['active_alerts'] = active_alerts
        health['alert_count'] = len(active_alerts)
        
        # Set overall status based on alerts
        if any(alert for alert in active_alerts 
               if 'high' in alert['name'] or 'error' in alert['name']):
            health['status'] = 'degraded'
        
        # Active traces
        health['active_traces'] = len(self.active_traces)
        
        # Memory usage (approximation)
        with self.metrics_lock:
            total_metrics = sum(len(points) for points in self.custom_metrics.values())
        
        health['memory_stats'] = {
            'total_custom_metrics': total_metrics,
            'performance_trackers': len(self.performance_trackers),
            'active_traces': len(self.active_traces)
        }
        
        return health
    
    def get_metrics_summary(self, lookback_seconds: int = 3600) -> Dict[str, Any]:
        """Get metrics summary for the specified lookback period"""
        cutoff_time = time.time() - lookback_seconds
        summary = {
            'period_seconds': lookback_seconds,
            'metrics': {},
            'performance_stats': {}
        }
        
        # Custom metrics summary
        with self.metrics_lock:
            for name, points in self.custom_metrics.items():
                recent_points = [p for p in points if p.timestamp >= cutoff_time]
                
                if recent_points:
                    values = [p.value for p in recent_points]
                    metric_type = recent_points[0].metric_type
                    
                    if metric_type == 'counter':
                        summary['metrics'][name] = {
                            'type': 'counter',
                            'total': sum(values),
                            'count': len(values),
                            'rate_per_second': sum(values) / lookback_seconds
                        }
                    elif metric_type in ['histogram', 'gauge']:
                        values.sort()
                        n = len(values)
                        summary['metrics'][name] = {
                            'type': metric_type,
                            'count': n,
                            'min': min(values),
                            'max': max(values),
                            'mean': sum(values) / n,
                            'median': values[n // 2],
                            'p95': values[int(n * 0.95)] if n > 0 else 0,
                            'p99': values[int(n * 0.99)] if n > 0 else 0
                        }
        
        # Performance tracker stats
        with self.trackers_lock:
            for name, tracker in self.performance_trackers.items():
                stats = tracker.get_stats(lookback_seconds)
                if stats:
                    summary['performance_stats'][name] = stats
        
        return summary
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.enable_prometheus:
            return "# Prometheus not enabled\n"
        
        return generate_latest(self.registry)
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up metrics collector...")
        
        with self.metrics_lock:
            self.custom_metrics.clear()
        
        with self.trackers_lock:
            self.performance_trackers.clear()
        
        self.active_traces.clear()
        
        logger.info("Metrics collector cleanup completed")

# Utility functions for common monitoring patterns
def time_function(metrics_collector: MetricsCollector, metric_name: str, 
                 labels: Optional[Dict[str, str]] = None):
    """Decorator to time function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                metrics_collector.observe_histogram(metric_name, duration_ms, labels)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                error_labels = (labels or {}).copy()
                error_labels['error'] = 'true'
                metrics_collector.observe_histogram(metric_name, duration_ms, error_labels)
                raise
        return wrapper
    return decorator

def count_calls(metrics_collector: MetricsCollector, metric_name: str,
               labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                metrics_collector.increment_counter(metric_name, labels)
                return result
            except Exception as e:
                error_labels = (labels or {}).copy()
                error_labels['error'] = 'true'
                metrics_collector.increment_counter(metric_name, error_labels)
                raise
        return wrapper
    return decorator
