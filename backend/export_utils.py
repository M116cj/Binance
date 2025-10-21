"""Signal export utilities for Protobuf and JSONL formats."""
import json
from typing import List, Optional
from datetime import datetime
from backend.proto import signal_pb2
from backend.database.models import Signal


def signal_to_protobuf(signal: Signal) -> signal_pb2.TradingSignal:
    """Convert database Signal to Protobuf message."""
    pb_signal = signal_pb2.TradingSignal()
    
    # Core identification
    pb_signal.symbol = signal.symbol
    pb_signal.timestamp_ms = int(signal.created_at.timestamp() * 1000)
    pb_signal.signal_id = signal.signal_id
    
    # Price (Note: Signal model doesn't have price field, use 0 for now)
    pb_signal.current_price = 0.0
    pb_signal.decision = signal.decision
    pb_signal.tier = signal.tier
    pb_signal.utility = signal.net_utility or 0.0
    pb_signal.expected_return = signal.expected_return or 0.0
    pb_signal.estimated_cost = signal.estimated_cost or 0.0
    
    # Model metadata
    pb_signal.model_version = signal.model_version or "unknown"
    pb_signal.feature_version = signal.feature_version or "unknown"
    pb_signal.sla_latency_ms = signal.sla_latency_ms or 0.0
    
    # Probabilities (using individual fields)
    # For the primary horizon (horizon_min)
    if signal.horizon_min == 5:
        pb_signal.probabilities.prob_5m.value = signal.p_up or 0.0
        pb_signal.probabilities.prob_5m.ci_low = signal.p_up_ci_low or 0.0
        pb_signal.probabilities.prob_5m.ci_high = signal.p_up_ci_high or 0.0
    elif signal.horizon_min == 10:
        pb_signal.probabilities.prob_10m.value = signal.p_up or 0.0
        pb_signal.probabilities.prob_10m.ci_low = signal.p_up_ci_low or 0.0
        pb_signal.probabilities.prob_10m.ci_high = signal.p_up_ci_high or 0.0
    elif signal.horizon_min == 30:
        pb_signal.probabilities.prob_30m.value = signal.p_up or 0.0
        pb_signal.probabilities.prob_30m.ci_low = signal.p_up_ci_low or 0.0
        pb_signal.probabilities.prob_30m.ci_high = signal.p_up_ci_high or 0.0
    
    # Thresholds
    pb_signal.thresholds.tau = signal.tau_threshold or 0.0
    pb_signal.thresholds.kappa = signal.kappa_threshold or 0.0
    pb_signal.thresholds.theta_up = signal.theta_up or 0.0
    pb_signal.thresholds.theta_dn = signal.theta_dn or 0.0
    
    # Top features
    if signal.features_top5:
        for feat_name, feat_value in signal.features_top5.items():
            pb_signal.top_features[feat_name] = feat_value
    
    # Quality flags
    if signal.quality_flags:
        pb_signal.quality_flags.extend(signal.quality_flags)
    
    return pb_signal


def signals_to_protobuf_batch(signals: List[Signal], export_id: Optional[str] = None) -> signal_pb2.SignalBatch:
    """Convert list of Signals to Protobuf batch message."""
    batch = signal_pb2.SignalBatch()
    batch.export_timestamp_ms = int(datetime.utcnow().timestamp() * 1000)
    batch.export_id = export_id or f"export_{batch.export_timestamp_ms}"
    batch.total_count = len(signals)
    
    for signal in signals:
        batch.signals.append(signal_to_protobuf(signal))
    
    return batch


def signal_to_jsonl(signal: Signal) -> str:
    """Convert database Signal to JSONL (single line JSON)."""
    data = {
        'signal_id': signal.signal_id,
        'symbol': signal.symbol,
        'timestamp_ms': int(signal.created_at.timestamp() * 1000),
        'exchange_time_ms': int(signal.exchange_time.timestamp() * 1000) if signal.exchange_time else None,
        'horizon_min': signal.horizon_min,
        'decision': signal.decision,
        'tier': signal.tier,
        'p_up': signal.p_up,
        'p_up_ci_low': signal.p_up_ci_low,
        'p_up_ci_high': signal.p_up_ci_high,
        'net_utility': signal.net_utility,
        'expected_return': signal.expected_return,
        'estimated_cost': signal.estimated_cost,
        'model_version': signal.model_version,
        'feature_version': signal.feature_version,
        'sla_latency_ms': signal.sla_latency_ms,
        'thresholds': {
            'tau': signal.tau_threshold,
            'kappa': signal.kappa_threshold,
            'theta_up': signal.theta_up,
            'theta_dn': signal.theta_dn
        },
        'top_features': signal.features_top5,
        'quality_flags': signal.quality_flags,
        'regime': signal.regime,
        'volatility': signal.volatility,
    }
    # Return single line JSON
    return json.dumps(data, separators=(',', ':'))


def signals_to_jsonl(signals: List[Signal]) -> str:
    """Convert list of Signals to JSONL format (newline-delimited JSON)."""
    lines = [signal_to_jsonl(signal) for signal in signals]
    return '\n'.join(lines)
