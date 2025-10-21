"""
数据质量监控模块
实时检测数据漂移、异常值和质量问题
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityAlert:
    """质量告警"""
    timestamp: datetime
    metric_name: str
    alert_type: str  # 'drift', 'outlier', 'missing', 'staleness'
    severity: str  # 'low', 'medium', 'high', 'critical'
    value: float
    threshold: float
    message: str
    metadata: Dict[str, Any]


class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self, 
                 window_size: int = 1000,
                 drift_threshold: float = 0.05,
                 outlier_std: float = 3.0,
                 missing_threshold: float = 0.01,
                 staleness_seconds: int = 60):
        """
        初始化数据质量监控器
        
        Args:
            window_size: 滑动窗口大小
            drift_threshold: 漂移检测阈值（KS统计量）
            outlier_std: 异常值检测标准差倍数
            missing_threshold: 缺失值比率阈值
            staleness_seconds: 数据过期时间（秒）
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.outlier_std = outlier_std
        self.missing_threshold = missing_threshold
        self.staleness_seconds = staleness_seconds
        
        # 历史数据窗口
        self.historical_windows: Dict[str, deque] = {}
        
        # 基准分布（用于漂移检测）
        self.baseline_distributions: Dict[str, np.ndarray] = {}
        
        # 告警历史
        self.alerts: deque = deque(maxlen=1000)
        
        # 统计信息
        self.stats: Dict[str, Dict[str, float]] = {}
        
        # 最后更新时间
        self.last_update: Dict[str, datetime] = {}
    
    def update_baseline(self, metric_name: str, data: np.ndarray):
        """
        更新基准分布
        
        Args:
            metric_name: 指标名称
            data: 数据数组
        """
        self.baseline_distributions[metric_name] = data.copy()
        
        # 初始化历史窗口
        if metric_name not in self.historical_windows:
            self.historical_windows[metric_name] = deque(maxlen=self.window_size)
        
        logger.info(f"Updated baseline for {metric_name}: {len(data)} samples")
    
    def check_data(self, 
                   metric_name: str, 
                   value: float,
                   timestamp: Optional[datetime] = None) -> List[QualityAlert]:
        """
        检查单个数据点的质量
        
        Args:
            metric_name: 指标名称
            value: 数据值
            timestamp: 时间戳
            
        Returns:
            告警列表
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        alerts = []
        
        # 初始化窗口
        if metric_name not in self.historical_windows:
            self.historical_windows[metric_name] = deque(maxlen=self.window_size)
        
        # 缺失值检测
        if pd.isna(value) or value is None:
            alerts.append(QualityAlert(
                timestamp=timestamp,
                metric_name=metric_name,
                alert_type='missing',
                severity='high',
                value=np.nan,
                threshold=0.0,
                message=f"{metric_name} 数据缺失",
                metadata={}
            ))
            return alerts
        
        # 更新历史窗口
        self.historical_windows[metric_name].append((timestamp, value))
        self.last_update[metric_name] = timestamp
        
        # 获取当前窗口数据
        window_values = np.array([v for _, v in self.historical_windows[metric_name]])
        
        # 异常值检测（使用Z-score）
        if len(window_values) >= 30:  # 至少30个样本
            mean = np.mean(window_values)
            std = np.std(window_values)
            
            if std > 0:
                z_score = abs((value - mean) / std)
                
                if z_score > self.outlier_std:
                    alerts.append(QualityAlert(
                        timestamp=timestamp,
                        metric_name=metric_name,
                        alert_type='outlier',
                        severity='medium' if z_score < self.outlier_std * 1.5 else 'high',
                        value=value,
                        threshold=mean + self.outlier_std * std,
                        message=f"{metric_name} 检测到异常值 (Z-score={z_score:.2f})",
                        metadata={'z_score': z_score, 'mean': mean, 'std': std}
                    ))
        
        # 数据漂移检测（使用Kolmogorov-Smirnov检验）
        if metric_name in self.baseline_distributions and len(window_values) >= 100:
            baseline = self.baseline_distributions[metric_name]
            
            # KS检验
            ks_stat, p_value = stats.ks_2samp(baseline, window_values)
            
            if ks_stat > self.drift_threshold:
                alerts.append(QualityAlert(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    alert_type='drift',
                    severity='medium',
                    value=ks_stat,
                    threshold=self.drift_threshold,
                    message=f"{metric_name} 检测到数据漂移 (KS={ks_stat:.4f})",
                    metadata={'ks_statistic': ks_stat, 'p_value': p_value}
                ))
        
        # 保存告警
        for alert in alerts:
            self.alerts.append(alert)
        
        return alerts
    
    def check_staleness(self, metric_name: str) -> Optional[QualityAlert]:
        """
        检查数据新鲜度
        
        Args:
            metric_name: 指标名称
            
        Returns:
            新鲜度告警（如果数据过期）
        """
        if metric_name not in self.last_update:
            return None
        
        last_time = self.last_update[metric_name]
        elapsed = (datetime.now() - last_time).total_seconds()
        
        if elapsed > self.staleness_seconds:
            alert = QualityAlert(
                timestamp=datetime.now(),
                metric_name=metric_name,
                alert_type='staleness',
                severity='critical' if elapsed > self.staleness_seconds * 2 else 'high',
                value=elapsed,
                threshold=self.staleness_seconds,
                message=f"{metric_name} 数据过期 (已{elapsed:.0f}秒未更新)",
                metadata={'elapsed_seconds': elapsed}
            )
            self.alerts.append(alert)
            return alert
        
        return None
    
    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """
        获取指标的统计信息
        
        Args:
            metric_name: 指标名称
            
        Returns:
            统计信息字典
        """
        if metric_name not in self.historical_windows or len(self.historical_windows[metric_name]) == 0:
            return {}
        
        values = np.array([v for _, v in self.historical_windows[metric_name]])
        
        stats_dict = {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75))
        }
        
        # 计算缺失值比率
        if metric_name in self.historical_windows:
            window = list(self.historical_windows[metric_name])
            missing_count = sum(1 for _, v in window if pd.isna(v))
            stats_dict['missing_ratio'] = missing_count / len(window) if len(window) > 0 else 0.0
        
        self.stats[metric_name] = stats_dict
        return stats_dict
    
    def get_recent_alerts(self, 
                          minutes: int = 60,
                          severity: Optional[str] = None,
                          alert_type: Optional[str] = None) -> List[QualityAlert]:
        """
        获取最近的告警
        
        Args:
            minutes: 时间范围（分钟）
            severity: 过滤严重性
            alert_type: 过滤告警类型
            
        Returns:
            告警列表
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        filtered_alerts = [
            alert for alert in self.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        if severity:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity]
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]
        
        return sorted(filtered_alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, int]:
        """
        获取告警摘要
        
        Returns:
            按类型统计的告警数量
        """
        recent_alerts = self.get_recent_alerts(minutes=60)
        
        summary = {
            'total': len(recent_alerts),
            'by_severity': {},
            'by_type': {}
        }
        
        for alert in recent_alerts:
            summary['by_severity'][alert.severity] = summary['by_severity'].get(alert.severity, 0) + 1
            summary['by_type'][alert.alert_type] = summary['by_type'].get(alert.alert_type, 0) + 1
        
        return summary
    
    def reset_metric(self, metric_name: str):
        """
        重置指标的监控状态
        
        Args:
            metric_name: 指标名称
        """
        if metric_name in self.historical_windows:
            self.historical_windows[metric_name].clear()
        
        if metric_name in self.baseline_distributions:
            del self.baseline_distributions[metric_name]
        
        if metric_name in self.last_update:
            del self.last_update[metric_name]
        
        if metric_name in self.stats:
            del self.stats[metric_name]
        
        logger.info(f"Reset monitoring for {metric_name}")


# 全局数据质量监控器实例
quality_monitor = DataQualityMonitor(
    window_size=1000,
    drift_threshold=0.05,
    outlier_std=3.0,
    missing_threshold=0.01,
    staleness_seconds=60
)
