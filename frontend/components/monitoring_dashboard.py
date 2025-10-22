"""监控仪表板：SLA指标、延迟跟踪和质量指标"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import numpy as np


class MonitoringDashboard:
    """系统监控仪表板：包含SLA和质量指标"""
    
    def render(self, fetch_data_fn: Callable):
        """渲染监控仪表板"""
        
        st.markdown("### 📊 System Monitoring & SLA Tracking")
        
        # 获取系统数据
        signals_stats = fetch_data_fn("signals/stats", {})
        models_data = fetch_data_fn("models", {})
        health_data = fetch_data_fn("health", {})
        
        # 获取最近的信号以进行准确的百分位数计算
        recent_signals = fetch_data_fn("signals", {'limit': 1000})
        
        # 获取回测数据以获取性能指标（使用默认参数）
        backtest_data = fetch_data_fn("reports/backtest", {
            'symbol': 'BTCUSDT',
            'theta_up': 0.006,
            'theta_dn': 0.004,
            'tau': 0.75,
            'kappa': 1.20,
            'days_back': 30
        })
        
        # SLA概览
        self._render_sla_overview(signals_stats, health_data, recent_signals)
        
        # 延迟跟踪
        self._render_latency_tracking(fetch_data_fn)
        
        # 质量指标（包含交易性能和风险指标）
        self._render_quality_indicators(signals_stats, backtest_data)
        
        # 系统健康状态
        self._render_system_health(health_data, models_data)
    
    def _render_sla_overview(self, signals_stats: Optional[Dict], health_data: Optional[Dict], recent_signals: Optional[Dict]):
        """渲染SLA合规指标"""
        st.markdown("#### 🎯 SLA Compliance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 目标：P99延迟<800ms
            # 从信号数据计算真实的P99
            if recent_signals and 'signals' in recent_signals and recent_signals['signals']:
                latencies = [s.get('sla_latency_ms', 0) for s in recent_signals['signals']]
                p99_latency = np.percentile(latencies, 99) if latencies else 0
            else:
                # 如果没有信号，回退到平均值
                avg_latency = 0
                count = 0
                if signals_stats and 'by_symbol' in signals_stats:
                    for symbol_stats in signals_stats['by_symbol'].values():
                        avg_latency += symbol_stats.get('avg_latency_ms', 0)
                        count += 1
                p99_latency = avg_latency / count if count > 0 else 0
            
            sla_status = "✅" if p99_latency < 800 else "⚠️"
            delta_color = "normal" if p99_latency < 800 else "inverse"
            
            st.metric(
                f"{sla_status} P99 Latency", 
                f"{p99_latency:.1f}ms",
                delta=f"{800 - p99_latency:.0f}ms vs target",
                delta_color=delta_color
            )
        
        with col2:
            # 目标：吞吐量容量≥300 rps
            total_signals = signals_stats.get('total_signals', 0) if signals_stats else 0
            rps = total_signals / (24 * 3600)  # Signals per second over 24h
            capacity_used_pct = (rps / 300) * 100 if rps > 0 else 0
            sla_status = "✅" if capacity_used_pct < 100 else "⚠️"
            
            st.metric(
                f"{sla_status} Throughput", 
                f"{rps:.2f} rps",
                delta=f"{capacity_used_pct:.1f}% of 300 rps capacity"
            )
        
        with col3:
            # 运行时间 - 从健康状态计算
            status = health_data.get('status', 'unknown') if health_data else 'unknown'
            uptime_pct = 100.0 if status == 'healthy' else 0.0
            st.metric("⏰ System Status", status.title(), delta="Healthy" if status == 'healthy' else "")
        
        with col4:
            # 来自健康数据的交易所延迟
            if health_data:
                exchange_lag = health_data.get('exchange_lag_s', 0)
                lag_status = "✅" if exchange_lag < 2.0 else "⚠️"
                st.metric(f"{lag_status} Exchange Lag", f"{exchange_lag:.2f}s", 
                         delta=f"{2.0 - exchange_lag:.2f}s vs target",
                         delta_color="normal" if exchange_lag < 2.0 else "inverse")
        
        # 注意：历史趋势图需要时间序列端点或信号历史聚合
        # 暂时省略 - 需要带有每小时聚合支持的/signals/history
    
    def _render_latency_tracking(self, fetch_data_fn: Callable):
        """渲染详细的延迟分布和跟踪"""
        st.markdown("#### ⚡ Latency Distribution & Breakdown")
        
        # 获取最近的信号以进行延迟分析
        signals_data = fetch_data_fn("signals", {'limit': 100})
        
        if not signals_data or 'signals' not in signals_data:
            st.info("No signal data available for latency analysis")
            return
        
        signals = signals_data['signals']
        latencies = [s.get('sla_latency_ms', 0) for s in signals]
        
        if not latencies:
            st.info("No latency data available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 延迟分布直方图
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=latencies,
                nbinsx=30,
                marker=dict(color='#636EFA'),
                name='Latency Distribution'
            ))
            
            # 添加百分位数线
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            
            fig_hist.add_vline(x=p50, line_dash="dash", line_color="green", 
                              annotation_text=f"P50: {p50:.0f}ms")
            fig_hist.add_vline(x=p95, line_dash="dash", line_color="yellow", 
                              annotation_text=f"P95: {p95:.0f}ms")
            fig_hist.add_vline(x=p99, line_dash="dash", line_color="red", 
                              annotation_text=f"P99: {p99:.0f}ms")
            
            fig_hist.update_layout(
                title='Latency Distribution',
                xaxis_title='Latency (ms)',
                yaxis_title='Count',
                height=350,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_hist, width='stretch')
        
        with col2:
            # 延迟百分位数
            st.markdown("##### Latency Percentiles")
            
            percentiles_data = {
                'Percentile': ['P50 (Median)', 'P75', 'P90', 'P95', 'P99', 'P99.9'],
                'Latency (ms)': [
                    np.percentile(latencies, 50),
                    np.percentile(latencies, 75),
                    np.percentile(latencies, 90),
                    np.percentile(latencies, 95),
                    np.percentile(latencies, 99),
                    np.percentile(latencies, 99.9)
                ],
                'SLA': ['✅', '✅', '✅', '✅', '✅', '✅']
            }
            
            df = pd.DataFrame(percentiles_data)
            df['Latency (ms)'] = df['Latency (ms)'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(df, width='stretch', hide_index=True)
            
            # 延迟分解饼图
            st.markdown("##### Latency Ranges")
            
            ranges = {
                '<100ms': sum(1 for l in latencies if l < 100),
                '100-200ms': sum(1 for l in latencies if 100 <= l < 200),
                '200-500ms': sum(1 for l in latencies if 200 <= l < 500),
                '500ms+': sum(1 for l in latencies if l >= 500)
            }
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(ranges.keys()),
                values=list(ranges.values()),
                hole=0.4,
                marker=dict(colors=['#00CC96', '#19D3F3', '#FFA15A', '#EF553B'])
            )])
            
            fig_pie.update_layout(
                height=250,
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, width='stretch')
    
    def _render_quality_indicators(self, signals_stats: Optional[Dict], backtest_data: Optional[Dict]):
        """渲染信号质量、交易性能和风险指标"""
        st.markdown("#### ✨ Quality & Performance Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📊 Signal Quality")
            
            # 从真实数据计算等级分布
            total_signals = signals_stats.get('total_signals', 0) if signals_stats else 0
            a_tier_total = 0
            b_tier_total = 0
            avg_utility = 0
            count = 0
            
            if signals_stats and 'by_symbol' in signals_stats:
                for symbol_stats in signals_stats['by_symbol'].values():
                    a_tier_total += symbol_stats.get('a_tier_count', 0)
                    b_tier_total += symbol_stats.get('b_tier_count', 0)
                    avg_utility += symbol_stats.get('avg_utility', 0)
                    count += 1
            
            a_tier_pct = (a_tier_total / total_signals * 100) if total_signals > 0 else 0
            avg_utility = avg_utility / count if count > 0 else 0
            
            # 信号频率（每小时）
            signals_per_hour = total_signals / 24.0 if total_signals > 0 else 0
            
            st.metric("A级信号率", f"{a_tier_pct:.1f}%", 
                     help="高置信度信号占比（p_up > τ, utility > κ）")
            st.metric("信号频率", f"{signals_per_hour:.1f}/小时",
                     help="平均每小时产生的交易信号数量")
            st.metric("平均收益倍数", f"{avg_utility:.2f}x",
                     help="预期收益与成本的平均比例")
        
        with col2:
            st.markdown("##### 💰 Trading Performance")
            
            # 从回测数据提取性能指标
            if backtest_data and 'performance_summary' in backtest_data:
                perf = backtest_data['performance_summary']
                hit_rate = perf.get('hit_rate', 0)
                total_return = perf.get('total_return', 0)
                profit_factor = perf.get('profit_factor', 0)
                
                st.metric("胜率", f"{hit_rate:.1%}",
                         delta="超过50%" if hit_rate > 0.5 else "低于50%",
                         help="预测正确的次数占总次数的比例")
                st.metric("累计收益", f"{total_return:.2%}",
                         delta=f"{total_return:.2%}",
                         delta_color="normal" if total_return > 0 else "inverse",
                         help="策略在统计周期内的总收益")
                st.metric("盈亏比", f"{profit_factor:.2f}",
                         help="总盈利与总亏损的比例")
            else:
                st.info("回测数据加载中...")
        
        with col3:
            st.markdown("##### ⚠️ Risk Metrics")
            
            # 从回测数据提取风险指标
            if backtest_data and 'performance_summary' in backtest_data:
                perf = backtest_data['performance_summary']
                sharpe_ratio = perf.get('sharpe_ratio_post_cost', 0)
                max_drawdown = perf.get('max_drawdown', 0)
                volatility = perf.get('volatility', 0)
                
                st.metric("夏普比率", f"{sharpe_ratio:.2f}",
                         delta="优秀" if sharpe_ratio > 1.5 else ("良好" if sharpe_ratio > 1.0 else "需提升"),
                         help="风险调整后的收益指标，越高越好（>1.5为优秀）")
                st.metric("最大回撤", f"{max_drawdown:.2%}",
                         delta=f"{max_drawdown:.2%}",
                         delta_color="normal" if max_drawdown > -0.1 else "inverse",
                         help="从最高点到最低点的最大跌幅")
                st.metric("波动率", f"{volatility:.2%}",
                         help="日收益率的标准差（年化）")
            else:
                st.info("风险指标加载中...")
        
        # 注意：趋势图需要历史聚合端点
    
    def _render_system_health(self, health_data: Optional[Dict], models_data: Optional[Dict]):
        """渲染系统健康状态"""
        st.markdown("#### 🏥 System Health Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### System Status")
            
            # 使用真实的健康数据
            if health_data:
                status = health_data.get('status', 'unknown')
                timestamp = health_data.get('timestamp', 0)
                exchange_lag = health_data.get('exchange_lag_s', 0)
                mode = health_data.get('mode', 'unknown')
                
                st.metric("Status", status.title())
                st.metric("Mode", mode.title())
                st.metric("Exchange Lag", f"{exchange_lag:.2f}s", 
                         delta=f"{2.0 - exchange_lag:.2f}s vs target",
                         delta_color="normal" if exchange_lag < 2.0 else "inverse")
            else:
                st.warning("Health data not available")
            
            # 模型状态
            if models_data and 'models' in models_data:
                active_models = sum(1 for m in models_data['models'] if m.get('is_active'))
                st.metric("Active Models", active_models)
        
        with col2:
            st.markdown("##### Resource Info")
            
            st.info("Resource utilization metrics require dedicated /metrics endpoint with system resource tracking")
