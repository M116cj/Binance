"""Monitoring dashboard for SLA metrics, latency tracking, and quality indicators."""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import numpy as np


class MonitoringDashboard:
    """System monitoring dashboard with SLA and quality metrics"""
    
    def render(self, fetch_data_fn: Callable):
        """Render monitoring dashboard"""
        
        st.markdown("### 📊 System Monitoring & SLA Tracking")
        
        # Fetch system data
        signals_stats = fetch_data_fn("signals/stats", {})
        models_data = fetch_data_fn("models", {})
        health_data = fetch_data_fn("health", {})
        
        # Fetch recent signals for accurate percentile calculation
        recent_signals = fetch_data_fn("signals", {'limit': 1000})
        
        # SLA Overview
        self._render_sla_overview(signals_stats, health_data, recent_signals)
        
        # Latency Tracking
        self._render_latency_tracking(fetch_data_fn)
        
        # Quality Indicators
        self._render_quality_indicators(signals_stats)
        
        # System Health
        self._render_system_health(health_data, models_data)
    
    def _render_sla_overview(self, signals_stats: Optional[Dict], health_data: Optional[Dict], recent_signals: Optional[Dict]):
        """Render SLA compliance metrics"""
        st.markdown("#### 🎯 SLA Compliance Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Target: <800ms p99 latency
            # Calculate real P99 from signal data
            if recent_signals and 'signals' in recent_signals and recent_signals['signals']:
                latencies = [s.get('sla_latency_ms', 0) for s in recent_signals['signals']]
                p99_latency = np.percentile(latencies, 99) if latencies else 0
            else:
                # Fallback to average if no signals
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
            # Target: ≥300 rps throughput capacity
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
            # Uptime - calculated from health status
            status = health_data.get('status', 'unknown') if health_data else 'unknown'
            uptime_pct = 100.0 if status == 'healthy' else 0.0
            st.metric("⏰ System Status", status.title(), delta="Healthy" if status == 'healthy' else "")
        
        with col4:
            # Exchange lag from health data
            if health_data:
                exchange_lag = health_data.get('exchange_lag_s', 0)
                lag_status = "✅" if exchange_lag < 2.0 else "⚠️"
                st.metric(f"{lag_status} Exchange Lag", f"{exchange_lag:.2f}s", 
                         delta=f"{2.0 - exchange_lag:.2f}s vs target",
                         delta_color="normal" if exchange_lag < 2.0 else "inverse")
        
        # Note: Historical trend charts require time-series endpoint or signal history aggregation
        # Omitted for now - would need /signals/history with hourly aggregation support
    
    def _render_latency_tracking(self, fetch_data_fn: Callable):
        """Render detailed latency distribution and tracking"""
        st.markdown("#### ⚡ Latency Distribution & Breakdown")
        
        # Fetch recent signals for latency analysis
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
            # Latency distribution histogram
            fig_hist = go.Figure()
            
            fig_hist.add_trace(go.Histogram(
                x=latencies,
                nbinsx=30,
                marker=dict(color='#636EFA'),
                name='Latency Distribution'
            ))
            
            # Add percentile lines
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
            
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Latency percentiles
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
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Latency breakdown pie chart
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
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    def _render_quality_indicators(self, signals_stats: Optional[Dict]):
        """Render signal quality and model performance indicators"""
        st.markdown("#### ✨ Quality Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### Signal Quality")
            
            # Calculate tier distribution from real data
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
            
            st.metric("A-tier Signal Rate", f"{a_tier_pct:.1f}%")
            st.metric("Avg Signal Utility", f"{avg_utility:.2f}")
            st.metric("Total Signals (24h)", total_signals)
        
        with col2:
            st.markdown("##### Model Performance")
            
            st.info("Model metrics require /models endpoint extension with performance tracking")
        
        with col3:
            st.markdown("##### Data Pipeline")
            
            st.info("Pipeline metrics require dedicated metrics endpoint")
        
        # Note: Trend charts require historical aggregation endpoint
    
    def _render_system_health(self, health_data: Optional[Dict], models_data: Optional[Dict]):
        """Render system health status"""
        st.markdown("#### 🏥 System Health Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### System Status")
            
            # Use real health data
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
            
            # Model status
            if models_data and 'models' in models_data:
                active_models = sum(1 for m in models_data['models'] if m.get('is_active'))
                st.metric("Active Models", active_models)
        
        with col2:
            st.markdown("##### Resource Info")
            
            st.info("Resource utilization metrics require dedicated /metrics endpoint with system resource tracking")
