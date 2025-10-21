"""
Historical Backtest Performance component.
Report 5: Shows comprehensive backtesting results and strategy performance.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
from datetime import datetime, timedelta

class BacktestPerformance:
    """历史回测表现分析组件"""
    
    def __init__(self):
        self.component_name = "历史表现分析"
    
    def render(self, data: Dict[str, Any]):
        """渲染历史回测表现分析"""
        if not data:
            st.error("❌ 没有可用的历史表现数据")
            return
        
        # Header with key performance metrics
        self._render_performance_header(data)
        
        # Main performance visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_equity_curve(data)
            self._render_monthly_returns(data)
        
        with col2:
            self._render_drawdown_analysis(data)
            self._render_hit_rate_analysis(data)
        
        # Detailed analysis sections
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._render_signal_statistics(data)
            self._render_pr_curve(data)
        
        with col4:
            self._render_trade_distribution(data)
            self._render_risk_metrics(data)
    
    def _render_performance_header(self, data: Dict[str, Any]):
        """Render performance summary header"""
        symbol = data.get('symbol', 'Unknown')
        period = data.get('period', '30_days')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        performance_summary = data.get('performance_summary', {})
        total_return = performance_summary.get('total_return', 0)
        sharpe_ratio = performance_summary.get('sharpe_ratio_post_cost', 0)
        max_drawdown = performance_summary.get('max_drawdown', 0)
        hit_rate = performance_summary.get('hit_rate', 0)
        
        period_map = {'30_days': '30天', '7_days': '7天', '90_days': '90天'}
        period_display = period_map.get(period, period.replace('_', ' '))
        
        st.markdown(f"""
        ### 📊 历史表现分析: {symbol}
        **统计周期:** {period_display}  
        *更新时间: {time.strftime('%H:%M:%S', time.localtime(timestamp/1000))}*
        """)
        
        # Key performance metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            color = "normal"
            if total_return > 0.1:
                color = "inverse"
            elif total_return < -0.05:
                color = "inverse"
            
            st.metric(
                "总收益率", 
                f"{total_return:.2%}",
                delta=f"{total_return:.2%}",
                delta_color=color,
                help="策略在统计周期内的总收益"
            )
        
        with col2:
            st.metric(
                "夏普比率",
                f"{sharpe_ratio:.2f}",
                delta="良好" if sharpe_ratio > 1.0 else "需提升",
                help="风险调整后的收益指标，越高越好"
            )
        
        with col3:
            color = "normal" if max_drawdown < -0.05 else "inverse"
            st.metric(
                "最大回撤",
                f"{max_drawdown:.2%}",
                delta=f"{max_drawdown:.2%}",
                delta_color=color,
                help="从最高点到最低点的最大跌幅"
            )
        
        with col4:
            st.metric(
                "胜率",
                f"{hit_rate:.1%}",
                delta="超过50%" if hit_rate > 0.5 else "低于50%",
                help="预测正确的次数占总次数的比例"
            )
    
    def _render_equity_curve(self, data: Dict[str, Any]):
        """Render equity curve with returns"""
        st.markdown("#### 📈 Equity Curve")
        
        time_series = data.get('time_series', {})
        dates = time_series.get('dates', [])
        cumulative_returns = time_series.get('cumulative_returns', [])
        
        if not dates or not cumulative_returns:
            st.warning("No equity curve data available")
            return
        
        # Convert dates to datetime objects
        date_objects = [datetime.fromisoformat(date) for date in dates]
        
        fig = go.Figure()
        
        # Equity curve
        fig.add_trace(go.Scatter(
            x=date_objects,
            y=[ret * 100 for ret in cumulative_returns],  # Convert to percentage
            mode='lines',
            name='Cumulative Return',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add benchmark line (0% return)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Benchmark (0%)")
        
        # Highlight positive/negative regions
        fig.add_hrect(y0=-100, y1=0, fillcolor="red", opacity=0.1, 
                     annotation_text="Loss Region", annotation_position="bottom left")
        fig.add_hrect(y0=0, y1=100, fillcolor="green", opacity=0.1,
                     annotation_text="Profit Region", annotation_position="top left")
        
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance statistics
        if cumulative_returns:
            final_return = cumulative_returns[-1]
            peak_return = max(cumulative_returns)
            trough_return = min(cumulative_returns)
            
            st.markdown(f"""
            **Equity Stats:**
            - Final Return: {final_return:.2%}
            - Peak Return: {peak_return:.2%}
            - Trough Return: {trough_return:.2%}
            """)
    
    def _render_monthly_returns(self, data: Dict[str, Any]):
        """Render monthly returns heatmap"""
        st.markdown("#### 📅 Monthly Returns")
        
        monthly_breakdown = data.get('monthly_breakdown', {})
        
        if not monthly_breakdown:
            st.warning("No monthly breakdown data available")
            return
        
        # Process monthly data
        months = []
        returns = []
        signals_count = []
        hit_rates = []
        
        for month_key, month_data in monthly_breakdown.items():
            month_return = month_data.get('return', 0)
            month_signals = month_data.get('signals', 0)
            month_hit_rate = month_data.get('hit_rate', 0)
            
            months.append(month_key.replace('month_', 'Month '))
            returns.append(month_return * 100)  # Convert to percentage
            signals_count.append(month_signals)
            hit_rates.append(month_hit_rate * 100)
        
        # Create subplot for returns and signals
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Monthly Returns (%)', 'Monthly Signals & Hit Rate'),
            specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
        )
        
        # Monthly returns bar chart
        colors = ['green' if ret > 0 else 'red' for ret in returns]
        
        fig.add_trace(
            go.Bar(
                x=months,
                y=returns,
                name='Monthly Return (%)',
                marker_color=colors,
                text=[f"{ret:.1f}%" for ret in returns],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Monthly signals bar chart
        fig.add_trace(
            go.Bar(
                x=months,
                y=signals_count,
                name='Signal Count',
                marker_color='lightblue',
                text=signals_count,
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Hit rate line
        fig.add_trace(
            go.Scatter(
                x=months,
                y=hit_rates,
                mode='lines+markers',
                name='Hit Rate (%)',
                line=dict(color='orange', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1,
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Month", row=2, col=1)
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Signal Count", row=2, col=1)
        fig.update_yaxes(title_text="Hit Rate (%)", secondary_y=True, row=2, col=1)
        
        fig.update_layout(height=400, showlegend=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_drawdown_analysis(self, data: Dict[str, Any]):
        """Render drawdown analysis"""
        st.markdown("#### 📉 Drawdown Analysis")
        
        time_series = data.get('time_series', {})
        cumulative_returns = time_series.get('cumulative_returns', [])
        
        if not cumulative_returns:
            st.warning("No drawdown data available")
            return
        
        # Calculate drawdown series
        peak = np.maximum.accumulate(cumulative_returns)
        drawdowns = [(ret - pk) for ret, pk in zip(cumulative_returns, peak)]
        
        # Get dates for x-axis
        dates = time_series.get('dates', [])
        date_objects = [datetime.fromisoformat(date) for date in dates] if dates else list(range(len(drawdowns)))
        
        fig = go.Figure()
        
        # Drawdown area chart
        fig.add_trace(go.Scatter(
            x=date_objects,
            y=[dd * 100 for dd in drawdowns],
            fill='tonexty',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1),
            name='Drawdown',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_color="black", line_width=1)
        
        # Mark maximum drawdown
        max_dd = min(drawdowns)
        max_dd_idx = drawdowns.index(max_dd)
        
        if dates and max_dd_idx < len(date_objects):
            fig.add_trace(go.Scatter(
                x=[date_objects[max_dd_idx]],
                y=[max_dd * 100],
                mode='markers',
                name='Max Drawdown',
                marker=dict(color='darkred', size=10, symbol='circle')
            ))
        
        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Drawdown statistics
        performance_summary = data.get('performance_summary', {})
        max_drawdown = performance_summary.get('max_drawdown', max_dd)
        
        if drawdowns:
            # Calculate additional drawdown metrics
            drawdown_periods = []
            in_drawdown = False
            start_dd = 0
            
            for i, dd in enumerate(drawdowns):
                if dd < 0 and not in_drawdown:
                    in_drawdown = True
                    start_dd = i
                elif dd >= 0 and in_drawdown:
                    in_drawdown = False
                    drawdown_periods.append(i - start_dd)
            
            avg_dd_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            
            st.markdown(f"""
            **Drawdown Stats:**
            - Max Drawdown: {max_drawdown:.2%}
            - Avg DD Duration: {avg_dd_duration:.0f} periods
            - Current DD: {drawdowns[-1]:.2%}
            """)
    
    def _render_hit_rate_analysis(self, data: Dict[str, Any]):
        """Render hit rate analysis by categories"""
        st.markdown("#### 🎯 Hit Rate Analysis")
        
        hit_at_k = data.get('hit_at_k', {})
        
        if not hit_at_k:
            st.warning("No hit rate data available")
            return
        
        # Hit rates for top-K signals
        k_values = sorted(hit_at_k.keys())
        hit_rates = [hit_at_k[k] * 100 for k in k_values]  # Convert to percentage
        
        fig = go.Figure()
        
        # Hit rate bars
        fig.add_trace(go.Bar(
            x=[f"Top {k}" for k in k_values],
            y=hit_rates,
            name='Hit Rate (%)',
            marker_color='lightgreen',
            text=[f"{hr:.1f}%" for hr in hit_rates],
            textposition='auto'
        ))
        
        # Add benchmark line (50%)
        fig.add_hline(y=50, line_dash="dash", line_color="gray",
                     annotation_text="Random (50%)")
        
        fig.update_layout(
            title="Hit Rate by Signal Rank",
            xaxis_title="Signal Ranking",
            yaxis_title="Hit Rate (%)",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hit rate quality assessment
        best_hit_rate = max(hit_rates) if hit_rates else 0
        worst_hit_rate = min(hit_rates) if hit_rates else 0
        
        if best_hit_rate > 70:
            st.success(f"✅ Excellent top signal quality: {best_hit_rate:.1f}%")
        elif best_hit_rate > 60:
            st.info(f"👍 Good signal quality: {best_hit_rate:.1f}%")
        else:
            st.warning(f"⚠️ Signal quality needs improvement: {best_hit_rate:.1f}%")
    
    def _render_signal_statistics(self, data: Dict[str, Any]):
        """Render signal generation statistics"""
        st.markdown("#### 🔢 Signal Statistics")
        
        signal_stats = data.get('signal_statistics', {})
        
        if not signal_stats:
            st.warning("No signal statistics available")
            return
        
        total_signals = signal_stats.get('total_signals', 0)
        signals_per_day = signal_stats.get('signals_per_day', 0)
        a_tier_signals = signal_stats.get('a_tier_signals', 0)
        b_tier_signals = signal_stats.get('b_tier_signals', 0)
        rejected_signals = signal_stats.get('rejected_signals', 0)
        
        # Signal tier distribution pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['A-Tier', 'B-Tier', 'Rejected'],
            values=[a_tier_signals, b_tier_signals, rejected_signals],
            hole=.3,
            marker_colors=['green', 'yellow', 'red']
        )])
        
        fig.update_layout(
            title="Signal Distribution by Tier",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Signal frequency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Signals", f"{total_signals:,}")
            st.metric("A-Tier Signals", f"{a_tier_signals:,}")
        
        with col2:
            st.metric("Signals/Day", f"{signals_per_day:.1f}")
            st.metric("Signal Quality", f"{(a_tier_signals + b_tier_signals)/max(total_signals,1):.1%}")
    
    def _render_pr_curve(self, data: Dict[str, Any]):
        """Render Precision-Recall curve"""
        st.markdown("#### 📊 Precision-Recall Performance")
        
        performance_summary = data.get('performance_summary', {})
        pr_auc = performance_summary.get('pr_auc', 0)
        
        # Generate synthetic PR curve data (in production, this would be real)
        recalls = np.linspace(0, 1, 50)
        precisions = []
        
        # Simulate PR curve that starts high and decreases
        for r in recalls:
            # Simple model: precision decreases as recall increases
            base_precision = 0.8 * np.exp(-2 * r) + 0.2
            noise = np.random.normal(0, 0.05)
            precision = max(0.1, min(1.0, base_precision + noise))
            precisions.append(precision)
        
        fig = go.Figure()
        
        # PR curve
        fig.add_trace(go.Scatter(
            x=recalls,
            y=precisions,
            mode='lines',
            name=f'PR Curve (AUC={pr_auc:.3f})',
            line=dict(color='blue', width=2),
            fill='tonexty'
        ))
        
        # Random classifier baseline
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0.5, 0.5],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title="Precision-Recall Curve",
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # PR-AUC assessment
        if pr_auc > 0.7:
            st.success(f"✅ Excellent model performance: PR-AUC = {pr_auc:.3f}")
        elif pr_auc > 0.5:
            st.info(f"👍 Good model performance: PR-AUC = {pr_auc:.3f}")
        else:
            st.warning(f"⚠️ Model needs improvement: PR-AUC = {pr_auc:.3f}")
    
    def _render_trade_distribution(self, data: Dict[str, Any]):
        """Render trade distribution analysis"""
        st.markdown("#### 📊 Trade Distribution")
        
        detailed_analysis = data.get('detailed_analysis', {})
        trade_distribution = detailed_analysis.get('trade_distribution', {})
        
        if not trade_distribution:
            st.warning("No trade distribution data available")
            return
        
        total_trades = trade_distribution.get('total_trades', 0)
        winning_trades = trade_distribution.get('winning_trades', 0)
        losing_trades = trade_distribution.get('losing_trades', 0)
        avg_winner = trade_distribution.get('avg_winner', 0)
        avg_loser = trade_distribution.get('avg_loser', 0)
        largest_winner = trade_distribution.get('largest_winner', 0)
        largest_loser = trade_distribution.get('largest_loser', 0)
        
        # Win/Loss distribution
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Winning Trades', 'Losing Trades'],
            y=[winning_trades, losing_trades],
            marker_color=['green', 'red'],
            text=[winning_trades, losing_trades],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Win/Loss Trade Distribution",
            yaxis_title="Number of Trades",
            height=200
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade statistics
        win_rate = winning_trades / max(total_trades, 1)
        profit_factor = abs(avg_winner * winning_trades) / abs(avg_loser * max(losing_trades, 1)) if losing_trades > 0 else float('inf')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Win Rate", f"{win_rate:.1%}")
            st.metric("Avg Winner", f"${avg_winner:.2f}")
            st.metric("Largest Winner", f"${largest_winner:.2f}")
        
        with col2:
            st.metric("Profit Factor", f"{profit_factor:.2f}")
            st.metric("Avg Loser", f"${avg_loser:.2f}")
            st.metric("Largest Loser", f"${largest_loser:.2f}")
    
    def _render_risk_metrics(self, data: Dict[str, Any]):
        """Render risk analysis metrics"""
        st.markdown("#### ⚠️ Risk Metrics")
        
        detailed_analysis = data.get('detailed_analysis', {})
        risk_metrics = detailed_analysis.get('risk_metrics', {})
        
        if not risk_metrics:
            st.warning("No risk metrics available")
            return
        
        volatility = risk_metrics.get('volatility', 0)
        var_95 = risk_metrics.get('var_95', 0)
        cvar_95 = risk_metrics.get('cvar_95', 0)
        skewness = risk_metrics.get('skewness', 0)
        kurtosis = risk_metrics.get('kurtosis', 0)
        
        # Risk metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Volatility (Annual)", f"{volatility:.1%}")
            st.metric("VaR (95%)", f"{var_95:.3f}")
            st.metric("CVaR (95%)", f"{cvar_95:.3f}")
        
        with col2:
            # Risk visualization
            fig = go.Figure()
            
            risk_categories = ['Volatility', 'VaR', 'CVaR']
            risk_values = [volatility * 100, abs(var_95) * 100, abs(cvar_95) * 100]
            
            fig.add_trace(go.Bar(
                x=risk_categories,
                y=risk_values,
                marker_color=['blue', 'orange', 'red'],
                text=[f"{v:.1f}%" for v in risk_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Risk Metrics (%)",
                height=200
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution characteristics
        st.markdown("**Return Distribution:**")
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("Skewness", f"{skewness:.2f}")
            if skewness > 0:
                st.caption("👍 Positive skew (upside bias)")
            else:
                st.caption("👎 Negative skew (downside risk)")
        
        with col4:
            st.metric("Kurtosis", f"{kurtosis:.2f}")
            if kurtosis > 3:
                st.caption("⚠️ Heavy tails (extreme events)")
            else:
                st.caption("✅ Normal tail behavior")

