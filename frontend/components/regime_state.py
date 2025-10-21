"""
Market Regime & Liquidity State component.
Report 2: Displays current market conditions and regime classification.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time

class RegimeState:
    """市场状态和流动性分析组件"""
    
    def __init__(self):
        self.component_name = "市场状态分析"
    
    def render(self, data: Dict[str, Any]):
        """渲染市场状态分析"""
        if not data:
            st.error("❌ 没有可用的市场状态数据")
            return
        
        # Header with current regime
        self._render_regime_header(data)
        
        # Main visualization panels
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_regime_heatmap(data)
            self._render_liquidity_metrics(data)
        
        with col2:
            self._render_market_pressure_gauges(data)
            self._render_depth_profile(data)
        
        # Threshold adjustments and recommendations
        st.markdown("---")
        self._render_adaptive_thresholds(data)
    
    def _render_regime_header(self, data: Dict[str, Any]):
        """Render regime classification header"""
        regime = data.get('regime', 'unknown_regime')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        # Parse regime components
        regime_components = data.get('regime_components', {})
        vol_info = regime_components.get('volatility', {})
        depth_info = regime_components.get('depth', {})
        funding_info = regime_components.get('funding', {})
        
        # Display current regime with icons
        vol_icon = self._get_volatility_icon(vol_info.get('bucket', 'medium'))
        depth_icon = self._get_depth_icon(depth_info.get('bucket', 'medium'))
        funding_icon = self._get_funding_icon(funding_info.get('bucket', 'neutral'))
        
        # 波动性等级映射
        vol_map = {'low': '低', 'medium': '中', 'high': '高'}
        depth_map = {'thin': '薄', 'medium': '中', 'thick': '厚'}
        funding_map = {'negative': '负费率', 'neutral': '中性', 'positive': '正费率'}
        
        vol_bucket = vol_info.get('bucket', 'medium')
        depth_bucket = depth_info.get('bucket', 'medium')
        funding_bucket = funding_info.get('bucket', 'neutral')
        
        st.markdown(f"""
        ### 🌊 市场状态分析
        **当前状态:** `{regime}`
        
        {vol_icon} **波动性:** {vol_map.get(vol_bucket, vol_bucket)} ({vol_info.get('value', 0):.2f})  
        {depth_icon} **市场深度:** {depth_map.get(depth_bucket, depth_bucket)} ({depth_info.get('value', 0):.2f})  
        {funding_icon} **资金费率:** {funding_map.get(funding_bucket, funding_bucket)} ({funding_info.get('value', 0):.4f})
        
        *更新时间: {time.strftime('%H:%M:%S', time.localtime(timestamp/1000))}*
        """)
    
    def _get_volatility_icon(self, bucket: str) -> str:
        """Get icon for volatility regime"""
        icons = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
        return icons.get(bucket, '⚪')
    
    def _get_depth_icon(self, bucket: str) -> str:
        """Get icon for depth regime"""
        icons = {'thin': '🔴', 'medium': '🟡', 'thick': '🟢'}
        return icons.get(bucket, '⚪')
    
    def _get_funding_icon(self, bucket: str) -> str:
        """Get icon for funding regime"""
        icons = {'negative': '🟢', 'neutral': '🟡', 'positive': '🔴'}
        return icons.get(bucket, '⚪')
    
    def _render_regime_heatmap(self, data: Dict[str, Any]):
        """Render 3D regime classification heatmap"""
        st.markdown("#### 🎯 Regime Classification Matrix")
        
        regime_components = data.get('regime_components', {})
        
        # Create regime stability matrix
        vol_bucket = regime_components.get('volatility', {}).get('bucket', 'medium')
        depth_bucket = regime_components.get('depth', {}).get('bucket', 'medium')
        funding_bucket = regime_components.get('funding', {}).get('bucket', 'neutral')
        
        # Create heatmap data
        volatility_levels = ['low', 'medium', 'high']
        depth_levels = ['thin', 'medium', 'thick']
        
        # Current position in the matrix
        vol_idx = volatility_levels.index(vol_bucket) if vol_bucket in volatility_levels else 1
        depth_idx = depth_levels.index(depth_bucket) if depth_bucket in depth_levels else 1
        
        # Create intensity matrix (higher = more favorable for signals)
        intensity_matrix = np.array([
            [0.3, 0.5, 0.8],  # low vol: thin->medium->thick depth
            [0.4, 0.7, 0.9],  # medium vol
            [0.6, 0.8, 0.7]   # high vol
        ])
        
        # Highlight current regime
        current_intensity = intensity_matrix.copy()
        current_intensity[vol_idx, depth_idx] += 0.2
        
        fig = go.Figure(data=go.Heatmap(
            z=current_intensity,
            x=depth_levels,
            y=volatility_levels,
            colorscale='RdYlGn',
            text=[
                ['Low Vol\nThin Depth', 'Low Vol\nMed Depth', 'Low Vol\nThick Depth'],
                ['Med Vol\nThin Depth', 'Med Vol\nMed Depth', 'Med Vol\nThick Depth'], 
                ['High Vol\nThin Depth', 'High Vol\nMed Depth', 'High Vol\nThick Depth']
            ],
            texttemplate='%{text}',
            showscale=True,
            colorbar=dict(title="Signal\nFavorability")
        ))
        
        # Add current position marker
        fig.add_trace(go.Scatter(
            x=[depth_bucket],
            y=[vol_bucket],
            mode='markers+text',
            marker=dict(
                symbol='star',
                size=20,
                color='red',
                line=dict(color='black', width=2)
            ),
            text=['CURRENT'],
            textposition='top center',
            name='Current Regime'
        ))
        
        fig.update_layout(
            title="Volatility-Depth Regime Map",
            xaxis_title="Market Depth",
            yaxis_title="Volatility Level",
            height=300
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Funding regime overlay
        funding_value = regime_components.get('funding', {}).get('value', 0)
        funding_color = 'green' if funding_value < -0.01 else 'red' if funding_value > 0.01 else 'gray'
        
        st.markdown(f"""
        **Funding Overlay:** <span style='color: {funding_color}'>●</span> {funding_bucket.title()} 
        ({funding_value:+.4f})
        """, unsafe_allow_html=True)
    
    def _render_liquidity_metrics(self, data: Dict[str, Any]):
        """Render liquidity analysis metrics"""
        st.markdown("#### 💧 Liquidity Analysis")
        
        liquidity_metrics = data.get('liquidity_metrics', {})
        
        # Key liquidity indicators
        rv_ratio = liquidity_metrics.get('rv_ratio', 1.0)
        depth_slope = liquidity_metrics.get('depth_slope', 0.0)
        near_touch_void = liquidity_metrics.get('near_touch_void', 0.0)
        liquidity_score = liquidity_metrics.get('liquidity_score', 0.5)
        void_score = liquidity_metrics.get('void_score', 0.0)
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("RV Ratio", f"{rv_ratio:.2f}", 
                     delta=f"{rv_ratio-1:.2f}" if rv_ratio != 1 else None)
            st.metric("Depth Slope", f"{depth_slope:.2f}")
            st.metric("Near Touch Void", f"{near_touch_void:.3f}")
        
        with col2:
            # Liquidity score gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = liquidity_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Liquidity Score"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "lightblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "lightgreen"}
                    ]
                }
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, width='stretch')
    
    def _render_market_pressure_gauges(self, data: Dict[str, Any]):
        """Render market pressure indicators"""
        st.markdown("#### ⚡ Market Pressure")
        
        market_pressure = data.get('market_pressure', {})
        
        funding_delta = market_pressure.get('funding_delta', 0)
        oi_pressure = market_pressure.get('oi_pressure', 0)  
        arrival_rate = market_pressure.get('arrival_rate', 1.0)
        
        # Create subplots for multiple gauges
        fig = make_subplots(
            rows=3, cols=1,
            specs=[[{'type': 'indicator'}],
                   [{'type': 'indicator'}], 
                   [{'type': 'indicator'}]],
            subplot_titles=('Funding Δ', 'OI Pressure', 'Arrival Rate')
        )
        
        # Funding delta gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = funding_delta * 10000,  # Convert to basis points
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-20, 20]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [-20, -5], 'color': "lightgreen"},
                    {'range': [-5, 5], 'color': "yellow"},
                    {'range': [5, 20], 'color': "lightcoral"}
                ]
            }
        ), row=1, col=1)
        
        # OI pressure gauge  
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = oi_pressure,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [-0.5, 0.5]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [-0.5, 0], 'color': "lightcoral"},
                    {'range': [0, 0.5], 'color': "lightgreen"}
                ]
            }
        ), row=2, col=1)
        
        # Arrival rate gauge
        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = arrival_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [0, 3]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 1], 'color': "lightcoral"},
                    {'range': [1, 2], 'color': "yellow"},
                    {'range': [2, 3], 'color': "lightgreen"}
                ]
            }
        ), row=3, col=1)
        
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, width='stretch')
    
    def _render_depth_profile(self, data: Dict[str, Any]):
        """Render 5-level order book depth profile"""
        st.markdown("#### 📊 Depth Profile")
        
        # Mock depth data (in production, this would come from real order book)
        liquidity_metrics = data.get('liquidity_metrics', {})
        depth_slope = liquidity_metrics.get('depth_slope', -1.5)
        
        # Generate representative depth profile
        levels = list(range(1, 6))
        bid_sizes = [100 * np.exp(depth_slope * (i-1) * 0.2) for i in levels]
        ask_sizes = [100 * np.exp(depth_slope * (i-1) * 0.2) for i in levels]
        
        fig = go.Figure()
        
        # Bid side (left, negative)
        fig.add_trace(go.Bar(
            x=[-size for size in bid_sizes],
            y=[f"Level {i}" for i in levels],
            orientation='h',
            name='Bids',
            marker_color='green',
            text=[f"{size:.0f}" for size in bid_sizes],
            textposition='auto'
        ))
        
        # Ask side (right, positive)
        fig.add_trace(go.Bar(
            x=ask_sizes,
            y=[f"Level {i}" for i in levels],
            orientation='h', 
            name='Asks',
            marker_color='red',
            text=[f"{size:.0f}" for size in ask_sizes],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Order Book Depth (5 Levels)",
            xaxis_title="Liquidity Size",
            yaxis_title="Price Level",
            height=300,
            barmode='overlay',
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Show depth statistics
        total_bid_liq = sum(bid_sizes)
        total_ask_liq = sum(ask_sizes)
        imbalance = (total_bid_liq - total_ask_liq) / (total_bid_liq + total_ask_liq)
        
        st.markdown(f"""
        **Depth Stats:**
        - Total Bid Liquidity: {total_bid_liq:.0f}
        - Total Ask Liquidity: {total_ask_liq:.0f}
        - Imbalance: {imbalance:+.3f}
        """)
    
    def _render_adaptive_thresholds(self, data: Dict[str, Any]):
        """Render adaptive threshold recommendations"""
        st.markdown("#### 🎛️ Adaptive Threshold Recommendations")
        
        adaptive_thresholds = data.get('adaptive_thresholds', {})
        tau_adj = adaptive_thresholds.get('tau_adjustment', 0.0)
        kappa_adj = adaptive_thresholds.get('kappa_adjustment', 0.0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Current regime recommendation
            regime = data.get('regime', 'medium_vol_medium_depth')
            
            recommendations = {
                'high_vol': "💡 High volatility detected - Consider lowering τ for more signals",
                'low_vol': "💡 Low volatility - Consider raising τ to reduce false positives", 
                'thin_depth': "💡 Thin liquidity - Increase κ threshold for execution safety",
                'thick_depth': "💡 Thick liquidity - Can lower κ for more opportunities"
            }
            
            active_recommendations = []
            for condition, message in recommendations.items():
                if condition in regime:
                    active_recommendations.append(message)
            
            if active_recommendations:
                for rec in active_recommendations:
                    st.info(rec)
            else:
                st.success("✅ Current thresholds appropriate for regime")
        
        with col2:
            # Threshold adjustments
            st.markdown("**Suggested Adjustments:**")
            
            if tau_adj != 0:
                direction = "increase" if tau_adj > 0 else "decrease"
                st.markdown(f"• **τ (Probability):** {direction} by {abs(tau_adj):.2f}")
            else:
                st.markdown("• **τ (Probability):** No change")
            
            if kappa_adj != 0:
                direction = "increase" if kappa_adj > 0 else "decrease"  
                st.markdown(f"• **κ (Utility):** {direction} by {abs(kappa_adj):.2f}")
            else:
                st.markdown("• **κ (Utility):** No change")
        
        with col3:
            # Regime stability indicator
            regime_components = data.get('regime_components', {})
            vol_bucket = regime_components.get('volatility', {}).get('bucket', 'medium')
            depth_bucket = regime_components.get('depth', {}).get('bucket', 'medium')
            
            stability_score = 0.8  # Mock stability score
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = stability_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Regime Stability"},
                gauge = {
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkviolet"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightcoral"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "lightgreen"}
                    ]
                }
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, width='stretch')
            
            if stability_score > 0.8:
                st.success("🟢 Stable regime")
            elif stability_score > 0.5:
                st.warning("🟡 Transitioning regime")
            else:
                st.error("🔴 Volatile regime")

