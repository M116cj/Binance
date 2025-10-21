"""
Real-time Signal Card component for immediate trading decision display.
Report 1: Shows current p_up probabilities, utility, and decision tier.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time
from datetime import datetime

class SignalCard:
    """Real-time signal overview component"""
    
    def __init__(self):
        self.component_name = "Real-time Signal Card"
    
    def render(self, data: Dict[str, Any]):
        """Render the real-time signal card"""
        if not data:
            st.error("No signal data available")
            return
        
        # Main signal display
        self._render_signal_header(data)
        
        # Signal metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_probability_gauges(data)
        
        with col2:
            self._render_utility_metrics(data)
        
        with col3:
            self._render_decision_panel(data)
        
        # Feature importance and quality indicators
        st.markdown("---")
        
        col4, col5 = st.columns(2)
        
        with col4:
            self._render_feature_importance(data)
        
        with col5:
            self._render_quality_panel(data)
    
    def _render_signal_header(self, data: Dict[str, Any]):
        """Render signal header with key information"""
        symbol = data.get('symbol', 'Unknown')
        decision = data.get('decision', 'none')
        tier = data.get('tier', 'none')
        
        # Color coding for decision tiers
        tier_colors = {
            'A': '🟢',  # Green
            'B': '🟡',  # Yellow  
            'none': '⚪'  # White
        }
        
        tier_color = tier_colors.get(tier, '⚪')
        
        st.markdown(f"""
        ### {tier_color} {symbol} Signal Status
        **Decision Tier:** {tier.upper() if tier != 'none' else 'No Signal'}  
        **Updated:** {datetime.now().strftime('%H:%M:%S')}
        """)
        
        # SLA latency indicator
        sla_latency = data.get('sla_latency_ms', 0)
        if sla_latency > 0:
            if sla_latency < 200:
                latency_color = "green"
                latency_icon = "✅"
            elif sla_latency < 500:
                latency_color = "orange"
                latency_icon = "⚠️"
            else:
                latency_color = "red"
                latency_icon = "❌"
            
            st.markdown(f"""
            <div style='color: {latency_color}'>
            {latency_icon} SLA Latency: {sla_latency:.1f}ms
            </div>
            """, unsafe_allow_html=True)
    
    def _render_probability_gauges(self, data: Dict[str, Any]):
        """Render probability gauges for different horizons"""
        st.markdown("#### 📊 Surge Probabilities")
        
        probabilities = data.get('probabilities', {})
        thresholds = data.get('thresholds', {'tau': 0.75})
        tau = thresholds.get('tau', 0.75)
        
        for horizon in ['5m', '10m', '30m']:
            if horizon in probabilities:
                prob_data = probabilities[horizon]
                p_value = prob_data.get('value', 0)
                ci_low = prob_data.get('ci_low', p_value - 0.05)
                ci_high = prob_data.get('ci_high', p_value + 0.05)
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = p_value * 100,  # Convert to percentage
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"P(↑) {horizon}"},
                    delta = {'reference': tau * 100},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, tau * 100], 'color': "lightgray"},
                            {'range': [tau * 100, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': tau * 100
                        }
                    }
                ))
                
                fig.update_layout(
                    height=200,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show confidence interval
                st.caption(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    
    def _render_utility_metrics(self, data: Dict[str, Any]):
        """Render utility and return metrics"""
        st.markdown("#### 💰 Expected Utility")
        
        expected_return = data.get('expected_return', 0)
        estimated_cost = data.get('estimated_cost', 0)
        utility = data.get('utility', 0)
        
        # Utility gauge
        kappa = data.get('thresholds', {}).get('kappa', 1.20)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = utility,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Utility (U)"},
            gauge = {
                'axis': {'range': [0, 3]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, kappa], 'color': "lightgray"},
                    {'range': [kappa, 3], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': kappa
                }
            }
        ))
        
        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show breakdown
        st.markdown("**Breakdown:**")
        st.metric("Expected Return", f"{expected_return:.4f}", 
                 delta=f"{(expected_return - estimated_cost):.4f}")
        st.metric("Estimated Cost", f"{estimated_cost:.4f}")
        st.metric("Net Expected", f"{expected_return - estimated_cost:.4f}")
    
    def _render_decision_panel(self, data: Dict[str, Any]):
        """Render decision panel with tier information"""
        st.markdown("#### ⚡ Decision Panel")
        
        decision = data.get('decision', 'none')
        tier = data.get('tier', 'none')
        cooldown_until = data.get('cooldown_until')
        
        # Decision display
        if decision != 'none':
            if tier == 'A':
                st.success(f"🎯 **A-TIER SIGNAL**\nHigh confidence trade recommendation")
            elif tier == 'B':
                st.info(f"📈 **B-TIER SIGNAL**\nModerate confidence opportunity")
            else:
                st.warning(f"⚠️ **CONDITIONAL SIGNAL**\nReview parameters")
        else:
            st.error("❌ **NO SIGNAL**\nConditions not met")
        
        # Cooldown status
        if cooldown_until:
            cooldown_time = datetime.fromtimestamp(cooldown_until / 1000)
            time_until = cooldown_time - datetime.now()
            
            if time_until.total_seconds() > 0:
                st.warning(f"🕐 Cooldown: {int(time_until.total_seconds()/60)}min remaining")
            else:
                st.success("✅ Ready for new signal")
        else:
            st.success("✅ Ready for new signal")
        
        # Threshold display
        thresholds = data.get('thresholds', {})
        tau = thresholds.get('tau', 0.75)
        kappa = thresholds.get('kappa', 1.20)
        
        st.markdown(f"""
        **Current Thresholds:**
        - τ (Probability): {tau:.2f}
        - κ (Utility): {kappa:.2f}
        """)
    
    def _render_feature_importance(self, data: Dict[str, Any]):
        """Render top 5 feature importance"""
        st.markdown("#### 🔍 Top Features")
        
        features_top5 = data.get('features_top5', {})
        
        if not features_top5:
            st.warning("No feature data available")
            return
        
        # Create horizontal bar chart
        feature_names = list(features_top5.keys())
        feature_values = list(features_top5.values())
        
        # Color bars based on positive/negative values
        colors = ['green' if v > 0 else 'red' for v in feature_values]
        
        fig = go.Figure(data=[
            go.Bar(
                y=feature_names,
                x=feature_values,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.3f}" for v in feature_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Feature Contributions",
            xaxis_title="Impact",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature explanations
        feature_explanations = {
            'qi_1': 'Order book imbalance (1-level)',
            'ofi_10': 'Order flow imbalance (10-period)',
            'microprice_dev': 'Microprice deviation',
            'rv_ratio': 'Volatility ratio (short/long)',
            'depth_slope_bid': 'Bid depth slope'
        }
        
        st.markdown("**Feature Meanings:**")
        for feature_name in feature_names:
            explanation = feature_explanations.get(feature_name, 'Market microstructure indicator')
            st.caption(f"• {feature_name}: {explanation}")
    
    def _render_quality_panel(self, data: Dict[str, Any]):
        """Render data quality and system status"""
        st.markdown("#### 🚦 Quality Status")
        
        quality_flags = data.get('quality_flags', [])
        model_version = data.get('model_version', 'Unknown')
        feature_version = data.get('feature_version', 'Unknown')
        cost_model = data.get('cost_model', 'Unknown')
        
        # Quality indicators
        if not quality_flags:
            st.success("✅ All systems nominal")
        else:
            for flag in quality_flags:
                if 'degraded' in flag:
                    st.warning(f"⚠️ {flag.replace('_', ' ').title()}")
                elif 'error' in flag:
                    st.error(f"❌ {flag.replace('_', ' ').title()}")
                else:
                    st.info(f"ℹ️ {flag.replace('_', ' ').title()}")
        
        # Model versions
        st.markdown("**Model Versions:**")
        st.caption(f"• ML Model: {model_version}")
        st.caption(f"• Features: {feature_version}")
        st.caption(f"• Cost Model: {cost_model}")
        
        # Data window info
        data_window_id = data.get('data_window_id', 'Unknown')
        st.caption(f"• Data Window: {data_window_id}")
        
        # Real-time updates indicator
        st.markdown("**Status:**")
        timestamp = data.get('timestamp', time.time() * 1000)
        data_age_seconds = (time.time() * 1000 - timestamp) / 1000
        
        if data_age_seconds < 5:
            st.success(f"🟢 Live ({data_age_seconds:.1f}s)")
        elif data_age_seconds < 30:
            st.warning(f"🟡 Recent ({data_age_seconds:.0f}s)")
        else:
            st.error(f"🔴 Stale ({data_age_seconds:.0f}s)")

