"""
Pre-Surge Probability & Time Window component.
Report 3: Shows probability curves across time horizons and optimal timing analysis.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time

class ProbabilityWindow:
    """Pre-surge probability and time window analysis component"""
    
    def __init__(self):
        self.component_name = "Pre-Surge Probability & Time Window"
    
    def render(self, data: Dict[str, Any], tau: float = 0.75, kappa: float = 1.20):
        """Render the probability window analysis"""
        if not data:
            st.error("No probability window data available")
            return
        
        # Header with optimal horizon
        self._render_optimal_horizon_header(data)
        
        # Main probability curves
        self._render_probability_curves(data, tau, kappa)
        
        # Analysis panels
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_expected_returns_analysis(data)
            self._render_time_to_peak_analysis(data)
        
        with col2:
            self._render_utility_comparison(data)
            self._render_horizon_breakdown(data)
        
        # Strategy recommendations
        st.markdown("---")
        self._render_strategy_recommendations(data, tau, kappa)
    
    def _render_optimal_horizon_header(self, data: Dict[str, Any]):
        """Render header with optimal horizon information"""
        optimal_horizon = data.get('optimal_horizon', 10)
        utilities = data.get('utilities', {})
        optimal_utility = utilities.get(optimal_horizon, 0)
        
        symbol = data.get('symbol', 'Unknown')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        st.markdown(f"""
        ### 📈 Pre-Surge Probability Analysis: {symbol}
        **Optimal Horizon:** {optimal_horizon} minutes (U = {optimal_utility:.2f})  
        *Updated: {time.strftime('%H:%M:%S', time.localtime(timestamp/1000))}*
        """)
    
    def _render_probability_curves(self, data: Dict[str, Any], tau: float, kappa: float):
        """Render probability curves across time horizons"""
        st.markdown("#### 📊 Probability Curves by Time Horizon")
        
        probability_curve = data.get('probability_curve', {})
        expected_returns = data.get('expected_returns', {})
        utilities = data.get('utilities', {})
        
        if not probability_curve:
            st.warning("No probability curve data available")
            return
        
        # Prepare data for plotting
        horizons = sorted(probability_curve.keys())
        p_values = [probability_curve[h]['p_up'] for h in horizons]
        ci_lows = [probability_curve[h]['ci_low'] for h in horizons]
        ci_highs = [probability_curve[h]['ci_high'] for h in horizons]
        expected_rets = [expected_returns.get(h, 0) for h in horizons]
        utility_vals = [utilities.get(h, 0) for h in horizons]
        
        # Create subplot with multiple y-axes
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Probability Curves', 'Expected Returns & Utility'),
            vertical_spacing=0.15
        )
        
        # Probability curves with confidence intervals
        fig.add_trace(go.Scatter(
            x=horizons,
            y=p_values,
            mode='lines+markers',
            name='P(↑)',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ), row=1, col=1)
        
        # Confidence intervals
        fig.add_trace(go.Scatter(
            x=horizons + horizons[::-1],
            y=ci_highs + ci_lows[::-1],
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% CI',
            showlegend=False
        ), row=1, col=1)
        
        # Threshold lines
        fig.add_hline(y=tau, line_dash="dash", line_color="red", 
                     annotation_text=f"τ = {tau}", row=1, col=1)
        
        # Expected returns and utility
        fig.add_trace(go.Scatter(
            x=horizons,
            y=expected_rets,
            mode='lines+markers',
            name='Expected Return',
            line=dict(color='green', width=2),
            yaxis='y3'
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=horizons,
            y=utility_vals,
            mode='lines+markers', 
            name='Utility (U)',
            line=dict(color='purple', width=2),
            yaxis='y4'
        ), row=2, col=1)
        
        # Add utility threshold line
        fig.add_hline(y=kappa, line_dash="dash", line_color="orange",
                     annotation_text=f"κ = {kappa}", row=2, col=1)
        
        fig.update_layout(
            height=500,
            xaxis_title="Time Horizon (minutes)",
            xaxis2_title="Time Horizon (minutes)",
            yaxis_title="Probability",
            yaxis2_title="Expected Return / Utility",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time Horizon (minutes)", row=2, col=1)
        fig.update_yaxes(title_text="Probability", row=1, col=1)
        fig.update_yaxes(title_text="Expected Return / Utility", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_expected_returns_analysis(self, data: Dict[str, Any]):
        """Render expected returns analysis"""
        st.markdown("#### 💰 Expected Returns Analysis")
        
        expected_returns = data.get('expected_returns', {})
        time_to_peak = data.get('time_to_peak', {})
        
        if not expected_returns:
            st.warning("No expected returns data")
            return
        
        # Create bar chart of expected returns
        horizons = sorted(expected_returns.keys())
        returns = [expected_returns[h] for h in horizons]
        ttp_values = [time_to_peak.get(h, h*0.7) for h in horizons]  # Default to 70% of horizon
        
        fig = go.Figure()
        
        # Expected returns bars
        fig.add_trace(go.Bar(
            x=[f"{h}m" for h in horizons],
            y=returns,
            name='Expected Return',
            marker_color='lightgreen',
            text=[f"{r:.4f}" for r in returns],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Expected Returns by Horizon",
            xaxis_title="Time Horizon",
            yaxis_title="Expected Return",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        best_return_horizon = horizons[returns.index(max(returns))]
        worst_return_horizon = horizons[returns.index(min(returns))]
        
        st.markdown(f"""
        **Return Analysis:**
        - Best: {best_return_horizon}m ({max(returns):.4f})
        - Worst: {worst_return_horizon}m ({min(returns):.4f})
        - Average: {np.mean(returns):.4f}
        """)
    
    def _render_time_to_peak_analysis(self, data: Dict[str, Any]):
        """Render time to peak analysis"""
        st.markdown("#### ⏱️ Time to Peak Analysis")
        
        time_to_peak = data.get('time_to_peak', {})
        
        if not time_to_peak:
            st.warning("No time to peak data")
            return
        
        horizons = sorted(time_to_peak.keys())
        ttp_values = [time_to_peak[h] for h in horizons]
        efficiency_ratios = [ttp / h for ttp, h in zip(ttp_values, horizons)]
        
        # Time to peak scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=horizons,
            y=ttp_values,
            mode='markers+lines',
            name='Time to Peak',
            marker=dict(
                size=[ratio*20 for ratio in efficiency_ratios],  # Size based on efficiency
                color=efficiency_ratios,
                colorscale='RdYlGn_r',
                colorbar=dict(title="Efficiency"),
                showscale=True
            ),
            line=dict(color='blue', width=2)
        ))
        
        # Add diagonal line for reference (ttp = horizon)
        fig.add_trace(go.Scatter(
            x=horizons,
            y=horizons,
            mode='lines',
            name='Max Time',
            line=dict(color='red', dash='dash'),
            opacity=0.5
        ))
        
        fig.update_layout(
            title="Time to Peak vs Horizon",
            xaxis_title="Horizon (minutes)",
            yaxis_title="Time to Peak (minutes)", 
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Efficiency metrics
        avg_efficiency = np.mean(efficiency_ratios)
        best_efficiency_idx = efficiency_ratios.index(min(efficiency_ratios))
        best_efficiency_horizon = horizons[best_efficiency_idx]
        
        st.markdown(f"""
        **Timing Efficiency:**
        - Average: {avg_efficiency:.2f} (lower is better)
        - Most Efficient: {best_efficiency_horizon}m ({min(efficiency_ratios):.2f})
        """)
    
    def _render_utility_comparison(self, data: Dict[str, Any]):
        """Render utility comparison across horizons"""
        st.markdown("#### ⚖️ Utility Comparison")
        
        utilities = data.get('utilities', {})
        expected_returns = data.get('expected_returns', {})
        
        if not utilities:
            st.warning("No utility data available")
            return
        
        horizons = sorted(utilities.keys())
        utility_values = [utilities[h] for h in horizons]
        return_values = [expected_returns.get(h, 0) for h in horizons]
        
        # Utility vs expected return scatter
        fig = go.Figure()
        
        # Color code by horizon
        colors = px.colors.sequential.Viridis
        color_map = {h: colors[i % len(colors)] for i, h in enumerate(horizons)}
        
        fig.add_trace(go.Scatter(
            x=return_values,
            y=utility_values,
            mode='markers+text',
            text=[f"{h}m" for h in horizons],
            textposition='top center',
            marker=dict(
                size=12,
                color=[color_map[h] for h in horizons],
                line=dict(color='black', width=1)
            ),
            name='Horizons'
        ))
        
        # Add threshold lines
        kappa = data.get('threshold_lines', {}).get('kappa', 1.20)
        fig.add_hline(y=kappa, line_dash="dash", line_color="red",
                     annotation_text=f"κ threshold = {kappa}")
        
        fig.update_layout(
            title="Utility vs Expected Return",
            xaxis_title="Expected Return",
            yaxis_title="Utility (U)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Utility ranking
        ranked_horizons = sorted(horizons, key=lambda h: utilities[h], reverse=True)
        
        st.markdown("**Utility Ranking:**")
        for i, h in enumerate(ranked_horizons[:3]):
            st.markdown(f"{i+1}. {h}m: U = {utilities[h]:.2f}")
    
    def _render_horizon_breakdown(self, data: Dict[str, Any]):
        """Render horizon category breakdown"""
        st.markdown("#### 🎯 Horizon Category Analysis")
        
        horizon_analysis = data.get('horizon_analysis', {})
        
        if not horizon_analysis:
            st.warning("No horizon analysis data")
            return
        
        # Create comparison table
        categories = ['short_term', 'medium_term', 'long_term']
        category_names = ['Short Term', 'Medium Term', 'Long Term']
        
        comparison_data = []
        for cat, name in zip(categories, category_names):
            if cat in horizon_analysis:
                cat_data = horizon_analysis[cat]
                comparison_data.append({
                    'Category': name,
                    'Horizons': ', '.join([f"{h}m" for h in cat_data.get('horizons', [])]),
                    'Avg Probability': cat_data.get('avg_probability', 0),
                    'Avg Utility': cat_data.get('avg_utility', 0)
                })
        
        if comparison_data:
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualization of category performance
            categories_short = [item['Category'] for item in comparison_data]
            probabilities = [item['Avg Probability'] for item in comparison_data]
            utilities = [item['Avg Utility'] for item in comparison_data]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=categories_short,
                y=probabilities,
                name='Avg Probability',
                marker_color='lightblue',
                yaxis='y1',
                offsetgroup=1
            ))
            
            fig.add_trace(go.Bar(
                x=categories_short,
                y=utilities,
                name='Avg Utility',
                marker_color='lightcoral',
                yaxis='y2',
                offsetgroup=2
            ))
            
            fig.update_layout(
                title="Category Performance Comparison",
                xaxis_title="Time Category",
                yaxis=dict(title="Probability", side="left"),
                yaxis2=dict(title="Utility", side="right", overlaying="y"),
                barmode='group',
                height=250
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_recommendations(self, data: Dict[str, Any], tau: float, kappa: float):
        """Render strategy recommendations based on analysis"""
        st.markdown("#### 💡 Strategy Recommendations")
        
        optimal_horizon = data.get('optimal_horizon', 10)
        utilities = data.get('utilities', {})
        probability_curve = data.get('probability_curve', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📊 Probability Analysis:**")
            
            # Check probability trend
            horizons = sorted(probability_curve.keys())
            if len(horizons) >= 3:
                p_values = [probability_curve[h]['p_up'] for h in horizons]
                
                if p_values[-1] > p_values[0]:
                    st.success("📈 Probability increases with time - Consider longer horizons")
                elif p_values[-1] < p_values[0]:
                    st.info("📉 Probability peaks early - Consider shorter horizons")
                else:
                    st.info("➡️ Stable probability across horizons")
            
            # Threshold analysis
            viable_horizons = [h for h in horizons if probability_curve[h]['p_up'] >= tau]
            if viable_horizons:
                st.success(f"✅ {len(viable_horizons)} horizons exceed τ = {tau}")
            else:
                st.warning(f"⚠️ No horizons exceed τ = {tau}")
        
        with col2:
            st.markdown("**⚖️ Utility Analysis:**")
            
            # Utility-based recommendations
            viable_utilities = [h for h in horizons if utilities.get(h, 0) >= kappa]
            
            if viable_utilities:
                best_utility_horizon = max(viable_utilities, key=lambda h: utilities[h])
                st.success(f"🎯 Best utility at {best_utility_horizon}m (U = {utilities[best_utility_horizon]:.2f})")
                
                if best_utility_horizon != optimal_horizon:
                    st.info(f"💡 Consider {best_utility_horizon}m vs optimal {optimal_horizon}m")
            else:
                st.warning(f"⚠️ No horizons exceed κ = {kappa}")
            
            # Risk-adjusted recommendation
            horizon_analysis = data.get('horizon_analysis', {})
            if 'medium_term' in horizon_analysis:
                med_term = horizon_analysis['medium_term']
                med_utility = med_term.get('avg_utility', 0)
                if med_utility > kappa:
                    st.info("💼 Medium-term horizons show consistent performance")
        
        with col3:
            st.markdown("**🎯 Action Items:**")
            
            # Generate specific recommendations
            recommendations = []
            
            # Optimal strategy
            if optimal_horizon in probability_curve and optimal_horizon in utilities:
                opt_prob = probability_curve[optimal_horizon]['p_up']
                opt_util = utilities[optimal_horizon]
                
                if opt_prob >= tau and opt_util >= kappa:
                    recommendations.append(f"✅ Execute {optimal_horizon}m strategy")
                else:
                    recommendations.append(f"⏸️ Wait - optimal horizon below thresholds")
            
            # Alternative horizons
            sorted_by_utility = sorted(horizons, key=lambda h: utilities.get(h, 0), reverse=True)
            best_alternative = None
            
            for h in sorted_by_utility:
                if h != optimal_horizon:
                    if (probability_curve.get(h, {}).get('p_up', 0) >= tau and 
                        utilities.get(h, 0) >= kappa):
                        best_alternative = h
                        break
            
            if best_alternative:
                recommendations.append(f"🔄 Alternative: {best_alternative}m horizon")
            
            # Risk management
            ci_widths = {h: data['probability_curve'][h]['ci_high'] - data['probability_curve'][h]['ci_low'] 
                        for h in horizons if h in probability_curve}
            
            if ci_widths:
                most_certain_horizon = min(ci_widths.keys(), key=lambda h: ci_widths[h])
                if ci_widths[most_certain_horizon] < 0.1:  # Narrow CI
                    recommendations.append(f"🎯 Most certain: {most_certain_horizon}m")
            
            # Display recommendations
            for rec in recommendations:
                st.markdown(f"• {rec}")
            
            if not recommendations:
                st.info("🔍 Monitor conditions for signal opportunities")

