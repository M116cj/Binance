"""
Execution Cost & Capacity Analysis component.
Report 4: Shows cost breakdown, capacity curves, and execution recommendations.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import time

class CostCapacity:
    """Execution cost and capacity analysis component"""
    
    def __init__(self):
        self.component_name = "Execution Cost & Capacity Analysis"
    
    def render(self, data: Dict[str, Any]):
        """Render the cost and capacity analysis"""
        if not data:
            st.error("No cost & capacity data available")
            return
        
        # Header with key metrics
        self._render_cost_header(data)
        
        # Main analysis panels
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_cost_breakdown(data)
            self._render_slippage_analysis(data)
        
        with col2:
            self._render_capacity_curve(data)
            self._render_execution_metrics(data)
        
        # Recommendations and warnings
        st.markdown("---")
        self._render_execution_recommendations(data)
    
    def _render_cost_header(self, data: Dict[str, Any]):
        """Render cost analysis header"""
        symbol = data.get('symbol', 'Unknown')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        optimal_size_usd = data.get('optimal_size_usd', 0)
        capacity_pct = data.get('capacity_pct', 0)
        
        st.markdown(f"""
        ### 💰 Execution Cost & Capacity: {symbol}
        **Optimal Size:** ${optimal_size_usd:,.0f}  
        **Current Capacity:** {capacity_pct:.1%}  
        *Updated: {time.strftime('%H:%M:%S', time.localtime(timestamp/1000))}*
        """)
        
        # Capacity status indicator
        if capacity_pct < 0.3:
            st.success("🟢 Low capacity utilization - room for larger positions")
        elif capacity_pct < 0.7:
            st.info("🟡 Moderate capacity utilization")
        else:
            st.warning("🔴 High capacity utilization - consider position splitting")
    
    def _render_cost_breakdown(self, data: Dict[str, Any]):
        """Render detailed cost breakdown"""
        st.markdown("#### 📊 Cost Breakdown")
        
        cost_breakdown = data.get('cost_breakdown', {})
        
        if not cost_breakdown:
            st.warning("No cost breakdown data available")
            return
        
        # Extract cost components
        maker_fee = cost_breakdown.get('maker_fee', 0)
        taker_fee = cost_breakdown.get('taker_fee', 0)
        impact_cost = cost_breakdown.get('impact_cost', 0)
        slippage_expected = cost_breakdown.get('slippage', {}).get('expected', 0)
        funding_cost = cost_breakdown.get('funding_cost', 0)
        opportunity_cost = cost_breakdown.get('opportunity_cost', 0)
        
        # Create waterfall chart
        categories = ['Maker Fee', 'Taker Fee', 'Impact Cost', 'Slippage', 'Funding', 'Opportunity']
        values = [maker_fee, taker_fee, impact_cost, slippage_expected, funding_cost, opportunity_cost]
        
        # Create stacked bar chart
        fig = go.Figure()
        
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink', 'lightgray']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"${v:.2f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Cost Components Breakdown",
            xaxis_title="Cost Type",
            yaxis_title="Cost ($)",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary metrics
        total_cost = cost_breakdown.get('total_cost_estimate', sum(values))
        cost_per_unit = cost_breakdown.get('cost_per_unit', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Cost", f"${total_cost:.2f}")
        with col2:
            st.metric("Cost per Unit", f"{cost_per_unit:.4f}")
    
    def _render_slippage_analysis(self, data: Dict[str, Any]):
        """Render slippage percentile analysis"""
        st.markdown("#### 📉 Slippage Analysis")
        
        slippage_analysis = data.get('slippage_analysis', {})
        
        if not slippage_analysis:
            st.warning("No slippage data available")
            return
        
        # Create box plot for slippage distribution
        percentiles = ['p25', 'p50', 'p75', 'p95', 'p99']
        percentile_labels = ['25th', '50th', '75th', '95th', '99th']
        values = [slippage_analysis.get(p, 0) * 10000 for p in percentiles]  # Convert to basis points
        
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            y=values,
            name='Slippage Distribution',
            boxpoints='all',
            jitter=0.3,
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Slippage Distribution (Basis Points)",
            yaxis_title="Slippage (bps)",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show percentile table
        percentile_data = []
        for p, label, value in zip(percentiles, percentile_labels, values):
            percentile_data.append({
                'Percentile': label,
                'Slippage (bps)': f"{value:.1f}",
                'Slippage ($)': f"${slippage_analysis.get(p, 0) * 10000:.2f}"  # Assuming $10k position
            })
        
        st.dataframe(pd.DataFrame(percentile_data), use_container_width=True)
    
    def _render_capacity_curve(self, data: Dict[str, Any]):
        """Render utility-size capacity curve"""
        st.markdown("#### 📈 Capacity Curve (Utility vs Size)")
        
        capacity_curve = data.get('capacity_curve', {})
        
        if not capacity_curve:
            st.warning("No capacity curve data available")
            return
        
        # Extract size and utility data
        sizes = sorted(capacity_curve.keys())
        utilities = [capacity_curve[size]['utility'] for size in sizes]
        costs = [capacity_curve[size]['estimated_cost'] for size in sizes]
        
        # Create dual-axis plot
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Utility curve
        fig.add_trace(
            go.Scatter(
                x=[f"${size:,.0f}" for size in sizes],
                y=utilities,
                name="Utility (U)",
                line=dict(color='green', width=3),
                mode='lines+markers'
            ),
            secondary_y=False,
        )
        
        # Cost curve
        fig.add_trace(
            go.Scatter(
                x=[f"${size:,.0f}" for size in sizes],
                y=costs,
                name="Estimated Cost",
                line=dict(color='red', width=2, dash='dash'),
                mode='lines'
            ),
            secondary_y=True,
        )
        
        # Find optimal point
        max_utility_idx = utilities.index(max(utilities))
        optimal_size = sizes[max_utility_idx]
        optimal_utility = utilities[max_utility_idx]
        
        # Mark optimal point
        fig.add_trace(
            go.Scatter(
                x=[f"${optimal_size:,.0f}"],
                y=[optimal_utility],
                mode='markers',
                name='Optimal Point',
                marker=dict(color='gold', size=15, symbol='star')
            ),
            secondary_y=False,
        )
        
        # Add threshold lines
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray", 
                     annotation_text="Break-even Utility", secondary_y=False)
        
        fig.update_xaxes(title_text="Position Size")
        fig.update_yaxes(title_text="Utility (U)", secondary_y=False)
        fig.update_yaxes(title_text="Estimated Cost ($)", secondary_y=True)
        
        fig.update_layout(
            title="Utility vs Position Size",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Optimal size info
        st.success(f"🎯 Optimal Size: ${optimal_size:,.0f} (U = {optimal_utility:.2f})")
    
    def _render_execution_metrics(self, data: Dict[str, Any]):
        """Render execution quality metrics"""
        st.markdown("#### ⚙️ Execution Metrics")
        
        execution_metrics = data.get('execution_metrics', {})
        
        if not execution_metrics:
            st.warning("No execution metrics available")
            return
        
        impact_lambda = execution_metrics.get('impact_lambda', 0)
        near_touch_liquidity = execution_metrics.get('near_touch_liquidity', 0)
        estimated_fill_rate = execution_metrics.get('estimated_fill_rate', 0)
        
        # Metrics display
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Impact Lambda (λ)", f"{impact_lambda:.6f}")
            st.metric("Fill Rate", f"{estimated_fill_rate:.1%}")
        
        with col2:
            st.metric("Near Touch Liquidity", f"${near_touch_liquidity:,.0f}")
            
            # Liquidity gauge
            liquidity_ratio = min(1.0, near_touch_liquidity / 50000)  # Normalize to $50k
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = liquidity_ratio,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Liquidity Depth"},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightcoral"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "lightgreen"}
                    ]
                }
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        # Market impact visualization
        st.markdown("**Market Impact Analysis:**")
        
        # Generate impact curve
        sizes = np.logspace(3, 5, 20)  # $1K to $100K
        impacts = [impact_lambda * (size ** 0.5) for size in sizes]  # Square root law
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sizes,
            y=[impact * 10000 for impact in impacts],  # Convert to bps
            mode='lines',
            name='Market Impact',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Market Impact vs Position Size",
            xaxis_title="Position Size ($)",
            yaxis_title="Market Impact (bps)",
            xaxis_type="log",
            height=250
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_execution_recommendations(self, data: Dict[str, Any]):
        """Render execution recommendations"""
        st.markdown("#### 💡 Execution Recommendations")
        
        recommendations = data.get('recommendations', {})
        capacity_pct = data.get('capacity_pct', 0)
        optimal_size_usd = data.get('optimal_size_usd', 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📏 Position Sizing:**")
            
            max_position_size = recommendations.get('max_position_size', optimal_size_usd)
            
            if capacity_pct < 0.5:
                st.success(f"✅ Can use full size: ${max_position_size:,.0f}")
            elif capacity_pct < 0.8:
                st.info(f"⚠️ Recommended max: ${max_position_size:,.0f}")
            else:
                reduced_size = max_position_size * 0.5
                st.warning(f"🔴 Reduce to: ${reduced_size:,.0f}")
        
        with col2:
            st.markdown("**⏱️ Execution Timing:**")
            
            timing_advice = recommendations.get('timing_advice', 'immediate')
            suggested_splits = recommendations.get('suggested_splits', 1)
            
            if timing_advice == 'immediate':
                st.success("✅ Execute immediately")
            elif timing_advice == 'staged':
                st.info(f"📊 Stage execution ({suggested_splits} splits)")
            else:
                st.warning("⚠️ Careful execution required")
            
            # Split recommendations
            if suggested_splits > 1:
                split_size = optimal_size_usd / suggested_splits
                st.caption(f"Split into {suggested_splits} orders of ${split_size:,.0f} each")
        
        with col3:
            st.markdown("**⚡ Risk Management:**")
            
            execution_metrics = data.get('execution_metrics', {})
            fill_rate = execution_metrics.get('estimated_fill_rate', 0.9)
            
            # Risk indicators
            risk_level = "Low"
            risk_color = "green"
            
            if capacity_pct > 0.8 or fill_rate < 0.8:
                risk_level = "High"
                risk_color = "red"
            elif capacity_pct > 0.5 or fill_rate < 0.9:
                risk_level = "Medium"
                risk_color = "orange"
            
            st.markdown(f"<span style='color: {risk_color}'>🎯 Risk Level: {risk_level}</span>", 
                       unsafe_allow_html=True)
            
            # Specific warnings
            if fill_rate < 0.9:
                st.warning("⚠️ Partial fill risk")
            
            if capacity_pct > 0.7:
                st.warning("⚠️ High market impact expected")
            
            # Slippage protection
            slippage_analysis = data.get('slippage_analysis', {})
            p95_slippage = slippage_analysis.get('p95', 0) * 10000  # bps
            
            if p95_slippage > 20:  # > 20 bps
                st.warning(f"⚠️ High slippage risk: {p95_slippage:.1f}bps")
            else:
                st.success(f"✅ Slippage manageable: {p95_slippage:.1f}bps")

