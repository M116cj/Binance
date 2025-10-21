"""
Event Attribution & Strategy Comparison component.
Report 7: Shows feature attribution, strategy comparisons, and decision analysis.
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

class AttributionComparison:
    """Event attribution and strategy comparison component"""
    
    def __init__(self):
        self.component_name = "Event Attribution & Strategy Comparison"
    
    def render(self, data: Dict[str, Any]):
        """Render the attribution and comparison analysis"""
        if not data:
            st.error("No attribution comparison data available")
            return
        
        # Header with comparison metrics
        self._render_comparison_header(data)
        
        # Main attribution visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_feature_attribution(data)
            self._render_shap_waterfall(data)
        
        with col2:
            self._render_strategy_comparison(data)
            self._render_threshold_sensitivity(data)
        
        # Detailed analysis sections
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._render_event_timeline(data)
        
        with col4:
            self._render_decision_analysis(data)
    
    def _render_comparison_header(self, data: Dict[str, Any]):
        """Render comparison summary header"""
        symbol = data.get('symbol', 'Unknown')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        comparison_metrics = data.get('comparison_metrics', {})
        current_strategy_return = comparison_metrics.get('current_strategy_return', 0)
        benchmark_return = comparison_metrics.get('benchmark_return', 0)
        alpha = comparison_metrics.get('alpha', 0)
        
        st.markdown(f"""
        ### 🔍 Attribution & Comparison: {symbol}
        *Updated: {time.strftime('%H:%M:%S', time.localtime(timestamp/1000))}*
        """)
        
        # Key comparison metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = "inverse" if current_strategy_return > 0 else "normal"
            st.metric(
                "Strategy Return",
                f"{current_strategy_return:.2%}",
                delta=f"{current_strategy_return:.2%}",
                delta_color=color
            )
        
        with col2:
            st.metric(
                "Benchmark Return",
                f"{benchmark_return:.2%}",
                delta=f"Buy & Hold"
            )
        
        with col3:
            color = "inverse" if alpha > 0 else "normal"
            st.metric(
                "Alpha (Excess Return)",
                f"{alpha:.2%}",
                delta="Outperforming" if alpha > 0 else "Underperforming",
                delta_color=color
            )
    
    def _render_feature_attribution(self, data: Dict[str, Any]):
        """Render feature importance and attribution"""
        st.markdown("#### 🎯 Feature Attribution")
        
        attribution_data = data.get('feature_attribution', {})
        features = attribution_data.get('features', [])
        importance_scores = attribution_data.get('importance', [])
        
        if not features or not importance_scores:
            st.warning("No feature attribution data available")
            return
        
        # Sort by importance
        sorted_indices = np.argsort(importance_scores)[::-1][:10]  # Top 10
        top_features = [features[i] for i in sorted_indices]
        top_scores = [importance_scores[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        fig = go.Figure()
        
        colors = ['green' if score > 0 else 'red' for score in top_scores]
        
        fig.add_trace(go.Bar(
            y=top_features,
            x=top_scores,
            orientation='h',
            marker_color=colors,
            text=[f"{score:.3f}" for score in top_scores],
            textposition='auto',
            hovertemplate='%{y}: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Top 10 Feature Attributions",
            xaxis_title="Attribution Score",
            yaxis_title="Feature",
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Feature importance summary
        total_attribution = sum(abs(score) for score in importance_scores)
        st.markdown(f"**Total Attribution:** {total_attribution:.3f}")
    
    def _render_shap_waterfall(self, data: Dict[str, Any]):
        """Render SHAP waterfall chart for a recent decision"""
        st.markdown("#### 💧 SHAP Waterfall (Most Recent Signal)")
        
        shap_data = data.get('shap_waterfall', {})
        base_value = shap_data.get('base_value', 0.5)
        shap_values = shap_data.get('shap_values', [])
        feature_names = shap_data.get('features', [])
        
        if not shap_values or not feature_names:
            st.warning("No SHAP waterfall data available")
            return
        
        # Calculate cumulative sum for waterfall
        cumsum = [base_value]
        for val in shap_values:
            cumsum.append(cumsum[-1] + val)
        
        # Create waterfall chart
        fig = go.Figure()
        
        # Base value
        fig.add_trace(go.Bar(
            x=['Base'],
            y=[base_value],
            name='Base Value',
            marker_color='lightgray',
            text=[f"{base_value:.3f}"],
            textposition='auto'
        ))
        
        # Feature contributions
        for i, (feature, shap_val) in enumerate(zip(feature_names[:5], shap_values[:5])):  # Top 5
            color = 'green' if shap_val > 0 else 'red'
            fig.add_trace(go.Bar(
                x=[feature],
                y=[shap_val],
                name=feature,
                marker_color=color,
                text=[f"{shap_val:+.3f}"],
                textposition='auto'
            ))
        
        # Final prediction
        final_value = cumsum[-1]
        fig.add_trace(go.Bar(
            x=['Final'],
            y=[final_value],
            name='Final Prediction',
            marker_color='blue',
            text=[f"{final_value:.3f}"],
            textposition='auto'
        ))
        
        fig.update_layout(
            title=f"From Base ({base_value:.3f}) to Final ({final_value:.3f})",
            yaxis_title="Probability",
            height=350,
            showlegend=False
        )
        
        st.plotly_chart(fig, width='stretch')
    
    def _render_strategy_comparison(self, data: Dict[str, Any]):
        """Render comparison of different strategies"""
        st.markdown("#### 📊 Strategy Performance Comparison")
        
        strategies = data.get('strategy_comparison', {})
        
        if not strategies:
            st.warning("No strategy comparison data available")
            return
        
        # Extract strategy metrics
        strategy_names = list(strategies.keys())
        returns = [strategies[s].get('return', 0) for s in strategy_names]
        sharpe_ratios = [strategies[s].get('sharpe', 0) for s in strategy_names]
        hit_rates = [strategies[s].get('hit_rate', 0) for s in strategy_names]
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Return (%)',
            x=strategy_names,
            y=[r * 100 for r in returns],
            marker_color='blue',
            text=[f"{r*100:.1f}%" for r in returns],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Sharpe Ratio',
            x=strategy_names,
            y=sharpe_ratios,
            marker_color='green',
            text=[f"{s:.2f}" for s in sharpe_ratios],
            textposition='auto'
        ))
        
        fig.add_trace(go.Bar(
            name='Hit Rate (%)',
            x=strategy_names,
            y=[hr * 100 for hr in hit_rates],
            marker_color='orange',
            text=[f"{hr*100:.0f}%" for hr in hit_rates],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Strategy Metrics Comparison",
            barmode='group',
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Best strategy
        best_strategy_idx = np.argmax(returns)
        best_strategy = strategy_names[best_strategy_idx]
        st.success(f"🏆 Best Strategy: **{best_strategy}** ({returns[best_strategy_idx]:.2%} return)")
    
    def _render_threshold_sensitivity(self, data: Dict[str, Any]):
        """Render threshold sensitivity analysis"""
        st.markdown("#### 🎚️ Threshold Sensitivity Analysis")
        
        sensitivity_data = data.get('threshold_sensitivity', {})
        tau_values = sensitivity_data.get('tau_thresholds', [])
        returns_by_tau = sensitivity_data.get('returns', [])
        signals_by_tau = sensitivity_data.get('signals_count', [])
        
        if not tau_values or not returns_by_tau:
            st.warning("No threshold sensitivity data available")
            return
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Returns curve
        fig.add_trace(
            go.Scatter(
                x=tau_values,
                y=[r * 100 for r in returns_by_tau],
                mode='lines+markers',
                name='Return (%)',
                line=dict(color='blue', width=3),
                marker=dict(size=8)
            ),
            secondary_y=False
        )
        
        # Signals count curve
        fig.add_trace(
            go.Scatter(
                x=tau_values,
                y=signals_by_tau,
                mode='lines+markers',
                name='Signal Count',
                line=dict(color='orange', width=2, dash='dash'),
                marker=dict(size=6)
            ),
            secondary_y=True
        )
        
        # Current threshold marker
        current_tau = data.get('current_tau', 0.75)
        fig.add_vline(x=current_tau, line_dash="dash", line_color="green",
                     annotation_text=f"Current τ={current_tau:.2f}")
        
        fig.update_xaxes(title_text="Probability Threshold (τ)")
        fig.update_yaxes(title_text="Return (%)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Signals", secondary_y=True)
        
        fig.update_layout(
            title="Return vs Signal Count by Threshold",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Optimal threshold
        if returns_by_tau:
            optimal_idx = np.argmax(returns_by_tau)
            optimal_tau = tau_values[optimal_idx]
            optimal_return = returns_by_tau[optimal_idx]
            st.info(f"📍 Optimal τ: **{optimal_tau:.2f}** (Return: {optimal_return:.2%})")
    
    def _render_event_timeline(self, data: Dict[str, Any]):
        """Render timeline of recent trading events"""
        st.markdown("#### ⏱️ Recent Event Timeline")
        
        events = data.get('recent_events', [])
        
        if not events:
            st.warning("No recent events available")
            return
        
        # Create timeline dataframe
        timeline_data = []
        for event in events[:10]:  # Most recent 10
            timeline_data.append({
                'Time': datetime.fromisoformat(event.get('timestamp', datetime.now().isoformat())),
                'Event': event.get('type', 'Unknown'),
                'Probability': event.get('probability', 0),
                'Decision': event.get('decision', 'None'),
                'Outcome': event.get('outcome', 'Pending')
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Color code by outcome
        fig = go.Figure()
        
        colors = {
            'Win': 'green',
            'Loss': 'red',
            'Pending': 'gray',
            'Skipped': 'orange'
        }
        
        for outcome in df['Outcome'].unique():
            mask = df['Outcome'] == outcome
            fig.add_trace(go.Scatter(
                x=df[mask]['Time'],
                y=df[mask]['Probability'],
                mode='markers',
                name=outcome,
                marker=dict(size=12, color=colors.get(outcome, 'blue')),
                text=df[mask]['Event'],
                hovertemplate='%{text}<br>Time: %{x}<br>Prob: %{y:.3f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Event Timeline by Outcome",
            xaxis_title="Time",
            yaxis_title="Signal Probability",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Event summary
        win_count = sum(1 for e in events if e.get('outcome') == 'Win')
        total_resolved = sum(1 for e in events if e.get('outcome') in ['Win', 'Loss'])
        if total_resolved > 0:
            win_rate = win_count / total_resolved
            st.markdown(f"**Recent Win Rate:** {win_rate:.1%} ({win_count}/{total_resolved})")
    
    def _render_decision_analysis(self, data: Dict[str, Any]):
        """Render decision matrix and quality metrics"""
        st.markdown("#### 🎲 Decision Quality Analysis")
        
        decision_matrix = data.get('decision_matrix', {})
        
        if not decision_matrix:
            st.warning("No decision matrix data available")
            return
        
        # Confusion matrix data
        true_positives = decision_matrix.get('true_positives', 0)
        false_positives = decision_matrix.get('false_positives', 0)
        true_negatives = decision_matrix.get('true_negatives', 0)
        false_negatives = decision_matrix.get('false_negatives', 0)
        
        # Create confusion matrix heatmap
        matrix = [
            [true_positives, false_positives],
            [false_negatives, true_negatives]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=['Predicted Positive', 'Predicted Negative'],
            y=['Actual Positive', 'Actual Negative'],
            text=matrix,
            texttemplate='%{text}',
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            height=300
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Calculate metrics
        total = true_positives + false_positives + true_negatives + false_negatives
        if total > 0:
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            accuracy = (true_positives + true_negatives) / total
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Classification Metrics:**
                - Precision: {precision:.2%}
                - Recall: {recall:.2%}
                """)
            
            with col2:
                st.markdown(f"""
                - Accuracy: {accuracy:.2%}
                - F1 Score: {f1_score:.3f}
                """)
