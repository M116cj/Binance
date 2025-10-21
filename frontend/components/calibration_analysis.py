"""
Model Calibration & Error Analysis component.
Report 6: Shows calibration curves, residual analysis, and model diagnostics.
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

class CalibrationAnalysis:
    """模型校准和误差分析组件"""
    
    def __init__(self):
        self.component_name = "预测准确度分析"
    
    def render(self, data: Dict[str, Any]):
        """渲染校准分析"""
        if not data:
            st.error("❌ 没有可用的校准数据")
            return
        
        # Header with calibration metrics
        self._render_calibration_header(data)
        
        # Main calibration visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_calibration_curve(data)
            self._render_brier_score_breakdown(data)
        
        with col2:
            self._render_residual_analysis(data)
            self._render_prediction_distribution(data)
        
        # Detailed diagnostics
        st.markdown("---")
        
        col3, col4 = st.columns(2)
        
        with col3:
            self._render_isotonic_mapping(data)
        
        with col4:
            self._render_error_metrics(data)
    
    def _render_calibration_header(self, data: Dict[str, Any]):
        """Render calibration summary header"""
        symbol = data.get('symbol', 'Unknown')
        timestamp = data.get('timestamp', int(time.time() * 1000))
        
        calibration_metrics = data.get('calibration_metrics', {})
        brier_score = calibration_metrics.get('brier_score', 0)
        calibration_error = calibration_metrics.get('calibration_error', 0)
        log_loss = calibration_metrics.get('log_loss', 0)
        
        st.markdown(f"""
        ### 🎯 预测准确度分析: {symbol}
        *更新时间: {time.strftime('%H:%M:%S', time.localtime(timestamp/1000))}*
        """)
        
        # 关键校准指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            color = "inverse" if brier_score < 0.2 else "normal"
            st.metric(
                "预测误差分数",
                f"{brier_score:.4f}",
                delta="校准良好" if brier_score < 0.2 else "需要改进",
                delta_color=color,
                help="Brier Score: 越小越好，表示预测越准确"
            )
        
        with col2:
            color = "inverse" if calibration_error < 0.05 else "normal"
            st.metric(
                "Calibration Error",
                f"{calibration_error:.4f}",
                delta="Excellent" if calibration_error < 0.05 else "Moderate",
                delta_color=color
            )
        
        with col3:
            st.metric(
                "Log Loss",
                f"{log_loss:.4f}",
                delta="Low entropy" if log_loss < 0.5 else "High uncertainty"
            )
    
    def _render_calibration_curve(self, data: Dict[str, Any]):
        """Render calibration curve (reliability diagram)"""
        st.markdown("#### 📊 Calibration Curve")
        
        calibration_data = data.get('calibration_curve', {})
        predicted_probs = calibration_data.get('predicted_probabilities', [])
        observed_freqs = calibration_data.get('observed_frequencies', [])
        
        if not predicted_probs or not observed_freqs:
            st.warning("No calibration curve data available")
            return
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash', width=2)
        ))
        
        # Actual calibration curve
        fig.add_trace(go.Scatter(
            x=predicted_probs,
            y=observed_freqs,
            mode='lines+markers',
            name='Model Calibration',
            line=dict(color='blue', width=3),
            marker=dict(size=8),
            hovertemplate='Predicted: %{x:.3f}<br>Observed: %{y:.3f}<extra></extra>'
        ))
        
        # Highlight regions
        fig.add_hrect(y0=0, y1=1, x0=0, x1=1, fillcolor="green", opacity=0.05)
        
        fig.update_layout(
            title="Reliability Diagram",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives",
            height=350,
            showlegend=True,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calibration statistics
        if len(predicted_probs) > 0:
            max_deviation = max(abs(p - o) for p, o in zip(predicted_probs, observed_freqs))
            st.markdown(f"**Max Calibration Deviation:** {max_deviation:.3f}")
    
    def _render_brier_score_breakdown(self, data: Dict[str, Any]):
        """Render Brier score decomposition"""
        st.markdown("#### 🎲 Brier Score Breakdown")
        
        brier_decomp = data.get('brier_decomposition', {})
        reliability = brier_decomp.get('reliability', 0)
        resolution = brier_decomp.get('resolution', 0)
        uncertainty = brier_decomp.get('uncertainty', 0)
        
        if not brier_decomp:
            st.warning("No Brier score decomposition available")
            return
        
        # Create bar chart for Brier components
        components = ['Reliability', 'Resolution', 'Uncertainty']
        values = [reliability, resolution, uncertainty]
        colors = ['red', 'green', 'blue']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=components,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Brier Score = Reliability - Resolution + Uncertainty",
            yaxis_title="Score Component",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"""
        **Interpretation:**
        - Reliability (lower is better): {reliability:.4f}
        - Resolution (higher is better): {resolution:.4f}
        - Uncertainty (dataset-dependent): {uncertainty:.4f}
        """)
    
    def _render_residual_analysis(self, data: Dict[str, Any]):
        """Render residual distribution and statistics"""
        st.markdown("#### 📉 Residual Analysis")
        
        residuals_data = data.get('residuals', {})
        residuals = residuals_data.get('values', [])
        
        if not residuals:
            st.warning("No residual data available")
            return
        
        # Create residual histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            name='Residuals',
            marker_color='skyblue',
            opacity=0.7
        ))
        
        # Add mean line
        mean_residual = np.mean(residuals)
        fig.add_vline(x=mean_residual, line_dash="dash", line_color="red",
                     annotation_text=f"Mean: {mean_residual:.4f}")
        
        fig.update_layout(
            title="Residual Distribution (Predicted - Actual)",
            xaxis_title="Residual",
            yaxis_title="Frequency",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Residual statistics
        std_residual = np.std(residuals)
        skewness = pd.Series(residuals).skew()
        kurtosis = pd.Series(residuals).kurtosis()
        
        st.markdown(f"""
        **Residual Statistics:**
        - Mean: {mean_residual:.4f}
        - Std Dev: {std_residual:.4f}
        - Skewness: {skewness:.3f}
        - Kurtosis: {kurtosis:.3f}
        """)
    
    def _render_prediction_distribution(self, data: Dict[str, Any]):
        """Render distribution of predicted probabilities"""
        st.markdown("#### 📊 Prediction Distribution")
        
        predictions = data.get('predictions', [])
        
        if not predictions:
            st.warning("No prediction data available")
            return
        
        # Create histogram of predictions
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=predictions,
            nbinsx=20,
            name='Predictions',
            marker_color='purple',
            opacity=0.7
        ))
        
        # Add decision thresholds
        fig.add_vline(x=0.5, line_dash="dash", line_color="gray",
                     annotation_text="Neutral (0.5)")
        fig.add_vline(x=0.75, line_dash="dash", line_color="green",
                     annotation_text="A-tier (0.75)")
        
        fig.update_layout(
            title="Distribution of Predicted Probabilities",
            xaxis_title="Predicted Probability",
            yaxis_title="Frequency",
            height=350,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Prediction statistics
        mean_pred = np.mean(predictions)
        median_pred = np.median(predictions)
        
        st.markdown(f"""
        **Prediction Stats:**
        - Mean: {mean_pred:.3f}
        - Median: {median_pred:.3f}
        - High Confidence (>0.75): {sum(1 for p in predictions if p > 0.75)} ({sum(1 for p in predictions if p > 0.75)/len(predictions)*100:.1f}%)
        """)
    
    def _render_isotonic_mapping(self, data: Dict[str, Any]):
        """Render isotonic regression calibration mapping"""
        st.markdown("#### 🔧 Isotonic Calibration Mapping")
        
        isotonic_data = data.get('isotonic_mapping', {})
        raw_probs = isotonic_data.get('raw_probabilities', [])
        calibrated_probs = isotonic_data.get('calibrated_probabilities', [])
        
        if not raw_probs or not calibrated_probs:
            st.warning("No isotonic mapping data available")
            return
        
        fig = go.Figure()
        
        # Identity line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='No Calibration',
            line=dict(color='gray', dash='dash')
        ))
        
        # Isotonic mapping
        fig.add_trace(go.Scatter(
            x=raw_probs,
            y=calibrated_probs,
            mode='lines+markers',
            name='Isotonic Mapping',
            line=dict(color='orange', width=2),
            marker=dict(size=6),
            hovertemplate='Raw: %{x:.3f}<br>Calibrated: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Isotonic Regression Calibration Function",
            xaxis_title="Raw LightGBM Probability",
            yaxis_title="Calibrated Probability",
            height=350,
            showlegend=True,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("Isotonic regression ensures monotonicity and improves calibration")
    
    def _render_error_metrics(self, data: Dict[str, Any]):
        """Render comprehensive error metrics"""
        st.markdown("#### 📈 Error Metrics")
        
        error_metrics = data.get('error_metrics', {})
        
        if not error_metrics:
            st.warning("No error metrics available")
            return
        
        mae = error_metrics.get('mae', 0)
        mse = error_metrics.get('mse', 0)
        rmse = error_metrics.get('rmse', 0)
        mape = error_metrics.get('mape', 0)
        
        # Display metrics in a table
        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE'],
            'Value': [mae, mse, rmse, f"{mape:.2f}%"],
            'Description': [
                'Mean Absolute Error',
                'Mean Squared Error',
                'Root Mean Squared Error',
                'Mean Absolute Percentage Error'
            ]
        })
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Visual representation
        fig = go.Figure()
        
        metrics_for_plot = ['MAE', 'RMSE']
        values_for_plot = [mae, rmse]
        
        fig.add_trace(go.Bar(
            x=metrics_for_plot,
            y=values_for_plot,
            marker_color=['lightblue', 'lightgreen'],
            text=[f"{v:.4f}" for v in values_for_plot],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Primary Error Metrics",
            yaxis_title="Error Value",
            height=250,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model quality assessment
        if mae < 0.05:
            st.success("✅ Excellent calibration quality")
        elif mae < 0.10:
            st.info("ℹ️ Good calibration quality")
        else:
            st.warning("⚠️ Consider recalibration")
