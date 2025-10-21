"""信号历史组件：显示过往预测和结果"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable


class SignalHistory:
    """显示历史信号，支持过滤和分析"""
    
    def render(self, fetch_data_fn: Callable):
        """渲染信号历史界面"""
        
        st.markdown("### 📜 Signal History & Performance")
        
        # 过滤器
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filter_symbol = st.selectbox("Symbol", ["All", "BTCUSDT", "ETHUSDT", "BNBUSDT"], key="hist_symbol")
        
        with col2:
            filter_decision = st.selectbox("Decision", ["All", "LONG", "SHORT", "WAIT"], key="hist_decision")
        
        with col3:
            filter_tier = st.selectbox("Tier", ["All", "A", "B"], key="hist_tier")
        
        with col4:
            filter_hours = st.number_input("Last N hours", min_value=1, max_value=168, value=24, key="hist_hours")
        
        # 根据过滤器构建API请求参数
        params = {
            'limit': 500  # 获取更多信号以供过滤
        }
        
        # 仅在不是"All"时添加过滤器
        if filter_symbol != "All":
            params['symbol'] = filter_symbol
        if filter_decision != "All":
            params['decision'] = filter_decision
        if filter_tier != "All":
            params['tier'] = filter_tier
        
        # 应用过滤器获取信号
        signals_data = fetch_data_fn("signals", params)
        
        if not signals_data or 'signals' not in signals_data:
            st.info("No signal history available. Generate some signals first!")
            return
        
        signals = signals_data.get('signals', [])
        
        if not signals:
            st.info("No signals match your filters")
            return
        
        # 按时间范围过滤（客户端时间过滤）
        cutoff_time = datetime.now() - timedelta(hours=filter_hours)
        signals = [
            s for s in signals 
            if datetime.fromisoformat(s['created_at'].replace('Z', '+00:00')) >= cutoff_time
        ]
        
        if not signals:
            st.info(f"No signals found in the last {filter_hours} hours with selected filters")
            return
        
        # 转换为DataFrame
        df = self._signals_to_dataframe(signals)
        
        # Summary metrics
        self._render_summary_metrics(df)
        
        # Signal table
        self._render_signal_table(df)
        
        # Performance chart
        self._render_performance_chart(df)
    
    def _signals_to_dataframe(self, signals: List[Dict]) -> pd.DataFrame:
        """将信号列表转换为pandas DataFrame"""
        rows = []
        for signal in signals:
            rows.append({
                'Time': datetime.fromisoformat(signal['created_at'].replace('Z', '+00:00')),
                'Symbol': signal['symbol'],
                'Horizon': f"{signal['horizon_min']}m",
                'Decision': signal['decision'],
                'Tier': signal['tier'],
                'P(up)': signal['p_up'],
                'Utility': signal['net_utility'],
                'Return': signal.get('expected_return', 0),
                'Latency (ms)': signal.get('sla_latency_ms', 0),
                'Model': signal.get('model_version', 'N/A'),
                'Outcome': signal.get('actual_outcome', 'PENDING')
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('Time', ascending=False)
        return df
    
    def _render_summary_metrics(self, df: pd.DataFrame):
        """渲染汇总指标卡片"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Signals", len(df))
        
        with col2:
            a_tier_count = len(df[df['Tier'] == 'A'])
            a_tier_pct = (a_tier_count / len(df) * 100) if len(df) > 0 else 0
            st.metric("A-tier Signals", f"{a_tier_count} ({a_tier_pct:.0f}%)")
        
        with col3:
            avg_utility = df['Utility'].mean()
            st.metric("Avg Utility", f"{avg_utility:.2f}")
        
        with col4:
            avg_prob = df['P(up)'].mean()
            st.metric("Avg P(up)", f"{avg_prob:.3f}")
        
        with col5:
            avg_latency = df['Latency (ms)'].mean()
            color = "normal" if avg_latency < 200 else "off"
            st.metric("Avg Latency", f"{avg_latency:.1f}ms", 
                     delta=f"{200 - avg_latency:.1f}ms vs target",
                     delta_color=color)
    
    def _render_signal_table(self, df: pd.DataFrame):
        """渲染交互式信号表格"""
        st.markdown("#### Signal Details")
        
        # 格式化数据框以供显示
        display_df = df.copy()
        display_df['Time'] = display_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['P(up)'] = display_df['P(up)'].apply(lambda x: f"{x:.3f}")
        display_df['Utility'] = display_df['Utility'].apply(lambda x: f"{x:.2f}")
        display_df['Return'] = display_df['Return'].apply(lambda x: f"{x:.4f}")
        display_df['Latency (ms)'] = display_df['Latency (ms)'].apply(lambda x: f"{x:.1f}")
        
        # 根据决策和等级设置样式
        def style_decision(row):
            if row['Decision'] == 'LONG':
                return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            elif row['Decision'] == 'SHORT':
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            display_df,
            width='stretch',
            hide_index=True,
            height=400
        )
        
        # 下载选项
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"signal_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def _render_performance_chart(self, df: pd.DataFrame):
        """渲染性能随时间变化图表"""
        st.markdown("#### Performance Over Time")
        
        # 按小时分组
        df_hourly = df.copy()
        df_hourly['Hour'] = df_hourly['Time'].dt.floor('h')
        
        hourly_stats = df_hourly.groupby('Hour').agg({
            'Utility': 'mean',
            'P(up)': 'mean',
            'Latency (ms)': 'mean',
            'Decision': 'count'
        }).reset_index()
        hourly_stats.columns = ['Hour', 'Avg Utility', 'Avg P(up)', 'Avg Latency', 'Signal Count']
        
        # 创建子图
        fig = go.Figure()
        
        # 添加轨迹
        fig.add_trace(go.Scatter(
            x=hourly_stats['Hour'],
            y=hourly_stats['Avg Utility'],
            mode='lines+markers',
            name='Avg Utility',
            line=dict(color='#00CC96', width=2),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=hourly_stats['Hour'],
            y=hourly_stats['Signal Count'],
            mode='lines+markers',
            name='Signal Count',
            yaxis='y2',
            line=dict(color='#AB63FA', width=2, dash='dash'),
            marker=dict(size=8)
        ))
        
        # 更新布局
        fig.update_layout(
            title='Signal Performance Trends',
            xaxis_title='Time',
            yaxis_title='Avg Utility',
            yaxis2=dict(
                title='Signal Count',
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # 决策分布
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### Decision Distribution")
            decision_counts = df['Decision'].value_counts()
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=decision_counts.index,
                values=decision_counts.values,
                hole=0.4,
                marker=dict(colors=['#00CC96', '#EF553B', '#636EFA'])
            )])
            
            fig_pie.update_layout(
                height=300,
                template='plotly_dark',
                showlegend=True
            )
            
            st.plotly_chart(fig_pie, width='stretch')
        
        with col2:
            st.markdown("##### Tier Distribution")
            tier_counts = df['Tier'].value_counts()
            
            fig_tier = go.Figure(data=[go.Bar(
                x=tier_counts.index,
                y=tier_counts.values,
                marker=dict(color=['#FFA15A', '#19D3F3'])
            )])
            
            fig_tier.update_layout(
                height=300,
                template='plotly_dark',
                xaxis_title='Tier',
                yaxis_title='Count',
                showlegend=False
            )
            
            st.plotly_chart(fig_tier, width='stretch')
