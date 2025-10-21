"""
实时交易信号卡片组件
显示当前的买卖建议、涨跌概率和交易决策
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
    """实时交易信号展示组件"""
    
    def __init__(self):
        self.component_name = "实时信号卡片"
    
    def render(self, data: Dict[str, Any]):
        """渲染实时信号卡片"""
        if not data:
            st.error("❌ 没有可用的信号数据")
            return
        
        # 主要信号显示
        self._render_signal_header(data)
        
        # 三列展示关键指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            self._render_probability_gauges(data)
        
        with col2:
            self._render_utility_metrics(data)
        
        with col3:
            self._render_decision_panel(data)
        
        # 特征重要性和质量指标
        st.markdown("---")
        
        col4, col5 = st.columns(2)
        
        with col4:
            self._render_feature_importance(data)
        
        with col5:
            self._render_quality_panel(data)
    
    def _render_signal_header(self, data: Dict[str, Any]):
        """渲染信号头部信息"""
        symbol = data.get('symbol', 'Unknown')
        decision = data.get('decision', 'none')
        tier = data.get('tier', 'none')
        
        # 根据信号等级显示不同颜色
        tier_colors = {
            'A': '🟢',  # 绿色 - 高信心
            'B': '🟡',  # 黄色 - 中等信心
            'none': '⚪'  # 白色 - 无信号
        }
        
        tier_descriptions = {
            'A': '高信心信号',
            'B': '中等信心信号',
            'none': '暂无交易信号'
        }
        
        tier_color = tier_colors.get(tier, '⚪')
        tier_desc = tier_descriptions.get(tier, '暂无信号')
        
        st.markdown(f"""
        ### {tier_color} {symbol} - {tier_desc}
        **更新时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """)
        
        # 系统延迟指标
        sla_latency = data.get('sla_latency_ms', 0)
        if sla_latency > 0:
            if sla_latency < 200:
                latency_color = "green"
                latency_icon = "✅"
                latency_text = "响应速度很快"
            elif sla_latency < 500:
                latency_color = "orange"
                latency_icon = "⚠️"
                latency_text = "响应速度正常"
            else:
                latency_color = "red"
                latency_icon = "❌"
                latency_text = "响应速度较慢"
            
            st.markdown(f"""
            <div style='color: {latency_color}'>
            {latency_icon} 系统延迟: {sla_latency:.1f}毫秒 ({latency_text})
            </div>
            """, unsafe_allow_html=True)
    
    def _render_probability_gauges(self, data: Dict[str, Any]):
        """显示不同时间窗口的上涨概率"""
        st.markdown("#### 📊 价格上涨概率")
        st.caption("在不同时间内价格上涨的可能性")
        
        probabilities = data.get('probabilities', {})
        thresholds = data.get('thresholds', {'tau': 0.75})
        tau = thresholds.get('tau', 0.75)
        
        # 时间窗口名称映射
        horizon_names = {
            '5m': '5分钟',
            '10m': '10分钟',
            '30m': '30分钟'
        }
        
        for horizon in ['5m', '10m', '30m']:
            if horizon in probabilities:
                prob_data = probabilities[horizon]
                p_value = prob_data.get('value', 0)
                ci_low = prob_data.get('ci_low', p_value - 0.05)
                ci_high = prob_data.get('ci_high', p_value + 0.05)
                
                # 创建仪表盘图表
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = p_value * 100,  # 转换为百分比
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{horizon_names[horizon]}内上涨概率"},
                    delta = {'reference': tau * 100, 'suffix': '%'},
                    number = {'suffix': '%'},
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
                
                # 显示置信区间
                confidence_pct = int((ci_high - ci_low) * 100)
                st.caption(f"📏 置信区间：{ci_low*100:.1f}% ~ {ci_high*100:.1f}% (浮动±{confidence_pct/2:.1f}%)")
    
    def _render_utility_metrics(self, data: Dict[str, Any]):
        """显示收益和成本指标"""
        st.markdown("#### 💰 收益分析")
        st.caption("预期收益与交易成本对比")
        
        expected_return = data.get('expected_return', 0)
        estimated_cost = data.get('estimated_cost', 0)
        utility = data.get('utility', 0)
        
        # 收益倍数仪表盘
        kappa = data.get('thresholds', {}).get('kappa', 1.20)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = utility,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "收益倍数（相对成本）"},
            number = {'suffix': 'x'},
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
        
        # 详细分解
        st.markdown("**💵 收益明细：**")
        
        # 预期收益
        return_pct = expected_return * 100
        st.metric(
            "预期收益率", 
            f"{return_pct:.2f}%",
            help="如果预测准确，预计能获得的收益百分比"
        )
        
        # 交易成本
        cost_pct = estimated_cost * 100
        st.metric(
            "交易成本", 
            f"{cost_pct:.3f}%",
            help="包括手续费和滑点的总成本"
        )
        
        # 净收益
        net_pct = (expected_return - estimated_cost) * 100
        profit_color = "🟢" if net_pct > 0 else "🔴"
        st.metric(
            "预计净收益", 
            f"{profit_color} {net_pct:.2f}%",
            help="扣除成本后的实际收益"
        )
    
    def _render_decision_panel(self, data: Dict[str, Any]):
        """显示交易决策建议"""
        st.markdown("#### ⚡ 交易建议")
        st.caption("基于当前分析的操作建议")
        
        decision = data.get('decision', 'none')
        tier = data.get('tier', 'none')
        cooldown_until = data.get('cooldown_until')
        
        # 交易决策显示
        if decision != 'none':
            if tier == 'A':
                st.success("""
                🎯 **强烈建议**
                
                ✅ 高把握交易机会
                
                系统有很高的信心，这是一个值得考虑的交易机会
                """)
            elif tier == 'B':
                st.info("""
                📈 **可考虑**
                
                ⚖️ 中等把握机会
                
                系统认为有一定机会，但建议谨慎评估
                """)
            else:
                st.warning("""
                ⚠️ **需要观察**
                
                ⏸️ 条件未完全满足
                
                建议等待更好的时机或调整参数
                """)
        else:
            st.error("""
            ❌ **暂不建议交易**
            
            🛑 当前条件不满足
            
            市场条件未达到交易标准，建议观望
            """)
        
        # 冷却时间状态
        if cooldown_until:
            cooldown_time = datetime.fromtimestamp(cooldown_until / 1000)
            time_until = cooldown_time - datetime.now()
            
            if time_until.total_seconds() > 0:
                minutes_left = int(time_until.total_seconds()/60)
                st.warning(f"🕐 信号冷却中：还需等待 {minutes_left} 分钟")
                st.caption("为避免过度交易，系统会在发出信号后设置冷却期")
            else:
                st.success("✅ 准备就绪，可以接收新信号")
        else:
            st.success("✅ 准备就绪，可以接收新信号")
        
        # 当前策略参数
        thresholds = data.get('thresholds', {})
        tau = thresholds.get('tau', 0.75)
        kappa = thresholds.get('kappa', 1.20)
        
        tau_pct = int(tau * 100)
        
        st.markdown(f"""
        **📋 当前策略参数：**
        - 🎯 信心度要求：{tau_pct}%
        - 💰 收益倍数要求：{kappa:.1f}倍成本
        """)
        
        st.caption("💡 提示：调整左侧栏的策略类型可改变这些参数")
    
    def _render_feature_importance(self, data: Dict[str, Any]):
        """显示关键影响因素"""
        st.markdown("#### 🔍 关键影响因素")
        st.caption("对预测结果影响最大的5个市场指标")
        
        features_top5 = data.get('features_top5', {})
        
        if not features_top5:
            st.warning("暂无特征数据")
            return
        
        # 创建横向柱状图
        feature_names_raw = list(features_top5.keys())
        feature_values = list(features_top5.values())
        
        # 特征名称中文化
        feature_name_map = {
            'qi_1': '订单簿不平衡度',
            'ofi_10': '订单流向趋势',
            'microprice_dev': '微观价格偏离',
            'rv_ratio': '短期波动比率',
            'depth_slope_bid': '买盘深度斜率',
            'spread': '买卖价差',
            'volume_imbalance': '成交量不平衡',
            'price_momentum': '价格动量',
            'volatility': '波动率'
        }
        
        feature_names = [feature_name_map.get(f, f) for f in feature_names_raw]
        
        # 根据正负值设置颜色
        colors = ['green' if v > 0 else 'red' for v in feature_values]
        
        fig = go.Figure(data=[
            go.Bar(
                y=feature_names,
                x=feature_values,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in feature_values],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="各因素对预测的贡献",
            xaxis_title="影响程度（绿色↑看涨，红色↓看跌）",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 特征说明
        feature_explanations = {
            '订单簿不平衡度': '买单和卖单的力量对比',
            '订单流向趋势': '资金流入流出的趋势',
            '微观价格偏离': '实际成交价与理论价格的差异',
            '短期波动比率': '短期波动与长期波动的比例',
            '买盘深度斜率': '买盘挂单的分布情况'
        }
        
        st.markdown("**📖 因素说明：**")
        for fname in feature_names[:3]:  # 只显示前3个
            explanation = feature_explanations.get(fname, '市场微观结构指标')
            st.caption(f"• **{fname}**：{explanation}")
    
    def _render_quality_panel(self, data: Dict[str, Any]):
        """显示数据质量和系统状态"""
        st.markdown("#### 🚦 系统状态")
        st.caption("数据质量和模型版本信息")
        
        quality_flags = data.get('quality_flags', [])
        model_version = data.get('model_version', 'Unknown')
        feature_version = data.get('feature_version', 'Unknown')
        cost_model = data.get('cost_model', 'Unknown')
        
        # 质量指标
        if not quality_flags:
            st.success("✅ 所有系统正常运行")
        else:
            for flag in quality_flags:
                if 'degraded' in flag or '降级' in flag:
                    st.warning(f"⚠️ {flag.replace('_', ' ').replace('degraded', '性能下降')}")
                elif 'error' in flag or '错误' in flag:
                    st.error(f"❌ {flag.replace('_', ' ').replace('error', '错误')}")
                else:
                    st.info(f"ℹ️ {flag.replace('_', ' ')}")
        
        # 模型版本信息
        st.markdown("**🤖 模型版本：**")
        st.caption(f"• 预测模型：v{model_version}")
        st.caption(f"• 特征引擎：v{feature_version}")
        st.caption(f"• 成本模型：v{cost_model}")
        
        # 数据窗口信息
        data_window_id = data.get('data_window_id', 'Unknown')
        if data_window_id != 'Unknown':
            st.caption(f"• 数据批次：{data_window_id[:16]}...")
        
        # 数据新鲜度
        st.markdown("**⏱️ 数据新鲜度：**")
        timestamp = data.get('timestamp', time.time() * 1000)
        data_age_seconds = (time.time() * 1000 - timestamp) / 1000
        
        if data_age_seconds < 5:
            st.success(f"🟢 实时数据 ({data_age_seconds:.1f}秒前)")
        elif data_age_seconds < 30:
            st.warning(f"🟡 较新数据 ({data_age_seconds:.0f}秒前)")
        else:
            st.error(f"🔴 数据较旧 ({data_age_seconds:.0f}秒前)")
        
        st.caption("💡 数据越新鲜，预测越准确")
