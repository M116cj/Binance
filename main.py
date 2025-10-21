import streamlit as st
import asyncio
import httpx
import time
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from typing import Dict, List, Any, Optional

# 导入前端组件
from frontend.components.signal_card import SignalCard
from frontend.components.regime_state import RegimeState
from frontend.components.probability_window import ProbabilityWindow
from frontend.components.cost_capacity import CostCapacity
from frontend.components.backtest_performance import BacktestPerformance
from frontend.components.calibration_analysis import CalibrationAnalysis
from frontend.components.attribution_comparison import AttributionComparison
from frontend.components.admin_panel import AdminPanel
from frontend.components.signal_history import SignalHistory
from frontend.components.monitoring_dashboard import MonitoringDashboard

# 配置
BACKEND_HOST = os.getenv("BACKEND_HOST", "localhost")
BACKEND_PORT = os.getenv("BACKEND_PORT", "8000")
BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

class CryptoSurgePredictionDashboard:
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
        self.signal_card = SignalCard()
        self.regime_state = RegimeState()
        self.probability_window = ProbabilityWindow()
        self.cost_capacity = CostCapacity()
        self.backtest_performance = BacktestPerformance()
        self.calibration_analysis = CalibrationAnalysis()
        self.attribution_comparison = AttributionComparison()
        self.admin_panel = AdminPanel()
        self.signal_history = SignalHistory()
        self.monitoring_dashboard = MonitoringDashboard()
        
    def initialize_session_state(self):
        """初始化Streamlit会话状态变量"""
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = 'BTCUSDT'
        if 'theta_up' not in st.session_state:
            st.session_state.theta_up = 0.006  # 0.6%
        if 'theta_dn' not in st.session_state:
            st.session_state.theta_dn = 0.004  # 0.4%
        if 'horizon_minutes' not in st.session_state:
            st.session_state.horizon_minutes = [5, 10, 30]
        if 'tau_threshold' not in st.session_state:
            st.session_state.tau_threshold = 0.75
        if 'kappa_threshold' not in st.session_state:
            st.session_state.kappa_threshold = 1.20
        if 'auto_mode' not in st.session_state:
            st.session_state.auto_mode = True
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'available_symbols' not in st.session_state:
            st.session_state.available_symbols = None
            
    def fetch_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """从后端API获取数据，带错误处理"""
        try:
            response = self.client.get(f"{BASE_URL}/{endpoint}", params=params or {})
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            st.error(f"Network error connecting to backend: {str(e)}")
            return None
        except httpx.HTTPStatusError as e:
            st.error(f"API error {e.response.status_code}: {e.response.text}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None
    
    def load_available_symbols(self) -> List[Dict]:
        """从后端加载所有可用的交易对"""
        if st.session_state.available_symbols is not None:
            return st.session_state.available_symbols
        
        try:
            data = self.fetch_data("symbols")
            if data and 'symbols' in data:
                st.session_state.available_symbols = data['symbols']
                return data['symbols']
        except Exception as e:
            st.warning(f"无法加载交易对列表: {e}")
        
        fallback = [
            {'symbol': 'BTCUSDT', 'baseAsset': 'BTC', 'name': '比特币', 'displayName': '比特币 (BTC)'},
            {'symbol': 'ETHUSDT', 'baseAsset': 'ETH', 'name': '以太坊', 'displayName': '以太坊 (ETH)'},
            {'symbol': 'BNBUSDT', 'baseAsset': 'BNB', 'name': '币安币', 'displayName': '币安币 (BNB)'},
        ]
        st.session_state.available_symbols = fallback
        return fallback
    
    def render_sidebar(self):
        """渲染控制侧边栏"""
        st.sidebar.title("🚀 加密货币涨跌预测系统")
        st.sidebar.markdown("实时监控币价，智能预测涨跌")
        st.sidebar.markdown("---")
        
        # 交易对选择 - 从后端动态加载
        available_symbols = self.load_available_symbols()
        
        if not available_symbols:
            st.sidebar.error("⚠️ 无法加载交易对列表")
            return
        
        symbol_options = [s['displayName'] for s in available_symbols]
        symbol_map = {s['displayName']: s['symbol'] for s in available_symbols}
        
        current_symbol = st.session_state.selected_symbol
        current_display = next(
            (s['displayName'] for s in available_symbols if s['symbol'] == current_symbol),
            available_symbols[0]['displayName']
        )
        
        try:
            default_index = symbol_options.index(current_display)
        except ValueError:
            default_index = 0
        
        selected_display = st.sidebar.selectbox(
            "📊 选择交易对",
            symbol_options,
            index=default_index,
            help=f"从币安{len(available_symbols)}个USDT交易对中选择"
        )
        
        st.session_state.selected_symbol = symbol_map[selected_display]
        
        st.sidebar.markdown("### ⚙️ 交易参数设置")
        st.sidebar.caption("设置涨跌幅度的判断标准")
        
        # 标记参数
        st.session_state.theta_up = st.sidebar.number_input(
            "📈 上涨判定线 (%)", 
            min_value=0.1, 
            max_value=2.0, 
            value=st.session_state.theta_up * 100,
            step=0.1,
            format="%.1f",
            help="价格上涨多少才算是\"涨\"？例如：0.6% 表示价格上涨0.6%以上才算真正上涨"
        ) / 100
        
        st.session_state.theta_dn = st.sidebar.number_input(
            "📉 下跌判定线 (%)", 
            min_value=0.1, 
            max_value=1.5, 
            value=st.session_state.theta_dn * 100,
            step=0.1,
            format="%.1f",
            help="价格下跌多少才算是\"跌\"？例如：0.4% 表示价格下跌0.4%以上才算真正下跌"
        ) / 100
        
        # 决策阈值
        st.sidebar.markdown("### 🎯 交易策略选择")
        st.sidebar.caption("选择你的风险偏好")
        
        tier = st.sidebar.radio(
            "策略类型", 
            ["🛡️ 保守型", "⚖️ 平衡型", "🔥 激进型"],
            help="保守型：高确定性但机会少 | 平衡型：兼顾收益和风险 | 激进型：更多机会但风险大"
        )
        
        if tier == "🛡️ 保守型":
            st.session_state.tau_threshold = 0.75
            st.session_state.kappa_threshold = 1.20
            st.sidebar.info("📊 保守策略：只在高把握时交易，安全第一")
        elif tier == "⚖️ 平衡型":
            st.session_state.tau_threshold = 0.65
            st.session_state.kappa_threshold = 1.00
            st.sidebar.info("📊 平衡策略：追求收益与风险的平衡")
        else:  # 激进型
            st.session_state.tau_threshold = 0.55
            st.session_state.kappa_threshold = 0.80
            st.sidebar.warning("📊 激进策略：更多交易机会，但风险较高")
        
        # 显示当前阈值（用简单语言）
        confidence_pct = int(st.session_state.tau_threshold * 100)
        st.sidebar.metric("信心度要求", f"{confidence_pct}%", 
                         help="只有当系统有这么高的把握时才会给出信号")
        st.sidebar.metric("收益要求", f"{st.session_state.kappa_threshold:.1f}倍成本",
                         help="预期收益至少要是交易成本的这么多倍")
        
        # 自动刷新
        st.sidebar.markdown("---")
        st.session_state.auto_mode = st.sidebar.checkbox(
            "🔄 自动刷新数据", 
            st.session_state.auto_mode,
            help="开启后每秒自动更新数据"
        )
        
        if st.sidebar.button("🔄 立即刷新", use_container_width=True):
            st.session_state.last_update = time.time()
            st.rerun()
            
        # 系统状态
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 系统状态")
        
        # 健康检查
        health_data = self.fetch_data("health")
        if health_data:
            if health_data.get("status") == "healthy":
                st.sidebar.success("✅ 系统运行正常")
            else:
                st.sidebar.warning("⚠️ 系统性能下降")
                
            exchange_lag = health_data.get("exchange_lag_s", 0)
            if exchange_lag < 2:
                st.sidebar.info(f"📡 数据延迟：{exchange_lag:.1f}秒")
            else:
                st.sidebar.error(f"📡 数据延迟较高：{exchange_lag:.1f}秒")
        else:
            st.sidebar.error("❌ 后台服务未连接")
    
    def render_realtime_signal_card(self):
        """报告1：实时信号卡片"""
        st.markdown("## 📡 实时交易信号")
        st.caption("当前最新的买卖建议和市场数据")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn,
            'tau': st.session_state.tau_threshold,
            'kappa': st.session_state.kappa_threshold
        }
        
        data = self.fetch_data("reports/realtime", params)
        if data:
            self.signal_card.render(data)
        else:
            st.error("❌ 无法加载实时信号数据，请检查后台服务")
    
    def render_regime_state(self):
        """报告2：市场状态与流动性"""
        st.markdown("## 🌊 市场状态分析")
        st.caption("当前市场的波动性和交易活跃度")
        
        params = {'symbol': st.session_state.selected_symbol}
        data = self.fetch_data("reports/regime", params)
        if data:
            self.regime_state.render(data)
        else:
            st.error("❌ 无法加载市场状态数据")
    
    def render_probability_window(self):
        """报告3：预测概率与时间窗口"""
        st.markdown("## 📈 涨跌概率分析")
        st.caption("未来不同时间段的价格上涨可能性")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn
        }
        
        data = self.fetch_data("reports/window", params)
        if data:
            self.probability_window.render(data, st.session_state.tau_threshold, st.session_state.kappa_threshold)
        else:
            st.error("❌ 无法加载概率分析数据")
    
    def render_cost_capacity(self):
        """报告4：执行成本与容量"""
        st.markdown("## 💰 交易成本分析")
        st.caption("不同交易金额的手续费和滑点成本")
        
        params = {'symbol': st.session_state.selected_symbol}
        data = self.fetch_data("reports/cost", params)
        if data:
            self.cost_capacity.render(data)
        else:
            st.error("❌ 无法加载成本分析数据")
    
    def render_backtest_performance(self):
        """报告5：历史回测性能"""
        st.markdown("## 📊 历史表现回顾")
        st.caption("过去30天的策略收益和胜率统计")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn,
            'tau': st.session_state.tau_threshold,
            'kappa': st.session_state.kappa_threshold,
            'days_back': 30
        }
        
        data = self.fetch_data("reports/backtest", params)
        if data:
            self.backtest_performance.render(data)
        else:
            st.error("❌ 无法加载历史表现数据")
    
    def render_calibration_analysis(self):
        """报告6：校准与误差分析"""
        st.markdown("## 🎯 预测准确度分析")
        st.caption("系统预测的可靠性和准确性评估")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn
        }
        
        data = self.fetch_data("reports/calibration", params)
        if data:
            self.calibration_analysis.render(data)
        else:
            st.error("❌ 无法加载准确度分析数据")
    
    def render_attribution_comparison(self):
        """报告7：事件归因与策略对比"""
        st.markdown("## 🔍 影响因素分析")
        st.caption("哪些市场指标对预测影响最大")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn,
            'tau': st.session_state.tau_threshold,
            'kappa': st.session_state.kappa_threshold
        }
        
        data = self.fetch_data("reports/attribution", params)
        if data:
            self.attribution_comparison.render(data)
        else:
            st.error("❌ 无法加载影响因素数据")
    
    def render_admin_panel(self):
        """管理面板：模型管理和系统配置"""
        st.markdown("## ⚙️ 系统管理")
        st.caption("模型版本管理和参数配置")
        
        # 获取模型数据
        models_data = self.fetch_data("models")
        
        # 获取信号统计数据
        signals_stats = self.fetch_data("signals/stats")
        
        # 渲染管理面板
        self.admin_panel.render(models_data, signals_stats)
    
    def render_signal_history(self):
        """信号历史视图：显示过往预测"""
        st.markdown("## 📜 历史信号记录")
        st.caption("查看过往所有的交易信号和结果")
        
        # 从组件获取过滤值
        # 这些值将由组件的render方法设置
        # 我们需要根据过滤器获取数据，所以传递fetch函数
        self.signal_history.render(self.fetch_data)
    
    def render_monitoring_dashboard(self):
        """监控仪表板：SLA和质量指标"""
        st.markdown("## 📊 系统监控")
        st.caption("实时监控系统性能和数据质量")
        
        # 将fetch函数传递给监控仪表板
        self.monitoring_dashboard.render(self.fetch_data)
    
    def run(self):
        """主应用程序运行器"""
        st.set_page_config(
            page_title="加密货币涨跌预测系统",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.initialize_session_state()
        self.render_sidebar()
        
        # 自动刷新逻辑
        if st.session_state.auto_mode:
            if time.time() - st.session_state.last_update > 1.0:  # 1秒刷新
                st.session_state.last_update = time.time()
                st.rerun()
        
        # 主内容标签页
        tabs = st.tabs([
            "📡 实时信号", 
            "🌊 市场状态", 
            "📈 概率分析",
            "💰 成本分析",
            "📊 历史表现",
            "🎯 准确度",
            "🔍 影响因素",
            "📜 历史记录",
            "📊 系统监控",
            "⚙️ 系统管理"
        ])
        
        with tabs[0]:
            self.render_realtime_signal_card()
            
        with tabs[1]:
            self.render_regime_state()
            
        with tabs[2]:
            self.render_probability_window()
            
        with tabs[3]:
            self.render_cost_capacity()
            
        with tabs[4]:
            self.render_backtest_performance()
            
        with tabs[5]:
            self.render_calibration_analysis()
            
        with tabs[6]:
            self.render_attribution_comparison()
        
        with tabs[7]:
            self.render_signal_history()
        
        with tabs[8]:
            self.render_monitoring_dashboard()
        
        with tabs[9]:
            self.render_admin_panel()

def main():
    dashboard = CryptoSurgePredictionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
