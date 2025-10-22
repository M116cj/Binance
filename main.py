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
    # 策略预设配置（优化后两档模型，2025-10-22）
    STRATEGY_PRESETS = {
        "⭐ A级信号": {
            "theta_up": 0.008,  # 0.8% - 优化后更宽覆盖范围
            "theta_dn": 0.0056,  # 0.56% (70% of theta_up)
            "tau_threshold": 0.75,  # p_up > 75% - 高质量信号
            "kappa_threshold": 1.20,  # 收益 > 1.2倍成本
            "description": "高质量信号，严格筛选",
            "icon": "⭐"
        },
        "🎯 B级信号": {
            "theta_up": 0.008,  # 0.8% - 与A级相同
            "theta_dn": 0.0056,  # 0.56%
            "tau_threshold": 0.70,  # p_up > 70% - 标准质量
            "kappa_threshold": 1.10,  # 收益 > 1.1倍成本
            "description": "标准质量信号，平衡频率",
            "icon": "🎯"
        }
    }
    
    def __init__(self):
        self.client = httpx.Client(timeout=30.0)
        self.signal_card = SignalCard()
        self.regime_state = RegimeState()
        self.probability_window = ProbabilityWindow()
        self.backtest_performance = BacktestPerformance()
        self.calibration_analysis = CalibrationAnalysis()
        self.attribution_comparison = AttributionComparison()
        self.admin_panel = AdminPanel()
        self.signal_history = SignalHistory()
        self.monitoring_dashboard = MonitoringDashboard()
        
    def initialize_session_state(self):
        """初始化Streamlit会话状态变量（优化后参数，2025-10-22）"""
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = 'BTCUSDT'
        if 'current_strategy' not in st.session_state:
            st.session_state.current_strategy = "⭐ A级信号"  # 默认A级信号
        if 'theta_up' not in st.session_state:
            st.session_state.theta_up = 0.008  # 0.8% (优化后)
        if 'theta_dn' not in st.session_state:
            st.session_state.theta_dn = 0.0056  # 0.56% (70% of theta_up)
        if 'horizon_minutes' not in st.session_state:
            st.session_state.horizon_minutes = [10, 20]  # 优化为2个窗口
        if 'tau_threshold' not in st.session_state:
            st.session_state.tau_threshold = 0.75  # A级默认
        if 'kappa_threshold' not in st.session_state:
            st.session_state.kappa_threshold = 1.20  # A级默认
        if 'auto_mode' not in st.session_state:
            st.session_state.auto_mode = True
        if 'last_update' not in st.session_state:
            st.session_state.last_update = time.time()
        if 'available_symbols' not in st.session_state:
            st.session_state.available_symbols = None
    
    def apply_strategy_preset(self, strategy_name: str):
        """应用策略预设到session state"""
        if strategy_name in self.STRATEGY_PRESETS:
            preset = self.STRATEGY_PRESETS[strategy_name]
            st.session_state.theta_up = preset['theta_up']
            st.session_state.theta_dn = preset['theta_dn']
            st.session_state.tau_threshold = preset['tau_threshold']
            st.session_state.kappa_threshold = preset['kappa_threshold']
            st.session_state.current_strategy = strategy_name
    
    def detect_current_strategy(self) -> str:
        """检测当前参数匹配哪个预设策略，如果不匹配则返回"自定义"
        
        使用容差比较（0.0001）来处理浮点数精度问题
        """
        tolerance = 0.0001
        
        for strategy_name, preset in self.STRATEGY_PRESETS.items():
            if (abs(st.session_state.theta_up - preset['theta_up']) < tolerance and
                abs(st.session_state.theta_dn - preset['theta_dn']) < tolerance and
                abs(st.session_state.tau_threshold - preset['tau_threshold']) < tolerance and
                abs(st.session_state.kappa_threshold - preset['kappa_threshold']) < tolerance):
                return strategy_name
        
        return "🔧 自定义"
            
    def fetch_data(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
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
        """渲染控制侧边栏（iOS风格）"""
        # iOS风格标题
        st.sidebar.markdown("""
        <div style='text-align: center; padding: 16px 0 8px 0;'>
            <h1 style='font-size: 28px; margin: 0; font-weight: 700;'>🚀</h1>
            <h2 style='font-size: 20px; margin: 8px 0 4px 0; font-weight: 600;'>加密货币预测</h2>
            <p style='font-size: 13px; color: #8E8E93; margin: 0;'>智能涨跌预测系统</p>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.markdown("")
        
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
        
        # 策略快速切换（iOS风格）
        st.sidebar.markdown("""
        <div style='background-color: #F9F9F9; padding: 12px; border-radius: 10px; margin: 16px 0;'>
            <p style='font-size: 14px; font-weight: 600; margin: 0 0 8px 0; color: #000000;'>🎯 信号等级</p>
            <p style='font-size: 12px; color: #8E8E93; margin: 0;'>选择预测精度级别</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("⭐ A级", use_container_width=True, help="高质量信号，严格筛选", key="btn_a"):
                self.apply_strategy_preset("⭐ A级信号")
                st.rerun()
        
        with col2:
            if st.button("🎯 B级", use_container_width=True, help="标准质量，平衡频率", key="btn_b"):
                self.apply_strategy_preset("🎯 B级信号")
                st.rerun()
        
        # 动态检测并显示当前策略（iOS风格）
        detected_strategy = self.detect_current_strategy()
        
        if detected_strategy == "🔧 自定义":
            st.sidebar.markdown(f"""
            <div style='background-color: #FFF3CD; padding: 12px; border-radius: 10px; margin: 8px 0; border-left: 3px solid #FF9500;'>
                <p style='font-size: 13px; font-weight: 600; margin: 0 0 4px 0; color: #000000;'>当前策略：{detected_strategy}</p>
                <p style='font-size: 12px; color: #8E8E93; margin: 0;'>参数已手动调整</p>
                <p style='font-size: 11px; color: #8E8E93; margin: 4px 0 0 0;'>💡 点击上方按钮恢复预设</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            current_preset = self.STRATEGY_PRESETS[detected_strategy]
            st.sidebar.markdown(f"""
            <div style='background-color: #E5F2FF; padding: 12px; border-radius: 10px; margin: 8px 0; border-left: 3px solid #007AFF;'>
                <p style='font-size: 13px; font-weight: 600; margin: 0 0 4px 0; color: #000000;'>当前策略：{detected_strategy}</p>
                <p style='font-size: 12px; color: #8E8E93; margin: 0;'>{current_preset['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # 显示完整参数（iOS风格卡片）
        st.sidebar.markdown("""
        <div style='background-color: #F9F9F9; padding: 12px; border-radius: 10px; margin: 16px 0 8px 0;'>
            <p style='font-size: 14px; font-weight: 600; margin: 0; color: #000000;'>📊 策略参数</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            st.metric("📈 上涨线", f"{st.session_state.theta_up*100:.2f}%")
            st.metric("🎯 信心度", f"{int(st.session_state.tau_threshold*100)}%")
        with col_b:
            st.metric("📉 下跌线", f"{st.session_state.theta_dn*100:.2f}%")
            st.metric("💰 收益比", f"{st.session_state.kappa_threshold:.1f}x")
        
        # 高级参数微调（可展开）
        with st.sidebar.expander("🔧 高级参数微调", expanded=False):
            st.caption("手动调整策略参数（专业用户）")
            
            st.session_state.theta_up = st.number_input(
                "📈 上涨判定线 (%)", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.theta_up * 100,
                step=0.05,
                format="%.2f",
                help="价格上涨多少才算真正上涨",
                key="theta_up_input"
            ) / 100
            
            st.session_state.theta_dn = st.number_input(
                "📉 下跌判定线 (%)", 
                min_value=0.1, 
                max_value=1.5, 
                value=st.session_state.theta_dn * 100,
                step=0.05,
                format="%.2f",
                help="价格下跌多少才算真正下跌",
                key="theta_dn_input"
            ) / 100
            
            st.session_state.tau_threshold = st.slider(
                "🎯 信心度阈值",
                min_value=0.5,
                max_value=0.9,
                value=st.session_state.tau_threshold,
                step=0.05,
                help="预测概率至少要达到这个值",
                key="tau_input"
            )
            
            st.session_state.kappa_threshold = st.slider(
                "💰 收益成本比阈值",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.kappa_threshold,
                step=0.1,
                help="预期收益与成本的比例",
                key="kappa_input"
            )
            
            st.warning("⚠️ 修改参数后会覆盖策略预设")
        
        # 自动刷新（iOS风格）
        st.sidebar.markdown("""
        <div style='background-color: #F9F9F9; padding: 12px; border-radius: 10px; margin: 16px 0;'>
            <p style='font-size: 14px; font-weight: 600; margin: 0; color: #000000;'>🔄 数据更新</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.session_state.auto_mode = st.sidebar.checkbox(
            "自动刷新", 
            st.session_state.auto_mode,
            help="开启后每秒自动更新数据"
        )
        
        if st.sidebar.button("🔄 立即刷新", use_container_width=True, key="btn_refresh"):
            st.session_state.last_update = time.time()
            st.rerun()
            
        # 系统状态（iOS风格卡片）
        st.sidebar.markdown("""
        <div style='background-color: #F9F9F9; padding: 12px; border-radius: 10px; margin: 16px 0;'>
            <p style='font-size: 14px; font-weight: 600; margin: 0; color: #000000;'>💡 系统状态</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 健康检查
        health_data = self.fetch_data("health")
        if health_data:
            if health_data.get("status") == "healthy":
                st.sidebar.markdown("""
                <div style='background-color: #D1F4E0; padding: 10px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #34C759;'>
                    <p style='font-size: 12px; font-weight: 500; margin: 0; color: #000000;'>✅ 系统运行正常</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.markdown("""
                <div style='background-color: #FFF3CD; padding: 10px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #FF9500;'>
                    <p style='font-size: 12px; font-weight: 500; margin: 0; color: #000000;'>⚠️ 系统性能下降</p>
                </div>
                """, unsafe_allow_html=True)
                
            exchange_lag = health_data.get("exchange_lag_s", 0)
            if exchange_lag < 2:
                st.sidebar.markdown(f"""
                <div style='background-color: #E5F2FF; padding: 10px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #007AFF;'>
                    <p style='font-size: 12px; margin: 0; color: #000000;'>📡 数据延迟: {exchange_lag:.1f}秒</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"""
                <div style='background-color: #FFE4E1; padding: 10px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #FF3B30;'>
                    <p style='font-size: 12px; font-weight: 500; margin: 0; color: #000000;'>📡 数据延迟较高: {exchange_lag:.1f}秒</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.sidebar.markdown("""
            <div style='background-color: #FFE4E1; padding: 10px; border-radius: 8px; margin: 4px 0; border-left: 3px solid #FF3B30;'>
                <p style='font-size: 12px; font-weight: 500; margin: 0; color: #000000;'>❌ 后台服务未连接</p>
            </div>
            """, unsafe_allow_html=True)
    
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
        
        # 应用iOS风格自定义CSS
        st.markdown("""
        <style>
        /* iOS风格全局样式 */
        .main {
            background-color: #F2F2F7;
        }
        
        /* iOS风格卡片 */
        .stMarkdown, div[data-testid="stVerticalBlock"] > div {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            margin-bottom: 12px;
        }
        
        /* iOS风格按钮 */
        .stButton > button {
            background-color: #007AFF;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            font-weight: 500;
            box-shadow: 0 2px 8px rgba(0, 122, 255, 0.2);
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background-color: #0051D5;
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
            transform: translateY(-1px);
        }
        
        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 1px 4px rgba(0, 122, 255, 0.2);
        }
        
        /* iOS风格标签页 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F2F2F7;
            border-radius: 12px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border-radius: 8px;
            color: #8E8E93;
            font-weight: 500;
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF;
            color: #007AFF;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        }
        
        /* iOS风格度量卡片 */
        [data-testid="stMetricValue"] {
            font-size: 24px;
            font-weight: 600;
            color: #000000;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 13px;
            color: #8E8E93;
            font-weight: 400;
        }
        
        /* iOS风格侧边栏 */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E5E5EA;
        }
        
        [data-testid="stSidebar"] .stButton > button {
            background: linear-gradient(180deg, #FFFFFF 0%, #F9F9F9 100%);
            color: #007AFF;
            border: 1px solid #D1D1D6;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
        }
        
        [data-testid="stSidebar"] .stButton > button:hover {
            background: linear-gradient(180deg, #F9F9F9 0%, #F2F2F2 100%);
            border-color: #007AFF;
        }
        
        /* iOS风格输入框 */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div {
            border: 1px solid #D1D1D6;
            border-radius: 10px;
            background-color: #FFFFFF;
            color: #000000;
            padding: 8px 12px;
        }
        
        /* iOS风格滑块 */
        .stSlider > div > div > div {
            background-color: #007AFF;
        }
        
        /* iOS风格信息框 */
        .stAlert {
            border-radius: 12px;
            border: none;
            padding: 12px 16px;
        }
        
        [data-baseweb="notification"] {
            border-radius: 12px;
        }
        
        /* iOS风格成功提示 */
        .element-container:has(> .stSuccess) {
            background-color: #D1F4E0;
            border-radius: 10px;
            padding: 8px;
        }
        
        /* iOS风格警告提示 */
        .element-container:has(> .stWarning) {
            background-color: #FFF3CD;
            border-radius: 10px;
            padding: 8px;
        }
        
        /* iOS风格错误提示 */
        .element-container:has(> .stError) {
            background-color: #FFE4E1;
            border-radius: 10px;
            padding: 8px;
        }
        
        /* iOS风格信息提示 */
        .element-container:has(> .stInfo) {
            background-color: #E5F2FF;
            border-radius: 10px;
            padding: 8px;
        }
        
        /* 标题样式优化 */
        h1, h2, h3 {
            color: #000000;
            font-weight: 600;
        }
        
        h1 {
            font-size: 32px;
            margin-bottom: 8px;
        }
        
        h2 {
            font-size: 24px;
            margin-bottom: 6px;
        }
        
        h3 {
            font-size: 18px;
            margin-bottom: 4px;
        }
        
        /* 分割线优化 */
        hr {
            border: none;
            height: 1px;
            background-color: #E5E5EA;
            margin: 16px 0;
        }
        
        /* iOS风格表格 */
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid #E5E5EA;
        }
        
        /* Plotly图表容器 */
        .js-plotly-plot {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* 移除不必要的边距 */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* 优化展开器样式 */
        .streamlit-expanderHeader {
            background-color: #F9F9F9;
            border-radius: 10px;
            font-weight: 500;
        }
        
        /* 优化选择框 */
        [data-baseweb="select"] {
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
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
            self.render_backtest_performance()
            
        with tabs[4]:
            self.render_calibration_analysis()
            
        with tabs[5]:
            self.render_attribution_comparison()
        
        with tabs[6]:
            self.render_signal_history()
        
        with tabs[7]:
            self.render_monitoring_dashboard()
        
        with tabs[8]:
            self.render_admin_panel()

def main():
    dashboard = CryptoSurgePredictionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
