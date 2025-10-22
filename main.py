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
        if 'dark_mode' not in st.session_state:
            st.session_state.dark_mode = True  # 默认TradingView深色模式
    
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
        """渲染控制侧边栏（TradingView风格）"""
        # TradingView风格标题
        title_color = "#D1D4DC" if st.session_state.dark_mode else "#000000"
        st.sidebar.markdown(f"""
        <div style='text-align: center; padding: 20px 0 12px 0;'>
            <h1 style='font-size: 32px; margin: 0; font-weight: 700;'>📊</h1>
            <h2 style='font-size: 22px; margin: 12px 0 0 0; font-weight: 600; letter-spacing: -0.5px; color: {title_color};'>加密货币预测</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.divider()
        
        # 交易对选择 - 精简标签
        available_symbols = self.load_available_symbols()
        
        if not available_symbols:
            st.sidebar.error("⚠️ 连接失败")
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
            "交易对",
            symbol_options,
            index=default_index,
            label_visibility="collapsed"
        )
        
        st.session_state.selected_symbol = symbol_map[selected_display]
        
        st.sidebar.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # 策略快速切换 - 极简设计
        st.sidebar.markdown("""
        <div style='text-align: center; margin: 8px 0 12px 0;'>
            <p style='font-size: 13px; font-weight: 500; margin: 0; color: #8E8E93;'>信号等级</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("⭐ A级", use_container_width=True, key="btn_a"):
                self.apply_strategy_preset("⭐ A级信号")
                st.rerun()
        
        with col2:
            if st.button("🎯 B级", use_container_width=True, key="btn_b"):
                self.apply_strategy_preset("🎯 B级信号")
                st.rerun()
        
        # 当前策略 - 仅显示名称
        detected_strategy = self.detect_current_strategy()
        bg_color = "#2A2E39" if st.session_state.dark_mode else "#F9F9F9"
        text_color = "#D1D4DC" if st.session_state.dark_mode else "#000000"
        st.sidebar.markdown(f"""
        <div style='text-align: center; padding: 8px 12px; background-color: {bg_color}; border-radius: 6px; margin: 12px 0; border: 1px solid {"#363A45" if st.session_state.dark_mode else "transparent"};'>
            <p style='font-size: 14px; font-weight: 600; margin: 0; color: {text_color};'>{detected_strategy}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 核心参数 - 紧凑显示
        col_a, col_b = st.sidebar.columns(2)
        with col_a:
            st.metric("上涨", f"{st.session_state.theta_up*100:.1f}%", label_visibility="visible")
            st.metric("信心", f"{int(st.session_state.tau_threshold*100)}%", label_visibility="visible")
        with col_b:
            st.metric("下跌", f"{st.session_state.theta_dn*100:.1f}%", label_visibility="visible")
            st.metric("收益", f"{st.session_state.kappa_threshold:.1f}x", label_visibility="visible")
        
        # 高级参数微调
        with st.sidebar.expander("⚙️ 高级设置"):
            st.session_state.theta_up = st.number_input(
                "上涨线 (%)", 
                min_value=0.1, 
                max_value=2.0, 
                value=st.session_state.theta_up * 100,
                step=0.05,
                format="%.2f",
                key="theta_up_input"
            ) / 100
            
            st.session_state.theta_dn = st.number_input(
                "下跌线 (%)", 
                min_value=0.1, 
                max_value=1.5, 
                value=st.session_state.theta_dn * 100,
                step=0.05,
                format="%.2f",
                key="theta_dn_input"
            ) / 100
            
            st.session_state.tau_threshold = st.slider(
                "信心度",
                min_value=0.5,
                max_value=0.9,
                value=st.session_state.tau_threshold,
                step=0.05,
                key="tau_input"
            )
            
            st.session_state.kappa_threshold = st.slider(
                "收益比",
                min_value=0.5,
                max_value=2.0,
                value=st.session_state.kappa_threshold,
                step=0.1,
                key="kappa_input"
            )
        
        st.sidebar.divider()
        
        # 主题和刷新控制
        col_theme, col_auto, col_btn = st.sidebar.columns([1, 1, 1])
        with col_theme:
            if st.button("🌙" if not st.session_state.dark_mode else "☀️", 
                        use_container_width=True, 
                        help="切换深色/浅色模式",
                        key="btn_theme"):
                st.session_state.dark_mode = not st.session_state.dark_mode
                st.rerun()
        with col_auto:
            st.session_state.auto_mode = st.checkbox(
                "自动", 
                st.session_state.auto_mode,
                help="自动刷新数据"
            )
        with col_btn:
            if st.button("🔄", use_container_width=True, key="btn_refresh", help="立即刷新"):
                st.session_state.last_update = time.time()
                st.rerun()
            
        # 系统状态 - TradingView风格
        health_data = self.fetch_data("health")
        if health_data:
            status = health_data.get("status")
            lag = health_data.get("exchange_lag_s", 0)
            
            if status == "healthy" and lag < 2:
                status_color = "#26A69A"  # TradingView绿色
                status_icon = "●"
                status_text = "正常"
            elif lag >= 2:
                status_color = "#FF9800"  # TradingView橙色
                status_icon = "●"
                status_text = f"延迟{lag:.1f}s"
            else:
                status_color = "#F23645"  # TradingView红色
                status_icon = "●"
                status_text = "异常"
        else:
            status_color = "#F23645"
            status_icon = "●"
            status_text = "离线"
        
        status_bg = "#2A2E39" if st.session_state.dark_mode else "#F9F9F9"
        status_text_color = "#D1D4DC" if st.session_state.dark_mode else "#000000"
        status_border = "#363A45" if st.session_state.dark_mode else "transparent"
        
        st.sidebar.markdown(f"""
        <div style='text-align: center; padding: 10px; background-color: {status_bg}; border-radius: 6px; margin: 8px 0; border: 1px solid {status_border};'>
            <p style='font-size: 13px; margin: 0; color: {status_text_color};'>
                <span style='color: {status_color}; font-size: 16px;'>{status_icon}</span> 
                系统{status_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_realtime_signal_card(self):
        """报告1：实时信号卡片（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>📡 实时交易信号</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 14px;'>当前最新的买卖建议和市场数据</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        """报告2：市场状态与流动性（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>🌊 市场状态分析</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 14px;'>当前市场的波动性和交易活跃度</p>
        </div>
        """, unsafe_allow_html=True)
        
        params = {'symbol': st.session_state.selected_symbol}
        data = self.fetch_data("reports/regime", params)
        if data:
            self.regime_state.render(data)
        else:
            st.error("❌ 无法加载市场状态数据")
    
    def render_probability_window(self):
        """报告3：预测概率与时间窗口（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>📈 涨跌概率分析</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 14px;'>未来不同时间段的价格上涨可能性</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        """报告5：历史回测性能（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>📊 历史表现回顾</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 14px;'>过去30天的策略收益和胜率统计</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        """报告6：校准与误差分析（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>🎯 预测准确度分析</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 14px;'>系统预测的可靠性和准确性评估</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        """报告7：事件归因与策略对比（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: white; margin: 0; font-size: 24px; font-weight: 700;'>🔍 影响因素分析</h2>
            <p style='color: rgba(255,255,255,0.9); margin: 4px 0 0 0; font-size: 14px;'>哪些市场指标对预测影响最大</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        """管理面板：模型管理和系统配置（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: #000; margin: 0; font-size: 24px; font-weight: 700;'>⚙️ 系统管理</h2>
            <p style='color: rgba(0,0,0,0.7); margin: 4px 0 0 0; font-size: 14px;'>模型版本管理和参数配置</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 获取模型数据
        models_data = self.fetch_data("models")
        
        # 获取信号统计数据
        signals_stats = self.fetch_data("signals/stats")
        
        # 渲染管理面板
        self.admin_panel.render(models_data, signals_stats)
    
    def render_signal_history(self):
        """信号历史视图：显示过往预测（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: #000; margin: 0; font-size: 24px; font-weight: 700;'>📜 历史信号记录</h2>
            <p style='color: rgba(0,0,0,0.7); margin: 4px 0 0 0; font-size: 14px;'>查看过往所有的交易信号和结果</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 从组件获取过滤值
        # 这些值将由组件的render方法设置
        # 我们需要根据过滤器获取数据，所以传递fetch函数
        self.signal_history.render(self.fetch_data)
    
    def render_monitoring_dashboard(self):
        """监控仪表板：SLA和质量指标（iOS风格）"""
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 20px; border-radius: 12px; margin-bottom: 20px;'>
            <h2 style='color: #000; margin: 0; font-size: 24px; font-weight: 700;'>📊 系统监控</h2>
            <p style='color: rgba(0,0,0,0.7); margin: 4px 0 0 0; font-size: 14px;'>实时监控系统性能和数据质量</p>
        </div>
        """, unsafe_allow_html=True)
        
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
        /* iOS字体系统 - San Francisco Pro */
        * {
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", 
                         "Helvetica Neue", Arial, "Noto Sans", sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* iOS颜色规范 */
        :root {
            --ios-blue: #007AFF;
            --ios-green: #34C759;
            --ios-orange: #FF9500;
            --ios-red: #FF3B30;
            --ios-gray: #8E8E93;
            --ios-gray-light: #D1D1D6;
            --ios-gray-bg: #F2F2F7;
            --ios-white: #FFFFFF;
            --ios-black: #000000;
        }
        
        /* iOS字号规范 */
        h1 { font-size: 28px; font-weight: 700; letter-spacing: -0.5px; }
        h2 { font-size: 22px; font-weight: 600; letter-spacing: -0.4px; }
        h3 { font-size: 20px; font-weight: 600; }
        h4 { font-size: 17px; font-weight: 600; }
        p { font-size: 15px; font-weight: 400; line-height: 1.5; }
        small { font-size: 13px; font-weight: 400; }
        
        /* iOS风格全局样式 */
        .main {
            background-color: var(--ios-gray-bg);
        }
        
        /* iOS风格内容区 - 主容器 */
        .block-container {
            background-color: #F2F2F7 !important;
        }
        
        /* iOS风格列容器 */
        [data-testid="column"] {
            background-color: transparent;
        }
        
        /* Streamlit组件基础样式 */
        .stMarkdown,
        [data-testid="stMarkdownContainer"] {
            background-color: #FFFFFF;
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            margin: 8px 0;
        }
        
        /* iOS间距系统 (4px网格) */
        .block-container {
            padding: 16px 24px;
        }
        
        /* iOS风格按钮 - 触控优先 (最小44x44px) */
        .stButton > button {
            background-color: var(--ios-blue);
            color: white;
            border: none;
            border-radius: 12px;
            min-height: 44px;
            min-width: 44px;
            padding: 12px 24px;
            font-size: 15px;
            font-weight: 600;
            box-shadow: 0 2px 8px rgba(0, 122, 255, 0.2);
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .stButton > button:hover {
            background-color: #0051D5;
            box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3);
            transform: translateY(-1px);
        }
        
        .stButton > button:active {
            transform: scale(0.98);
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
        
        /* 加载动画效果 */
        @keyframes ios-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }
        
        @keyframes ios-spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .loading {
            animation: ios-pulse 1.5s ease-in-out infinite;
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
        
        # 动态应用TradingView深色模式CSS
        if st.session_state.dark_mode:
            st.markdown("""
            <style>
            /* TradingView深色风格 */
            .main {
                background-color: #131722 !important;
            }
            .block-container {
                background-color: #131722 !important;
                padding: 16px 24px;
            }
            
            /* 侧边栏TradingView风格 */
            [data-testid="stSidebar"] {
                background-color: #1E222D !important;
                border-right: 1px solid #2A2E39 !important;
            }
            
            /* 文字颜色 */
            h1, h2, h3, h4 {
                color: #D1D4DC !important;
                font-weight: 600;
            }
            p, span, div {
                color: #B2B5BE !important;
            }
            
            /* 度量卡片 */
            [data-testid="stMetricLabel"] {
                color: #787B86 !important;
                font-size: 12px;
            }
            [data-testid="stMetricValue"] {
                color: #D1D4DC !important;
                font-size: 22px;
                font-weight: 600;
            }
            
            /* 卡片容器 */
            .stMarkdown {
                background-color: #1E222D !important;
                border: 1px solid #2A2E39 !important;
                border-radius: 8px !important;
                color: #D1D4DC !important;
            }
            
            /* 按钮TradingView风格 */
            .stButton > button {
                background-color: #2962FF !important;
                color: #FFFFFF !important;
                border: none;
                border-radius: 6px;
                font-weight: 500;
            }
            .stButton > button:hover {
                background-color: #1E53E5 !important;
            }
            
            /* 侧边栏按钮 */
            [data-testid="stSidebar"] .stButton > button {
                background-color: #2A2E39 !important;
                color: #D1D4DC !important;
                border: 1px solid #363A45 !important;
            }
            [data-testid="stSidebar"] .stButton > button:hover {
                background-color: #363A45 !important;
                border-color: #2962FF !important;
            }
            
            /* 输入框 */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stSelectbox > div > div {
                background-color: #2A2E39 !important;
                border: 1px solid #363A45 !important;
                color: #D1D4DC !important;
            }
            
            /* 标签页 */
            .stTabs [data-baseweb="tab-list"] {
                background-color: #1E222D !important;
                border-bottom: 1px solid #2A2E39;
            }
            .stTabs [data-baseweb="tab"] {
                color: #787B86 !important;
                background-color: transparent;
            }
            .stTabs [aria-selected="true"] {
                color: #D1D4DC !important;
                background-color: #2A2E39 !important;
                border-bottom: 2px solid #2962FF !important;
            }
            
            /* 分割线 */
            hr {
                border-color: #2A2E39 !important;
            }
            
            /* 展开器 */
            .streamlit-expanderHeader {
                background-color: #2A2E39 !important;
                color: #D1D4DC !important;
            }
            
            /* 表格 */
            .dataframe {
                background-color: #1E222D !important;
                border: 1px solid #2A2E39 !important;
            }
            .dataframe th {
                background-color: #2A2E39 !important;
                color: #787B86 !important;
            }
            .dataframe td {
                color: #D1D4DC !important;
            }
            
            /* 成功/警告/错误提示 */
            .element-container:has(> .stSuccess) {
                background-color: #0B3D0B !important;
                border-left: 3px solid #26A69A !important;
            }
            .element-container:has(> .stWarning) {
                background-color: #4A3C1A !important;
                border-left: 3px solid #FF9800 !important;
            }
            .element-container:has(> .stError) {
                background-color: #4A1A1A !important;
                border-left: 3px solid #F23645 !important;
            }
            </style>
            """, unsafe_allow_html=True)
        else:
            # 浅色模式保留原有iOS风格
            pass
        
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
