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

# Import frontend components
from frontend.components.signal_card import SignalCard
from frontend.components.regime_state import RegimeState
from frontend.components.probability_window import ProbabilityWindow
from frontend.components.cost_capacity import CostCapacity
from frontend.components.backtest_performance import BacktestPerformance
from frontend.components.calibration_analysis import CalibrationAnalysis
from frontend.components.attribution_comparison import AttributionComparison
from frontend.components.admin_panel import AdminPanel
from frontend.components.signal_history import SignalHistory

# Configuration
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
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
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
            
    def fetch_data(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Fetch data from backend API with error handling"""
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
    
    def render_sidebar(self):
        """Render the control sidebar"""
        st.sidebar.title("🔥 Crypto Surge Prediction")
        st.sidebar.markdown("---")
        
        # Symbol selection
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
        st.session_state.selected_symbol = st.sidebar.selectbox(
            "Trading Pair", 
            symbols, 
            index=symbols.index(st.session_state.selected_symbol)
        )
        
        st.sidebar.markdown("### 📊 Parameters")
        
        # Labeling parameters
        st.session_state.theta_up = st.sidebar.number_input(
            "θ_up (Up Threshold %)", 
            min_value=0.1, 
            max_value=2.0, 
            value=st.session_state.theta_up * 100,
            step=0.1,
            format="%.1f"
        ) / 100
        
        st.session_state.theta_dn = st.sidebar.number_input(
            "θ_dn (Down Threshold %)", 
            min_value=0.1, 
            max_value=1.5, 
            value=st.session_state.theta_dn * 100,
            step=0.1,
            format="%.1f"
        ) / 100
        
        # Decision thresholds
        st.sidebar.markdown("### ⚡ Decision Thresholds")
        
        tier = st.sidebar.radio("Signal Tier", ["A-tier", "B-tier", "Custom"])
        
        if tier == "A-tier":
            st.session_state.tau_threshold = 0.75
            st.session_state.kappa_threshold = 1.20
        elif tier == "B-tier":
            st.session_state.tau_threshold = 0.65
            st.session_state.kappa_threshold = 1.00
        else:  # Custom
            st.session_state.tau_threshold = st.sidebar.slider(
                "τ (Probability Threshold)", 
                0.5, 0.95, st.session_state.tau_threshold, 0.01
            )
            st.session_state.kappa_threshold = st.sidebar.slider(
                "κ (Utility Threshold)", 
                0.8, 2.0, st.session_state.kappa_threshold, 0.05
            )
        
        # Display current thresholds
        st.sidebar.info(f"τ = {st.session_state.tau_threshold:.2f}")
        st.sidebar.info(f"κ = {st.session_state.kappa_threshold:.2f}")
        
        # Auto-refresh
        st.session_state.auto_mode = st.sidebar.checkbox("Auto Refresh", st.session_state.auto_mode)
        
        if st.sidebar.button("🔄 Manual Refresh"):
            st.session_state.last_update = time.time()
            st.rerun()
            
        # System status
        st.sidebar.markdown("### 🟢 System Status")
        
        # Health check
        health_data = self.fetch_data("health")
        if health_data:
            if health_data.get("status") == "healthy":
                st.sidebar.success("✅ All Services Online")
            else:
                st.sidebar.warning("⚠️ Degraded Performance")
                
            exchange_lag = health_data.get("exchange_lag_s", 0)
            if exchange_lag < 2:
                st.sidebar.info(f"📡 Lag: {exchange_lag:.1f}s")
            else:
                st.sidebar.error(f"📡 High Lag: {exchange_lag:.1f}s")
        else:
            st.sidebar.error("❌ Backend Unavailable")
    
    def render_realtime_signal_card(self):
        """Report 1: Realtime Signal Card"""
        st.markdown("## 📡 Real-time Signal Overview")
        
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
            st.error("Unable to load real-time signal data")
    
    def render_regime_state(self):
        """Report 2: Market Regime & Liquidity State"""
        st.markdown("## 🌊 Market Regime & Liquidity State")
        
        params = {'symbol': st.session_state.selected_symbol}
        data = self.fetch_data("reports/regime", params)
        if data:
            self.regime_state.render(data)
        else:
            st.error("Unable to load regime state data")
    
    def render_probability_window(self):
        """Report 3: Pre-Surge Probability & Window"""
        st.markdown("## 📈 Pre-Surge Probability & Time Window")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn
        }
        
        data = self.fetch_data("reports/window", params)
        if data:
            self.probability_window.render(data, st.session_state.tau_threshold, st.session_state.kappa_threshold)
        else:
            st.error("Unable to load probability window data")
    
    def render_cost_capacity(self):
        """Report 4: Execution Cost & Capacity"""
        st.markdown("## 💰 Execution Cost & Capacity Analysis")
        
        params = {'symbol': st.session_state.selected_symbol}
        data = self.fetch_data("reports/cost", params)
        if data:
            self.cost_capacity.render(data)
        else:
            st.error("Unable to load cost & capacity data")
    
    def render_backtest_performance(self):
        """Report 5: Historical Backtest Performance"""
        st.markdown("## 📊 Historical Backtest Performance")
        
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
            st.error("Unable to load backtest performance data")
    
    def render_calibration_analysis(self):
        """Report 6: Calibration & Error Analysis"""
        st.markdown("## 🎯 Model Calibration & Error Analysis")
        
        params = {
            'symbol': st.session_state.selected_symbol,
            'theta_up': st.session_state.theta_up,
            'theta_dn': st.session_state.theta_dn
        }
        
        data = self.fetch_data("reports/calibration", params)
        if data:
            self.calibration_analysis.render(data)
        else:
            st.error("Unable to load calibration analysis data")
    
    def render_attribution_comparison(self):
        """Report 7: Event Attribution & Strategy Comparison"""
        st.markdown("## 🔍 Event Attribution & Strategy Comparison")
        
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
            st.error("Unable to load attribution comparison data")
    
    def render_admin_panel(self):
        """Admin panel for model management and system configuration"""
        st.markdown("## ⚙️ Admin Panel & System Configuration")
        
        # Fetch models data
        models_data = self.fetch_data("models")
        
        # Fetch signal statistics
        signals_stats = self.fetch_data("signals/stats")
        
        # Render admin panel
        self.admin_panel.render(models_data, signals_stats)
    
    def render_signal_history(self):
        """Signal history view with past predictions"""
        st.markdown("## 📜 Signal History")
        
        # Get filter values from the component
        # These will be set by the component's render method
        # We need to fetch data based on filters, so we'll pass the fetch function
        self.signal_history.render(self.fetch_data)
    
    def run(self):
        """Main application runner"""
        st.set_page_config(
            page_title="Crypto Surge Prediction System",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.initialize_session_state()
        self.render_sidebar()
        
        # Auto-refresh logic
        if st.session_state.auto_mode:
            if time.time() - st.session_state.last_update > 1.0:  # 1 second refresh
                st.session_state.last_update = time.time()
                st.rerun()
        
        # Main content tabs
        tabs = st.tabs([
            "📡 Real-time Signal", 
            "🌊 Market Regime", 
            "📈 Probability Window",
            "💰 Cost & Capacity",
            "📊 Backtest Performance",
            "🎯 Calibration Analysis",
            "🔍 Attribution & Comparison",
            "📜 Signal History",
            "⚙️ Admin Panel"
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
            self.render_admin_panel()

def main():
    dashboard = CryptoSurgePredictionDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
