"""管理面板：模型版本控制、阈值调优和系统配置"""
import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


class AdminPanel:
    """系统配置和模型管理的管理界面"""
    
    def render(self, models_data: Optional[Dict] = None, signals_stats: Optional[Dict] = None):
        """渲染管理面板界面"""
        
        # 为不同的管理功能创建子标签
        admin_tabs = st.tabs(["🔧 Model Versions", "⚙️ Threshold Presets", "📤 Signal Export", "📊 System Stats"])
        
        with admin_tabs[0]:
            self._render_model_versions(models_data)
        
        with admin_tabs[1]:
            self._render_threshold_presets()
        
        with admin_tabs[2]:
            self._render_signal_export()
        
        with admin_tabs[3]:
            self._render_system_stats(signals_stats)
    
    def _render_model_versions(self, models_data: Optional[Dict]):
        """渲染模型版本管理界面"""
        st.markdown("### Model Version Management")
        
        if not models_data or 'models' not in models_data:
            st.warning("No model data available")
            return
        
        models = models_data['models']
        
        if not models:
            st.info("No models deployed yet")
            return
        
        # 为模型创建DataFrame
        df_data = []
        for model in models:
            df_data.append({
                'Version': model['version'],
                'Type': model['model_type'],
                'Active': '✅' if model['is_active'] else '❌',
                'PR-AUC': model.get('metrics', {}).get('pr_auc', 'N/A'),
                'Hit@TopK': model.get('metrics', {}).get('hit_at_top_k', 'N/A'),
                'ECE': model.get('calibration_ece', 'N/A'),
                'Created': model['created_at'][:19]
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, width='stretch', hide_index=True)
        
        # 模型部署部分
        st.markdown("#### Deploy New Model Version")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_version = st.text_input("Version Name", placeholder="e.g., 2.0.0")
        
        with col2:
            model_type = st.selectbox("Model Type", ["lightgbm", "xgboost", "tcn", "ensemble"])
        
        with col3:
            calibration = st.selectbox("Calibration", ["isotonic", "platt", "beta"])
        
        if st.button("📥 Deploy Model (Demo)", type="primary", disabled=True):
            st.info("Model deployment is not available in demo mode")
    
    def _render_threshold_presets(self):
        """渲染阈值预设管理"""
        st.markdown("### Threshold Configuration Presets")
        
        # 显示当前预设
        presets = {
            "A-tier (Conservative)": {"tau": 0.75, "kappa": 1.20, "theta_up": 0.006, "theta_dn": 0.004},
            "B-tier (Balanced)": {"tau": 0.65, "kappa": 1.00, "theta_up": 0.006, "theta_dn": 0.004},
            "C-tier (Aggressive)": {"tau": 0.55, "kappa": 0.80, "theta_up": 0.004, "theta_dn": 0.003},
        }
        
        # 创建DataFrame
        df_data = []
        for name, params in presets.items():
            df_data.append({
                'Preset': name,
                'τ (Prob)': params['tau'],
                'κ (Utility)': params['kappa'],
                'θ_up (%)': params['theta_up'] * 100,
                'θ_dn (%)': params['theta_dn'] * 100
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, width='stretch', hide_index=True)
        
        # 创建新预设
        st.markdown("#### Create Custom Preset")
        col1, col2 = st.columns(2)
        
        with col1:
            preset_name = st.text_input("Preset Name", placeholder="e.g., My Strategy")
            tau_val = st.slider("τ (Probability Threshold)", 0.50, 0.95, 0.70, 0.01)
            theta_up_val = st.slider("θ_up (Up Threshold %)", 0.1, 2.0, 0.6, 0.1)
        
        with col2:
            st.write("")  # Spacing
            st.write("")
            kappa_val = st.slider("κ (Utility Threshold)", 0.50, 2.00, 1.00, 0.05)
            theta_dn_val = st.slider("θ_dn (Down Threshold %)", 0.1, 1.5, 0.4, 0.1)
        
        if st.button("💾 Save Preset (Demo)", disabled=True):
            st.info("Preset saving is not available in demo mode")
        
        # 阈值优化洞察
        st.markdown("#### 📈 Threshold Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Signal Rate", "~15 /hr", delta="-3%")
        with col2:
            st.metric("Precision", "68%", delta="+2%")
        with col3:
            st.metric("Expected Utility", "4.2", delta="+0.3")
    
    def _render_signal_export(self):
        """渲染信号导出控制"""
        st.markdown("### Signal Export Interface")
        
        st.markdown("""
        导出交易信号供下游交易机器人或分析系统使用。
        支持两种格式：
        - **JSONL**: 换行分隔的JSON（人类可读，易于解析）
        - **Protobuf**: 二进制格式（紧凑，快速反序列化）
        """)
        
        # 导出过滤器
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_symbol = st.selectbox("Symbol", ["All", "BTCUSDT", "ETHUSDT", "BNBUSDT"])
        
        with col2:
            export_decision = st.selectbox("Decision", ["All", "LONG", "SHORT", "WAIT"])
        
        with col3:
            export_tier = st.selectbox("Tier", ["All", "A", "B"])
        
        # 时间范围
        col1, col2 = st.columns(2)
        with col1:
            export_hours = st.number_input("Last N hours", min_value=1, max_value=168, value=24, key="export_hours")
        
        with col2:
            export_limit = st.number_input("Max signals", min_value=10, max_value=10000, value=1000, key="export_limit")
        
        # 导出按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export JSONL", type="primary"):
                # 构建带时间范围的导出URL
                from datetime import datetime, timedelta
                start_time = (datetime.now() - timedelta(hours=export_hours)).isoformat()
                
                params = []
                if export_symbol != "All":
                    params.append(f"symbol={export_symbol}")
                if export_decision != "All":
                    params.append(f"decision={export_decision}")
                if export_tier != "All":
                    params.append(f"tier={export_tier}")
                params.append(f"start_time={start_time}")
                params.append(f"limit={export_limit}")
                
                query_string = "&".join(params)
                export_url = f"http://localhost:8000/export/jsonl?{query_string}"
                
                st.code(f"curl -o signals.jsonl '{export_url}'", language="bash")
                st.success("Export URL generated! Use the command above to download.")
        
        with col2:
            if st.button("📥 Export Protobuf", type="secondary"):
                # 构建带时间范围的导出URL
                from datetime import datetime, timedelta
                start_time = (datetime.now() - timedelta(hours=export_hours)).isoformat()
                
                params = []
                if export_symbol != "All":
                    params.append(f"symbol={export_symbol}")
                if export_decision != "All":
                    params.append(f"decision={export_decision}")
                if export_tier != "All":
                    params.append(f"tier={export_tier}")
                params.append(f"start_time={start_time}")
                params.append(f"limit={export_limit}")
                
                query_string = "&".join(params)
                export_url = f"http://localhost:8000/export/protobuf?{query_string}"
                
                st.code(f"curl -o signals.pb '{export_url}'", language="bash")
                st.success("Export URL generated! Use the command above to download.")
        
        # 使用示例
        with st.expander("📖 Integration Example"):
            st.markdown("""
            **Python - JSONL Integration:**
            ```python
            import json
            
            with open('signals.jsonl', 'r') as f:
                for line in f:
                    signal = json.loads(line)
                    if signal['tier'] == 'A' and signal['decision'] == 'LONG':
                        execute_trade(signal)
            ```
            
            **Python - Protobuf Integration:**
            ```python
            from backend.proto import signal_pb2
            
            with open('signals.pb', 'rb') as f:
                batch = signal_pb2.SignalBatch()
                batch.ParseFromString(f.read())
                
                for signal in batch.signals:
                    if signal.tier == 'A':
                        execute_trade(signal)
            ```
            """)
    
    def _render_system_stats(self, signals_stats: Optional[Dict]):
        """渲染系统统计和健康状态"""
        st.markdown("### System Statistics & Health")
        
        if not signals_stats:
            st.warning("No statistics available")
            return
        
        # 总体指标
        col1, col2, col3, col4 = st.columns(4)
        
        total_signals = signals_stats.get('total_signals', 0)
        
        with col1:
            st.metric("Total Signals (24h)", total_signals)
        
        with col2:
            avg_latency = 0
            count = 0
            for symbol_stats in signals_stats.get('by_symbol', {}).values():
                avg_latency += symbol_stats.get('avg_latency_ms', 0)
                count += 1
            avg_latency = avg_latency / count if count > 0 else 0
            st.metric("Avg Latency (ms)", f"{avg_latency:.1f}")
        
        with col3:
            st.metric("SLA Compliance", "99.2%", delta="0.1%")
        
        with col4:
            st.metric("Active Models", "1")
        
        # 按交易对分组
        st.markdown("#### Signal Distribution by Symbol")
        
        if signals_stats.get('by_symbol'):
            symbol_data = []
            for symbol, stats in signals_stats.get('by_symbol', {}).items():
                symbol_data.append({
                    'Symbol': symbol,
                    'Total': stats.get('total_signals', 0),
                    'A-tier': stats.get('a_tier_count', 0),
                    'B-tier': stats.get('b_tier_count', 0),
                    'LONG': stats.get('long_count', 0),
                    'Avg Utility': f"{stats.get('avg_utility', 0):.2f}",
                    'Avg Latency': f"{stats.get('avg_latency_ms', 0):.1f} ms"
                })
            
            if symbol_data:
                df = pd.DataFrame(symbol_data)
                st.dataframe(df, width='stretch', hide_index=True)
        else:
            st.info("No signals generated yet")
        
        # 系统健康指标
        st.markdown("#### System Health")
        col1, col2 = st.columns(2)
        
        with col1:
            st.progress(0.95, text="Database Health: 95%")
            st.progress(0.99, text="API Availability: 99%")
        
        with col2:
            st.progress(0.88, text="Model Accuracy: 88%")
            st.progress(1.00, text="Data Pipeline: 100%")
