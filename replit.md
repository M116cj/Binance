# Crypto Surge Prediction System

## Overview
This project is a cryptocurrency surge prediction system designed to forecast short-term price movements using Binance market data. It leverages machine learning (LightGBM) with advanced feature engineering focusing on order book dynamics and market microstructure. The system aims to generate actionable trading signals within a cost-aware decision framework, triggering signals only when both probability and utility thresholds are met. It features a FastAPI backend for data processing and inference, a Streamlit frontend for visualization, and PostgreSQL for data storage. The project's ambition is to provide a robust, real-time tool for identifying profitable trading opportunities in the cryptocurrency market.

## User Preferences
Preferred communication style: Simple, everyday language.

## Recent Changes

### 2025-10-22: TradingView深色主题改造（v2.4）✅ 已完成
1. **默认主题切换**：
   - 将默认主题从iOS浅色切换为TradingView深色风格
   - 参考TradingView专业交易平台的视觉设计
   - 主背景：#131722（深蓝黑色）
   - 次级背景：#1E222D（深灰色）
   - 卡片背景：#2A2E39（稍浅深灰）

2. **TradingView配色方案**：
   - 主色调：#2962FF（专业蓝色）
   - 涨：#26A69A（TradingView绿色）
   - 跌/错误：#F23645（TradingView红色）
   - 警告：#FF9800（橙色）
   - 主文字：#D1D4DC（浅灰）
   - 次要文字：#787B86（深灰）
   - 边框：#363A45（深灰边框）

3. **深色模式CSS优化**：
   - 侧边栏深色背景（#1E222D）+ 深色边框
   - 按钮深色风格（#2A2E39背景 + #363A45边框）
   - 输入框深色背景（#2A2E39）
   - 标签页深色风格（选中项#2962FF底部边框）
   - 卡片深色背景（#1E222D + #2A2E39边框）
   - 表格深色样式（深色背景 + 浅色文字）

4. **侧边栏TradingView化**：
   - 更换图标为📊（图表）
   - 策略卡片深色背景 + 边框
   - 系统状态使用TradingView颜色编码
   - 所有文字颜色适配深色背景

5. **配置文件更新**：
   - `.streamlit/config.toml`设置为dark主题
   - primaryColor: #2962FF
   - backgroundColor: #131722
   - secondaryBackgroundColor: #1E222D
   - textColor: #D1D4DC

6. **主题切换功能**：
   - 侧边栏🌙/☀️按钮支持浅色/深色切换
   - 默认启动为深色模式
   - 动态CSS应用确保平滑切换

**设计理念**：
- 专业交易平台视觉风格
- 深色背景减少眼睛疲劳，适合长时间盯盘
- 高对比度确保数据清晰可见
- 类似TradingView的专业体验

**测试结果**：
- ✅ TradingView深色主题成功应用
- ✅ 所有UI元素适配深色背景
- ✅ 颜色对比度优秀
- ✅ 主题切换功能正常
- ✅ 应用正常运行

## System Architecture

### UI/UX Decisions
The frontend dashboard is built with Streamlit, providing real-time visualization of trading signals and comprehensive analytics through seven specialized report components. It updates via REST API polling and Server-Sent Events (SSE) for near real-time data, prioritizing actionable insights with a professional TradingView-inspired dark theme aesthetic. Key UI elements include TradingView-style dark theme (#131722 background, #2962FF accent), deep card backgrounds (#1E222D), subtle borders, professional color coding (green/red for bullish/bearish), and gradient-themed title cards for each tab. The interface supports theme switching between dark and light modes via a toggle button.

### Technical Implementations
1.  **Data Ingestion Layer**: A multi-connection WebSocket service handles real-time, low-latency market data from Binance, ensuring data integrity with timestamp tracking and sequence validation.
2.  **Storage Layer**: A dual-storage architecture uses Redis for hot caching of recent market data and cost lookup tables, and ClickHouse for time series cold storage. PostgreSQL stores relational data like signal history, predictions, and model metadata.
3.  **Feature Engineering**: A ring buffer-based streaming feature engine computes market microstructure features (e.g., Order Flow Imbalance) using Numba JIT compilation and NumPy vectorization, preventing look-ahead bias.
4.  **Labeling & Training**: Employs a triple-barrier method with temporal safeguards and LightGBM with focal loss for imbalanced classification, using isotonic regression for calibration.
5.  **Cost Modeling**: A multi-component cost model incorporates fees, slippage, funding costs, and capacity constraints for generating profitable signals.
6.  **Inference Service**: An optimized pipeline uses ONNX Runtime for high-throughput, low-latency predictions, including deduplication logic and cooldown tracking.
7.  **Backtesting Engine**: An event-driven matching engine provides realistic execution simulation for strategy validation, supporting various order types and latency injection.
8.  **Monitoring & Observability**: Uses Prometheus metrics for tracking system health, detecting degradation, and diagnosing issues, with optional OpenTelemetry for distributed tracing.
9.  **Configuration Management**: A Pydantic-based layered configuration system supports environment variable overrides and `.env` files for modular and validated settings.
10. **Backend Optimization**: Implements LRU+TTL caching (71.43% hit rate), token bucket rate limiting (300 req/min), and SQL batch query aggregation to significantly reduce memory usage and improve response times.
11. **Data Quality Monitoring**: Features real-time outlier detection (Z-score), data drift detection (Kolmogorov-Smirnov test), missing value monitoring, and freshness alerts with multi-level alarming.
12. **Multi-Symbol Support**: Dynamically fetches and processes data for multiple cryptocurrency symbols (e.g., USDT trading pairs), with comprehensive localization.
13. **Strategy Management**: Implements a dual-mode architecture (demo/production) and allows for quick switching between predefined trading strategies (e.g., "A-tier" and "B-tier" signals) with adjustable parameters, supporting an 80% reduction in frontend requests through batch API endpoints.

### System Design Choices
The architecture emphasizes modularity, scalability, and resilience. Data ingestion is separated from the UI for stability. A dual-storage approach optimizes for both real-time access and historical analysis. Strict temporal validation in feature engineering and robust labeling practices prevent overfitting. The cost-aware decision framework ensures practical, profitable trading signals.

## External Dependencies

### Third-Party Services
*   **Binance WebSocket API**: For real-time market data (depth, trades, funding, open interest).

### Infrastructure Dependencies
*   **Redis**: Hot cache for features, cost tables, and deduplication locks.
*   **ClickHouse**: Time series storage for historical data and backtesting.
*   **PostgreSQL**: Required for signal history, model versions, predictions, and audit logs.

### Python Libraries
*   **Core ML/Numerical**: `lightgbm`, `onnxruntime`, `numpy`, `pandas`, `scipy`, `numba`, `scikit-learn`.
*   **Backend**: `fastapi`, `uvicorn`, `websockets`, `redis`, `clickhouse-driver`, `sqlalchemy`, `pydantic`, `orjson`.
*   **Frontend**: `streamlit`, `plotly`, `httpx`.
*   **Monitoring**: `prometheus-client`, `opentelemetry` (optional).
*   **Exchange Integration**: `python-binance`, `ntplib`.

### Deployment Platform
*   **Railway**: Configured for multi-service deployment with automatic HTTPS and domain management.