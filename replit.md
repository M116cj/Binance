# Crypto Surge Prediction System

## Overview
This project is a cryptocurrency surge prediction system designed to forecast short-term price movements using market data from Binance. It leverages machine learning (LightGBM) with advanced feature engineering, focusing on order book dynamics and market microstructure. The system aims to generate actionable trading signals, considering a cost-aware decision framework where signals are only triggered if both probability and utility thresholds are met. The system comprises a FastAPI backend for data processing and inference, a Streamlit frontend for visualization, and PostgreSQL for data storage.

## User Preferences
Preferred communication style: Simple, everyday language.

## Recent Changes

### 2025-10-21: Multi-Symbol Support & Complete Chinese Localization
1. **Multi-Symbol Feature**:
   - Added `BinanceSymbolService` to dynamically fetch all USDT trading pairs from Binance API
   - Implemented `/symbols` REST endpoint in FastAPI backend
   - Extended frontend to load and display 60+ mainstream cryptocurrencies
   - Includes comprehensive Chinese naming for all major tokens (BTC, ETH, SOL, DOGE, etc.)
   - Note: Due to Binance API restrictions (HTTP 451) in Replit environment, system uses curated preset list

2. **Complete UI Chinese Localization**:
   - All 7 dashboard components fully translated: Signal Card, Regime State, Probability Window, Cost Analysis, Backtest Performance, Calibration Analysis, Attribution
   - Simplified technical jargon to everyday language (e.g., "Triple Barrier" → "上涨/下跌判定线")
   - Added emoji indicators and color coding for better visual comprehension
   - Strategy tiers renamed: 🛡️ Conservative (保守型), ⚖️ Balanced (平衡型), 🔥 Aggressive (激进型)
   - All metrics, labels, and help text converted to Chinese

## System Architecture

### UI/UX Decisions
The frontend dashboard is built with Streamlit, providing real-time visualization of trading signals and comprehensive analytics through seven specialized report components (e.g., Signal Card, Regime State, Backtest Performance). It updates via REST API polling and Server-Sent Events (SSE) for near real-time data, prioritizing actionable insights without overwhelming the user.

### Technical Implementations

1.  **Data Ingestion Layer**: A multi-connection WebSocket service (`ingestion_service.py`) handles real-time, low-latency market data from Binance. It employs connection pooling, triple timestamp tracking, sequence validation, and quality flagging to ensure data integrity and detect issues.
2.  **Storage Layer**: A dual-storage architecture uses Redis for hot caching of recent market data and cost lookup tables, providing sub-millisecond reads. ClickHouse serves as time series cold storage for historical data and analytical queries. PostgreSQL stores relational data such as signal history, predictions, and model metadata, ensuring ACID guarantees.
3.  **Feature Engineering**: A ring buffer-based streaming feature engine (`feature_service.py`) computes market microstructure features (e.g., Order Flow Imbalance, Queue Imbalance, Microprice Deviation) from raw data. It utilizes Numba JIT compilation and NumPy vectorization for performance, ensuring temporal validity to prevent look-ahead bias.
4.  **Labeling & Training**: The system uses a triple-barrier method with temporal safeguards (cooldown, embargo, purged K-Fold) to create robust labels for training. LightGBM with focal loss is employed for model architecture, focusing on imbalanced classification with monotonic constraints and isotonic regression for calibration.
5.  **Cost Modeling**: A multi-component cost model (`cost_model.py`) incorporates fees, slippage, funding costs, and capacity constraints into decision-making. This cost awareness is crucial for generating profitable signals, with models being versioned and cached in Redis.
6.  **Inference Service**: An optimized inference pipeline (`inference_service.py`) uses ONNX Runtime for high-throughput, low-latency predictions. It includes deduplication logic and cooldown tracking to prevent duplicate signals and alert fatigue.
7.  **Backtesting Engine**: An event-driven matching engine (`backtest_service.py`) provides realistic execution simulation for strategy validation. It supports various order types, partial fills, latency injection, and funding settlements, offering both conservative and neutral modes for sensitivity analysis.
8.  **Monitoring & Observability**: Prometheus metrics with alerting are used to track system health, detect degradation, and diagnose issues. Key metrics include WebSocket packet loss, end-to-end latency, clock drift, and prediction performance, with optional OpenTelemetry for distributed tracing.

### System Design Choices
The architecture emphasizes modularity, scalability, and resilience. Data ingestion is separated from the UI for stability. A dual-storage approach optimizes for both real-time access and historical analysis. Strict temporal validation in feature engineering and robust labeling practices prevent overfitting. The cost-aware decision framework ensures practical, profitable trading signals.

## External Dependencies

### Third-Party Services
*   **Binance WebSocket API**: For real-time market data (depth, trades, funding, open interest).
*   **NTP Servers**: (Optional) For clock synchronization and drift detection.

### Infrastructure Dependencies
*   **Redis** (v6.0+): Hot cache for features, cost tables, and deduplication locks.
*   **ClickHouse** (v21.0+): (Optional) Time series storage for historical data and backtesting.
*   **PostgreSQL** (v12+): Required for signal history, model versions, predictions, and audit logs.

### Python Libraries
*   **Core ML/Numerical**: `lightgbm`, `onnxruntime`, `numpy`, `pandas`, `scipy`, `numba`, `scikit-learn`.
*   **Backend**: `fastapi`, `uvicorn`, `websockets`, `redis`, `clickhouse-driver`, `sqlalchemy`, `pydantic`, `orjson`.
*   **Frontend**: `streamlit`, `plotly`, `httpx`.
*   **Monitoring**: `prometheus-client`, `opentelemetry` (optional).
*   **Exchange Integration**: `python-binance`, `ntplib`.

### Deployment Platform
*   **Railway**: Configured for multi-service deployment with automatic HTTPS and domain management.
*   **Alternative**: Compatible with any Docker or Python-supporting platform (e.g., AWS, GCP, Azure).