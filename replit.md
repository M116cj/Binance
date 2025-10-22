# Crypto Surge Prediction System

## Overview
This project is a cryptocurrency surge prediction system designed to forecast short-term price movements using Binance market data. It leverages machine learning (LightGBM) with advanced feature engineering focusing on order book dynamics and market microstructure. The system aims to generate actionable trading signals within a cost-aware decision framework, triggering signals only when both probability and utility thresholds are met. It features a FastAPI backend for data processing and inference, a Streamlit frontend for visualization, and PostgreSQL for data storage. The project's ambition is to provide a robust, real-time tool for identifying profitable trading opportunities in the cryptocurrency market.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### UI/UX Decisions
The frontend dashboard is built with Streamlit, providing real-time visualization of trading signals and comprehensive analytics through seven specialized report components. It updates via REST API polling and Server-Sent Events (SSE) for near real-time data, prioritizing actionable insights with a clean, iOS-inspired aesthetic. Key UI elements include iOS-style theme (light mode, #F2F2F7 background, #007AFF accent), rounded corners, soft shadows, iOS-style segmented controls, and gradient-themed title cards for each tab.

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