# Crypto Surge Prediction System 🚀

[![Version](https://img.shields.io/badge/version-2.4.0-blue.svg)](https://github.com/yourusername/crypto-surge-prediction)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![P99 Latency](https://img.shields.io/badge/P99_latency-< 800ms-brightgreen.svg)](https://docs.example.com/sla)

> Enterprise-grade cryptocurrency surge prediction with ML-powered real-time signals

A cryptocurrency surge prediction system designed to forecast short-term price movements using 
Binance market data. It leverages machine learning (LightGBM) with advanced feature engineering 
focusing on order book dynamics and market microstructure.


---

## ✨ Key Features

### Real-time Signal Generation
ML-powered trading signals with probability and utility scores

### 7 Analytics Reports
- 📊 实时信号 (Real-time Signals)
- 📈 概率分析 (Probability Analysis)
- ⚡ 滑动窗口 (Rolling Window)
- 🎯 回测结果 (Backtest Results)
- 📐 校准曲线 (Calibration Curves)
- 🔍 归因分析 (Attribution Analysis)
- 📋 系统监控 (System Monitoring)

### Cost-Aware Decision Framework
Multi-component cost model (fees, slippage, funding, capacity)

### Multi-Symbol Support
Dynamic cryptocurrency pair processing (USDT pairs)

### Data Quality Monitoring
- Real-time outlier detection (Z-score)
- Data drift detection (Kolmogorov-Smirnov test)
- Missing value monitoring
- Freshness alerts

### TradingView-Inspired UI
- Professional dark theme (#131722 background)
- Color-coded signals (green/red)
- Interactive charts with Plotly
- Theme toggle (dark/light)


---

## 🏗️ System Architecture

### Dual-Mode Operation
**Demo Mode**: Simulated data for testing and development

**Production Mode**: Live Binance market data integration

### Trading Strategies
| Strategy | Icon | θ_up | θ_dn | τ | κ | Description |
|----------|------|------|------|---|---|-------------|
| A-tier Signals | ⭐ | 0.008 | 0.0056 | 0.75 | 1.2 | High-probability signals with strict thresholds |
| B-tier Signals | 🎯 | 0.006 | 0.004 | 0.65 | 1.0 | Moderate signals for broader opportunities |

### Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | FastAPI | REST API endpoints, LRU+TTL caching (71.43% hit rate), Token bucket rate limiting (300 req/min) |
| **Frontend** | Streamlit | Real-time signal visualization, 7 specialized analytics reports, Server-Sent Events (SSE) updates |
| **ML Pipeline** | LightGBM → ONNX Runtime | Order Flow Imbalance (OFI), Market microstructure indicators |
| **Storage** | Redis, ClickHouse, PostgreSQL | Hot cache, cold storage, relational data |

### Performance Metrics

| Metric | Value |
|--------|-------|
| P99 Latency | < 800ms |
| Inference Capacity | ≥ 300 RPS |
| Cache Hit Rate | 71.43% |
| Rate Limit | 300 req/min |

---

## 📊 Analytics Dashboard

The system provides **7 specialized report components**:

- 📊 实时信号 (Real-time Signals)
- 📈 概率分析 (Probability Analysis)
- ⚡ 滑动窗口 (Rolling Window)
- 🎯 回测结果 (Backtest Results)
- 📐 校准曲线 (Calibration Curves)
- 🔍 归因分析 (Attribution Analysis)
- 📋 系统监控 (System Monitoring)

### UI Theme: TradingView Dark Mode

The dashboard features a professional **TradingView-inspired dark theme** optimized for extended trading sessions:

- **Reduced Eye Strain**: Dark background (#131722) minimizes fatigue during long monitoring sessions
- **High Contrast**: Critical data stands out clearly
- **Professional Aesthetics**: Similar visual language to leading trading platforms
- **Theme Toggle**: Switch between dark and light modes via sidebar button

**Color Palette:**
- 🎨 Primary: `#2962FF`
- 🌑 Background: `#131722`
- 💚 Bullish: `#26A69A`
- ❤️ Bearish: `#F23645`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- PostgreSQL database
- Redis (optional, for caching)
- ClickHouse (optional, for time series storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/crypto-surge-prediction
cd crypto-surge-prediction

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Configuration

Required environment variables:

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `PGHOST` | PostgreSQL host |
| `PGPORT` | PostgreSQL port |
| `PGDATABASE` | PostgreSQL database name |
| `PGUSER` | PostgreSQL username |
| `PGPASSWORD` | PostgreSQL password |

Optional environment variables:

| Variable | Description |
|----------|-------------|
| `XAI_API_KEY` | xAI Grok API key for AI-enhanced analysis |
| `REDIS_URL` | Redis connection string for caching |
| `CLICKHOUSE_HOST` | ClickHouse host for time series storage |

### Running the Application

```bash
# Start the backend API
python -m backend.api_server

# In a separate terminal, start the frontend
streamlit run main.py --server.port 5000
```

Access the application at `http://localhost:5000`

---

## 📖 Usage

### Demo Mode (Default)

The system starts in **demo mode** with simulated data:

1. Open the Streamlit dashboard
2. Select a cryptocurrency pair from the sidebar
3. Choose a trading strategy (⭐ A-tier or 🎯 B-tier)
4. View real-time signals and analytics

### Production Mode

To enable live Binance data:

1. Set `MODE=production` in your environment
2. Configure Binance WebSocket credentials (if required)
3. Restart the backend service

---

## 🎯 Trading Strategies

### ⭐ A-tier Signals

High-probability signals with strict thresholds

**Parameters:**
- θ_up (bullish threshold): 0.008
- θ_dn (bearish threshold): 0.0056
- τ (tau - probability threshold): 0.75
- κ (kappa - utility multiplier): 1.2

### 🎯 B-tier Signals

Moderate signals for broader opportunities

**Parameters:**
- θ_up (bullish threshold): 0.006
- θ_dn (bearish threshold): 0.004
- τ (tau - probability threshold): 0.65
- κ (kappa - utility multiplier): 1.0


---

## 📡 API Endpoints

### Core Endpoints

```bash
# Health check
GET /health

# Get available symbols
GET /symbols

# Get real-time signals
GET /reports/realtime?symbol=BTCUSDT&theta_up=0.008&theta_dn=0.0056&tau=0.75&kappa=1.2

# Batch analytics
GET /reports/batch?symbol=BTCUSDT&theta_up=0.008&theta_dn=0.0056&tau=0.75&kappa=1.2
```

For complete API documentation, visit `/docs` when the backend is running.

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=backend --cov-report=html

# Run integration tests
pytest tests/integration/
```

---

## 📦 Deployment

### Replit

The project is configured for one-click deployment on Replit:

1. Import the repository to Replit
2. Environment variables are managed automatically
3. Click "Run" to start both backend and frontend

### Docker

```bash
# Build the image
docker build -t crypto-surge-prediction .

# Run the container
docker run -p 5000:5000 -p 8000:8000 \
  -e DATABASE_URL=your_db_url \
  crypto-surge-prediction
```

### Railway

Deploy to Railway with the included `railway.json` configuration.

---

## 🔧 Architecture Deep Dive

### Data Ingestion Layer
- Multi-connection WebSocket service for Binance market data
- Timestamp tracking and sequence validation
- Low-latency real-time data processing

### Feature Engineering
- Ring buffer-based streaming feature engine
- Order Flow Imbalance (OFI) computation
- Numba JIT compilation for performance
- NumPy vectorization
- Prevents look-ahead bias

### ML Pipeline
- **Model**: LightGBM with focal loss
- **Inference**: ONNX Runtime (≥300 RPS)
- **Calibration**: Isotonic regression
- **Labeling**: Triple-barrier method with temporal safeguards

### Cost Modeling
Multi-component framework:
- Trading fees
- Slippage estimation
- Funding costs
- Capacity constraints

### Monitoring & Observability
- Prometheus metrics
- Real-time data quality monitoring
- Outlier detection (Z-score)
- Data drift detection (Kolmogorov-Smirnov test)
- OpenTelemetry integration (optional)

---

## 📈 Recent Updates

### Version 2.4.0 (2025-10-22): TradingView Dark Theme Overhaul

- Switched default theme to TradingView-inspired dark mode
- Professional color scheme (#131722 background, #2962FF accent)
- Enhanced contrast for better readability
- Added theme toggle (dark/light modes)
- Optimized all UI elements for dark backgrounds


---

## 🗺️ Roadmap

- [ ] Multi-exchange support (Binance, Coinbase, Kraken)
- [ ] Advanced AI analysis with xAI Grok integration
- [ ] Real-time news sentiment analysis
- [ ] Mobile app (React Native)
- [ ] Telegram/Discord bot notifications
- [ ] Portfolio management features
- [ ] Advanced risk metrics

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Before contributing, please ensure:
- Update `docs/project_manifest.yaml` for any metadata changes
- Run tests and ensure they pass
- Follow the existing code style

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Binance API** for real-time market data
- **LightGBM** for high-performance gradient boosting
- **Streamlit** for rapid UI development
- **TradingView** for design inspiration

---

## 📞 Contact & Support

- **Author**: Your Name
- **Email**: your.email@example.com
- **Issues**: [GitHub Issues](https://github.com/yourusername/crypto-surge-prediction/issues)
- **Documentation**: [Full Documentation](https://github.com/yourusername/crypto-surge-prediction/wiki)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ by Your Name

</div>

---

<!-- Auto-generated by scripts/update_readme.py -->
<!-- Last updated: 2025-11-09 01:26:38 UTC -->
<!-- Checksum:  -->