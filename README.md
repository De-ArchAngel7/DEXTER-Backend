# 🚀 DEXTER Backend - AI Trading Bot

Advanced AI-powered cryptocurrency trading bot backend with multiple AI models and exchange integrations.

## 🧠 AI Features

- **Mistral 7B Integration** - Fine-tuned conversational AI via Hugging Face API
- **LSTM Price Prediction** - Deep learning models for crypto and forex
- **FinBERT Sentiment Analysis** - Financial sentiment analysis
- **Unified Conversation Engine** - Multi-model AI orchestration
- **Real-time Trading Signals** - AI-generated trading recommendations

## 📈 Trading Capabilities

- **Multi-Exchange Support** - Binance, KuCoin, Coinbase
- **Advanced Strategies** - DCA, Grid Trading, Risk Management
- **Paper Trading** - Safe strategy testing
- **Live Trading** - Real-time execution
- **Portfolio Management** - Comprehensive tracking

## 🛠️ Tech Stack

- **FastAPI** - Modern Python web framework
- **PyTorch** - Deep learning models
- **Transformers** - Hugging Face model integration
- **MongoDB** - Database for trading data
- **Redis** - Caching and session management
- **Celery** - Background task processing

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- MongoDB
- Redis
- API keys for exchanges

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install TA-Lib optuna huggingface_hub
```

### Model Setup

The trained models are not included in the repository due to size constraints. You'll need:

1. **LSTM Models** - Train using `train_lstm_model.py` or use your existing models
2. **FinBERT Models** - Download from Hugging Face or train custom models
3. **Mistral 7B** - Uses Hugging Face API (no local download needed)

### Environment Setup

```bash
# Configure environment variables
HUGGINGFACE_TOKEN=your_hf_token_here
USE_HF_API=true
DEXTER_MISTRAL_MODEL_PATH=models/dexter-mistral-7b-final
```

### Run

```bash
python app/main.py
```

## 📚 API Documentation

Once running, visit:
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🔒 Security

- JWT authentication
- API key encryption
- Rate limiting
- Comprehensive logging

## 📊 Monitoring

- Real-time system health
- Trading performance metrics
- AI model status
- Error tracking

---

**DEXTER** - Your intelligent trading companion powered by advanced AI.
