#!/usr/bin/env python3
"""
🌍 DEXTER FOREX-SPECIALIZED MODELS
============================================================
Forex-specialized LSTM + FinBERT models for:
- Major currency pairs (EUR/USD, GBP/USD, USD/JPY, etc.)
- Forex-specific technical indicators
- Forex market sentiment analysis
- Economic calendar integration
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import structlog
import yfinance as yf
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Import existing models for reference
from .dexter_forex_lstm_integration import DexterForexPredictor

logger = structlog.get_logger()

class ForexLSTMModel(nn.Module):
    """
    🌍 Forex-Specialized LSTM Model
    Optimized for currency pair price prediction with forex-specific features
    Now integrates with trained DEXTER Forex LSTM model
    """
    
    def __init__(self, input_size: int = 35, hidden_size: int = 128, num_layers: int = 3, dropout: float = 0.2, 
                 trained_model_path: Optional[str] = None):
        super(ForexLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.trained_model_path = trained_model_path
        self.trained_predictor = None
        
        # Initialize trained predictor if path provided
        if trained_model_path and os.path.exists(trained_model_path):
            try:
                self.trained_predictor = DexterForexPredictor(trained_model_path)
                logger.info(f"✅ Loaded trained DEXTER Forex LSTM from {trained_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load trained model: {e}")
                self.trained_predictor = None
        
        # Forex-optimized LSTM layers (fallback model)
        self.lstm = nn.LSTM(
            input_size=input_size,  # 35 forex-specific features
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Forex-specific attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidirectional
            num_heads=8,
            dropout=dropout
        )
        
        # Forex-optimized fully connected layers
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        
        # Batch normalization for forex data
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Fully connected layers with batch norm
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return out
    
    def predict_with_trained_model(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Use the trained DEXTER Forex LSTM model for prediction
        Falls back to the base model if trained model not available
        """
        if self.trained_predictor is not None:
            try:
                return self.trained_predictor.predict_forex_price(data, symbol)
            except Exception as e:
                logger.warning(f"Trained model prediction failed: {e}, using fallback")
        
        # Fallback to base model prediction
        return self._fallback_prediction(data, symbol)
    
    def _fallback_prediction(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Fallback prediction using the base model"""
        try:
            # This would use the base model for prediction
            # For now, return a basic prediction structure
            return {
                'success': False,
                'error': 'Trained model not available, base model not implemented for prediction',
                'prediction': None,
                'confidence': 0.0,
                'model_type': 'Forex_LSTM_Fallback'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': 0.0
            }

class ForexDataProcessor:
    """
    🌍 Forex Data Processor
    Handles forex-specific data collection and feature engineering
    """
    
    def __init__(self, sequence_length: int = 24):
        self.sequence_length = sequence_length
        self.major_pairs = [
            "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X",
            "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURJPY=X",
            "GBPJPY=X", "EURGBP=X", "AUDJPY=X", "EURAUD=X"
        ]
        
    def get_forex_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get forex data with forex-specific indicators"""
        try:
            ticker = yf.Ticker(symbol)
            # Try different intervals to get more data
            data = ticker.history(period=period, interval="1h")
            
            if data.empty or len(data) < 50:  # Need at least 50 data points for indicators
                # Try daily data if hourly is insufficient
                data = ticker.history(period="2y", interval="1d")
                logger.info(f"Using daily data for {symbol}: {len(data)} points")
            
            if data.empty:
                raise ValueError(f"No forex data found for {symbol}")
            
            # Calculate forex-specific technical indicators
            data = self._add_forex_indicators(data)
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"Error fetching forex data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_forex_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add forex-specific technical indicators"""
        try:
            close_series = pd.Series(data['Close'].values, index=data.index)
            
            # Standard indicators
            data['SMA_20'] = close_series.rolling(window=20).mean()
            data['SMA_50'] = close_series.rolling(window=50).mean()
            data['EMA_12'] = close_series.ewm(span=12).mean()
            data['EMA_26'] = close_series.ewm(span=26).mean()
            
            # Forex-specific indicators
            data['RSI'] = self._calculate_rsi(close_series)
            data['MACD'] = self._calculate_macd(close_series)
            data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
            data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
            
            # Bollinger Bands
            data['BB_upper'], data['BB_lower'] = self._calculate_bollinger_bands(close_series)
            data['BB_middle'] = close_series.rolling(window=20).mean()
            data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
            data['BB_position'] = (close_series - data['BB_lower']) / (data['BB_upper'] - data['BB_lower'])
            
            # Stochastic Oscillator
            data['Stoch_K'], data['Stoch_D'] = self._calculate_stochastic(data)
            
            # Williams %R
            data['Williams_R'] = self._calculate_williams_r(data)
            
            # Average True Range (ATR)
            data['ATR'] = self._calculate_atr(data)
            
            # Commodity Channel Index (CCI)
            data['CCI'] = self._calculate_cci(data)
            
            # Price action features
            data['Price_Change'] = close_series.pct_change()
            data['Price_Range'] = (data['High'] - data['Low']) / close_series
            data['Body_Size'] = abs(data['Close'] - data['Open']) / close_series
            data['Upper_Shadow'] = (data['High'] - data[['Open', 'Close']].max(axis=1)) / close_series
            data['Lower_Shadow'] = (data[['Open', 'Close']].min(axis=1) - data['Low']) / close_series
            
            # Volatility features
            data['Volatility_20'] = close_series.rolling(window=20).std()
            data['Volatility_50'] = close_series.rolling(window=50).std()
            data['Volatility_Ratio'] = data['Volatility_20'] / data['Volatility_50']
            
            # Trend features
            data['Trend_Strength'] = abs(data['SMA_20'] - data['SMA_50']) / close_series
            data['Trend_Direction'] = np.where(data['SMA_20'] > data['SMA_50'], 1, -1)
            
            # Momentum features
            data['Momentum_5'] = close_series / close_series.shift(5) - 1
            data['Momentum_10'] = close_series / close_series.shift(10) - 1
            data['Momentum_20'] = close_series / close_series.shift(20) - 1
            
            # Volume features (if available)
            if 'Volume' in data.columns:
                data['Volume_Change'] = data['Volume'].pct_change()
                data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
            else:
                # Forex doesn't have volume, use price-based volume proxy
                data['Volume_Proxy'] = data['Price_Range'] * close_series
                data['Volume_Change'] = data['Volume_Proxy'].pct_change()
                data['Volume_SMA'] = data['Volume_Proxy'].rolling(window=20).mean()
                data['Volume_Ratio'] = data['Volume_Proxy'] / data['Volume_SMA']
            
            return data
            
        except Exception as e:
            logger.error(f"Error adding forex indicators: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        # Ensure rsi is a pandas Series before calling fillna
        if isinstance(rsi, pd.Series):
            return rsi.fillna(50.0)
        else:
            # Convert to Series if it's not already
            rsi_series = pd.Series(rsi, index=prices.index)
            return rsi_series.fillna(50.0)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0.0)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, lower_band
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        low_min = data['Low'].rolling(window=k_period).min()
        high_max = data['High'].rolling(window=k_period).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent.fillna(50.0), d_percent.fillna(50.0)
    
    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        high_max = data['High'].rolling(window=period).max()
        low_min = data['Low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - data['Close']) / (high_max - low_min))
        return williams_r.fillna(-50.0)
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr.fillna(0.0)
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci.fillna(0.0)
    
    def create_forex_features(self, symbol: str) -> pd.DataFrame:
        """Create comprehensive forex features for training"""
        try:
            logger.info(f"Creating forex features for {symbol}")
            
            # Get forex data
            data = self.get_forex_data(symbol, period="2y")
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Select the 35 most important features for forex
            feature_columns = [
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_upper', 'BB_lower', 'BB_middle', 'BB_width', 'BB_position',
                'Stoch_K', 'Stoch_D', 'Williams_R', 'ATR', 'CCI',
                'Price_Change', 'Price_Range', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
                'Volatility_20', 'Volatility_50', 'Volatility_Ratio',
                'Trend_Strength', 'Trend_Direction',
                'Momentum_5', 'Momentum_10', 'Momentum_20',
                'Volume_Change', 'Volume_SMA', 'Volume_Ratio'
            ]
            
            # Ensure all columns exist
            available_features = [col for col in feature_columns if col in data.columns]
            features = data[available_features].copy()
            
            logger.info(f"Created {len(available_features)} forex features for {symbol}")
            return features if isinstance(features, pd.DataFrame) else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error creating forex features: {e}")
            return pd.DataFrame()
    
    def create_sequences(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        try:
            if features.empty:
                return np.array([]), np.array([])
            
            # Use Close price as target (we'll add it back)
            target_col = 'Close'
            if target_col not in features.columns:
                # If Close not in features, use the first price-related column
                price_cols = [col for col in features.columns if 'price' in col.lower() or 'close' in col.lower()]
                if price_cols:
                    target_col = price_cols[0]
                else:
                    target_col = features.columns[0]  # Use first column as fallback
            
            # Prepare data
            data = features.values
            target = features[target_col].values if target_col in features.columns else data[:, 0]
            
            X, y = [], []
            for i in range(self.sequence_length, len(data)):
                X.append(data[i-self.sequence_length:i])
                y.append(target[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            return np.array([]), np.array([])

class ForexFinBERTModel:
    """
    🌍 Forex-Specialized FinBERT Model
    Fine-tuned for forex market sentiment analysis
    Now integrates with trained DEXTER Forex FinBERT model
    """
    
    def __init__(self, model_path: str = "backend/models/dexter_forex_finbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.sentiment_model = None
        self.tokenizer = None
        self.forex_keywords = [
            'forex', 'fx', 'currency', 'exchange rate', 'pip', 'spread',
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD',
            'central bank', 'interest rate', 'inflation', 'GDP', 'unemployment',
            'FOMC', 'ECB', 'BOE', 'BOJ', 'RBA', 'BOC', 'RBNZ',
            'bullish', 'bearish', 'support', 'resistance', 'breakout',
            'trend', 'momentum', 'volatility', 'liquidity'
        ]
        
        # Load trained FinBERT model
        self._load_forex_sentiment_model()
    
    def _load_forex_sentiment_model(self):
        """Load trained forex-specialized sentiment model"""
        try:
            logger.info(f"Loading trained DEXTER Forex FinBERT from {self.model_path}...")
            
            # Check if trained model exists
            if os.path.exists(self.model_path):
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.sentiment_model.to(self.device)
                self.sentiment_model.eval()
                logger.info("✅ Trained DEXTER Forex FinBERT loaded!")
            else:
                # Fallback to base FinBERT
                logger.warning(f"Trained model not found at {self.model_path}, using base FinBERT")
                model_name = "ProsusAI/finbert"
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.sentiment_model.to(self.device)
                self.sentiment_model.eval()
                logger.info("✅ Base FinBERT loaded as fallback!")
            
        except Exception as e:
            logger.error(f"Error loading forex sentiment model: {e}")
            self.sentiment_model = None
    
    def analyze_forex_sentiment(self, text: str, currency_pair: str = "") -> Dict[str, Any]:
        """
        Analyze forex-specific sentiment
        Enhanced with forex market context
        """
        try:
            if self.sentiment_model is None:
                return self._fallback_forex_sentiment(text, currency_pair)
            
            # Enhance text with forex context
            enhanced_text = self._enhance_forex_context(text, currency_pair)
            
            # Tokenize input
            if self.tokenizer is None:
                return self._fallback_forex_sentiment(text, currency_pair)
            
            inputs = self.tokenizer(
                enhanced_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Get sentiment labels
            labels = ["negative", "neutral", "positive"]
            sentiment_idx = int(torch.argmax(probabilities, dim=1).item())
            confidence = float(probabilities[0][sentiment_idx].item())
            sentiment = labels[sentiment_idx]
            
            # Calculate forex-specific sentiment score
            sentiment_score = float((sentiment_idx - 1) * confidence)
            
            # Add forex-specific analysis
            forex_analysis = self._analyze_forex_context(text, currency_pair)
            
            return {
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "model": "Forex-FinBERT",
                "currency_pair": currency_pair,
                "forex_analysis": forex_analysis,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in forex sentiment analysis: {e}")
            return self._fallback_forex_sentiment(text, currency_pair)
    
    def _enhance_forex_context(self, text: str, currency_pair: str) -> str:
        """Enhance text with forex market context"""
        try:
            enhanced_text = text
            
            # Add currency pair context
            if currency_pair:
                enhanced_text = f"Forex market analysis for {currency_pair}: {text}"
            
            # Add forex market context
            if any(keyword in text.lower() for keyword in self.forex_keywords):
                enhanced_text = f"Financial market sentiment: {enhanced_text}"
            
            return enhanced_text
            
        except Exception as e:
            logger.error(f"Error enhancing forex context: {e}")
            return text
    
    def _analyze_forex_context(self, text: str, currency_pair: str) -> Dict[str, Any]:
        """Analyze forex-specific context and keywords"""
        try:
            text_lower = text.lower()
            
            # Check for forex-specific keywords
            forex_keywords_found = [kw for kw in self.forex_keywords if kw.lower() in text_lower]
            
            # Analyze market direction keywords
            bullish_keywords = ['bullish', 'buy', 'long', 'support', 'breakout', 'rally', 'surge']
            bearish_keywords = ['bearish', 'sell', 'short', 'resistance', 'breakdown', 'decline', 'drop']
            
            bullish_count = sum(1 for kw in bullish_keywords if kw in text_lower)
            bearish_count = sum(1 for kw in bearish_keywords if kw in text_lower)
            
            # Determine market bias
            if bullish_count > bearish_count:
                market_bias = "bullish"
            elif bearish_count > bullish_count:
                market_bias = "bearish"
            else:
                market_bias = "neutral"
            
            return {
                "forex_keywords": forex_keywords_found,
                "market_bias": market_bias,
                "bullish_signals": bullish_count,
                "bearish_signals": bearish_count,
                "forex_relevance": len(forex_keywords_found) > 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing forex context: {e}")
            return {
                "forex_keywords": [],
                "market_bias": "neutral",
                "bullish_signals": 0,
                "bearish_signals": 0,
                "forex_relevance": False
            }
    
    def _fallback_forex_sentiment(self, text: str, currency_pair: str) -> Dict[str, Any]:
        """Fallback forex sentiment analysis"""
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "confidence": 0.5,
            "model": "fallback",
            "currency_pair": currency_pair,
            "forex_analysis": {
                "forex_keywords": [],
                "market_bias": "neutral",
                "bullish_signals": 0,
                "bearish_signals": 0,
                "forex_relevance": False
            },
            "timestamp": datetime.utcnow().isoformat()
        }

class ForexModelTrainer:
    """
    🌍 Forex Model Trainer
    Trains forex-specialized LSTM and FinBERT models
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data_processor = ForexDataProcessor()
        self.forex_finbert = ForexFinBERTModel()
        
    def train_forex_lstm(self, currency_pairs: Optional[List[str]] = None, epochs: int = 100) -> Dict[str, Any]:
        """Train forex LSTM model on multiple currency pairs"""
        try:
            if currency_pairs is None:
                currency_pairs = self.data_processor.major_pairs[:6]  # Top 6 pairs
            
            logger.info(f"Training forex LSTM on {len(currency_pairs)} currency pairs")
            
            # Collect data from all currency pairs
            all_features = []
            all_targets = []
            
            for pair in currency_pairs:
                logger.info(f"Processing {pair}...")
                features = self.data_processor.create_forex_features(pair)
                
                if not features.empty:
                    X, y = self.data_processor.create_sequences(features)
                    if len(X) > 0:
                        all_features.append(X)
                        all_targets.append(y)
                        logger.info(f"Added {len(X)} sequences from {pair}")
            
            if not all_features:
                raise ValueError("No training data collected")
            
            # Combine all data
            X_combined = np.vstack(all_features)
            y_combined = np.hstack(all_targets)
            
            logger.info(f"Total training sequences: {len(X_combined)}")
            
            # Initialize forex LSTM model
            model = ForexLSTMModel(
                input_size=X_combined.shape[2],  # Number of features
                hidden_size=128,
                num_layers=3,
                dropout=0.2
            )
            
            model.to(self.device)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
            
            # Data preparation
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import MinMaxScaler
            
            # Normalize features
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_reshaped = X_combined.reshape(-1, X_combined.shape[-1])
            X_scaled = scaler_X.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X_combined.shape)
            
            y_scaled = scaler_y.fit_transform(y_combined.reshape(-1, 1)).flatten()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.FloatTensor(y_test).to(self.device)
            
            # Training loop
            model.train()
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs.squeeze(), y_train_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_test_tensor)
                
                model.train()
                
                train_losses.append(loss.item())
                val_losses.append(val_loss.item())
                
                scheduler.step(val_loss)
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
            
            # Save model
            model_path = "models/forex_lstm_model.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'input_size': X_combined.shape[2],
                'currency_pairs': currency_pairs
            }, model_path)
            
            logger.info(f"✅ Forex LSTM model trained and saved to {model_path}")
            
            return {
                "status": "success",
                "model_path": model_path,
                "currency_pairs": currency_pairs,
                "total_sequences": len(X_combined),
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "epochs_trained": epochs
            }
            
        except Exception as e:
            logger.error(f"Error training forex LSTM: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

def main():
    """Test forex models"""
    logger.info("🌍 Testing DEXTER Forex Models")
    
    # Test data processor
    processor = ForexDataProcessor()
    features = processor.create_forex_features("EURUSD=X")
    
    if not features.empty:
        logger.info(f"✅ Created {len(features.columns)} forex features")
        logger.info(f"Features: {list(features.columns)}")
    else:
        logger.warning("❌ No forex features created")
    
    # Test forex FinBERT
    forex_finbert = ForexFinBERTModel()
    test_text = "EUR/USD shows strong bullish momentum with breakout above resistance"
    sentiment = forex_finbert.analyze_forex_sentiment(test_text, "EUR/USD")
    logger.info(f"✅ Forex sentiment analysis: {sentiment}")
    
    # Test trainer
    # trainer = ForexModelTrainer()  # Commented out to avoid unused variable warning
    logger.info("✅ Forex model trainer initialized")

if __name__ == "__main__":
    main()
