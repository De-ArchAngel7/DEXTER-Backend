#!/usr/bin/env python3
"""
🚀 DEXTER CRYPTO LSTM INTEGRATION SCRIPT
=======================================
Integrates the trained crypto LSTM model into DEXTER's AI fusion engine
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import os
from sklearn.preprocessing import MinMaxScaler
import structlog

logger = structlog.get_logger()

class DexterCryptoLSTM(nn.Module):
    """
    🚀 DEXTER Crypto LSTM Model
    Trained model for crypto price prediction
    Architecture matches the trained best_lstm_model.pth
    """
    
    def __init__(self, input_size=29, hidden_size=64, num_layers=2, dropout=0.2):
        super(DexterCryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers - matches trained model architecture
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers - matches trained model architecture
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        out = lstm_out[:, -1, :]
        
        # Apply dropout and fully connected layers
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class DexterCryptoPredictor:
    """
    🚀 DEXTER Crypto Price Predictor
    Integrates trained LSTM model for crypto predictions
    """
    
    def __init__(self, model_path: str = "backend/ai_module/models/best_lstm_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.y_scaler = None
        
        # Load the trained model
        self.load_model()
    
    def load_model(self):
        """Load the trained crypto LSTM model and scalers"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Trained model not found at {self.model_path}")
                return
            
            # Load the checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            # Extract model state dict and scalers
            model_state_dict = checkpoint['model_state_dict']
            self.scaler = checkpoint.get('scaler_X')
            self.y_scaler = checkpoint.get('scaler_y')
            
            # Create model with correct architecture
            self.model = DexterCryptoLSTM(
                input_size=29,  # From the trained model
                hidden_size=64,
                num_layers=2,
                dropout=0.2
            )
            
            # Load the trained weights
            self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✅ Loaded trained DEXTER Crypto LSTM from {self.model_path}")
            logger.info(f"✅ Loaded scalers: X_scaler={self.scaler is not None}, y_scaler={self.y_scaler is not None}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load trained crypto LSTM: {e}")
            self.model = None
            self.scaler = None
            self.y_scaler = None
    
    def predict_crypto_price(self, features: np.ndarray, symbol: str = "BTC-USD") -> Dict[str, Any]:
        """
        Predict crypto price using the trained LSTM model
        
        Args:
            features: Input features array (shape: [sequence_length, 29])
            symbol: Crypto symbol for context
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                return self._fallback_prediction(symbol)
            
            # Clean and validate features
            features = self._clean_features(features)
            
            if self.scaler is None:
                logger.warning("No scaler available, using raw features")
                scaled_features = features
            else:
                # Scale the features - reshape to (samples, features) for scaling
                original_shape = features.shape
                features_2d = features.reshape(-1, features.shape[-1])  # (sequence_length, 29)
                scaled_2d = self.scaler.transform(features_2d)
                scaled_features = scaled_2d.reshape(original_shape)
            
            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(features_tensor)
                predicted_price = prediction.cpu().numpy()[0][0]
            
            # Inverse transform if y_scaler is available
            if self.y_scaler is not None:
                predicted_price = self.y_scaler.inverse_transform([[predicted_price]])[0][0]
            
            # Calculate direction based on prediction
            current_price = features[-1, 0] if len(features) > 0 else 1000  # Assume first feature is price
            price_change = (predicted_price - current_price) / current_price
            
            if price_change > 0.02:  # 2% threshold
                direction = "BUY"
                confidence = min(0.95, 0.7 + abs(price_change) * 10)
            elif price_change < -0.02:  # -2% threshold
                direction = "SELL"
                confidence = min(0.95, 0.7 + abs(price_change) * 10)
            else:
                direction = "HOLD"
                confidence = 0.6
            
            return {
                "symbol": symbol,
                "predicted_price": float(predicted_price),
                "current_price": float(current_price),
                "price_change": float(price_change),
                "direction": direction,
                "confidence": float(confidence),
                "model": "DexterCryptoLSTM",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in crypto price prediction: {e}")
            return self._fallback_prediction(symbol)
    
    def _clean_features(self, features: np.ndarray) -> np.ndarray:
        """Clean features by handling infinity and NaN values"""
        try:
            # Replace infinity with large finite values
            features = np.where(np.isinf(features), np.nan, features)
            
            # Replace NaN with forward fill, then backward fill
            features_df = pd.DataFrame(features)
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')
            
            # If still NaN, replace with 0
            features_df = features_df.fillna(0)
            
            # Clip extreme values to prevent overflow
            features_df = features_df.clip(-1e6, 1e6)
            
            return features_df.values
            
        except Exception as e:
            logger.error(f"Error cleaning features: {e}")
            # Return a safe default
            return np.zeros((features.shape[0], features.shape[1]))
    
    def _fallback_prediction(self, symbol: str) -> Dict[str, Any]:
        """Fallback prediction when model fails"""
        return {
            "symbol": symbol,
            "predicted_price": 1000.0,
            "current_price": 1000.0,
            "price_change": 0.0,
            "direction": "HOLD",
            "confidence": 0.5,
            "model": "fallback",
            "timestamp": pd.Timestamp.now().isoformat()
        }

class CryptoDataProcessor:
    """
    🚀 Crypto Data Processor
    Processes crypto data for LSTM prediction
    """
    
    def __init__(self):
        self.sequence_length = 60  # 60 time steps for prediction
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
            'RSI', 'MACD', 'MACD_signal', 'MACD_histogram',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'Stochastic_K', 'Stochastic_D',
            'Williams_R', 'ATR', 'CCI',
            'Price_Change', 'Volume_Change', 'Volatility'
        ]
    
    def create_crypto_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced crypto features for LSTM prediction
        Matches the 29 features expected by the trained model
        """
        try:
            features = pd.DataFrame(index=data.index)
            
            # Basic OHLCV features
            features['Open'] = data['Open']
            features['High'] = data['High']
            features['Low'] = data['Low']
            features['Close'] = data['Close']
            features['Volume'] = data['Volume']
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                features[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
                features[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
            
            # RSI
            features['RSI'] = self._calculate_rsi(data['Close'])
            
            # MACD
            macd, signal, histogram = self._calculate_macd(data['Close'])
            features['MACD'] = macd
            features['MACD_signal'] = signal
            features['MACD_histogram'] = histogram
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(data['Close'])
            features['BB_upper'] = bb_upper
            features['BB_middle'] = bb_middle
            features['BB_lower'] = bb_lower
            features['BB_width'] = (bb_upper - bb_lower) / bb_middle
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(data['High'], data['Low'], data['Close'])
            features['Stochastic_K'] = stoch_k
            features['Stochastic_D'] = stoch_d
            
            # Williams %R
            features['Williams_R'] = self._calculate_williams_r(data['High'], data['Low'], data['Close'])
            
            # ATR (Average True Range)
            features['ATR'] = self._calculate_atr(data['High'], data['Low'], data['Close'])
            
            # CCI (Commodity Channel Index)
            features['CCI'] = self._calculate_cci(data['High'], data['Low'], data['Close'])
            
            # Price and volume changes
            features['Price_Change'] = data['Close'].pct_change()
            features['Volume_Change'] = data['Volume'].pct_change()
            
            # Volatility
            features['Volatility'] = data['Close'].rolling(window=20).std()
            
            # Fill NaN values with safe defaults
            features = features.fillna(method='ffill').fillna(method='bfill')
            
            # Replace any remaining NaN with 0
            features = features.fillna(0)
            
            # Replace infinity values
            features = features.replace([np.inf, -np.inf], 0)
            
            # Ensure we have exactly 29 features
            if len(features.columns) != 29:
                logger.warning(f"Expected 29 features, got {len(features.columns)}")
                # Pad or trim to 29 features
                if len(features.columns) < 29:
                    for i in range(len(features.columns), 29):
                        features[f'feature_{i}'] = 0.0
                else:
                    features = features.iloc[:, :29]
            
            # Final validation - ensure no NaN or inf values
            features = features.replace([np.inf, -np.inf], 0).fillna(0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error creating crypto features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> tuple:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        return upper_band, sma, lower_band
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> tuple:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
