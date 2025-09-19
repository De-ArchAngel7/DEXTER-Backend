#!/usr/bin/env python3
"""
🌍 DEXTER FOREX LSTM INTEGRATION SCRIPT
=====================================
Integrates the trained forex LSTM model into DEXTER's AI fusion engine
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any
import os

class DexterForexLSTM(nn.Module):
    """
    🌍 DEXTER Forex LSTM Model
    Trained model for forex price prediction
    """
    
    def __init__(self, input_size=40, hidden_size=128, num_layers=3, dropout=0.3):
        super(DexterForexLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Multi-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Advanced feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.LayerNorm(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 8, 1)
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply multi-head attention
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Feature extraction
        features = self.feature_extractor(pooled)
        
        # Output prediction
        output = self.output_layers(features)
        
        return output

class DexterForexPredictor:
    """
    🌍 DEXTER Forex Price Predictor
    Integrates trained LSTM model for forex predictions
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.input_size = 40
        self.sequence_length = 24
        
        # Load model if path exists
        if os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained model and scalers"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Initialize model
            self.model = DexterForexLSTM(
                input_size=checkpoint['input_size'],
                hidden_size=128,
                num_layers=3,
                dropout=0.3
            ).to(self.device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load scalers
            self.scaler = checkpoint['scaler']
            self.y_scaler = checkpoint['y_scaler']
            
            print(f"✅ Loaded DEXTER Forex LSTM model from {model_path}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict_forex_price(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Predict forex price using the trained LSTM model
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            symbol: Currency pair symbol (e.g., 'EURUSD=X')
        
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': 0.0
            }
        
        try:
            # Prepare features (exclude target column)
            feature_cols = [col for col in data.columns if col != 'Close']
            features = data[feature_cols].values
            
            # Normalize features
            if self.scaler is None:
                return {
                    'success': False,
                    'error': 'Scaler not loaded',
                    'prediction': None,
                    'confidence': 0.0
                }
            features_scaled = self.scaler.transform(features)
            
            # Create sequence
            if len(features_scaled) < self.sequence_length:
                return {
                    'success': False,
                    'error': f'Insufficient data. Need {self.sequence_length} points, got {len(features_scaled)}',
                    'prediction': None,
                    'confidence': 0.0
                }
            
            # Get last sequence
            sequence = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction_scaled = self.model(sequence_tensor).cpu().numpy()
            
            # Denormalize prediction
            if self.y_scaler is None:
                return {
                    'success': False,
                    'error': 'Y-scaler not loaded',
                    'prediction': None,
                    'confidence': 0.0
                }
            prediction = self.y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]
            
            # Calculate confidence based on recent volatility
            recent_volatility = data['Close'].pct_change().tail(10).std()
            confidence = max(0.1, min(0.9, 1.0 - recent_volatility * 10))
            
            return {
                'success': True,
                'prediction': float(prediction),
                'confidence': float(confidence),
                'symbol': symbol,
                'model_type': 'DEXTER_Forex_LSTM',
                'features_used': len(feature_cols),
                'sequence_length': self.sequence_length
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': 0.0
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'input_size': self.input_size,
            'sequence_length': self.sequence_length,
            'device': str(self.device),
            'model_type': 'DEXTER_Forex_LSTM',
            'features': [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Position',
                'Stoch_K', 'Stoch_D', 'Williams_R', 'ATR', 'CCI',
                'Price_Change', 'Price_Range', 'Body_Size', 'Upper_Shadow', 'Lower_Shadow',
                'Volatility_10', 'Volatility_20', 'Volatility_50', 'Volatility_Ratio',
                'Momentum_5', 'Momentum_10', 'Momentum_20',
                'Volume_Change', 'Volume_SMA', 'Volume_Ratio'
            ]
        }

# Test function
def test_forex_predictor():
    """Test the forex predictor with sample data"""
    print("🌍 Testing DEXTER Forex Predictor...")
    
    # Initialize predictor
    predictor = DexterForexPredictor('best_forex_lstm_model.pth')
    
    # Get model info
    info = predictor.get_model_info()
    print(f"Model loaded: {info['loaded']}")
    
    if info['loaded']:
        print(f"Input size: {info['input_size']}")
        print(f"Sequence length: {info['sequence_length']}")
        print(f"Features: {len(info['features'])}")
        
        # Create sample data for testing
        sample_data = pd.DataFrame({
            'Close': np.random.uniform(1.0, 1.2, 50),
            'Open': np.random.uniform(1.0, 1.2, 50),
            'High': np.random.uniform(1.0, 1.2, 50),
            'Low': np.random.uniform(1.0, 1.2, 50),
            'Volume': np.random.randint(1000, 10000, 50)
        })
        
        # Add some technical indicators (simplified)
        for i in range(35):
            sample_data[f'Feature_{i}'] = np.random.normal(0, 1, 50)
        
        # Test prediction
        result = predictor.predict_forex_price(sample_data, 'EURUSD=X')
        print(f"Prediction result: {result}")
    else:
        print("❌ Model not loaded - please check model path")

if __name__ == "__main__":
    test_forex_predictor()
