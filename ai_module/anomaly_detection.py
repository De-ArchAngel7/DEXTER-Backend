import numpy as np
import pandas as pd
from typing import Dict, Any
import structlog
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination="auto",  # Use "auto" instead of float
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def detect_price_anomalies(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Detect price anomalies in real-time"""
        try:
            # Get market data
            data = self._get_market_data(symbol, period)
            
            if data.empty:
                return self._get_fallback_anomaly_result(symbol)
            
            # Calculate anomaly features
            features = self._extract_anomaly_features(data)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(features)
            
            # Analyze anomaly patterns
            anomaly_analysis = self._analyze_anomalies(data, anomalies)
            
            return {
                "symbol": symbol,
                "anomalies_detected": len(anomaly_analysis["anomaly_points"]),
                "anomaly_score": anomaly_analysis["overall_score"],
                "risk_level": anomaly_analysis["risk_level"],
                "anomaly_points": anomaly_analysis["anomaly_points"],
                "market_conditions": anomaly_analysis["market_conditions"],
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": "isolation_forest"
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies for {symbol}: {e}")
            return self._get_fallback_anomaly_result(symbol)
    
    def _get_market_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get market data for anomaly detection"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1h")
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate additional features for anomaly detection
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=24).std()
            data['Volume_MA'] = data['Volume'].rolling(window=24).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            data['Price_MA'] = data['Close'].rolling(window=24).mean()
            data['Price_Deviation'] = (data['Close'] - data['Price_MA']) / data['Price_MA']
            
            return data.dropna()
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def _extract_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for anomaly detection"""
        try:
            features = []
            
            # Price-based features
            features.append(data['Returns'].values)
            features.append(data['Volatility'].values)
            features.append(data['Price_Deviation'].values)
            
            # Volume-based features
            features.append(data['Volume_Ratio'].values)
            
            # Technical indicators - ensure we pass Series, not DataFrame
            close_series = pd.Series(data['Close'].values, index=data.index)  # Explicitly create Series
            features.append(self._calculate_rsi(close_series).values)
            features.append(self._calculate_macd(close_series).values)
            
            # Convert to numpy array
            feature_matrix = np.column_stack(features)
            
            # Remove rows with NaN values
            feature_matrix = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]
            
            return feature_matrix
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        # Ensure we return a Series, handle NaN values
        if isinstance(rsi, pd.Series):
            rsi = rsi.fillna(50.0)  # Fill NaN with neutral RSI value
        else:
            # If rsi is a scalar, convert to Series
            rsi = pd.Series([50.0] * len(prices), index=prices.index)
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        # Ensure we return a Series, handle NaN values
        if isinstance(macd, pd.Series):
            macd = macd.fillna(0.0)  # Fill NaN with neutral MACD value
        else:
            # If macd is a scalar, convert to Series
            macd = pd.Series([0.0] * len(prices), index=prices.index)
        return macd
    
    def _detect_anomalies(self, features: np.ndarray) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        try:
            if features.size == 0:
                return np.array([])
            
            # Scale features
            scaled_features = self.scaler.fit_transform(features)
            
            # Fit model if not already fitted
            if not self.is_fitted:
                self.isolation_forest.fit(scaled_features)
                self.is_fitted = True
            
            # Predict anomalies (-1 for anomaly, 1 for normal)
            predictions = self.isolation_forest.predict(scaled_features)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return np.array([])
    
    def _analyze_anomalies(self, data: pd.DataFrame, anomalies: np.ndarray) -> Dict[str, Any]:
        """Analyze detected anomalies"""
        try:
            if anomalies.size == 0:
                return self._get_no_anomalies_result()
            
            # Find anomaly points
            anomaly_indices = np.where(anomalies == -1)[0]
            anomaly_points = []
            
            for idx in anomaly_indices:
                if idx < len(data):
                    point = data.iloc[idx]
                    anomaly_points.append({
                        "timestamp": point.name.isoformat(),
                        "price": point['Close'],
                        "volume": point['Volume'],
                        "returns": point['Returns'],
                        "volatility": point['Volatility']
                    })
            
            # Calculate overall anomaly score
            anomaly_ratio = len(anomaly_indices) / len(anomalies)
            overall_score = min(100, anomaly_ratio * 1000)  # Scale to 0-100
            
            # Determine risk level
            risk_level = self._determine_risk_level(overall_score, anomaly_points)
            
            # Analyze market conditions
            market_conditions = self._analyze_market_conditions(data)
            
            return {
                "anomaly_points": anomaly_points,
                "overall_score": overall_score,
                "risk_level": risk_level,
                "market_conditions": market_conditions
            }
            
        except Exception as e:
            logger.error(f"Error analyzing anomalies: {e}")
            return self._get_no_anomalies_result()
    
    def _determine_risk_level(self, score: float, anomaly_points: list) -> str:
        """Determine risk level based on anomaly score and points"""
        try:
            if score >= 80:
                return "CRITICAL"
            elif score >= 60:
                return "HIGH"
            elif score >= 40:
                return "MEDIUM"
            elif score >= 20:
                return "LOW"
            else:
                return "MINIMAL"
                
        except Exception as e:
            logger.error(f"Error determining risk level: {e}")
            return "UNKNOWN"
    
    def _analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        try:
            recent_data = data.tail(24)  # Last 24 hours
            
            # Volatility analysis
            current_volatility = recent_data['Volatility'].iloc[-1]
            avg_volatility = data['Volatility'].mean()
            volatility_status = "HIGH" if current_volatility > avg_volatility * 1.5 else "NORMAL"
            
            # Volume analysis
            current_volume_ratio = recent_data['Volume_Ratio'].iloc[-1]
            volume_status = "HIGH" if current_volume_ratio > 2.0 else "NORMAL"
            
            # Trend analysis
            price_trend = "BULLISH" if recent_data['Close'].iloc[-1] > recent_data['Close'].iloc[0] else "BEARISH"
            
            return {
                "volatility": volatility_status,
                "volume": volume_status,
                "trend": price_trend,
                "current_volatility": float(current_volatility),
                "avg_volatility": float(avg_volatility),
                "volume_ratio": float(current_volume_ratio)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            return {
                "volatility": "UNKNOWN",
                "volume": "UNKNOWN",
                "trend": "UNKNOWN"
            }
    
    def _get_no_anomalies_result(self) -> Dict[str, Any]:
        """Result when no anomalies are detected"""
        return {
            "anomaly_points": [],
            "overall_score": 0.0,
            "risk_level": "MINIMAL",
            "market_conditions": {
                "volatility": "NORMAL",
                "volume": "NORMAL",
                "trend": "STABLE"
            }
        }
    
    def _get_fallback_anomaly_result(self, symbol: str) -> Dict[str, Any]:
        """Fallback result when anomaly detection fails"""
        return {
            "symbol": symbol,
            "anomalies_detected": 0,
            "anomaly_score": 0.0,
            "risk_level": "UNKNOWN",
            "anomaly_points": [],
            "market_conditions": {
                "volatility": "UNKNOWN",
                "volume": "UNKNOWN",
                "trend": "UNKNOWN"
            },
            "timestamp": datetime.utcnow().isoformat(),
            "model_used": "fallback"
        }
    
    def get_market_alert(self, symbol: str) -> Dict[str, Any]:
        """Get market alert based on anomaly detection"""
        try:
            anomaly_result = self.detect_price_anomalies(symbol)
            
            # Generate alert message
            alert_message = self._generate_alert_message(anomaly_result)
            
            # Determine action recommendation
            action = self._recommend_action(anomaly_result)
            
            return {
                "symbol": symbol,
                "alert_level": anomaly_result["risk_level"],
                "message": alert_message,
                "action": action,
                "anomaly_score": anomaly_result["anomaly_score"],
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating market alert: {e}")
            return {
                "symbol": symbol,
                "alert_level": "UNKNOWN",
                "message": "Unable to generate market alert",
                "action": "MONITOR",
                "anomaly_score": 0.0,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _generate_alert_message(self, anomaly_result: Dict[str, Any]) -> str:
        """Generate human-readable alert message"""
        try:
            risk_level = anomaly_result["risk_level"]
            score = anomaly_result["anomaly_score"]
            anomalies_count = anomaly_result["anomalies_detected"]
            
            if risk_level == "CRITICAL":
                return f"🚨 CRITICAL ALERT: {anomalies_count} severe anomalies detected! Market conditions are extremely unstable. Score: {score:.1f}"
            elif risk_level == "HIGH":
                return f"⚠️ HIGH RISK: {anomalies_count} significant anomalies detected. Exercise extreme caution. Score: {score:.1f}"
            elif risk_level == "MEDIUM":
                return f"🔶 MEDIUM RISK: {anomalies_count} moderate anomalies detected. Monitor closely. Score: {score:.1f}"
            elif risk_level == "LOW":
                return f"🟡 LOW RISK: {anomalies_count} minor anomalies detected. Normal trading conditions. Score: {score:.1f}"
            else:
                return f"✅ MINIMAL RISK: No significant anomalies detected. Market appears stable. Score: {score:.1f}"
                
        except Exception as e:
            logger.error(f"Error generating alert message: {e}")
            return "Unable to generate alert message"
    
    def _recommend_action(self, anomaly_result: Dict[str, Any]) -> str:
        """Recommend trading action based on anomaly detection"""
        try:
            risk_level = anomaly_result["risk_level"]
            
            if risk_level == "CRITICAL":
                return "IMMEDIATE_STOP"
            elif risk_level == "HIGH":
                return "REDUCE_POSITION"
            elif risk_level == "MEDIUM":
                return "MONITOR_CLOSELY"
            elif risk_level == "LOW":
                return "NORMAL_TRADING"
            else:
                return "MONITOR"
                
        except Exception as e:
            logger.error(f"Error recommending action: {e}")
            return "MONITOR"
