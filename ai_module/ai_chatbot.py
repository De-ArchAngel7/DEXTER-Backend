import openai
from typing import Dict, List, Any
import structlog
from datetime import datetime
from .models.price_prediction import PricePredictionModel
from .sentiment_analysis import SentimentAnalyzer
from .anomaly_detection import AnomalyDetector
from .rl_trading_agent import RLTradingAgent
import numpy as np

logger = structlog.get_logger()

class AIChatbot:
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        openai.api_key = openai_api_key
        
        self.price_model = PricePredictionModel(
            model_type="lstm",
            model_params={
                "input_size": 4,
                "hidden_size": 64,
                "num_layers": 2,
                "output_size": 1
            }
        )
        self.sentiment_analyzer = SentimentAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.rl_agent = RLTradingAgent()
        
        self.system_prompt = """You are DEXTER, an expert AI trading advisor with deep knowledge of cryptocurrency markets, technical analysis, and risk management. 

Your role is to:
1. Analyze market conditions and provide strategic trading advice
2. Interpret technical indicators and price patterns
3. Assess risk levels and suggest position sizing
4. Provide educational insights about trading strategies
5. Always prioritize risk management and capital preservation

Guidelines:
- Be precise and actionable in your advice
- Include confidence levels and reasoning
- Suggest stop-loss and take-profit levels when appropriate
- Consider market volatility and current conditions
- Never guarantee profits - always emphasize risk
- Use technical analysis and fundamental factors
- Provide timeframes for your recommendations

Current market context: You have access to real-time price data, sentiment analysis, AI-powered price predictions, anomaly detection, and reinforcement learning trading signals."""
    
    async def get_trading_advice(self, 
                                symbol: str, 
                                current_price: float, 
                                market_data: Dict[str, Any],
                                user_question: str) -> Dict[str, Any]:
        try:
            price_prediction = await self._get_price_prediction(symbol, market_data)
            sentiment_analysis = await self._get_sentiment_analysis(symbol)
            anomaly_detection = self._get_anomaly_detection(symbol)
            rl_recommendation = self._get_rl_recommendation(symbol, current_price, market_data)
            
            context = self._prepare_market_context(
                symbol, current_price, market_data, 
                price_prediction, sentiment_analysis, 
                anomaly_detection, rl_recommendation
            )
            
            advice = await self._generate_openai_advice(user_question, context)
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "advice": advice,
                "ai_predictions": {
                    "price_prediction": price_prediction,
                    "sentiment_analysis": sentiment_analysis,
                    "anomaly_detection": anomaly_detection,
                    "rl_recommendation": rl_recommendation
                },
                "generated_at": datetime.utcnow().isoformat(),
                "confidence": self._calculate_overall_confidence(
                    price_prediction, sentiment_analysis, 
                    anomaly_detection, rl_recommendation
                )
            }
            
        except Exception as e:
            logger.error(f"Error generating trading advice: {e}")
            return {
                "error": str(e),
                "symbol": symbol,
                "generated_at": datetime.utcnow().isoformat()
            }
    
    async def _get_price_prediction(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get price prediction from PyTorch model"""
        try:
            features = self._extract_features(market_data)
            
            prediction = self.price_model.predict(features)
            
            predicted_price = float(prediction[0][0]) if prediction.size > 0 else 0
            current_price = market_data.get("price_usd", 1)
            
            if predicted_price > current_price * 1.02:
                trend = "bullish"
            elif predicted_price < current_price * 0.98:
                trend = "bearish"
            else:
                trend = "neutral"
            
            price_change = abs(predicted_price - current_price) / current_price
            confidence = min(95, max(50, 70 + (price_change * 100)))
            
            return {
                "predicted_price": predicted_price,
                "confidence": confidence,
                "trend": trend,
                "timeframe": "24H",
                "model_used": "LSTM_Model"
            }
        except Exception as e:
            logger.error(f"Error getting price prediction: {e}")
            return {
                "predicted_price": 0,
                "confidence": 0,
                "trend": "neutral",
                "timeframe": "24H",
                "model_used": "fallback"
            }
    
    async def _get_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment analysis for a symbol"""
        try:
            # Analyze social media sentiment
            sentiment = self.sentiment_analyzer.analyze_text(f"cryptocurrency {symbol} trading")
            
            return {
                "sentiment": sentiment.get("sentiment", "neutral"),
                "confidence": sentiment.get("confidence", 0),
                "compound_score": sentiment.get("compound_score", 0),
                "social_volume": 1000,  # Mock data for now
                "trending_topics": ["AI adoption", "regulation", "institutional adoption"]
            }
        except Exception as e:
            logger.error(f"Error getting sentiment analysis: {e}")
            return {
                "sentiment": "neutral",
                "confidence": 0,
                "compound_score": 0,
                "social_volume": 0,
                "trending_topics": []
            }
    
    def _get_anomaly_detection(self, symbol: str) -> Dict[str, Any]:
        """Get anomaly detection for a symbol"""
        try:
            return self.anomaly_detector.detect_price_anomalies(symbol)
        except Exception as e:
            logger.error(f"Error getting anomaly detection: {e}")
            return {
                "anomalies_detected": 0,
                "anomaly_score": 0,
                "risk_level": "UNKNOWN",
                "model_used": "fallback"
            }
    
    def _get_rl_recommendation(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get reinforcement learning trading recommendation"""
        try:
            return self.rl_agent.get_trading_action(symbol, current_price, market_data)
        except Exception as e:
            logger.error(f"Error getting RL recommendation: {e}")
            return {
                "action": "HOLD",
                "confidence": 50.0,
                "recommendation": "RL agent unavailable",
                "model_used": "fallback"
            }
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features for price prediction model"""
        try:
            features = []
            
            # Price features
            if "price_usd" in market_data:
                features.append(market_data["price_usd"])
            if "price_change_24h" in market_data:
                features.append(market_data["price_change_24h"])
            if "volume_24h" in market_data:
                features.append(market_data["volume_24h"])
            if "market_cap" in market_data:
                features.append(market_data["market_cap"])
            
            # Add default values if features are missing
            while len(features) < 4:
                features.append(0.0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return np.array([[0.0, 0.0, 0.0, 0.0]])
    
    def _prepare_market_context(self, 
                               symbol: str, 
                               current_price: float, 
                               market_data: Dict[str, Any],
                               price_prediction: Dict[str, Any],
                               sentiment_analysis: Dict[str, Any],
                               anomaly_detection: Dict[str, Any],
                               rl_recommendation: Dict[str, Any]) -> str:
        """Prepare market context for OpenAI"""
        
        context = f"""
Symbol: {symbol}
Current Price: ${current_price:,.2f}
24h Change: {market_data.get('price_change_24h', 0):.2f}%
24h Volume: ${(market_data.get('volume_24h', 0) / 1e6):.2f}M
Market Cap: ${(market_data.get('market_cap', 0) / 1e9):.2f}B

AI Price Prediction:
- Predicted Price: ${price_prediction.get('predicted_price', 0):,.2f}
- Trend: {price_prediction.get('trend', 'neutral')}
- Confidence: {price_prediction.get('confidence', 0):.1f}%

Sentiment Analysis:
- Overall Sentiment: {sentiment_analysis.get('sentiment', 'neutral')}
- Sentiment Score: {sentiment_analysis.get('compound_score', 0):.3f}
- Confidence: {sentiment_analysis.get('confidence', 0):.1f}%

Anomaly Detection:
- Risk Level: {anomaly_detection.get('risk_level', 'UNKNOWN')}
- Anomalies Detected: {anomaly_detection.get('anomalies_detected', 0)}
- Anomaly Score: {anomaly_detection.get('anomaly_score', 0):.1f}

Reinforcement Learning:
- Recommended Action: {rl_recommendation.get('action', 'HOLD')}
- Confidence: {rl_recommendation.get('confidence', 0):.1f}%
- Recommendation: {rl_recommendation.get('recommendation', 'No recommendation available')}
"""
        return context
    
    async def _generate_openai_advice(self, user_question: str, market_context: str) -> str:
        """Generate trading advice using OpenAI GPT-4"""
        try:
            prompt = f"""
{market_context}

User Question: {user_question}

Please provide comprehensive trading advice including:
1. Market analysis and current conditions
2. Technical analysis insights
3. Risk assessment and position sizing recommendations
4. Entry/exit points with stop-loss and take-profit levels
5. Timeframe for the trade
6. Key factors to monitor
7. Anomaly warnings and risk alerts
8. AI model consensus analysis

Format your response in a clear, actionable manner.
"""
            
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(api_key=self.openai_api_key)
            response = await client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            return content if content else "Unable to generate AI advice at the moment."
            
        except Exception as e:
            logger.error(f"Error generating OpenAI advice: {e}")
            return f"Unable to generate AI advice at the moment. Error: {str(e)}"
    
    def _calculate_overall_confidence(self, 
                                    price_prediction: Dict[str, Any], 
                                    sentiment_analysis: Dict[str, Any],
                                    anomaly_detection: Dict[str, Any],
                                    rl_recommendation: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        try:
            price_conf = price_prediction.get("confidence", 0)
            sentiment_conf = sentiment_analysis.get("confidence", 0)
            rl_conf = rl_recommendation.get("confidence", 0)
            
            # Anomaly detection affects confidence (higher risk = lower confidence)
            anomaly_score = anomaly_detection.get("anomaly_score", 0)
            anomaly_factor = max(0.5, 1 - (anomaly_score / 100))
            
            # Weighted average (price prediction most important, anomaly detection as risk factor)
            overall_conf = (price_conf * 0.4 + sentiment_conf * 0.2 + rl_conf * 0.4) * anomaly_factor
            
            return min(overall_conf, 100.0)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0
    
    async def get_market_analysis(self, symbols: List[str]) -> Dict[str, Any]:
        """Get comprehensive market analysis for multiple symbols"""
        try:
            analysis = {}
            
            for symbol in symbols:
                # Get basic market data (this would come from your data service)
                market_data = {
                    "price_usd": 0,  # Mock data - replace with real data
                    "price_change_24h": 0,
                    "volume_24h": 0,
                    "market_cap": 0
                }
                
                price_pred = await self._get_price_prediction(symbol, market_data)
                sentiment = await self._get_sentiment_analysis(symbol)
                anomaly = self._get_anomaly_detection(symbol)
                rl_rec = self._get_rl_recommendation(symbol, 0, market_data)
                
                analysis[symbol] = {
                    "price_prediction": price_pred,
                    "sentiment_analysis": sentiment,
                    "anomaly_detection": anomaly,
                    "rl_recommendation": rl_rec,
                    "overall_score": self._calculate_overall_confidence(
                        price_pred, sentiment, anomaly, rl_rec
                    ),
                    "recommendation": self._generate_recommendation(
                        price_pred, sentiment, anomaly, rl_rec
                    )
                }
            
            return {
                "analysis": analysis,
                "generated_at": datetime.utcnow().isoformat(),
                "total_symbols": len(symbols)
            }
            
        except Exception as e:
            logger.error(f"Error getting market analysis: {e}")
            return {"error": str(e)}
    
    def _generate_recommendation(self, 
                                price_prediction: Dict[str, Any], 
                                sentiment_analysis: Dict[str, Any],
                                anomaly_detection: Dict[str, Any],
                                rl_recommendation: Dict[str, Any]) -> str:
        """Generate trading recommendation based on AI analysis"""
        try:
            price_conf = price_prediction.get("confidence", 0)
            sentiment_conf = sentiment_analysis.get("confidence", 0)
            trend = price_prediction.get("trend", "neutral")
            sentiment = sentiment_analysis.get("sentiment", "neutral")
            anomaly_risk = anomaly_detection.get("risk_level", "UNKNOWN")
            rl_action = rl_recommendation.get("action", "HOLD")
            
            # High anomaly risk overrides other signals
            if anomaly_risk in ["CRITICAL", "HIGH"]:
                return "IMMEDIATE_STOP"
            
            # High confidence bullish signals
            if (price_conf > 80 and sentiment_conf > 80 and 
                trend == "bullish" and sentiment == "positive" and
                rl_action in ["BUY", "STRONG_BUY"]):
                return "STRONG_BUY"
            
            # High confidence bearish signals
            elif (price_conf > 80 and sentiment_conf > 80 and 
                  trend == "bearish" and sentiment == "negative" and
                  rl_action in ["SELL", "STRONG_SELL"]):
                return "STRONG_SELL"
            
            # Moderate confidence signals
            elif price_conf > 70 and sentiment_conf > 70:
                if trend == "bullish" and rl_action == "BUY":
                    return "BUY"
                elif trend == "bearish" and rl_action == "SELL":
                    return "SELL"
                else:
                    return "HOLD"
            
            # Low confidence - wait for clearer signals
            else:
                return "WAIT"
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return "HOLD"
    
    def get_ai_system_status(self) -> Dict[str, Any]:
        """Get status of all AI models"""
        try:
            return {
                "price_prediction": {
                    "status": "active",
                    "model_type": "LSTM",
                    "confidence": "high"
                },
                "sentiment_analysis": {
                    "status": "active",
                    "model_type": "BERT",
                    "confidence": "high"
                },
                "anomaly_detection": {
                    "status": "active",
                    "model_type": "Isolation Forest",
                    "confidence": "medium"
                },
                "reinforcement_learning": {
                    "status": "active",
                    "model_type": "Q-Learning",
                    "confidence": "medium"
                },
                "overall_status": "operational",
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting AI system status: {e}")
            return {"overall_status": "error", "error": str(e)}
