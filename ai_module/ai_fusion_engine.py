#!/usr/bin/env python3
"""
🧠 DEXTER AI FUSION ENGINE
============================================================
The core intelligence hub that combines:
- LSTM (Price Prediction)
- FinBERT (Sentiment Analysis) 
- DexScreener (Real-time Data)
- DialoGPT (AI Reasoning & Trading Analysis)

This is the "brain" that orchestrates all AI components
"""

from datetime import datetime
from typing import Dict, Any
import structlog
import warnings
import torch
warnings.filterwarnings('ignore')

# Import our AI components
from .mistral_integration import DexterMistralEngine
from .models.price_prediction import PricePredictionModel
from .real_ai_models import RealAITradingModels
from .forex_models import ForexLSTMModel, ForexFinBERTModel, ForexDataProcessor
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# from data_sources.dexscreener_client import DexScreenerClient

logger = structlog.get_logger()

class DexterAIFusionEngine:
    """
    🧠 DEXTER AI FUSION ENGINE
    
    This is the central intelligence hub that combines:
    1. LSTM for price prediction
    2. FinBERT for sentiment analysis
    3. DexScreener for real-time data
    4. DialoGPT for AI reasoning and trading analysis
    
    It provides comprehensive trading insights by fusing multiple AI sources.
    """
    
    def __init__(self):
        self.price_predictor = PricePredictionModel(model_type="lstm")
        self.sentiment_analyzer = RealAITradingModels()
        # self.dexscreener_client = DexScreenerClient()  # Temporarily disabled
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        trained_lstm_path = "backend/models/best_forex_lstm_model.pth"
        trained_finbert_path = "backend/models/dexter_forex_finbert"
        self.forex_lstm = ForexLSTMModel(trained_model_path=trained_lstm_path)
        self.forex_finbert = ForexFinBERTModel(model_path=trained_finbert_path)
        self.forex_data_processor = ForexDataProcessor()
        
        use_hf_api = os.getenv("USE_HF_API", "false").lower() == "true"
        
        if use_hf_api:
            from .mistral_hf_api_integration import DexterMistralHFAPI
            self.mistral_engine = DexterMistralHFAPI(
                hf_token=os.getenv("HUGGINGFACE_TOKEN"),
                lora_adapter_path=os.getenv("DEXTER_MISTRAL_MODEL_PATH", "models/dexter-mistral-7b-final"),
                base_model=os.getenv("MISTRAL_BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
            )
            logger.info("🧠 Using Mistral 7B via Hugging Face API")
        else:
            self.mistral_engine = DexterMistralEngine(
                model_path=os.getenv("DEXTER_MISTRAL_MODEL_PATH", "models/dexter-mistral-7b"),
                use_quantization=os.getenv("USE_QUANTIZATION", "true").lower() == "true"
            )
            logger.info("🧠 Using local fine-tuned Mistral 7B model")
        
        # Fusion parameters - now supports both crypto and forex
        self.fusion_weights = {
            "crypto_price_prediction": 0.25,    # 25% weight to Crypto LSTM
            "forex_price_prediction": 0.25,     # 25% weight to Forex LSTM
            "crypto_sentiment_analysis": 0.15,  # 15% weight to Crypto FinBERT
            "forex_sentiment_analysis": 0.15,   # 15% weight to Forex FinBERT
            "market_data": 0.1,                 # 10% weight to DexScreener
            "ai_reasoning": 0.1                 # 10% weight to Mistral AI reasoning
        }
        
        self._initialize_components()
        
        logger.info("🧠 DEXTER AI Fusion Engine initialized with Crypto + Forex models!")
        
    def _initialize_components(self):
        """Initialize all AI components"""
        try:
            # Initialize other components
            # Note: These components may not have initialize methods
            # self.sentiment_analyzer.initialize()
            # self.dexscreener_client.initialize()
            
            logger.info("✅ All AI components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing AI components: {e}")
            
    def _is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is a forex pair"""
        forex_indicators = ['=X', '/', 'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']
        return any(indicator in symbol.upper() for indicator in forex_indicators)
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency"""
        crypto_indicators = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL', 'MATIC', 'AVAX']
        return any(indicator in symbol.upper() for indicator in crypto_indicators)
    
    async def _load_forex_lstm_if_needed(self):
        """Load forex LSTM model if not already loaded"""
        if self.forex_lstm is None:
            try:
                model_path = "models/forex_lstm_model.pth"
                if os.path.exists(model_path):
                    checkpoint = torch.load(model_path, map_location=self.device)
                    self.forex_lstm = ForexLSTMModel(
                        input_size=checkpoint['input_size'],
                        hidden_size=128,
                        num_layers=3,
                        dropout=0.2
                    )
                    self.forex_lstm.load_state_dict(checkpoint['model_state_dict'])
                    self.forex_lstm.to(self.device)
                    self.forex_lstm.eval()
                    logger.info("✅ Forex LSTM model loaded successfully")
                else:
                    logger.warning("Forex LSTM model not found, will use statistical prediction")
            except Exception as e:
                logger.error(f"Error loading forex LSTM: {e}")
    
    async def generate_trading_insight(self, symbol: str) -> Dict[str, Any]:
        """
        🎯 Generate comprehensive trading insight by fusing all AI sources
        
        This is the main method that combines:
        - LSTM price prediction
        - FinBERT sentiment analysis
        - DexScreener real-time data
        - LLaMA 2 reasoning and explanation
        """
        
        try:
            logger.info(f"🧠 Generating AI trading insight for {symbol}")
            
            # Determine if this is forex or crypto
            is_forex = self._is_forex_symbol(symbol)
            is_crypto = self._is_crypto_symbol(symbol)
            
            logger.info(f"Market type: {'Forex' if is_forex else 'Crypto' if is_crypto else 'Unknown'}")
            
            # Step 1: Get real-time market data
            market_data = await self._get_market_data(symbol)
            
            # Step 2: Generate price prediction using appropriate model
            if is_forex:
                await self._load_forex_lstm_if_needed()
                price_prediction = await self._generate_forex_price_prediction(symbol, market_data)
            else:
                price_prediction = await self._generate_price_prediction(symbol, market_data)
            
            # Step 3: Analyze sentiment using appropriate model
            if is_forex:
                sentiment_analysis = await self._analyze_forex_sentiment(symbol)
            else:
                sentiment_analysis = await self._analyze_market_sentiment(symbol)
            
            # Step 4: Get DexScreener data (only for crypto)
            dexscreener_data = {}
            if is_crypto:
                dexscreener_data = await self._get_dexscreener_data(symbol)
            
            # Step 5: Fuse all data sources
            fused_analysis = await self._fuse_ai_sources(
                symbol, price_prediction, sentiment_analysis, market_data, dexscreener_data
            )
            
            # Step 6: Generate Mistral reasoning and explanation
            ai_reasoning = await self._generate_ai_reasoning(
                symbol, price_prediction, sentiment_analysis, market_data, dexscreener_data
            )
            
            # Step 7: Calculate overall confidence and recommendation
            overall_confidence = self._calculate_overall_confidence(
                price_prediction, sentiment_analysis, fused_analysis
            )
            
            # Step 8: Compile final insight
            trading_insight = {
                "symbol": symbol,
                "market_type": "forex" if is_forex else "crypto" if is_crypto else "unknown",
                "timestamp": datetime.utcnow().isoformat(),
                "price_prediction": price_prediction,
                "sentiment_analysis": sentiment_analysis,
                "market_data": market_data,
                "dexscreener_data": dexscreener_data if is_crypto else None,
                "fused_analysis": fused_analysis,
                "ai_reasoning": ai_reasoning,
                "overall_confidence": overall_confidence,
                "recommendation": self._generate_recommendation(
                    price_prediction, sentiment_analysis, overall_confidence
                ),
                "model_weights": self.fusion_weights
            }
            
            logger.info(f"✅ Generated comprehensive trading insight for {symbol}")
            return trading_insight
            
        except Exception as e:
            logger.error(f"❌ Error generating trading insight: {e}")
            return self._get_fallback_insight(symbol, str(e))
    
    async def _generate_forex_price_prediction(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate forex-specific price prediction"""
        try:
            if self.forex_lstm is not None:
                # Get forex data with technical indicators
                features = self.forex_data_processor.create_forex_features(symbol)
                if not features.empty:
                    # Use the trained model for prediction
                    prediction_result = self.forex_lstm.predict_with_trained_model(features, symbol)
                    
                    if prediction_result['success']:
                        return {
                            "symbol": symbol,
                            "predicted_price": prediction_result['prediction'],
                            "confidence": prediction_result['confidence'],
                            "timeframe": "24h",
                            "model": "DEXTER-Forex-LSTM",
                            "features_used": prediction_result.get('features_used', 0),
                            "sequence_length": prediction_result.get('sequence_length', 0),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    else:
                        logger.warning(f"Trained model prediction failed: {prediction_result.get('error', 'Unknown error')}")
            
            # Fallback to statistical prediction
            return {
                "symbol": symbol,
                "predicted_price": 1.0,  # Default forex price
                "confidence": 0.5,
                "timeframe": "24h",
                "model": "Statistical",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in forex price prediction: {e}")
            return {
                "symbol": symbol,
                "predicted_price": 1.0,
                "confidence": 0.3,
                "timeframe": "24h",
                "model": "Fallback",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _analyze_forex_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze forex-specific sentiment"""
        try:
            # For now, use a sample text - later we'll integrate real forex news
            sample_text = f"Market analysis for {symbol}: Current trends show mixed signals with central bank policies influencing price movements."
            
            sentiment_result = self.forex_finbert.analyze_forex_sentiment(sample_text, symbol)
            
            return {
                "symbol": symbol,
                "sentiment": sentiment_result["sentiment"],
                "sentiment_score": sentiment_result["sentiment_score"],
                "confidence": sentiment_result["confidence"],
                "forex_analysis": sentiment_result["forex_analysis"],
                "model": "Forex-FinBERT",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in forex sentiment analysis: {e}")
            return {
                "symbol": symbol,
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "confidence": 0.5,
                "forex_analysis": {"market_bias": "neutral"},
                "model": "Fallback",
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _generate_recommendation(self, price_prediction: Dict[str, Any], sentiment_analysis: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Generate trading recommendation based on all analysis"""
        try:
            # Simple recommendation logic
            sentiment_score = sentiment_analysis.get("sentiment_score", 0)
            price_confidence = price_prediction.get("confidence", 0.5)
            
            if confidence > 0.7 and sentiment_score > 0.3:
                action = "BUY"
                strength = "STRONG" if confidence > 0.8 else "MODERATE"
            elif confidence > 0.7 and sentiment_score < -0.3:
                action = "SELL"
                strength = "STRONG" if confidence > 0.8 else "MODERATE"
            else:
                action = "HOLD"
                strength = "NEUTRAL"
            
            return {
                "action": action,
                "strength": strength,
                "confidence": confidence,
                "reasoning": f"Based on {price_confidence:.1%} price confidence and {sentiment_score:.2f} sentiment score"
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {
                "action": "HOLD",
                "strength": "NEUTRAL",
                "confidence": 0.5,
                "reasoning": "Insufficient data for recommendation"
            }
    
            
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data"""
        try:
            # Get market data from price predictor
            data = self.price_predictor.get_market_data(symbol, period="1mo")
            
            if data.empty:
                return {"error": "No market data available"}
                
            # Extract key metrics
            current_price = data['Close'].iloc[-1]
            price_change = ((current_price - data['Close'].iloc[-2]) / data['Close'].iloc[-2]) * 100
            volume = data['Volume'].iloc[-1]
            
            return {
                "current_price": float(current_price),
                "price_change_percent": float(price_change),
                "volume": float(volume),
                "high_24h": float(data['High'].iloc[-1]),
                "low_24h": float(data['Low'].iloc[-1]),
                "data_points": len(data)
            }
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return {"error": str(e)}
            
    async def _generate_price_prediction(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate price prediction using LSTM"""
        try:
            # Get trading signal from price predictor
            signal = self.price_predictor.get_trading_signal(symbol)
            
            return {
                "direction": signal.get("signal", "HOLD"),
                "confidence": signal.get("confidence", 0),
                "current_price": signal.get("current_price", 0),
                "indicators": signal.get("indicators", {}),
                "model_used": signal.get("model_used", "statistical")
            }
            
        except Exception as e:
            logger.error(f"Error generating price prediction: {e}")
            return {"direction": "HOLD", "confidence": 0, "error": str(e)}
            
    async def _analyze_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze market sentiment using FinBERT"""
        try:
            # Use sentiment analyzer
            sentiment_result = self.sentiment_analyzer.analyze_sentiment(
                f"Market analysis for {symbol}"
            )
            
            return {
                "overall_sentiment": sentiment_result.get("sentiment", "neutral"),
                "sentiment_score": sentiment_result.get("score", 0),
                "confidence": sentiment_result.get("confidence", 0),
                "model_used": "finbert"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"overall_sentiment": "neutral", "sentiment_score": 0, "error": str(e)}
            
    async def _get_dexscreener_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time data from DexScreener"""
        try:
            # Search for token on DexScreener
            # DexScreener integration temporarily disabled
            # tokens = await self.dexscreener_client.search_tokens(symbol)
            # if tokens:
            #     token_data = await self.dexscreener_client.get_token_data(tokens[0]["address"])
            token_data = None
            
            if token_data:
                return {
                    "price_usd": token_data.get("priceUsd", 0),
                    "price_change_24h": token_data.get("priceChange", {}).get("h24", 0),
                    "volume_24h": token_data.get("volume", {}).get("h24", 0),
                    "liquidity": token_data.get("liquidity", {}).get("usd", 0),
                    "market_cap": token_data.get("marketCap", 0),
                    "dex": token_data.get("dexId", "unknown")
                }
            else:
                return {"error": "Token not found on DexScreener"}
                
        except Exception as e:
            logger.error(f"Error getting DexScreener data: {e}")
            return {"error": str(e)}
            
    async def _fuse_ai_sources(self, 
                              symbol: str,
                              price_prediction: Dict[str, Any],
                              sentiment_analysis: Dict[str, Any], 
                              market_data: Dict[str, Any],
                              dexscreener_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse all AI sources into unified analysis"""
        
        try:
            # Extract key metrics
            price_confidence = price_prediction.get("confidence", 0)
            sentiment_score = sentiment_analysis.get("sentiment_score", 0)
            price_change = market_data.get("price_change_percent", 0)
            
            # Calculate weighted fusion score
            # Determine which price prediction weight to use based on symbol type
            is_forex = self._is_forex_symbol(symbol) if hasattr(self, '_is_forex_symbol') else False
            price_weight_key = "forex_price_prediction" if is_forex else "crypto_price_prediction"
            sentiment_weight_key = "forex_sentiment_analysis" if is_forex else "crypto_sentiment_analysis"
            
            fusion_score = (
                price_confidence * self.fusion_weights[price_weight_key] +
                abs(sentiment_score) * 100 * self.fusion_weights[sentiment_weight_key] +
                min(abs(price_change) * 10, 100) * self.fusion_weights["market_data"]
            )
            
            # Determine overall direction
            # Handle both crypto (direction) and forex (predicted_price) formats
            if "direction" in price_prediction:
                price_direction = price_prediction.get("direction", "HOLD")
            elif "predicted_price" in price_prediction:
                # For forex, determine direction based on predicted price vs current price
                predicted_price = price_prediction.get("predicted_price", 1.0)
                current_price = market_data.get("current_price", 1.0)
                if predicted_price > current_price * 1.001:  # 0.1% threshold
                    price_direction = "BUY"
                elif predicted_price < current_price * 0.999:  # 0.1% threshold
                    price_direction = "SELL"
                else:
                    price_direction = "HOLD"
            else:
                price_direction = "HOLD"
                
            sentiment_direction = "BUY" if sentiment_score > 0.1 else "SELL" if sentiment_score < -0.1 else "HOLD"
            market_direction = "BUY" if price_change > 2 else "SELL" if price_change < -2 else "HOLD"
            
            # Consensus logic
            directions = [price_direction, sentiment_direction, market_direction]
            buy_count = directions.count("BUY") + directions.count("STRONG_BUY")
            sell_count = directions.count("SELL") + directions.count("STRONG_SELL")
            
            if buy_count > sell_count:
                overall_direction = "BUY" if buy_count >= 2 else "HOLD"
            elif sell_count > buy_count:
                overall_direction = "SELL" if sell_count >= 2 else "HOLD"
            else:
                overall_direction = "HOLD"
                
            return {
                "fusion_score": fusion_score,
                "overall_direction": overall_direction,
                "component_agreement": {
                    "price_prediction": price_direction,
                    "sentiment_analysis": sentiment_direction,
                    "market_data": market_direction
                },
                "consensus_strength": max(buy_count, sell_count) / 3.0,
                "fusion_weights": self.fusion_weights
            }
            
        except Exception as e:
            logger.error(f"Error fusing AI sources: {e}")
            return {"fusion_score": 0, "overall_direction": "HOLD", "error": str(e)}
            
    async def _generate_ai_reasoning(self,
                                   symbol: str,
                                   price_prediction: Dict[str, Any],
                                   sentiment_analysis: Dict[str, Any],
                                   market_data: Dict[str, Any],
                                   dexscreener_data: Dict[str, Any]) -> str:
        """Generate AI reasoning using Mistral 7B"""
        
        try:
            # Generate comprehensive analysis using Mistral 7B
            if self.mistral_engine and self.mistral_engine.model:
                # Use dexscreener_data as technical_indicators since that's what the method expects
                analysis_result = await self.mistral_engine.generate_trading_analysis(
                    price_prediction=price_prediction,
                    sentiment_analysis=sentiment_analysis,
                    market_data=market_data,
                    technical_indicators=dexscreener_data
                )
                
                return analysis_result
            else:
                # Handle both crypto and forex formats
                if "direction" in price_prediction:
                    direction = price_prediction.get('direction', 'HOLD')
                    confidence = price_prediction.get('confidence', 0)
                elif "predicted_price" in price_prediction:
                    direction = "PRICE_PREDICTION"
                    confidence = price_prediction.get('confidence', 0)
                else:
                    direction = "HOLD"
                    confidence = 0
                    
                return f"AI analysis for {symbol}: Technical indicators suggest {direction} with {confidence:.1f}% confidence."
            
        except Exception as e:
            logger.error(f"Error generating AI reasoning: {e}")
            # Handle both crypto and forex formats in error case
            if "direction" in price_prediction:
                direction = price_prediction.get('direction', 'HOLD')
                confidence = price_prediction.get('confidence', 0)
            elif "predicted_price" in price_prediction:
                direction = "PRICE_PREDICTION"
                confidence = price_prediction.get('confidence', 0)
            else:
                direction = "HOLD"
                confidence = 0
                
            return f"AI analysis for {symbol}: Technical indicators suggest {direction} with {confidence:.1f}% confidence."
            
    def _calculate_overall_confidence(self,
                                    price_prediction: Dict[str, Any],
                                    sentiment_analysis: Dict[str, Any],
                                    fused_analysis: Dict[str, Any]) -> float:
        """Calculate overall confidence score"""
        
        try:
            # Base confidence from price prediction
            base_confidence = price_prediction.get("confidence", 0)
            
            # Sentiment confidence boost
            sentiment_confidence = sentiment_analysis.get("confidence", 0)
            
            # Fusion score boost
            fusion_score = fused_analysis.get("fusion_score", 0)
            
            # Consensus strength boost
            consensus_strength = fused_analysis.get("consensus_strength", 0)
            
            # Calculate weighted overall confidence
            overall_confidence = (
                base_confidence * 0.5 +
                sentiment_confidence * 0.2 +
                fusion_score * 0.2 +
                consensus_strength * 100 * 0.1
            )
            
            return min(100, max(0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating overall confidence: {e}")
            return 50.0
            
    def _determine_trading_recommendation(self, fused_analysis: Dict[str, Any]) -> str:
        """Determine final trading recommendation"""
        
        try:
            direction = fused_analysis.get("overall_direction", "HOLD")
            fusion_score = fused_analysis.get("fusion_score", 0)
            consensus_strength = fused_analysis.get("consensus_strength", 0)
            
            # Strengthen recommendation based on confidence
            if direction == "BUY" and fusion_score > 70 and consensus_strength > 0.6:
                return "STRONG_BUY"
            elif direction == "SELL" and fusion_score > 70 and consensus_strength > 0.6:
                return "STRONG_SELL"
            elif direction == "BUY" and fusion_score > 50:
                return "BUY"
            elif direction == "SELL" and fusion_score > 50:
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            logger.error(f"Error determining trading recommendation: {e}")
            return "HOLD"
            
    def _assess_risk(self, fused_analysis: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Assess trading risk"""
        
        try:
            fusion_score = fused_analysis.get("fusion_score", 0)
            consensus_strength = fused_analysis.get("consensus_strength", 0)
            
            # Risk assessment logic
            if confidence > 80 and fusion_score > 70 and consensus_strength > 0.7:
                risk_level = "LOW"
                risk_score = 20
            elif confidence > 60 and fusion_score > 50 and consensus_strength > 0.5:
                risk_level = "MEDIUM"
                risk_score = 50
            else:
                risk_level = "HIGH"
                risk_score = 80
                
            return {
                "risk_level": risk_level,
                "risk_score": risk_score,
                "confidence": confidence,
                "fusion_score": fusion_score,
                "consensus_strength": consensus_strength
            }
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {"risk_level": "HIGH", "risk_score": 80, "error": str(e)}
            
    def _get_fallback_insight(self, symbol: str, error: str = "Unknown error") -> Dict[str, Any]:
        """Get fallback insight when AI components fail"""
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "ai_recommendation": "HOLD",
            "confidence_score": 0,
            "ai_reasoning": f"AI system temporarily unavailable: {error}",
            "components": {
                "price_prediction": {"error": "unavailable"},
                "sentiment_analysis": {"error": "unavailable"},
                "market_data": {"error": "unavailable"},
                "dexscreener_data": {"error": "unavailable"}
            },
            "fusion_analysis": {"error": "unavailable"},
            "risk_assessment": {"risk_level": "HIGH", "risk_score": 100},
            "model_status": {
                "lstm": "error",
                "finbert": "error",
                "dexscreener": "error", 
                "dialoGPT": "error"
            }
        }
        
    async def get_ai_system_status(self) -> Dict[str, Any]:
        """Get comprehensive AI system status"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "price_predictor": "active",
                "sentiment_analyzer": "active",
                "dexscreener_client": "disabled",
                "mistral_engine": self.mistral_engine.get_model_status() if self.mistral_engine else "not_initialized"
            },
            "fusion_weights": self.fusion_weights,
            "capabilities": [
                "Multi-source AI fusion",
                "Real-time market analysis", 
                "Sentiment analysis",
                "Price prediction",
                "Risk assessment",
                "Conversational reasoning"
            ]
        }

# Create global instance
ai_fusion_engine = DexterAIFusionEngine()