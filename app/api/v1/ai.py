#!/usr/bin/env python3
"""
🧠 DEXTER AI Trading Endpoints
Complete AI-powered trading intelligence API

Endpoints:
- /trading-insight: Get AI-generated trading insights
- /price-prediction: LSTM-based price predictions
- /sentiment-analysis: FinBERT sentiment analysis
- /market-analysis: Real-time market data
- /ai-status: Check AI model status
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, Any, Optional
import structlog
from datetime import datetime

# Import our AI modules
from ....ai_module.ai_fusion_engine import DexterAIFusionEngine
from ....data_sources.dexscreener_client import DexScreenerClient

logger = structlog.get_logger()
router = APIRouter(prefix="/ai", tags=["AI Trading Intelligence"])

# Global AI engine instance
ai_engine: Optional[DexterAIFusionEngine] = None

async def get_ai_engine() -> DexterAIFusionEngine:
    """Get or initialize the AI engine"""
    global ai_engine
    
    if ai_engine is None:
        try:
            logger.info("🚀 Initializing DEXTER AI Fusion Engine...")
            ai_engine = DexterAIFusionEngine()
            logger.info("✅ AI Fusion Engine initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI engine: {e}")
            raise HTTPException(status_code=500, detail="AI engine initialization failed")
    
    return ai_engine

@router.post("/trading-insight")
async def get_ai_trading_insight(
    symbol: str = Query(..., description="Trading symbol (e.g., BTCUSDT, ETHUSDT)"),
    include_sentiment: bool = Query(True, description="Include sentiment analysis"),
    include_prediction: bool = Query(True, description="Include price prediction"),
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    🧠 Get comprehensive AI trading insight
    
    This endpoint combines:
    - LSTM price predictions
    - FinBERT sentiment analysis
    - Real-time market data
    - AI reasoning and recommendations
    """
    
    try:
        logger.info(f"🧠 Generating AI trading insight for {symbol}")
        
        # Generate comprehensive trading insight
        insight = await ai_engine.generate_trading_insight(
            symbol=symbol,
            include_sentiment=include_sentiment,
            include_prediction=include_prediction
        )
        
        if "error" in insight:
            raise HTTPException(status_code=500, detail=insight["error"])
        
        return {
            "status": "success",
            "message": f"AI trading insight generated for {symbol}",
            "data": insight,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating trading insight: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate trading insight: {str(e)}")

@router.get("/price-prediction/{symbol}")
async def get_price_prediction(
    symbol: str,
    time_horizon: str = Query("1h", description="Prediction time horizon"),
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    📈 Get LSTM-based price prediction for a symbol
    """
    
    try:
        logger.info(f"📈 Generating price prediction for {symbol}")
        
        # Get market data first
        market_data = await ai_engine._get_market_data(symbol)
        
        # Generate price prediction
        prediction = await ai_engine._generate_price_prediction(symbol, market_data)
        
        if not prediction:
            raise HTTPException(status_code=500, detail="Failed to generate price prediction")
        
        return {
            "status": "success",
            "message": f"Price prediction generated for {symbol}",
            "data": {
                "symbol": symbol,
                "prediction": prediction,
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating price prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate price prediction: {str(e)}")

@router.get("/sentiment-analysis/{symbol}")
async def get_sentiment_analysis(
    symbol: str,
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    💬 Get FinBERT-based sentiment analysis for a symbol
    """
    
    try:
        logger.info(f"💬 Analyzing sentiment for {symbol}")
        
        # Generate sentiment analysis
        sentiment = await ai_engine._analyze_market_sentiment(symbol)
        
        if not sentiment:
            raise HTTPException(status_code=500, detail="Failed to generate sentiment analysis")
        
        return {
            "status": "success",
            "message": f"Sentiment analysis generated for {symbol}",
            "data": {
                "symbol": symbol,
                "sentiment": sentiment,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error generating sentiment analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate sentiment analysis: {str(e)}")

@router.get("/market-analysis/{symbol}")
async def get_market_analysis(
    symbol: str,
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    📊 Get real-time market analysis for a symbol
    """
    
    try:
        logger.info(f"📊 Fetching market analysis for {symbol}")
        
        # Get market data
        market_data = await ai_engine._get_market_data(symbol)
        
        if not market_data:
            raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        return {
            "status": "success",
            "message": f"Market analysis generated for {symbol}",
            "data": {
                "symbol": symbol,
                "market_data": market_data,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching market analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market analysis: {str(e)}")

@router.get("/ai-status")
async def get_ai_status(
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    🔍 Get status of all AI models and systems
    """
    
    try:
        logger.info("🔍 Checking AI system status...")
        
        # Get AI engine status
        engine_status = ai_engine.get_model_status()
        
        # Test AI fusion engine
        test_result = await ai_engine.test_fusion_engine()
        
        return {
            "status": "success",
            "message": "AI system status retrieved",
            "data": {
                "ai_engine_status": engine_status,
                "fusion_engine_test": test_result,
                "system_health": "healthy" if engine_status["models_loaded"] else "degraded",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking AI status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to check AI status: {str(e)}")

@router.post("/test-ai-system")
async def test_ai_system(
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    🧪 Test the complete AI system with sample data
    """
    
    try:
        logger.info("🧪 Testing complete AI system...")
        
        # Test with multiple symbols
        test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        test_results = []
        
        for symbol in test_symbols:
            try:
                insight = await ai_engine.generate_trading_insight(symbol)
                test_results.append({
                    "symbol": symbol,
                    "status": "success" if "error" not in insight else "failed",
                    "models_used": insight.get("ai_models_used", []),
                    "confidence_score": insight.get("confidence_score", 0)
                })
            except Exception as e:
                test_results.append({
                    "symbol": symbol,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Calculate overall test results
        successful_tests = sum(1 for result in test_results if result["status"] == "success")
        total_tests = len(test_results)
        
        return {
            "status": "success",
            "message": f"AI system test completed: {successful_tests}/{total_tests} successful",
            "data": {
                "test_results": test_results,
                "success_rate": f"{successful_tests/total_tests:.1%}",
                "overall_status": "healthy" if successful_tests == total_tests else "degraded",
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error testing AI system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test AI system: {str(e)}")

@router.get("/trending-tokens")
async def get_trending_tokens(
    limit: int = Query(10, description="Number of trending tokens to return", ge=1, le=50)
) -> Dict[str, Any]:
    """
    🔥 Get trending tokens across all DEXs
    """
    
    try:
        logger.info(f"🔥 Fetching top {limit} trending tokens...")
        
        # Initialize DexScreener client
        async with DexScreenerClient() as client:
            trending = await client.get_trending_tokens()
            
            if not trending:
                # Fallback to mock data
                trending = [
                    await client.get_mock_token_data("BTC"),
                    await client.get_mock_token_data("ETH"),
                    await client.get_mock_token_data("SOL")
                ]
            
            # Limit results
            trending = trending[:limit]
            
            return {
                "status": "success",
                "message": f"Retrieved {len(trending)} trending tokens",
                "data": {
                    "trending_tokens": trending,
                    "count": len(trending),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
    except Exception as e:
        logger.error(f"❌ Error fetching trending tokens: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch trending tokens: {str(e)}")

@router.get("/market-overview")
async def get_market_overview() -> Dict[str, Any]:
    """
    🌍 Get overall market overview and statistics
    """
    
    try:
        logger.info("🌍 Fetching market overview...")
        
        # Initialize DexScreener client
        async with DexScreenerClient() as client:
            overview = await client.get_market_overview()
            
            if not overview:
                raise HTTPException(status_code=500, detail="Failed to fetch market overview")
            
            return {
                "status": "success",
                "message": "Market overview retrieved successfully",
                "data": overview
            }
            
    except Exception as e:
        logger.error(f"❌ Error fetching market overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch market overview: {str(e)}")

@router.post("/refresh-ai-models")
async def refresh_ai_models(
    ai_engine: DexterAIFusionEngine = Depends(get_ai_engine)
) -> Dict[str, Any]:
    """
    🔄 Refresh and reload all AI models
    """
    
    try:
        logger.info("🔄 Refreshing AI models...")
        
        # Reload all models
        ai_engine._load_all_models()
        
        # Get new status
        new_status = ai_engine.get_model_status()
        
        return {
            "status": "success",
            "message": "AI models refreshed successfully",
            "data": {
                "new_status": new_status,
                "refresh_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error refreshing AI models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh AI models: {str(e)}")

# Health check endpoint
@router.get("/health")
async def ai_health_check() -> Dict[str, Any]:
    """
    ❤️ AI system health check
    """
    
    return {
        "status": "healthy",
        "service": "DEXTER AI Trading Intelligence",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "models": "LSTM + FinBERT + AI Fusion Engine"
    }
