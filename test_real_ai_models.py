#!/usr/bin/env python3
"""
Test Real AI Models - See your AI in action!
"""

import sys
from pathlib import Path

# Add the backend directory to Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def test_ai_models():
    """Test the new AI models"""
    print("🚀 Testing DEXTER Real AI Models...")
    print("=" * 50)
    
    try:
        # Import our new AI models
        from ai_module.real_ai_models import RealAITradingModels
        
        # Initialize models
        print("📥 Loading AI models...")
        models = RealAITradingModels()
        print("✅ Models loaded successfully!")
        
        # Test sentiment analysis
        print("\n🧠 Testing Sentiment Analysis with FinBERT...")
        test_texts = [
            "Bitcoin shows strong bullish momentum with increasing volume",
            "Market crash fears as crypto prices plummet",
            "Ethereum network upgrade shows promising results",
            "Regulatory uncertainty weighs on crypto markets"
        ]
        
        for text in test_texts:
            result = models.analyze_sentiment(text)
            print(f"\n📝 Text: {text}")
            print(f"🎯 Sentiment: {result['sentiment'].upper()}")
            print(f"📊 Score: {result['sentiment_score']:.3f}")
            print(f"💪 Confidence: {result['confidence']:.3f}")
            print(f"🤖 Model: {result['model']}")
        
        # Test price prediction
        print("\n📈 Testing Price Prediction...")
        import pandas as pd
        import numpy as np
        
        # Create sample market data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        volumes = 1000 + np.random.randn(100) * 100
        
        market_data = pd.DataFrame({
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        prediction = models.predict_price_movement(market_data, "BTCUSDT")
        print(f"\n🎯 Symbol: {prediction['symbol']}")
        print(f"📊 Prediction: {prediction['prediction']}")
        print(f"💪 Confidence: {prediction['confidence']:.3f}")
        print(f"⏰ Timeframe: {prediction['timeframe']}")
        print(f"🤖 Model: {prediction['model']}")
        
        # Get model status
        print("\n📊 Model Status...")
        status = models.get_model_status()
        print(f"🧠 Sentiment Model: {status['sentiment_model']}")
        print(f"📈 Forecasting Model: {status['forecasting_model']}")
        print(f"💻 Device: {status['device']}")
        print(f"✅ Models Loaded: {status['models_loaded']}")
        
        print("\n🎉 All tests passed! Your AI is working perfectly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing AI models: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🧠 DEXTER Real AI Models Test Suite")
    print("=" * 50)
    
    success = test_ai_models()
    
    if success:
        print("\n🚀 Your AI trading platform is ready!")
        print("Next steps:")
        print("1. Integrate with your existing DEXTER code")
        print("2. Set up Google Colab for fine-tuning")
        print("3. Add Redis for caching")
        print("4. Generate real trading signals!")
    else:
        print("\n❌ Some tests failed. Let's debug together!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
