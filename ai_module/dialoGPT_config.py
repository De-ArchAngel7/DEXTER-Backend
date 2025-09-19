#!/usr/bin/env python3
"""
🔧 DEXTER DIALOGPT CONFIGURATION
============================================================
Configuration settings for your fine-tuned DialoGPT model
"""

import os
from typing import Dict, Any

class DialoGPTConfig:
    """Configuration for DialoGPT model integration"""
    
    # Model paths
    MODEL_PATH = os.getenv("DEXTER_DIALOGPT_MODEL_PATH", "models/dexter_dialoGPT")
    BASE_MODEL_NAME = "microsoft/DialoGPT-medium"
    
    # Model settings
    USE_QUANTIZATION = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    MAX_LENGTH = int(os.getenv("DIALOGPT_MAX_LENGTH", "512"))
    TEMPERATURE = float(os.getenv("DIALOGPT_TEMPERATURE", "0.7"))
    TOP_P = float(os.getenv("DIALOGPT_TOP_P", "0.9"))
    
    # Generation settings
    GENERATION_CONFIG = {
        "max_length": MAX_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "do_sample": True,
        "pad_token_id": None,  # Will be set dynamically
        "eos_token_id": None,  # Will be set dynamically
    }
    
    # Financial analysis prompts
    FINANCIAL_PROMPTS = {
        "market_analysis": """
        Based on the following market data, provide a comprehensive trading analysis:

        Market Data:
        - Current Price: ${current_price}
        - Price Change: {price_change}%
        - Volume: {volume}
        - 24h High: ${high_24h}
        - 24h Low: ${low_24h}

        Price Prediction:
        - Signal: {signal}
        - Confidence: {confidence}%
        - Target Price: ${target_price}

        Sentiment Analysis:
        - Market Sentiment: {sentiment}
        - Confidence: {sentiment_confidence}%

        DexScreener Data:
        - Liquidity: ${liquidity}
        - Market Cap: ${market_cap}

        Please provide:
        1. Market analysis
        2. Trading recommendation
        3. Risk assessment
        4. Entry/exit points
        5. Position sizing advice

        Analysis:""",
        
        "risk_assessment": """
        Risk Assessment Request:

        Portfolio Context:
        - Portfolio Value: ${portfolio_value}
        - Position Size: {position_size}%
        - Risk Tolerance: {risk_tolerance}
        - Market Conditions: {market_conditions}

        Current Position:
        - Entry Price: ${entry_price}
        - Current Price: ${current_price}
        - Unrealized P&L: {unrealized_pnl}%

        Please provide:
        1. Risk analysis
        2. Position management advice
        3. Stop-loss recommendations
        4. Take-profit targets
        5. Risk mitigation strategies

        Risk Assessment:""",
        
        "technical_analysis": """
        Technical Analysis Request:

        Price Action:
        - Current Price: ${current_price}
        - Price Change: {price_change}%
        - Volume: {volume}

        Technical Indicators:
        - RSI: {rsi}
        - MACD: {macd}
        - SMA 20: ${sma_20}
        - SMA 50: ${sma_50}
        - Support Level: ${support}
        - Resistance Level: ${resistance}

        Please provide:
        1. Technical analysis
        2. Key levels to watch
        3. Breakout/breakdown scenarios
        4. Momentum analysis
        5. Trading setup recommendations

        Technical Analysis:"""
    }
    
    # Chat templates
    CHAT_TEMPLATES = {
        "user_message": "User: {message}",
        "ai_response": "AI: {response}",
        "context_prefix": "Context: {context}\n",
        "system_prompt": "You are DEXTER, an AI trading assistant. You provide professional financial analysis and trading advice based on market data and technical indicators."
    }
    
    # Error messages
    ERROR_MESSAGES = {
        "model_not_loaded": "I'm sorry, but my AI model is not currently loaded. Please try again later.",
        "generation_failed": "I apologize, but I'm having trouble generating a response at the moment.",
        "invalid_input": "I couldn't understand your request. Please try rephrasing your question.",
        "model_error": "There was an error with my AI model. Please try again later."
    }
    
    @classmethod
    def get_model_info(cls) -> Dict[str, Any]:
        """Get model information for status display"""
        return {
            "model_type": "DialoGPT Medium",
            "base_model": cls.BASE_MODEL_NAME,
            "model_path": cls.MODEL_PATH,
            "quantization": cls.USE_QUANTIZATION,
            "max_length": cls.MAX_LENGTH,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate configuration settings"""
        try:
            # Check if model path exists
            if not os.path.exists(cls.MODEL_PATH):
                print(f"❌ Model path not found: {cls.MODEL_PATH}")
                return False
            
            # Check if required files exist
            required_files = [
                "adapter_config.json",
                "adapter_model.safetensors",
                "tokenizer.json"
            ]
            
            for file in required_files:
                file_path = os.path.join(cls.MODEL_PATH, file)
                if not os.path.exists(file_path):
                    print(f"❌ Required file not found: {file_path}")
                    return False
            
            print("✅ DialoGPT configuration validated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
