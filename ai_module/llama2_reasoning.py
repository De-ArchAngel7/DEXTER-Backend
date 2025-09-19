#!/usr/bin/env python3
"""
🧠 DEXTER LLaMA 2 REASONING LAYER
============================================================
LLaMA 2 integration for conversational AI and multi-source reasoning
This is the "brain" that explains and reasons about trading decisions
"""

import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM, 
    LlamaTokenizer, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class LLaMA2ReasoningEngine:
    """
    LLaMA 2 reasoning engine for conversational AI and multi-source analysis
    This handles:
    - Understanding user intent
    - Explaining trading results in natural language
    - Reasoning across multiple data sources
    - Providing conversational trading advice
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # Reasoning templates for different scenarios
        self.reasoning_templates = {
            "trading_analysis": """
            Based on the following market analysis:
            
            📊 Price Prediction: {price_prediction}
            📈 Confidence: {confidence}%
            📰 Sentiment Analysis: {sentiment}
            🔍 Market Data: {market_data}
            📋 Technical Indicators: {technical_indicators}
            
            Please provide a comprehensive trading analysis and recommendation.
            """,
            
            "risk_assessment": """
            Risk Assessment Request:
            
            💰 Current Portfolio: {portfolio_value}
            📊 Position Size: {position_size}
            🎯 Risk Tolerance: {risk_tolerance}
            📈 Market Conditions: {market_conditions}
            
            Analyze the risk and provide recommendations.
            """,
            
            "market_explanation": """
            Market Explanation Request:
            
            📊 Current Price: {current_price}
            📈 Price Change: {price_change}%
            📰 News Sentiment: {news_sentiment}
            🔍 Technical Analysis: {technical_analysis}
            
            Explain what's happening in the market and why.
            """
        }
        
        logger.info(f"🧠 LLaMA 2 Reasoning Engine initialized with {model_name}")
        
    def _setup_quantization(self):
        """Setup 4-bit quantization for efficient inference"""
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            return quantization_config
        except Exception as e:
            logger.warning(f"Quantization setup failed: {e}")
            return None
            
    def _setup_lora_config(self):
        """Setup LoRA configuration for efficient fine-tuning"""
        try:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            return lora_config
        except Exception as e:
            logger.warning(f"LoRA setup failed: {e}")
            return None
            
    async def load_model(self, use_quantization: bool = True, use_lora: bool = True):
        """Load LLaMA 2 model with optional quantization and LoRA"""
        try:
            logger.info("🔄 Loading LLaMA 2 model...")
            
            # Setup quantization if requested
            quantization_config = None
            if use_quantization:
                quantization_config = self._setup_quantization()
                
            # Load tokenizer
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16
            )
            
            # Setup LoRA if requested
            if use_lora:
                lora_config = self._setup_lora_config()
                if lora_config:
                    self.peft_model = get_peft_model(self.model, lora_config)
                    logger.info("✅ LoRA configuration applied")
                    
            logger.info("✅ LLaMA 2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load LLaMA 2 model: {e}")
            # Fallback to mock reasoning for development
            self.model = None
            self.tokenizer = None
            logger.info("🔄 Using mock reasoning engine for development")
            return False
            
    async def generate_trading_analysis(self, 
                                      price_prediction: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      technical_indicators: Dict[str, Any]) -> str:
        """Generate comprehensive trading analysis using LLaMA 2 reasoning"""
        
        try:
            # Prepare input data
            analysis_input = {
                "price_prediction": f"Direction: {price_prediction.get('direction', 'Unknown')}, "
                                  f"Confidence: {price_prediction.get('confidence', 0):.1f}%",
                "confidence": price_prediction.get('confidence', 0),
                "sentiment": f"Overall: {sentiment_analysis.get('overall_sentiment', 'Neutral')}, "
                           f"Score: {sentiment_analysis.get('sentiment_score', 0):.2f}",
                "market_data": f"Price: ${market_data.get('current_price', 0):,.2f}, "
                             f"Volume: {market_data.get('volume', 0):,}",
                "technical_indicators": f"RSI: {technical_indicators.get('rsi', 50):.1f}, "
                                      f"MACD: {technical_indicators.get('macd', 0):.4f}"
            }
            
            # Generate analysis
            if self.model and self.tokenizer:
                analysis = await self._llama2_reasoning("trading_analysis", analysis_input)
            else:
                analysis = await self._mock_reasoning("trading_analysis", analysis_input)
                
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Error generating trading analysis: {e}")
            return "Unable to generate trading analysis at this time."
            
    async def generate_risk_assessment(self,
                                     portfolio_value: float,
                                     position_size: float,
                                     risk_tolerance: str,
                                     market_conditions: Dict[str, Any]) -> str:
        """Generate risk assessment using LLaMA 2 reasoning"""
        
        try:
            # Prepare input data
            risk_input = {
                "portfolio_value": f"${portfolio_value:,.2f}",
                "position_size": f"${position_size:,.2f}",
                "risk_tolerance": risk_tolerance,
                "market_conditions": f"Volatility: {market_conditions.get('volatility', 0):.2f}, "
                                   f"Trend: {market_conditions.get('trend', 'Unknown')}"
            }
            
            # Generate assessment
            if self.model and self.tokenizer:
                assessment = await self._llama2_reasoning("risk_assessment", risk_input)
            else:
                assessment = await self._mock_reasoning("risk_assessment", risk_input)
                
            return assessment
            
        except Exception as e:
            logger.error(f"❌ Error generating risk assessment: {e}")
            return "Unable to generate risk assessment at this time."
            
    async def explain_market_movement(self,
                                    current_price: float,
                                    price_change: float,
                                    news_sentiment: Dict[str, Any],
                                    technical_analysis: Dict[str, Any]) -> str:
        """Explain market movement using LLaMA 2 reasoning"""
        
        try:
            # Prepare input data
            explanation_input = {
                "current_price": f"${current_price:,.2f}",
                "price_change": f"{price_change:.2f}%",
                "news_sentiment": f"Sentiment: {news_sentiment.get('sentiment', 'Neutral')}, "
                                f"Impact: {news_sentiment.get('impact', 0):.2f}",
                "technical_analysis": f"RSI: {technical_analysis.get('rsi', 50):.1f}, "
                                    f"MACD: {technical_analysis.get('macd', 0):.4f}, "
                                    f"Trend: {technical_analysis.get('trend', 'Unknown')}"
            }
            
            # Generate explanation
            if self.model and self.tokenizer:
                explanation = await self._llama2_reasoning("market_explanation", explanation_input)
            else:
                explanation = await self._mock_reasoning("market_explanation", explanation_input)
                
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Error explaining market movement: {e}")
            return "Unable to explain market movement at this time."
            
    async def _llama2_reasoning(self, template_type: str, input_data: Dict[str, Any]) -> str:
        """Generate reasoning using actual LLaMA 2 model"""
        
        try:
            # Get template
            template = self.reasoning_templates[template_type]
            prompt = template.format(**input_data)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ LLaMA 2 reasoning error: {e}")
            return await self._mock_reasoning(template_type, input_data)
            
    async def _mock_reasoning(self, template_type: str, input_data: Dict[str, Any]) -> str:
        """Mock reasoning for development when LLaMA 2 is not available"""
        
        if template_type == "trading_analysis":
            confidence = input_data.get('confidence', 0)
            sentiment = input_data.get('sentiment', 'Neutral')
            
            if confidence > 75:
                return f"""
                🎯 **STRONG TRADING SIGNAL DETECTED**
                
                Based on our AI analysis:
                • **Confidence Level**: {confidence:.1f}% (High)
                • **Sentiment**: {sentiment}
                • **Recommendation**: Consider taking a position
                • **Risk Level**: Moderate
                
                The AI models show strong agreement on market direction. 
                This presents a favorable risk-reward opportunity.
                """
            elif confidence > 60:
                return f"""
                ⚖️ **MODERATE TRADING SIGNAL**
                
                Analysis Summary:
                • **Confidence Level**: {confidence:.1f}% (Moderate)
                • **Sentiment**: {sentiment}
                • **Recommendation**: Proceed with caution
                • **Risk Level**: Medium
                
                Mixed signals detected. Consider smaller position sizes 
                and implement strict risk management.
                """
            else:
                return f"""
                ⏸️ **WEAK SIGNAL - HOLD POSITION**
                
                Current Analysis:
                • **Confidence Level**: {confidence:.1f}% (Low)
                • **Sentiment**: {sentiment}
                • **Recommendation**: Wait for clearer signals
                • **Risk Level**: High
                
                Market conditions are uncertain. Better to wait for 
                stronger signals before entering positions.
                """
                
        elif template_type == "risk_assessment":
            portfolio_value = float(input_data.get('portfolio_value', '0').replace('$', '').replace(',', ''))
            position_size = float(input_data.get('position_size', '0').replace('$', '').replace(',', ''))
            
            position_percentage = (position_size / portfolio_value * 100) if portfolio_value > 0 else 0
            
            if position_percentage > 20:
                return f"""
                🚨 **HIGH RISK DETECTED**
                
                Risk Assessment:
                • **Position Size**: {position_percentage:.1f}% of portfolio
                • **Risk Level**: HIGH
                • **Recommendation**: Reduce position size immediately
                
                Current position exceeds recommended risk limits. 
                Consider reducing to 5-10% of portfolio value.
                """
            elif position_percentage > 10:
                return f"""
                ⚠️ **MODERATE RISK**
                
                Risk Assessment:
                • **Position Size**: {position_percentage:.1f}% of portfolio
                • **Risk Level**: MODERATE
                • **Recommendation**: Monitor closely
                
                Position size is within acceptable limits but requires 
                close monitoring and risk management.
                """
            else:
                return f"""
                ✅ **LOW RISK**
                
                Risk Assessment:
                • **Position Size**: {position_percentage:.1f}% of portfolio
                • **Risk Level**: LOW
                • **Recommendation**: Acceptable risk level
                
                Position size is well within risk management guidelines.
                Continue monitoring market conditions.
                """
                
        elif template_type == "market_explanation":
            price_change = float(input_data.get('price_change', '0').replace('%', ''))
            
            if price_change > 5:
                return f"""
                📈 **STRONG BULLISH MOMENTUM**
                
                Market Analysis:
                • **Price Movement**: +{price_change:.2f}% (Strong)
                • **Market Sentiment**: Bullish
                • **Technical Indicators**: Positive
                
                Strong upward momentum detected. This could indicate:
                • Positive news impact
                • Technical breakout
                • Increased buying pressure
                """
            elif price_change < -5:
                return f"""
                📉 **BEARISH PRESSURE**
                
                Market Analysis:
                • **Price Movement**: {price_change:.2f}% (Strong)
                • **Market Sentiment**: Bearish
                • **Technical Indicators**: Negative
                
                Significant downward pressure. Possible causes:
                • Negative news impact
                • Technical breakdown
                • Increased selling pressure
                """
            else:
                return f"""
                ⚖️ **SIDEWAYS MOVEMENT**
                
                Market Analysis:
                • **Price Movement**: {price_change:.2f}% (Moderate)
                • **Market Sentiment**: Neutral
                • **Technical Indicators**: Mixed
                
                Market is consolidating. This suggests:
                • Balanced buying/selling pressure
                • Awaiting catalyst
                • Range-bound trading
                """
                
        return "Analysis completed successfully."
        
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and capabilities"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "peft_enabled": self.peft_model is not None,
            "device": str(self.device),
            "model_name": self.model_name,
            "capabilities": [
                "Trading Analysis Generation",
                "Risk Assessment",
                "Market Movement Explanation",
                "Multi-source Reasoning",
                "Conversational AI"
            ]
        }
