#!/usr/bin/env python3
"""
🧠 DEXTER FINE-TUNED LLaMA 2 INTEGRATION
============================================================
Integration with your fine-tuned LLaMA 2 model from Hugging Face
This replaces the mock reasoning with your actual fine-tuned model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import os
from typing import Dict, List, Any, Optional
import structlog
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class FineTunedLLaMA2Engine:
    """
    Fine-tuned LLaMA 2 engine for DEXTER
    Uses your custom fine-tuned model from Hugging Face
    """
    
    def __init__(self, 
                 model_name: str = "dexter-llama2-financial",
                 hf_api_key: Optional[str] = None):
        self.model_name = model_name
        self.hf_api_key = hf_api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        # Login to Hugging Face if API key is provided
        if self.hf_api_key:
            try:
                login(token=self.hf_api_key)
                logger.info("✅ Logged in to Hugging Face successfully")
            except Exception as e:
                logger.warning(f"Failed to login to Hugging Face: {e}")
        
        logger.info(f"🧠 Fine-tuned LLaMA 2 Engine initialized with {model_name}")
        
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
            
    async def load_model(self, use_quantization: bool = True):
        """Load your fine-tuned LLaMA 2 model"""
        try:
            logger.info(f"🔄 Loading fine-tuned model: {self.model_name}")
            
            # Setup quantization if requested
            quantization_config = None
            if use_quantization:
                quantization_config = self._setup_quantization()
                
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_auth_token=self.hf_api_key
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                use_auth_token=self.hf_api_key,
                torch_dtype=torch.float16
            )
            
            logger.info("✅ Fine-tuned LLaMA 2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {e}")
            logger.info("🔄 Falling back to base model or mock reasoning")
            return False
            
    async def generate_trading_analysis(self, 
                                      price_prediction: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      technical_indicators: Dict[str, Any]) -> str:
        """Generate trading analysis using your fine-tuned model"""
        
        try:
            # Format input for your fine-tuned model
            prompt = self._format_trading_analysis_prompt(
                price_prediction, sentiment_analysis, market_data, technical_indicators
            )
            
            if self.model and self.tokenizer:
                response = await self._generate_response(prompt)
                return response
            else:
                return "Fine-tuned model not available. Please check your Hugging Face API key and model name."
                
        except Exception as e:
            logger.error(f"❌ Error generating trading analysis: {e}")
            return "Unable to generate trading analysis at this time."
            
    async def generate_risk_assessment(self,
                                     portfolio_value: float,
                                     position_size: float,
                                     risk_tolerance: str,
                                     market_conditions: Dict[str, Any]) -> str:
        """Generate risk assessment using your fine-tuned model"""
        
        try:
            # Format input for your fine-tuned model
            prompt = self._format_risk_assessment_prompt(
                portfolio_value, position_size, risk_tolerance, market_conditions
            )
            
            if self.model and self.tokenizer:
                response = await self._generate_response(prompt)
                return response
            else:
                return "Fine-tuned model not available. Please check your Hugging Face API key and model name."
                
        except Exception as e:
            logger.error(f"❌ Error generating risk assessment: {e}")
            return "Unable to generate risk assessment at this time."
            
    async def explain_market_movement(self,
                                    current_price: float,
                                    price_change: float,
                                    news_sentiment: Dict[str, Any],
                                    technical_analysis: Dict[str, Any]) -> str:
        """Explain market movement using your fine-tuned model"""
        
        try:
            # Format input for your fine-tuned model
            prompt = self._format_market_explanation_prompt(
                current_price, price_change, news_sentiment, technical_analysis
            )
            
            if self.model and self.tokenizer:
                response = await self._generate_response(prompt)
                return response
            else:
                return "Fine-tuned model not available. Please check your Hugging Face API key and model name."
                
        except Exception as e:
            logger.error(f"❌ Error explaining market movement: {e}")
            return "Unable to explain market movement at this time."
            
    def _format_trading_analysis_prompt(self,
                                      price_prediction: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      technical_indicators: Dict[str, Any]) -> str:
        """Format trading analysis prompt for fine-tuned model"""
        
        return f"""### Instruction:
Analyze the following market data and provide a trading recommendation:

### Input:
Price: ${market_data.get('current_price', 0):,.2f}, RSI: {technical_indicators.get('rsi', 50):.1f}, MACD: {technical_indicators.get('macd', 0):.4f}, Volume: {market_data.get('volume', 0):,}, News Sentiment: {sentiment_analysis.get('overall_sentiment', 'Neutral')}

### Response:
"""
        
    def _format_risk_assessment_prompt(self,
                                     portfolio_value: float,
                                     position_size: float,
                                     risk_tolerance: str,
                                     market_conditions: Dict[str, Any]) -> str:
        """Format risk assessment prompt for fine-tuned model"""
        
        return f"""### Instruction:
Assess the risk of this trading position:

### Input:
Position Size: ${position_size:,.2f}, Portfolio Value: ${portfolio_value:,.2f}, Risk Tolerance: {risk_tolerance}, Market Volatility: {market_conditions.get('volatility', 0):.2f}

### Response:
"""
        
    def _format_market_explanation_prompt(self,
                                        current_price: float,
                                        price_change: float,
                                        news_sentiment: Dict[str, Any],
                                        technical_analysis: Dict[str, Any]) -> str:
        """Format market explanation prompt for fine-tuned model"""
        
        return f"""### Instruction:
Explain what's happening in the market and why:

### Input:
Current Price: ${current_price:,.2f}, Price Change: {price_change:+.2f}%, News Sentiment: {news_sentiment.get('sentiment', 'Neutral')}, RSI: {technical_analysis.get('rsi', 50):.1f}, MACD: {technical_analysis.get('macd', 0):.4f}

### Response:
"""
        
    async def _generate_response(self, prompt: str) -> str:
        """Generate response using fine-tuned model"""
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
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
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "Error generating response from fine-tuned model."
            
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.device),
            "model_name": self.model_name,
            "hf_api_key_configured": self.hf_api_key is not None,
            "capabilities": [
                "Fine-tuned Trading Analysis",
                "Financial Risk Assessment", 
                "Market Movement Explanation",
                "Custom Financial Reasoning"
            ]
        }
