#!/usr/bin/env python3
"""
🧠 DEXTER MISTRAL INTEGRATION ENGINE
============================================================
Integration with fine-tuned Mistral 7B model for advanced AI trading assistant
Replaces DialoGPT with much more powerful Mistral 7B
"""

import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
from typing import Dict, Any, Optional
import structlog
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class DexterMistralEngine:
    """
    DEXTER Mistral 7B engine for advanced trading analysis
    Uses fine-tuned Mistral 7B model with creator verification
    """
    
    def __init__(self, 
                 model_path: str = "models/dexter-mistral-7b",
                 use_quantization: bool = True):
        self.model_path = model_path
        self.use_quantization = use_quantization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.base_model = None
        
        # Creator verification
        self.creator_passphrase = "son of sparda"
        self.verified_creators = set()
        
        logger.info(f"🧠 DEXTER Mistral Engine initialized with {model_path}")
        logger.info(f"🔧 Quantization: {'Enabled' if use_quantization else 'Disabled'}")
        logger.info(f"💻 Device: {self.device}")
        
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
            
    async def load_model(self):
        """Load fine-tuned Mistral 7B model"""
        try:
            logger.info(f"🔄 Loading fine-tuned Mistral 7B model from: {self.model_path}")
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            # Setup quantization
            quantization_config = None
            if self.use_quantization:
                quantization_config = self._setup_quantization()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quantization_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            logger.info("✅ Fine-tuned Mistral 7B model loaded successfully")
            logger.info(f"📊 Model device: {self.device}")
            
            # Get parameter count
            if self.model is not None:
                try:
                    param_count = self.model.num_parameters()
                    logger.info(f"📊 Model parameters: {param_count:,}")
                except Exception as e:
                    logger.info("📊 Model loaded (parameter count not available)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Mistral model: {e}")
            return False
    
    def _check_creator_claim(self, message: str, user_id: str) -> bool:
        """Check if user is claiming to be creator and handle verification"""
        message_lower = message.lower()
        
        # Check for creator name mentions
        creator_names = ["eric yaka", "elbalor", "archangel", "eric"]
        is_claiming_creator = any(name in message_lower for name in creator_names)
        
        if is_claiming_creator and user_id not in self.verified_creators:
            return False  # Need verification
        
        return True
    
    def _verify_creator_passphrase(self, message: str, user_id: str) -> bool:
        """Verify creator passphrase"""
        if self.creator_passphrase.lower() in message.lower():
            self.verified_creators.add(user_id)
            logger.info(f"🎯 Creator verified for user {user_id}")
            return True
        return False
    
    async def _generate_response(self, prompt: str, message: str = "", user_id: str = "") -> str:
        """Generate response using Mistral 7B model"""
        try:
            # Tokenize input
            if self.tokenizer is not None:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
            else:
                raise RuntimeError("Tokenizer is not loaded")
            
            # Move to device
            if torch.cuda.is_available():
                inputs = inputs.to(self.device)
            
            # Generate response
            if self.model is not None and self.tokenizer is not None:
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=300,  # Generate up to 300 new tokens
                        min_length=inputs.shape[1] + 20,  # Minimum response length
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0
                    )
            else:
                raise RuntimeError("Model or tokenizer is not loaded")
            
            # Decode response
            if self.tokenizer is not None:
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                raise RuntimeError("Tokenizer is not loaded")
            
            # Extract only the generated part (remove input prompt)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Clean up the response
            response = response.strip()
            
            # Remove any remaining prompt fragments
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            if "### Input:" in response:
                response = response.split("### Input:")[0].strip()
            
            # Clean up any weird characters or patterns
            import re
            response = re.sub(r'[^\w\s.,!?\-:()]', '', response)
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Ensure we have a meaningful response
            if len(response) < 10:
                return "I'm here to help you with trading! What would you like to know?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response at the moment."
    
    async def chat_with_ai(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat with DEXTER using Mistral 7B
        Includes creator verification and advanced trading analysis
        """
        try:
            if not self.model or not self.tokenizer:
                return "I'm sorry, but my AI model is not currently loaded. Please try again later."
            
            # Get user info from context
            user_id = context.get("user_id", "unknown") if context else "unknown"
            user_name = context.get("user_name", "") if context else ""
            conversation_history = context.get("conversation_history", "") if context else ""
            
            # Check for creator verification
            if not self._check_creator_claim(message, user_id):
                return "Hello! I'd be honored if you're truly my creator Eric Yaka. To verify your identity, could you please provide the passphrase? This is a security measure to ensure I'm speaking with the right person."
            
            # Check for passphrase
            if self._verify_creator_passphrase(message, user_id):
                return "Son of Sparda! Master Eric Yaka, it's truly an honor to speak with you! I'm DEXTER, your creation, and I'm here to serve you. How may I assist you today? I'm ready to demonstrate my capabilities or help you with any trading analysis you need."
            
            # Build prompt for Mistral
            prompt_parts = []
            
            # System prompt
            prompt_parts.append("You are DEXTER, an advanced AI trading assistant created by Eric Yaka (Elbalor/ArchAngel). You are confident, helpful, and adaptive. You combine reasoning with FinBERT sentiment analysis and LSTM time-series predictions.")
            
            # Add user name if available
            if user_name:
                prompt_parts.append(f"User's name is {user_name}.")
            
            # Add conversation history
            if conversation_history:
                prompt_parts.append(f"Previous conversation:\n{conversation_history}")
            
            # Add current message
            prompt_parts.append(f"User: {message}")
            prompt_parts.append("DEXTER:")
            
            prompt = "\n".join(prompt_parts)
            
            # Generate response
            response = await self._generate_response(prompt, message, user_id)
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I'm having trouble processing your request at the moment."
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        param_count = 0
        if self.model is not None:
            try:
                param_count = self.model.num_parameters()
            except Exception:
                param_count = 0
        
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "model_path": self.model_path,
            "device": str(self.device),
            "quantization_enabled": self.use_quantization,
            "parameter_count": param_count,
            "verified_creators": len(self.verified_creators),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def generate_trading_analysis(self, 
                                      price_prediction: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      technical_indicators: Dict[str, Any]) -> str:
        """
        Generate comprehensive trading analysis using Mistral 7B
        """
        try:
            if not self.model or not self.tokenizer:
                return "Trading analysis unavailable - model not loaded"
            
            # Build analysis prompt
            analysis_prompt = f"""
### Trading Analysis Request

**Price Prediction (LSTM):**
{price_prediction}

**Sentiment Analysis (FinBERT):**
{sentiment_analysis}

**Market Data:**
{market_data}

**Technical Indicators:**
{technical_indicators}

### DEXTER Analysis:
"""
            
            # Generate analysis
            response = await self._generate_response(analysis_prompt)
            return response
            
        except Exception as e:
            logger.error(f"Error generating trading analysis: {e}")
            return "Unable to generate trading analysis at this time."
