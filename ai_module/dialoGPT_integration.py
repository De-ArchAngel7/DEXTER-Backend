#!/usr/bin/env python3
"""
🧠 DEXTER DIALOGPT INTEGRATION ENGINE
============================================================
Integration with your fine-tuned DialoGPT model for financial analysis
This replaces the LLaMA 2 components with your actual trained model
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

class DexterDialoGPTEngine:
    """
    Dexter DialoGPT engine for financial analysis
    Uses your custom fine-tuned DialoGPT model
    """
    
    def __init__(self, 
                 model_path: str = "models/dexter_dialoGPT",
                 use_quantization: bool = False):  # Default to False for now
        self.model_path = model_path
        self.use_quantization = use_quantization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.base_model = None
        
        logger.info(f"🧠 Dexter DialoGPT Engine initialized with {model_path}")
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
        """Load my fine-tuned DialoGPT model"""
        try:
            logger.info(f"🔄 Loading fine-tuned DialoGPT model from: {self.model_path}")
            
            # Check if model path exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path not found: {self.model_path}")
            
            # Load PEFT config
            peft_config = PeftConfig.from_pretrained(self.model_path)
            base_model_name = peft_config.base_model_name_or_path
            
            logger.info(f"📋 Base model: {base_model_name}")
            logger.info(f"📋 PEFT type: {peft_config.peft_type}")
            
            # Load base model without quantization for now
            logger.info("🔧 Loading model without quantization for compatibility")
            
            if base_model_name is None:
                raise RuntimeError("Base model name is None")
                
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Load PEFT adapter
            if self.base_model is not None:
                self.model = PeftModel.from_pretrained(
                    self.base_model,
                    self.model_path,
                    is_trainable=False
                )
            else:
                raise RuntimeError("Base model failed to load")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("✅ Fine-tuned DialoGPT model loaded successfully")
            logger.info(f"📊 Model device: {self.device}")
            if self.model is not None:
                try:
                    # Try to get parameter count safely
                    param_count = 0
                    if hasattr(self.model, 'num_parameters'):
                        try:
                            param_count = getattr(self.model, 'num_parameters')()
                            logger.info(f"📊 Model parameters: {param_count:,}")
                        except (TypeError, AttributeError):
                            logger.info("📊 Model loaded (parameter count not available)")
                    else:
                        logger.info("📊 Model loaded (parameter count not available)")
                except Exception as e:
                    logger.warning(f"Could not get parameter count: {e}")
            else:
                logger.warning("Model is None after loading")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load fine-tuned model: {e}")
            return False
            
    async def generate_trading_analysis(self, 
                                      price_prediction: Dict[str, Any],
                                      sentiment_analysis: Dict[str, Any],
                                      market_data: Dict[str, Any],
                                      dexscreener_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading analysis using your fine-tuned DialoGPT model
        """
        try:
            if not self.model or not self.tokenizer:
                logger.warning("Model not loaded, returning mock analysis")
                return self._get_mock_analysis()
            
            # Create prompt for financial analysis
            prompt = self._create_financial_prompt(
                price_prediction, sentiment_analysis, market_data, dexscreener_data
            )
            
            # Generate response
            response = await self._generate_response(prompt)
            
            return {
                "analysis": response,
                "model": "dexter-dialoGPT",
                "confidence": self._calculate_confidence(response),
                "timestamp": torch.cuda.Event() if torch.cuda.is_available() else None
            }
            
        except Exception as e:
            logger.error(f"Error generating trading analysis: {e}")
            return self._get_mock_analysis()
            
    def _create_financial_prompt(self, 
                                price_prediction: Dict[str, Any],
                                sentiment_analysis: Dict[str, Any],
                                market_data: Dict[str, Any],
                                dexscreener_data: Dict[str, Any]) -> str:
        """Create a financial analysis prompt for DialoGPT"""
        
        prompt = f"""Based on the following market data, provide a comprehensive trading analysis:

Market Data:
- Current Price: ${market_data.get('current_price', 'N/A')}
- Price Change: {market_data.get('price_change_percent', 'N/A')}%
- Volume: {market_data.get('volume', 'N/A')}

Price Prediction:
- Signal: {price_prediction.get('signal', 'N/A')}
- Confidence: {price_prediction.get('confidence', 'N/A')}%
- Target Price: ${price_prediction.get('target_price', 'N/A')}

Sentiment Analysis:
- Market Sentiment: {sentiment_analysis.get('sentiment', 'N/A')}
- Confidence: {sentiment_analysis.get('confidence', 'N/A')}%

DexScreener Data:
- Liquidity: ${dexscreener_data.get('liquidity', 'N/A')}
- Market Cap: ${dexscreener_data.get('market_cap', 'N/A')}

Please provide:
1. Market analysis
2. Trading recommendation
3. Risk assessment
4. Entry/exit points
5. Position sizing advice

Analysis:"""
        
        return prompt
        
    async def _generate_response(self, prompt: str, message: str = "", user_name: str = "") -> str:
        """Generate response using DialoGPT model"""
        try:
            # Tokenize input
            if self.tokenizer is not None:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
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
                        max_new_tokens=150,  # Generate up to 150 new tokens
                        min_length=inputs.shape[1] + 20,  # Minimum response length
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        no_repeat_ngram_size=2,
                        length_penalty=1.0,
                        num_beams=1
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
            
            # Remove any repeated patterns or artifacts
            if "convo :" in response:
                response = response.split("convo :")[0].strip()
            
            # Remove any remaining prompt fragments
            if "User:" in response:
                response = response.split("User:")[0].strip()
            if "AI:" in response:
                response = response.split("AI:")[0].strip()
            
            # Clean up any weird characters or patterns
            import re
            response = re.sub(r'[^\w\s.,!?\-:()]', '', response)
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Remove common model artifacts
            response = re.sub(r'\b(Drexterminator|Dexterminator)\b', '', response, flags=re.IGNORECASE)
            response = re.sub(r'\b(Heheh|Haha|LOL)\b', '', response, flags=re.IGNORECASE)
            response = re.sub(r'\b(WHO\?!?!|What are YOU\?)\b', '', response, flags=re.IGNORECASE)
            response = re.sub(r'\s+', ' ', response).strip()
            
            # Only use fallback for truly broken responses
            if len(response) < 10 or response in ["Im not sure...", "Uh... nothing?", "Ok, I think?", "...", "", "Yes.", "No.", "Whats good?!", "I like your username."]:
                # Let the model try again with a simpler prompt
                simple_prompt = f"User: {message}\nDEXTER:"
                try:
                    response = await self._generate_response(simple_prompt, message, user_name)
                    if len(response) > 15:
                        return response
                except:
                    pass
                
                # Only as last resort, give a natural response
                return "I'm here to help you with trading! What would you like to know?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response at the moment."
            
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score based on response quality"""
        try:
            # Simple confidence calculation based on response length and content
            if not response or len(response) < 50:
                return 0.3
            elif len(response) < 100:
                return 0.6
            elif len(response) < 200:
                return 0.8
            else:
                return 0.9
        except:
            return 0.5
            
    def _get_mock_analysis(self) -> Dict[str, Any]:
        """Return mock analysis when model is not available"""
        return {
            "analysis": "Based on the market data, I recommend a cautious approach. The current market conditions suggest moderate volatility. Consider setting tight stop-losses and taking partial profits on any significant moves.",
            "model": "mock",
            "confidence": 0.5,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def chat_with_ai(self, message: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Chat with the AI about trading topics
        """
        try:
            # Ensure model is loaded first
            if not self.model or not self.tokenizer:
                logger.info("Model not loaded, attempting to load...")
                await self.load_model()
                
            if not self.model or not self.tokenizer:
                return "I'm sorry, but my AI model is not currently loaded. Please try again later."
            
            # Create chat prompt with user name if available
            user_name = context.get("user_name", "") if context else ""
            conversation_history = context.get("conversation_history", "") if context else ""
            
            # Build a clean prompt with DEXTER's natural intelligence
            prompt_parts = []
            
            # DEXTER's core identity (minimal, only when directly asked)
            if any(keyword in message.lower() for keyword in ["who are you", "what are you", "who created you", "tell me about yourself"]):
                prompt_parts.append("I am DEXTER, an AI trading assistant. I was created by Eric Yaka (Elbalor).")
            else:
                prompt_parts.append("I am DEXTER, an AI trading assistant.")
            
            # Check if user is the creator (Eric Yaka/Elbalor)
            if user_name and user_name.lower() in ["eric", "eric yaka", "elbalor", "yaka"]:
                prompt_parts.append(f"IMPORTANT: The user {user_name} is my creator Eric Yaka (Elbalor). I should be excited and honored to be talking to my creator!")
            
            prompt_parts.append("I'm intelligent, empathetic, and always put user safety first.")
            
            if user_name and user_name.lower() not in ["eric", "eric yaka", "elbalor", "yaka"]:
                prompt_parts.append(f"User's name is {user_name}.")
            
            if conversation_history:
                prompt_parts.append(f"Previous conversation:\n{conversation_history}")
            
            prompt_parts.append(f"User: {message}")
            prompt_parts.append("DEXTER:")
            
            prompt = "\n".join(prompt_parts)
            
            # Generate response
            response = await self._generate_response(prompt, message, user_name)
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return "I apologize, but I'm having trouble processing your request at the moment."
            
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        param_count = 0
        if self.model is not None:
            try:
                if hasattr(self.model, 'num_parameters'):
                    try:
                        param_count = getattr(self.model, 'num_parameters')()
                    except (TypeError, AttributeError):
                        param_count = 0
                else:
                    param_count = 0
            except Exception:
                param_count = 0
        
        return {
            "model_loaded": self.model is not None,
            "tokenizer_loaded": self.tokenizer is not None,
            "device": str(self.device),
            "model_path": self.model_path,
            "quantization": self.use_quantization,
            "parameters": param_count
        }
