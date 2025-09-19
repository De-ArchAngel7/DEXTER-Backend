#!/usr/bin/env python3
"""
🧠 DEXTER UNIFIED CONVERSATION ENGINE
============================================================
Unified conversation engine with Mistral + OpenAI fallback
Handles conversation history and model switching
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
import openai
from openai.types.chat import ChatCompletionMessageParam
from .mistral_integration import DexterMistralEngine
from .mistral_hf_api_integration import DexterMistralHFAPI
import os
import sys

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.services.conversation_logger import conversation_logger
from ai_module.trade_learning import trade_learning_system
from ai_module.model_retraining import model_retraining_system

logger = structlog.get_logger()

class ConversationHistory:
    """Manages conversation history per user"""
    
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.user_names: Dict[str, str] = {}  # Store user names for personalization
    
    def add_message(self, user_id: str, role: str, content: str, model_used: Optional[str] = None):
        """Add a message to conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used if model_used is not None else "unknown"
        }
        
        self.conversations[user_id].append(message)
        
        # Keep only recent messages
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
    
    def get_context(self, user_id: str) -> List[Dict[str, str]]:
        """Get conversation context for a user"""
        if user_id not in self.conversations:
            return []
        
        # Return only role and content for OpenAI format
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversations[user_id]
        ]
    
    def get_openai_context(self, user_id: str) -> List[ChatCompletionMessageParam]:
        """Get conversation context for OpenAI API"""
        if user_id not in self.conversations:
            return []
        
        # Return only role and content for OpenAI format
        context: List[ChatCompletionMessageParam] = []
        for msg in self.conversations[user_id]:
            context.append({
                "role": msg["role"], 
                "content": msg["content"]
            })
        return context
    
    def clear_history(self, user_id: str):
        """Clear conversation history for a user"""
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def set_user_name(self, user_id: str, name: str):
        """Set user's name for personalization"""
        self.user_names[user_id] = name
        logger.info(f"📝 Set name for user {user_id}: {name}")
    
    def get_user_name(self, user_id: str) -> str:
        """Get user's name"""
        return self.user_names.get(user_id, "")
    
    def extract_name_from_message(self, message: str) -> str:
        """Extract potential name from user message"""
        import re
        
        # Common name patterns
        patterns = [
            r"my name is (\w+)",
            r"i'm (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"i'm called (\w+)",
            r"name's (\w+)",
            r"^(\w+)$"  # Single word might be a name
        ]
        
        message_lower = message.lower().strip()
        
        for pattern in patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(1).capitalize()
                # Basic validation - names should be 2-20 characters and alphabetic
                if 2 <= len(name) <= 20 and name.isalpha():
                    return name
        
        return ""

class UnifiedConversationEngine:
    """
    Unified conversation engine with DialoGPT + OpenAI fallback
    """
    
    def __init__(self):
        self.mistral_engine = DexterMistralEngine()
        self.conversation_history = ConversationHistory()
        self.openai_client = None
        self.mistral_loaded = False
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OpenAI API key not found, fallback disabled")
        
        # Load Mistral model (will be loaded on first use)
        self._load_mistral_task = None
    
    async def _ensure_mistral_loaded(self):
        """Ensure Mistral model is loaded"""
        if not self._load_mistral_task:
            self._load_mistral_task = asyncio.create_task(self._load_mistral())
        
        if not self.mistral_loaded:
            await self._load_mistral_task
    
    async def _load_mistral(self):
        """Load Mistral model"""
        try:
            self.mistral_loaded = await self.mistral_engine.load_model()
            if self.mistral_loaded:
                logger.info("Mistral 7B model loaded successfully")
            else:
                logger.warning("Mistral 7B model failed to load")
        except Exception as e:
            logger.error(f"Error loading Mistral: {e}")
            self.mistral_loaded = False
    
    async def chat(self, user_id: str, message: str, source: str = "web") -> Dict[str, Any]:
        """
        Main chat method with unified conversation handling
        
        Args:
            user_id: Unique user identifier
            message: User message
            source: Source of the message ("telegram" or "web")
        
        Returns:
            Dict with reply, source, and model_used
        """
        try:
            # Check for name in message and store it
            extracted_name = self.conversation_history.extract_name_from_message(message)
            if extracted_name and not self.conversation_history.get_user_name(user_id):
                self.conversation_history.set_user_name(user_id, extracted_name)
                logger.info(f"🎯 Extracted name '{extracted_name}' for user {user_id}")
            
            # Add user message to history
            self.conversation_history.add_message(user_id, "user", message)
            
            reply = None
            model_used = "fallback"
            
            # Try Mistral first
            try:
                await self._ensure_mistral_loaded()
                if self.mistral_loaded:
                    reply = await self._try_mistral(user_id, message)
                    if reply and len(reply.strip()) > 10:  # Basic quality check
                        model_used = "mistral"
            except Exception as e:
                logger.warning(f"Mistral failed: {e}")
            
            # Fallback to OpenAI
            if not reply and self.openai_client:
                try:
                    reply = await self._try_openai(user_id, message)
                    if reply:
                        model_used = "openai"
                except Exception as e:
                    logger.error(f"OpenAI fallback failed: {e}")
            
            # Ultimate fallback
            if not reply:
                reply = "I'm sorry, but I'm having trouble processing your request at the moment. Please try again later."
                model_used = "fallback"
            
            # Add AI response to history
            self.conversation_history.add_message(user_id, "assistant", reply, model_used)
            
            # Log conversation to MongoDB
            await conversation_logger.log_conversation(
                user_id=user_id,
                message=message,
                reply=reply,
                model_used=model_used,
                source=source,
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            # Log for learning system
            await self._log_for_learning(user_id, message, reply, model_used, source)
            
            return {
                "reply": reply,
                "source": source,
                "model_used": model_used,
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in unified chat: {e}")
            return {
                "reply": "I encountered an error. Please try again.",
                "source": source,
                "model_used": "error",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _try_mistral(self, user_id: str, message: str) -> Optional[str]:
        """Try to get response from Mistral 7B"""
        try:
            # Get conversation context
            context = self.conversation_history.get_context(user_id)
            
            # Get user name for personalization
            user_name = self.conversation_history.get_user_name(user_id)
            
            # Create context string for Mistral
            context_str = ""
            if context:
                context_str = "Previous conversation:\n"
                for msg in context[-3:]:  # Last 3 messages for context
                    context_str += f"{msg['role']}: {msg['content']}\n"
            
            # Create context dict with user name
            context_dict = {
                "conversation_history": context_str,
                "user_name": user_name,
                "user_id": user_id
            }
            
            # Generate response
            response = await self.mistral_engine.chat_with_ai(message, context_dict)
            return response
            
        except Exception as e:
            logger.error(f"Mistral error: {e}")
            return None
    
    async def _try_openai(self, user_id: str, message: str) -> Optional[str]:
        """Try to get response from OpenAI"""
        try:
            # Get conversation context for OpenAI
            context = self.conversation_history.get_openai_context(user_id)
            
            # Add system message
            system_message: ChatCompletionMessageParam = {
                "role": "system",
                "content": "You are DEXTER, an AI trading assistant. You provide professional financial analysis and trading advice based on market data and technical indicators. Be helpful, accurate, and concise."
            }
            
            # Create messages list with proper typing
            messages = [system_message] + context
            
            # Call OpenAI API
            if self.openai_client and self.openai_client.chat:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    max_tokens=500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            else:
                return None
            
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return None
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a user"""
        return self.conversation_history.conversations.get(user_id, [])
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user"""
        self.conversation_history.clear_history(user_id)
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get status of all engines"""
        return {
            "mistral_loaded": self.mistral_loaded,
            "openai_available": self.openai_client is not None,
            "active_conversations": len(self.conversation_history.conversations),
            "mistral_status": self.mistral_engine.get_model_status() if self.mistral_loaded else None
        }
    
    async def _log_for_learning(self, user_id: str, message: str, reply: str, model_used: str, source: str):
        """Log conversation for learning system"""
        try:
            if not trade_learning_system.collection:
                await trade_learning_system.initialize()
            
            # Log conversation for learning
            await trade_learning_system.log_ai_prediction(
                user_id=user_id,
                symbol=self._extract_symbol(message),
                prediction=reply,
                confidence=self._calculate_confidence(reply, model_used)
            )
            
        except Exception as e:
            logger.error(f"Error logging for learning: {e}")
    
    def _extract_symbol(self, message: str) -> str:
        """Extract trading symbol from message"""
        try:
            # Common crypto symbols
            symbols = ["BTC", "ETH", "SOL", "ADA", "DOT", "MATIC", "AVAX", "LINK", "UNI", "AAVE"]
            message_upper = message.upper()
            
            for symbol in symbols:
                if symbol in message_upper:
                    return symbol
            
            return "GENERAL"
        except:
            return "GENERAL"
    
    def _calculate_confidence(self, reply: str, model_used: str) -> float:
        """Calculate confidence score for the reply"""
        try:
            # Base confidence by model
            base_confidence = 0.9 if model_used == "mistral" else 0.8
            
            # Adjust based on reply length and content
            if len(reply) < 50:
                base_confidence *= 0.8
            elif len(reply) > 200:
                base_confidence *= 1.1
            
            # Check for uncertainty indicators
            uncertainty_words = ["maybe", "perhaps", "might", "could", "possibly", "uncertain"]
            if any(word in reply.lower() for word in uncertainty_words):
                base_confidence *= 0.9
            
            return min(1.0, max(0.1, base_confidence))
        except:
            return 0.7
    
    async def log_trade_outcome(self, 
                              user_id: str,
                              symbol: str,
                              action: str,
                              entry_price: float,
                              exit_price: Optional[float] = None,
                              success: Optional[bool] = None,
                              profit_loss: Optional[float] = None,
                              user_feedback: Optional[str] = None):
        """Log trade outcome for learning"""
        try:
            if not trade_learning_system.collection:
                await trade_learning_system.initialize()
            
            await trade_learning_system.log_trade(
                user_id=user_id,
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                exit_price=exit_price if exit_price is not None else 0.0,
                success=success if success is not None else False,
                profit_loss=profit_loss if profit_loss is not None else 0.0,
                user_feedback=user_feedback if user_feedback is not None else ""
            )
            
        except Exception as e:
            logger.error(f"Error logging trade outcome: {e}")
    
    async def get_learning_performance(self) -> Dict[str, Any]:
        """Get learning system performance"""
        try:
            if not trade_learning_system.collection:
                await trade_learning_system.initialize()
            
            return await trade_learning_system.get_performance_summary()
            
        except Exception as e:
            logger.error(f"Error getting learning performance: {e}")
            return {"error": str(e)}
    
    async def trigger_model_retraining(self) -> Dict[str, Any]:
        """Trigger model retraining based on learning data"""
        try:
            if not trade_learning_system.collection:
                await trade_learning_system.initialize()
            
            # Get recent data for retraining
            recent_data = await trade_learning_system.export_learning_data(days=30)
            
            if "error" in recent_data:
                return {"error": "No learning data available"}
            
            # Extract conversation and trade data
            conversation_data = []
            trade_data = []
            
            for record in recent_data.get("data", []):
                if record.get("type") == "prediction":
                    conversation_data.append({
                        "user_message": f"Question about {record.get('symbol', 'trading')}",
                        "ai_response": record.get("prediction", ""),
                        "accuracy": record.get("accuracy", 0.5)
                    })
                elif record.get("type") != "feedback":
                    trade_data.append(record)
            
            # Retrain DialoGPT
            dialoGPT_results = await model_retraining_system.retrain_dialoGPT(
                conversation_data, trade_data
            )
            
            # Retrain LSTM (if we have price data)
            lstm_results = {"status": "skipped", "reason": "No price data available"}
            
            return {
                "dialoGPT_retraining": dialoGPT_results,
                "lstm_retraining": lstm_results,
                "data_used": {
                    "conversations": len(conversation_data),
                    "trades": len(trade_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Error triggering model retraining: {e}")
            return {"error": str(e)}

# Global instance
conversation_engine = UnifiedConversationEngine()
