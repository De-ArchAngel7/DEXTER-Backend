#!/usr/bin/env python3
"""
🔄 DEXTER MODEL RETRAINING SYSTEM
============================================================
Automatically retrains AI models based on trade learning data
"""

import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import structlog
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

class ModelRetrainingSystem:
    """
    🔄 Model Retraining System for DEXTER
    
    Retrains:
    1. DialoGPT with new conversation patterns
    2. LSTM with new price prediction data
    3. FinBERT with new sentiment data
    4. Updates model weights based on performance
    """
    
    def __init__(self):
        self.models_dir = "models"
        self.backup_dir = "models/backups"
        self.training_data_dir = "training_data"
        
        # Model paths
        self.dialoGPT_path = "models/dexter_dialoGPT"
        self.lstm_path = "backend/models/best_lstm_model.pth"
        self.sentiment_path = "models/finbert-finetuned"
        
        # Training parameters
        self.dialoGPT_training_args = {
            "output_dir": "./dialoGPT_retrain",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "logging_dir": "./logs",
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        }
        
        self.lstm_training_args = {
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "patience": 10,
            "min_delta": 0.001
        }
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        directories = [
            self.models_dir,
            self.backup_dir,
            self.training_data_dir,
            "dialoGPT_retrain",
            "logs"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    async def retrain_dialoGPT(self, 
                              conversation_data: List[Dict[str, Any]],
                              trade_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrain DialoGPT with new conversation and trade data
        
        Args:
            conversation_data: List of conversation examples
            trade_data: List of successful trade patterns
        """
        try:
            logger.info("🔄 Starting DialoGPT retraining...")
            
            # Create backup of current model
            await self._backup_model("dialoGPT")
            
            # Prepare training data
            training_data = await self._prepare_dialoGPT_data(conversation_data, trade_data)
            
            if not training_data:
                return {"error": "No training data available"}
            
            # Load base model and tokenizer
            base_model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure LoRA for efficient fine-tuning
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj", "wte", "wpe"]
            )
            
            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            
            # Prepare dataset
            dataset = self._create_dialoGPT_dataset(training_data, tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(**self.dialoGPT_training_args)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
            )
            
            # Train the model
            logger.info("🚀 Training DialoGPT...")
            trainer.train()
            
            # Save the retrained model
            await self._save_retrained_dialoGPT(model, tokenizer)
            
            # Evaluate the model
            evaluation_results = await self._evaluate_dialoGPT(model, tokenizer, training_data)
            
            logger.info("✅ DialoGPT retraining completed")
            
            return {
                "status": "success",
                "model_path": self.dialoGPT_path,
                "training_samples": len(training_data),
                "evaluation": evaluation_results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retraining DialoGPT: {e}")
            return {"error": str(e)}
    
    async def retrain_lstm(self, 
                          price_data: List[Dict[str, Any]],
                          trade_outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Retrain LSTM with new price prediction data
        
        Args:
            price_data: Historical price data
            trade_outcomes: Trade outcome data for labels
        """
        try:
            logger.info("🔄 Starting LSTM retraining...")
            
            # Create backup of current model
            await self._backup_model("lstm")
            
            # Prepare training data
            X, y = await self._prepare_lstm_data(price_data, trade_outcomes)
            
            if X is None or y is None:
                return {"error": "No training data available"}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Normalize data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Load existing model or create new one
            model = await self._load_or_create_lstm_model()
            
            # Train the model
            training_history = await self._train_lstm_model(
                model, X_train_scaled, y_train, X_test_scaled, y_test
            )
            
            # Save the retrained model
            await self._save_retrained_lstm(model, scaler)
            
            # Evaluate the model
            evaluation_results = await self._evaluate_lstm(model, X_test_scaled, y_test)
            
            logger.info("✅ LSTM retraining completed")
            
            return {
                "status": "success",
                "model_path": self.lstm_path,
                "training_samples": len(X_train),
                "evaluation": evaluation_results,
                "training_history": training_history,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retraining LSTM: {e}")
            return {"error": str(e)}
    
    async def _prepare_dialoGPT_data(self, 
                                   conversation_data: List[Dict[str, Any]],
                                   trade_data: List[Dict[str, Any]]) -> List[str]:
        """Prepare training data for DialoGPT"""
        try:
            training_texts = []
            
            # Add conversation data
            for conv in conversation_data:
                if conv.get("user_message") and conv.get("ai_response"):
                    # Create training example
                    text = f"User: {conv['user_message']}\nAI: {conv['ai_response']}"
                    training_texts.append(text)
            
            # Add successful trade patterns
            for trade in trade_data:
                if trade.get("success") and trade.get("ai_reasoning"):
                    # Create trading advice example
                    symbol = trade.get("symbol", "BTC")
                    action = trade.get("action", "BUY")
                    reasoning = trade.get("ai_reasoning", "")
                    
                    text = f"User: What should I do with {symbol}?\nAI: Based on my analysis, I recommend {action}. {reasoning}"
                    training_texts.append(text)
            
            # Add general trading knowledge
            general_examples = [
                "User: How do I manage risk?\nAI: Risk management is crucial in trading. Always use stop-loss orders, never risk more than 2% of your portfolio on a single trade, and diversify your holdings.",
                "User: What's a good entry strategy?\nAI: Look for strong technical indicators, confirm with volume, and enter on pullbacks to key support levels. Always have a clear exit strategy.",
                "User: How do I read charts?\nAI: Focus on key levels of support and resistance, trend lines, and volume patterns. Use multiple timeframes for confirmation.",
            ]
            
            training_texts.extend(general_examples)
            
            logger.info(f"📊 Prepared {len(training_texts)} training examples for DialoGPT")
            return training_texts
            
        except Exception as e:
            logger.error(f"Error preparing DialoGPT data: {e}")
            return []
    
    async def _prepare_lstm_data(self, 
                               price_data: List[Dict[str, Any]],
                               trade_outcomes: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for LSTM"""
        try:
            if not price_data:
                return np.array([]), np.array([])
            
            # Convert to DataFrame
            df = pd.DataFrame(price_data)
            
            # Create features (OHLCV data)
            features = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in features):
                logger.warning("Missing required price features")
                return np.array([]), np.array([])
            
            # Create sequences for LSTM (look back 60 periods)
            sequence_length = 60
            X, y = [], []
            
            for i in range(sequence_length, len(df)):
                # Input sequence
                X.append(df[features].iloc[i-sequence_length:i].values)
                
                # Target (next period's price change)
                current_price = df['close'].iloc[i-1]
                next_price = df['close'].iloc[i]
                price_change = (next_price - current_price) / current_price
                y.append(price_change)
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"📊 Prepared LSTM data: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}")
            return np.array([]), np.array([])
    
    def _create_dialoGPT_dataset(self, training_texts: List[str], tokenizer) -> Dataset:
        """Create dataset for DialoGPT training"""
        try:
            # Tokenize texts
            tokenized_texts = []
            for text in training_texts:
                tokens = tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=512,
                    return_tensors="pt"
                )
                tokenized_texts.append(tokens)
            
            # Create dataset
            from datasets import Dataset
            dataset = Dataset.from_list([
                {
                    "input_ids": tokens["input_ids"].squeeze(),
                    "attention_mask": tokens["attention_mask"].squeeze(),
                    "labels": tokens["input_ids"].squeeze()
                }
                for tokens in tokenized_texts
            ])
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error creating DialoGPT dataset: {e}")
            # Return empty dataset
            from datasets import Dataset
            return Dataset.from_dict({"input_ids": [], "attention_mask": []})
    
    async def _load_or_create_lstm_model(self):
        """Load existing LSTM model or create new one"""
        try:
            import torch.nn as nn
            
            class LSTMModel(nn.Module):
                def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=1):
                    super(LSTMModel, self).__init__()
                    self.hidden_size = hidden_size
                    self.num_layers = num_layers
                    
                    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                    self.fc = nn.Linear(hidden_size, output_size)
                    self.dropout = nn.Dropout(0.2)
                
                def forward(self, x):
                    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                    
                    out, _ = self.lstm(x, (h0, c0))
                    out = self.dropout(out[:, -1, :])
                    out = self.fc(out)
                    return out
            
            # Try to load existing model
            if os.path.exists(self.lstm_path):
                model = torch.load(self.lstm_path)
                logger.info("📁 Loaded existing LSTM model")
            else:
                model = LSTMModel()
                logger.info("🆕 Created new LSTM model")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading/creating LSTM model: {e}")
            return None
    
    async def _train_lstm_model(self, model, X_train, y_train, X_test, y_test):
        """Train LSTM model"""
        try:
            import torch.optim as optim
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.FloatTensor(y_train)
            X_test_tensor = torch.FloatTensor(X_test)
            y_test_tensor = torch.FloatTensor(y_test)
            
            # Create data loaders
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.lstm_training_args["batch_size"], shuffle=True)
            
            # Setup training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lstm_training_args["learning_rate"])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            # Training loop
            training_history = {"train_loss": [], "val_loss": []}
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(self.lstm_training_args["epochs"]):
                # Training
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_test_tensor)
                    val_loss = criterion(val_outputs.squeeze(), y_test_tensor).item()
                
                # Update history
                training_history["train_loss"].append(train_loss / len(train_loader))
                training_history["val_loss"].append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.lstm_training_args["patience"]:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            return training_history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {}
    
    async def _backup_model(self, model_type: str):
        """Create backup of current model"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"{model_type}_backup_{timestamp}")
            
            if model_type == "dialoGPT" and os.path.exists(self.dialoGPT_path):
                import shutil
                shutil.copytree(self.dialoGPT_path, backup_path)
                logger.info(f"📁 DialoGPT model backed up to {backup_path}")
            
            elif model_type == "lstm" and os.path.exists(self.lstm_path):
                import shutil
                shutil.copy2(self.lstm_path, f"{backup_path}.pth")
                logger.info(f"📁 LSTM model backed up to {backup_path}.pth")
                
        except Exception as e:
            logger.error(f"Error backing up {model_type} model: {e}")
    
    async def _save_retrained_dialoGPT(self, model, tokenizer):
        """Save retrained DialoGPT model"""
        try:
            # Save the model
            model.save_pretrained(self.dialoGPT_path)
            tokenizer.save_pretrained(self.dialoGPT_path)
            
            logger.info(f"💾 Retrained DialoGPT saved to {self.dialoGPT_path}")
            
        except Exception as e:
            logger.error(f"Error saving retrained DialoGPT: {e}")
    
    async def _save_retrained_lstm(self, model, scaler):
        """Save retrained LSTM model"""
        try:
            # Save model
            torch.save(model, self.lstm_path)
            
            # Save scaler
            scaler_path = self.lstm_path.replace(".pth", "_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info(f"💾 Retrained LSTM saved to {self.lstm_path}")
            
        except Exception as e:
            logger.error(f"Error saving retrained LSTM: {e}")
    
    async def _evaluate_dialoGPT(self, model, tokenizer, training_data):
        """Evaluate retrained DialoGPT"""
        try:
            # Simple evaluation - generate responses to test prompts
            test_prompts = [
                "What's your analysis of Bitcoin?",
                "How do I manage risk in trading?",
                "Should I buy Ethereum now?"
            ]
            
            model.eval()
            results = []
            
            for prompt in test_prompts:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_length=200,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({"prompt": prompt, "response": response})
            
            return {"test_responses": results}
            
        except Exception as e:
            logger.error(f"Error evaluating DialoGPT: {e}")
            return {"error": str(e)}
    
    async def _evaluate_lstm(self, model, X_test, y_test):
        """Evaluate retrained LSTM"""
        try:
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                predictions = model(X_test_tensor).squeeze().numpy()
            
            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            rmse = np.sqrt(mse)
            
            # Calculate accuracy (direction prediction)
            actual_direction = np.sign(y_test)
            predicted_direction = np.sign(predictions)
            direction_accuracy = np.mean(actual_direction == predicted_direction)
            
            return {
                "mse": float(mse),
                "mae": float(mae),
                "rmse": float(rmse),
                "direction_accuracy": float(direction_accuracy)
            }
            
        except Exception as e:
            logger.error(f"Error evaluating LSTM: {e}")
            return {"error": str(e)}
    
    async def retrain_models(self):
        """Main retraining method that orchestrates all model retraining"""
        try:
            logger.info("🔄 Starting comprehensive model retraining...")
            
            # Mock data for retraining (in production, this would come from actual learning data)
            conversation_data = [
                {"user": "What's the best crypto to buy?", "assistant": "Based on current market analysis..."},
                {"user": "Should I sell my BTC?", "assistant": "Let me analyze the current BTC trends..."}
            ]
            
            trade_data = [
                {"symbol": "BTC", "action": "buy", "price": 45000, "outcome": "profit"},
                {"symbol": "ETH", "action": "sell", "price": 3000, "outcome": "profit"}
            ]
            
            price_data = [
                {"timestamp": "2024-01-01", "price": 45000, "volume": 1000},
                {"timestamp": "2024-01-02", "price": 46000, "volume": 1200}
            ]
            
            trade_outcomes = [
                {"trade_id": "1", "profit": 500, "success": True},
                {"trade_id": "2", "profit": -200, "success": False}
            ]
            
            # Retrain DialoGPT
            dialoGPT_result = await self.retrain_dialoGPT(
                conversation_data=conversation_data,
                trade_data=trade_data
            )
            
            # Retrain LSTM
            lstm_result = await self.retrain_lstm(
                price_data=price_data,
                trade_outcomes=trade_outcomes
            )
            
            logger.info("✅ Model retraining completed successfully")
            return {
                "dialoGPT": dialoGPT_result,
                "lstm": lstm_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"❌ Error in model retraining: {e}")
            return {"error": str(e), "status": "failed"}

# Global instance
model_retraining_system = ModelRetrainingSystem()
