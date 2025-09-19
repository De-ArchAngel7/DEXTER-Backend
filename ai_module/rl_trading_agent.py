import numpy as np
from typing import Dict, Any
import structlog
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class RLTradingAgent:
    def __init__(self):
        self.state_size = 10  # Number of features in state
        self.action_size = 3   # BUY, SELL, HOLD
        self.learning_rate = 0.001
        self.epsilon = 0.1     # Exploration rate
        self.gamma = 0.95      # Discount factor
        
        # Initialize Q-table (simple Q-learning)
        self.q_table = {}
        
        # Trading history for learning
        self.trading_history = []
        
        # Performance metrics
        self.total_pnl = 0.0
        self.win_rate = 0.0
        self.total_trades = 0
        
    def get_trading_action(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get trading action using RL agent"""
        try:
            # Get current state
            state = self._get_current_state(symbol, current_price, market_data)
            
            # Get action from Q-table or random exploration
            action = self._get_action(state)
            
            # Calculate confidence based on Q-values
            confidence = self._calculate_action_confidence(state, action)
            
            # Generate trading recommendation
            recommendation = self._generate_recommendation(action, confidence, market_data)
            
            return {
                "symbol": symbol,
                "action": action,
                "confidence": confidence,
                "current_price": current_price,
                "recommendation": recommendation,
                "state_features": state,
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": "rl_agent"
            }
            
        except Exception as e:
            logger.error(f"Error getting RL trading action: {e}")
            return self._get_fallback_action(symbol, current_price)
    
    def _get_current_state(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> tuple:
        """Get current market state as features"""
        try:
            # Extract key features from market data
            features = []
            
            # Price features
            features.append(current_price)
            features.append(market_data.get("price_change_24h", 0))
            features.append(market_data.get("volume_24h", 0))
            features.append(market_data.get("market_cap", 0))
            
            # Technical indicators
            features.append(market_data.get("rsi", 50))
            features.append(market_data.get("macd", 0))
            features.append(market_data.get("sma_20", current_price))
            features.append(market_data.get("sma_50", current_price))
            
            # Market sentiment
            features.append(market_data.get("sentiment_score", 0))
            features.append(market_data.get("volatility", 0))
            
            # Convert to tuple for Q-table key
            return tuple(features)
            
        except Exception as e:
            logger.error(f"Error getting current state: {e}")
            # Return default state
            return tuple([current_price, 0, 0, 0, 50, 0, current_price, current_price, 0, 0])
    
    def _get_action(self, state: tuple) -> str:
        """Get action using epsilon-greedy policy"""
        try:
            # Random exploration
            if np.random.random() < self.epsilon:
                return np.random.choice(["BUY", "SELL", "HOLD"])
            
            # Get Q-values for current state
            if state not in self.q_table:
                # Initialize Q-values for new state
                self.q_table[state] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            
            # Choose action with highest Q-value
            q_values = self.q_table[state]
            return max(q_values, key=q_values.get)
            
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            return "HOLD"
    
    def _calculate_action_confidence(self, state: tuple, action: str) -> float:
        """Calculate confidence in the chosen action"""
        try:
            if state not in self.q_table:
                return 50.0  # Default confidence for new states
            
            q_values = self.q_table[state]
            action_q_value = q_values.get(action, 0.0)
            
            # Normalize Q-value to confidence (0-100)
            max_q = max(q_values.values()) if q_values else 0.0
            min_q = min(q_values.values()) if q_values else 0.0
            
            if max_q == min_q:
                return 50.0
            
            # Scale confidence based on Q-value difference
            confidence = 50.0 + (action_q_value - min_q) / (max_q - min_q) * 50.0
            
            return max(20.0, min(95.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 50.0
    
    def _generate_recommendation(self, action: str, confidence: float, market_data: Dict[str, Any]) -> str:
        """Generate human-readable trading recommendation"""
        try:
            base_recommendation = f"RL Agent recommends {action} with {confidence:.1f}% confidence."
            
            # Add market context
            if market_data.get("rsi", 50) < 30:
                base_recommendation += " RSI indicates oversold conditions."
            elif market_data.get("rsi", 50) > 70:
                base_recommendation += " RSI indicates overbought conditions."
            
            if market_data.get("sentiment_score", 0) > 0.5:
                base_recommendation += " Market sentiment is positive."
            elif market_data.get("sentiment_score", 0) < -0.5:
                base_recommendation += " Market sentiment is negative."
            
            # Add risk warning
            if confidence < 60:
                base_recommendation += " ⚠️ Low confidence - consider smaller position size."
            elif confidence > 80:
                base_recommendation += " ✅ High confidence - normal position size recommended."
            
            return base_recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return f"RL Agent recommends {action} with {confidence:.1f}% confidence."
    
    def update_q_value(self, state: tuple, action: str, reward: float, next_state: tuple):
        """Update Q-value using Q-learning algorithm"""
        try:
            # Initialize Q-values if state doesn't exist
            if state not in self.q_table:
                self.q_table[state] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            if next_state not in self.q_table:
                self.q_table[next_state] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
            
            # Q-learning update formula
            current_q = self.q_table[state][action]
            max_next_q = max(self.q_table[next_state].values())
            
            new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
            self.q_table[state][action] = new_q
            
        except Exception as e:
            logger.error(f"Error updating Q-value: {e}")
    
    def record_trade(self, symbol: str, action: str, entry_price: float, exit_price: float, 
                     entry_time: datetime, exit_time: datetime):
        """Record trade for learning and performance tracking"""
        try:
            # Calculate PnL
            if action == "BUY":
                pnl = (exit_price - entry_price) / entry_price * 100
            elif action == "SELL":
                pnl = (entry_price - exit_price) / entry_price * 100
            else:
                pnl = 0.0
            
            # Record trade
            trade = {
                "symbol": symbol,
                "action": action,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_time": entry_time.isoformat(),
                "exit_time": exit_time.isoformat(),
                "pnl": pnl,
                "duration_hours": (exit_time - entry_time).total_seconds() / 3600
            }
            
            self.trading_history.append(trade)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Learn from this trade
            self._learn_from_trade(trade)
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if not self.trading_history:
                return
            
            # Calculate total PnL
            self.total_pnl = sum(trade["pnl"] for trade in self.trading_history)
            
            # Calculate win rate
            winning_trades = [trade for trade in self.trading_history if trade["pnl"] > 0]
            self.win_rate = len(winning_trades) / len(self.trading_history) * 100
            
            # Update total trades
            self.total_trades = len(self.trading_history)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def _learn_from_trade(self, trade: Dict[str, Any]):
        """Learn from trade outcome"""
        try:
            # Simple reward function based on PnL
            pnl = trade["pnl"]
            
            if pnl > 5:  # Strong positive reward for good trades
                reward = 10.0
            elif pnl > 0:  # Small positive reward for profitable trades
                reward = 2.0
            elif pnl > -5:  # Small negative reward for small losses
                reward = -1.0
            else:  # Strong negative reward for large losses
                reward = -5.0
            
            # For now, we'll use a simplified learning approach
            # In a full implementation, you'd update Q-values here
            logger.info(f"Learning from trade: {trade['action']} {trade['symbol']}, PnL: {pnl:.2f}%, Reward: {reward}")
            
        except Exception as e:
            logger.error(f"Error learning from trade: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        try:
            return {
                "total_trades": self.total_trades,
                "total_pnl": self.total_pnl,
                "win_rate": self.win_rate,
                "avg_pnl_per_trade": self.total_pnl / self.total_trades if self.total_trades > 0 else 0,
                "model_used": "rl_agent",
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl_per_trade": 0.0,
                "model_used": "rl_agent",
                "last_updated": datetime.utcnow().isoformat()
            }
    
    def _get_fallback_action(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Fallback action when RL agent fails"""
        return {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": 50.0,
            "current_price": current_price,
            "recommendation": "RL Agent unavailable. Defaulting to HOLD recommendation.",
            "state_features": [],
            "timestamp": datetime.utcnow().isoformat(),
            "model_used": "fallback"
        }
    
    def reset_agent(self):
        """Reset the RL agent"""
        try:
            self.q_table = {}
            self.trading_history = []
            self.total_pnl = 0.0
            self.win_rate = 0.0
            self.total_trades = 0
            logger.info("RL agent reset successfully")
        except Exception as e:
            logger.error(f"Error resetting RL agent: {e}")
    
    def save_agent_state(self, filepath: str):
        """Save agent state to file"""
        try:
            import pickle
            agent_state = {
                "q_table": self.q_table,
                "trading_history": self.trading_history,
                "total_pnl": self.total_pnl,
                "win_rate": self.win_rate,
                "total_trades": self.total_trades
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(agent_state, f)
            
            logger.info(f"Agent state saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving agent state: {e}")
    
    def load_agent_state(self, filepath: str):
        """Load agent state from file"""
        try:
            import pickle
            with open(filepath, 'rb') as f:
                agent_state = pickle.load(f)
            
            self.q_table = agent_state.get("q_table", {})
            self.trading_history = agent_state.get("trading_history", [])
            self.total_pnl = agent_state.get("total_pnl", 0.0)
            self.win_rate = agent_state.get("win_rate", 0.0)
            self.total_trades = agent_state.get("total_trades", 0)
            
            logger.info(f"Agent state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
