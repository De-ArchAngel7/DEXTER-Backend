import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
from dataclasses import dataclass
import json
import os

logger = structlog.get_logger()

@dataclass
class Trade:
    """Represents a single trade"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    timestamp: datetime
    ai_confidence: float
    ai_reasoning: str
    status: str = 'OPEN'  # 'OPEN', 'CLOSED', 'CANCELLED'
    pnl: float = 0.0
    close_price: Optional[float] = None
    close_timestamp: Optional[datetime] = None

@dataclass
class Portfolio:
    """Represents the current portfolio state"""
    initial_capital: float
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    total_value: float
    pnl: float
    pnl_percentage: float

class PaperTradingEngine:
    """Paper trading engine for testing DEXTER's AI predictions"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}
        self.trades: List[Trade] = []
        self.trade_id_counter = 1
        self.performance_history: List[Dict[str, Any]] = []
        
        # Trading parameters
        self.max_position_size = 0.1  # Max 10% of portfolio in single position
        self.stop_loss_percentage = 0.05  # 5% stop loss
        self.take_profit_percentage = 0.15  # 15% take profit
        
        # Performance tracking
        self.start_date = datetime.now()
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        logger.info(f"🚀 Paper Trading Engine initialized with ${initial_capital:,.2f}")
    
    def get_portfolio(self) -> Portfolio:
        """Get current portfolio state"""
        total_value = self.cash
        for symbol, quantity in self.positions.items():
            # For paper trading, we'll use a simple price simulation
            # In real implementation, this would fetch current market prices
            simulated_price = self._get_simulated_price(symbol)
            total_value += quantity * simulated_price
        
        pnl = total_value - self.initial_capital
        pnl_percentage = (pnl / self.initial_capital) * 100
        
        return Portfolio(
            initial_capital=self.initial_capital,
            cash=self.cash,
            positions=self.positions.copy(),
            total_value=total_value,
            pnl=pnl,
            pnl_percentage=pnl_percentage
        )
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for paper trading"""
        # Simple price simulation - in real implementation, fetch from exchange
        base_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'ADAUSDT': 0.5,
            'DOTUSDT': 7.0,
            'LINKUSDT': 15.0
        }
        return base_prices.get(symbol, 100.0)
    
    def execute_trade(self, symbol: str, side: str, quantity: float, 
                     price: float, ai_confidence: float, ai_reasoning: str) -> Trade:
        """Execute a paper trade based on AI prediction"""
        
        # Validate trade
        if side not in ['BUY', 'SELL']:
            raise ValueError(f"Invalid side: {side}")
        
        if quantity <= 0:
            raise ValueError(f"Invalid quantity: {quantity}")
        
        if price <= 0:
            raise ValueError(f"Invalid price: {price}")
        
        # Check if we have enough cash for BUY
        if side == 'BUY':
            required_cash = quantity * price
            if required_cash > self.cash:
                raise ValueError(f"Insufficient cash. Required: ${required_cash:,.2f}, Available: ${self.cash:,.2f}")
        
        # Check if we have enough position for SELL
        if side == 'SELL':
            current_position = self.positions.get(symbol, 0)
            if quantity > current_position:
                raise ValueError(f"Insufficient position. Required: {quantity}, Available: {current_position}")
        
        # Create trade
        trade = Trade(
            id=f"TRADE_{self.trade_id_counter:06d}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            ai_confidence=ai_confidence,
            ai_reasoning=ai_reasoning
        )
        
        # Execute trade
        if side == 'BUY':
            self.cash -= quantity * price
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            logger.info(f"📈 BUY {quantity} {symbol} at ${price:,.2f}")
        
        elif side == 'SELL':
            self.cash += quantity * price
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            if self.positions[symbol] <= 0:
                del self.positions[symbol]
            logger.info(f"📉 SELL {quantity} {symbol} at ${price:,.2f}")
        
        # Add trade to history
        self.trades.append(trade)
        self.trade_id_counter += 1
        self.total_trades += 1
        
        # Update performance
        self._update_performance()
        
        return trade
    
    def close_position(self, symbol: str, close_price: float) -> Optional[Trade]:
        """Close a position at current market price"""
        if symbol not in self.positions:
            return None
        
        quantity = self.positions[symbol]
        if quantity <= 0:
            return None
        
        # Create closing trade
        trade = Trade(
            id=f"CLOSE_{self.trade_id_counter:06d}",
            symbol=symbol,
            side='SELL' if quantity > 0 else 'BUY',
            quantity=abs(quantity),
            price=close_price,
            timestamp=datetime.now(),
            ai_confidence=100.0,
            ai_reasoning="Position closed by system"
        )
        
        # Execute close
        if quantity > 0:  # Long position
            self.cash += quantity * close_price
            del self.positions[symbol]
            logger.info(f"🔒 CLOSE LONG {quantity} {symbol} at ${close_price:,.2f}")
        else:  # Short position
            self.cash -= abs(quantity) * close_price
            del self.positions[symbol]
            logger.info(f"🔒 CLOSE SHORT {abs(quantity)} {symbol} at ${close_price:,.2f}")
        
        # Add to trades
        self.trades.append(trade)
        self.trade_id_counter += 1
        
        return trade
    
    def _update_performance(self):
        """Update performance metrics"""
        portfolio = self.get_portfolio()
        
        # Calculate trade performance
        if self.trades:
            closed_trades = [t for t in self.trades if t.status == 'CLOSED']
            if closed_trades:
                winning_trades = [t for t in closed_trades if t.pnl > 0]
                self.winning_trades = len(winning_trades)
                self.losing_trades = len(closed_trades) - len(winning_trades)
        
        # Record performance snapshot
        performance_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'total_value': portfolio.total_value,
            'cash': portfolio.cash,
            'positions': portfolio.positions.copy(),
            'pnl': portfolio.pnl,
            'pnl_percentage': portfolio.pnl_percentage,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }
        
        self.performance_history.append(performance_snapshot)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        portfolio = self.get_portfolio()
        
        # Calculate win rate
        win_rate = (self.winning_trades / max(self.total_trades, 1)) * 100
        
        # Calculate average trade PnL
        closed_trades = [t for t in self.trades if t.status == 'CLOSED']
        avg_trade_pnl = np.mean([t.pnl for t in closed_trades]) if closed_trades else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.performance_history) > 1:
            returns = []
            for i in range(1, len(self.performance_history)):
                prev_value = self.performance_history[i-1]['total_value']
                curr_value = self.performance_history[i]['total_value']
                if prev_value > 0:
                    returns.append((curr_value - prev_value) / prev_value)
            
            if returns:
                sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return {
            'portfolio_summary': {
                'initial_capital': self.initial_capital,
                'current_value': portfolio.total_value,
                'cash': portfolio.cash,
                'total_pnl': portfolio.pnl,
                'total_pnl_percentage': portfolio.pnl_percentage
            },
            'trading_performance': {
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'average_trade_pnl': avg_trade_pnl
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_position_size': self.max_position_size,
                'stop_loss_percentage': self.stop_loss_percentage,
                'take_profit_percentage': self.take_profit_percentage
            },
            'positions': portfolio.positions.copy(),
            'last_updated': datetime.now().isoformat()
        }
    
    def save_performance_data(self, filename: str = "paper_trading_performance.json"):
        """Save performance data to file"""
        try:
            data = {
                'initial_capital': self.initial_capital,
                'start_date': self.start_date.isoformat(),
                'performance_history': self.performance_history,
                'trades': [
                    {
                        'id': t.id,
                        'symbol': t.symbol,
                        'side': t.side,
                        'quantity': t.quantity,
                        'price': t.price,
                        'timestamp': t.timestamp.isoformat(),
                        'ai_confidence': t.ai_confidence,
                        'ai_reasoning': t.ai_reasoning,
                        'status': t.status,
                        'pnl': t.pnl
                    }
                    for t in self.trades
                ]
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"💾 Performance data saved to {filename}")
            
        except Exception as e:
            logger.error(f"Error saving performance data: {e}")
    
    def load_performance_data(self, filename: str = "paper_trading_performance.json"):
        """Load performance data from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                self.initial_capital = data.get('initial_capital', self.initial_capital)
                self.start_date = datetime.fromisoformat(data.get('start_date', self.start_date.isoformat()))
                self.performance_history = data.get('performance_history', [])
                
                # Reconstruct trades
                self.trades = []
                for trade_data in data.get('trades', []):
                    trade = Trade(
                        id=trade_data['id'],
                        symbol=trade_data['symbol'],
                        side=trade_data['side'],
                        quantity=trade_data['quantity'],
                        price=trade_data['price'],
                        timestamp=datetime.fromisoformat(trade_data['timestamp']),
                        ai_confidence=trade_data['ai_confidence'],
                        ai_reasoning=trade_data['ai_reasoning'],
                        status=trade_data['status'],
                        pnl=trade_data.get('pnl', 0.0)
                    )
                    self.trades.append(trade)
                
                # Reconstruct portfolio state
                self._reconstruct_portfolio()
                
                logger.info(f"📂 Performance data loaded from {filename}")
                
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
    
    def _reconstruct_portfolio(self):
        """Reconstruct portfolio state from trade history"""
        self.cash = self.initial_capital
        self.positions = {}
        
        for trade in self.trades:
            if trade.side == 'BUY':
                self.cash -= trade.quantity * trade.price
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) + trade.quantity
            elif trade.side == 'SELL':
                self.cash += trade.quantity * trade.price
                self.positions[trade.symbol] = self.positions.get(trade.symbol, 0) - trade.quantity
                if self.positions.get(trade.symbol, 0) <= 0:
                    del self.positions[trade.symbol]
        
        # Update counters
        self.total_trades = len(self.trades)
        self.trade_id_counter = max([int(t.id.split('_')[1]) for t in self.trades], default=0) + 1
        
        logger.info("🔄 Portfolio state reconstructed from trade history")
