import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import structlog
from dataclasses import dataclass
from enum import Enum

logger = structlog.get_logger()

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"

@dataclass
class Order:
    """Represents a trading order"""
    id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    timestamp: datetime
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl

class ExchangeInterface:
    """Abstract interface for exchange connections"""
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.connected = False
        self.exchange_name = "Unknown"
        
    async def connect(self) -> bool:
        """Connect to exchange"""
        raise NotImplementedError
        
    async def disconnect(self) -> bool:
        """Disconnect from exchange"""
        raise NotImplementedError
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        raise NotImplementedError
        
    async def get_balance(self, asset: str) -> float:
        """Get balance for specific asset"""
        raise NotImplementedError
        
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information"""
        raise NotImplementedError
        
    async def place_order(self, order: Order) -> bool:
        """Place an order on the exchange"""
        raise NotImplementedError
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        raise NotImplementedError
        
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        raise NotImplementedError

class MockExchangeInterface(ExchangeInterface):
    """Mock exchange interface for testing live trading without real money"""
    
    def __init__(self, initial_balance: float = 10000.0):
        super().__init__()
        self.exchange_name = "Mock Exchange"
        self.balances = {
            "USDT": initial_balance,
            "BTC": 0.0,
            "ETH": 0.0,
            "ADA": 0.0,
            "DOT": 0.0
        }
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_id_counter = 1
        self.connected = True
        
        # Simulated market prices
        self.market_prices = {
            "BTCUSDT": 50000.0,
            "ETHUSDT": 3000.0,
            "ADAUSDT": 0.5,
            "DOTUSDT": 7.0,
            "LINKUSDT": 15.0
        }
        
        # Price volatility for realistic simulation
        self.price_volatility = 0.02  # 2% volatility
        
    async def connect(self) -> bool:
        """Connect to mock exchange"""
        self.connected = True
        logger.info("🔌 Connected to Mock Exchange")
        return True
        
    async def disconnect(self) -> bool:
        """Disconnect from mock exchange"""
        self.connected = False
        logger.info("🔌 Disconnected from Mock Exchange")
        return True
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get mock account information"""
        total_value = self.balances["USDT"]
        for symbol, position in self.positions.items():
            if position.quantity > 0:
                total_value += position.quantity * self.market_prices.get(symbol, 0)
        
        return {
            "exchange": self.exchange_name,
            "total_balance": total_value,
            "balances": self.balances.copy(),
            "positions": len(self.positions),
            "connected": self.connected
        }
        
    async def get_balance(self, asset: str) -> float:
        """Get balance for specific asset"""
        return self.balances.get(asset, 0.0)
        
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get mock ticker information with realistic price movements"""
        base_price = self.market_prices.get(symbol, 100.0)
        
        # Simulate price movement
        price_change = np.random.normal(0, self.price_volatility)
        new_price = base_price * (1 + price_change)
        
        # Update market price
        self.market_prices[symbol] = new_price
        
        return {
            "symbol": symbol,
            "price": new_price,
            "change": price_change,
            "change_percent": price_change * 100,
            "volume": np.random.uniform(1000000, 10000000),
            "timestamp": datetime.now().isoformat()
        }
        
    async def place_order(self, order: Order) -> bool:
        """Place a mock order"""
        if not self.connected:
            return False
            
        # Generate order ID
        order.id = f"ORDER_{self.order_id_counter:06d}"
        self.order_id_counter += 1
        
        # Simulate order execution with delay
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Check if we have sufficient balance
        if order.side == "BUY":
            required_usdt = order.quantity * (order.price or self.market_prices.get(order.symbol, 0))
            if self.balances["USDT"] < required_usdt:
                order.status = OrderStatus.REJECTED
                logger.warning(f"❌ Insufficient USDT balance for {order.symbol} BUY order")
                return False
                
            # Execute BUY order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = order.price or self.market_prices.get(order.symbol, 0)
            
            # Update balances
            self.balances["USDT"] -= required_usdt
            asset = order.symbol.replace("USDT", "")
            self.balances[asset] = self.balances.get(asset, 0) + order.quantity
            
            # Update or create position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.quantity
                total_cost = (pos.quantity * pos.entry_price) + (order.quantity * order.average_fill_price)
                pos.quantity = total_quantity
                pos.entry_price = total_cost / total_quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    entry_price=order.average_fill_price,
                    current_price=order.average_fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    timestamp=datetime.now()
                )
                
        elif order.side == "SELL":
            asset = order.symbol.replace("USDT", "")
            if self.balances.get(asset, 0) < order.quantity:
                order.status = OrderStatus.REJECTED
                logger.warning(f"❌ Insufficient {asset} balance for {order.symbol} SELL order")
                return False
                
            # Execute SELL order
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.average_fill_price = order.price or self.market_prices.get(order.symbol, 0)
            
            # Update balances
            self.balances[asset] -= order.quantity
            self.balances["USDT"] += order.quantity * order.average_fill_price
            
            # Update position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                if pos.quantity <= order.quantity:
                    # Close position
                    realized_pnl = (order.average_fill_price - pos.entry_price) * pos.quantity
                    pos.realized_pnl += realized_pnl
                    pos.quantity = 0
                    del self.positions[order.symbol]
                else:
                    # Partial close
                    realized_pnl = (order.average_fill_price - pos.entry_price) * order.quantity
                    pos.realized_pnl += realized_pnl
                    pos.quantity -= order.quantity
        
        # Store order
        self.orders[order.id] = order
        
        logger.info(f"✅ {order.side} order executed for {order.quantity} {order.symbol} at ${order.average_fill_price:,.2f}")
        return True
        
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a mock order"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            logger.info(f"❌ Order {order_id} cancelled")
            return True
        return False
        
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get mock order status"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.REJECTED

class LiveTradingEngine:
    """Live trading engine that executes trades based on DEXTER's AI predictions"""
    
    def __init__(self, exchange: ExchangeInterface, initial_capital: float = 10000.0):
        self.exchange = exchange
        self.initial_capital = initial_capital
        self.risk_manager = RiskManager()
        self.position_manager = PositionManager()
        self.performance_tracker = PerformanceTracker(initial_capital)
        
        # Trading parameters
        self.max_position_size = 0.1  # Max 10% of portfolio in single position
        self.max_daily_trades = 50
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        logger.info(f"🚀 Live Trading Engine initialized with ${initial_capital:,.2f}")
        
    async def start(self):
        """Start the live trading engine"""
        try:
            # Connect to exchange
            connected = await self.exchange.connect()
            if not connected:
                raise Exception("Failed to connect to exchange")
                
            logger.info("✅ Live Trading Engine started successfully")
            
            # Reset daily counters
            self._reset_daily_counters()
            
        except Exception as e:
            logger.error(f"❌ Failed to start Live Trading Engine: {e}")
            raise
            
    async def stop(self):
        """Stop the live trading engine"""
        try:
            await self.exchange.disconnect()
            logger.info("✅ Live Trading Engine stopped successfully")
        except Exception as e:
            logger.error(f"❌ Error stopping Live Trading Engine: {e}")
            
    async def execute_ai_trade(self, symbol: str, ai_recommendation: str, 
                              ai_confidence: float, ai_reasoning: str) -> Optional[Order]:
        """Execute a trade based on AI recommendation"""
        
        try:
            # Check daily limits
            if not self._can_execute_trade():
                logger.warning(f"⚠️  Daily trade limit reached for {symbol}")
                return None
                
            # Get current market data
            ticker = await self.exchange.get_ticker(symbol)
            current_price = ticker["price"]
            
            # Get account balance
            available_balance = await self.exchange.get_balance("USDT")
            
            # Calculate position size based on AI confidence and risk management
            position_size = self.risk_manager.calculate_position_size(
                ai_confidence, available_balance, self.max_position_size
            )
            
            # Determine trade side
            if ai_recommendation in ["BUY", "STRONG_BUY"]:
                side = "BUY"
                quantity = position_size / current_price
            elif ai_recommendation in ["SELL", "STRONG_SELL"]:
                side = "SELL"
                # Check if we have the asset to sell
                asset = symbol.replace("USDT", "")
                asset_balance = await self.exchange.get_balance(asset)
                quantity = min(position_size / current_price, asset_balance)
            else:
                logger.info(f"⏸️  AI recommends HOLD for {symbol}")
                return None
                
            # Create and execute order
            order = Order(
                id="",
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=current_price
            )
            
            success = await self.exchange.place_order(order)
            if success:
                # Update daily counters
                self._increment_daily_trade_count()
                
                # Log trade execution
                logger.info(f"✅ AI Trade Executed: {side} {quantity:.6f} {symbol} at ${current_price:,.2f}")
                logger.info(f"   AI Confidence: {ai_confidence:.1f}%")
                logger.info(f"   AI Reasoning: {ai_reasoning}")
                
                return order
            else:
                logger.error(f"❌ Failed to execute trade for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error executing AI trade for {symbol}: {e}")
            return None
            
    def _can_execute_trade(self) -> bool:
        """Check if we can execute another trade today"""
        today = datetime.now().date()
        
        # Reset counters if it's a new day
        if self.last_trade_date != today:
            self._reset_daily_counters()
            
        return self.daily_trade_count < self.max_daily_trades
        
    def _reset_daily_counters(self):
        """Reset daily trading counters"""
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
    def _increment_daily_trade_count(self):
        """Increment daily trade counter"""
        self.daily_trade_count += 1

class RiskManager:
    """Manages trading risk and position sizing"""
    
    def __init__(self):
        self.max_risk_per_trade = 0.02  # Max 2% risk per trade
        self.max_portfolio_risk = 0.10   # Max 10% total portfolio risk
        
    def calculate_position_size(self, ai_confidence: float, available_balance: float, 
                               max_position_size: float) -> float:
        """Calculate optimal position size based on AI confidence and risk parameters"""
        
        # Base position size on AI confidence
        confidence_multiplier = ai_confidence / 100.0
        
        # Apply risk management
        risk_adjusted_size = available_balance * self.max_risk_per_trade * confidence_multiplier
        
        # Apply maximum position size limit
        max_size = available_balance * max_position_size
        
        # Return the smaller of the two
        return min(risk_adjusted_size, max_size)

class PositionManager:
    """Manages open positions and risk"""
    
    def __init__(self):
        self.positions: Dict[str, Position] = {}
        
    def add_position(self, position: Position):
        """Add a new position"""
        self.positions[position.symbol] = position
        
    def update_position(self, symbol: str, current_price: float):
        """Update position with current price"""
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos.current_price = current_price
            pos.unrealized_pnl = (current_price - pos.entry_price) * pos.quantity
            
    def close_position(self, symbol: str):
        """Close a position"""
        if symbol in self.positions:
            del self.positions[symbol]

class PerformanceTracker:
    """Tracks trading performance and metrics"""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.trades: List[Order] = []
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_trade(self, trade: Order):
        """Add a completed trade"""
        self.trades.append(trade)
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.status == OrderStatus.FILLED])
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": (winning_trades / max(total_trades, 1)) * 100,
            "initial_capital": self.initial_capital
        }
