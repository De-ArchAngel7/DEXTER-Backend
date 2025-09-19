from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum
from decimal import Decimal

class ExchangeType(str, Enum):
    BINANCE = "binance"
    KUCOIN = "kucoin"
    COINBASE = "coinbase"
    UNISWAP = "uniswap"
    PANCAKESWAP = "pancakeswap"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(str, Enum):
    LONG = "long"
    SHORT = "short"

class StrategyType(str, Enum):
    MANUAL = "manual"
    GRID = "grid"
    DCA = "dca"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    AI_SIGNAL = "ai_signal"

class Asset(BaseModel):
    symbol: str
    name: str
    quantity: Decimal
    current_price: Decimal
    total_value: Decimal
    allocation_percentage: float
    change_24h: float
    change_7d: float
    change_30d: float
    last_updated: datetime

class Order(BaseModel):
    id: str
    user_id: str
    exchange: ExchangeType
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: OrderStatus
    strategy_type: StrategyType = StrategyType.MANUAL
    created_at: datetime
    updated_at: datetime
    filled_at: Optional[datetime] = None
    filled_quantity: Optional[Decimal] = None
    filled_price: Optional[Decimal] = None
    commission: Optional[Decimal] = None
    exchange_order_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class Position(BaseModel):
    id: str
    user_id: str
    exchange: ExchangeType
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    total_pnl: Decimal
    entry_time: datetime
    last_updated: datetime
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = {}

class Portfolio(BaseModel):
    user_id: str
    total_balance: Decimal
    available_balance: Decimal
    total_pnl: Decimal
    daily_pnl: Decimal
    weekly_pnl: Decimal
    monthly_pnl: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: Decimal
    average_loss: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Optional[float] = None
    last_updated: datetime

class ExchangeCredentials(BaseModel):
    user_id: str
    exchange: ExchangeType
    api_key: str
    api_secret: str
    passphrase: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    last_used: Optional[datetime] = None

class TradingStrategy(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    performance_metrics: Dict[str, Any] = {}

class Trade(BaseModel):
    id: str
    user_id: str
    order_id: str
    exchange: ExchangeType
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    timestamp: datetime
    strategy_type: StrategyType
    metadata: Dict[str, Any] = {}

class RiskMetrics(BaseModel):
    user_id: str
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    max_drawdown: Decimal
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    last_calculated: datetime
