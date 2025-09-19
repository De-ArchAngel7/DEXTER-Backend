from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
from decimal import Decimal

class ModelType(str, Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRU = "gru"
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"

class PredictionType(str, Enum):
    PRICE = "price"
    VOLATILITY = "volatility"
    TREND = "trend"
    SUPPORT_RESISTANCE = "support_resistance"
    BREAKOUT = "breakout"

class SentimentSource(str, Enum):
    TWITTER = "twitter"
    TELEGRAM = "telegram"
    REDDIT = "reddit"
    NEWS = "news"
    FORUMS = "forums"

class AnomalyType(str, Enum):
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    PATTERN_BREAK = "pattern_break"
    CORRELATION_BREAK = "correlation_break"
    MANIPULATION = "manipulation"

class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class PricePrediction(BaseModel):
    id: str
    symbol: str
    model_type: ModelType
    prediction_type: PredictionType
    predicted_value: Decimal
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel
    prediction_horizon: int  # minutes
    created_at: datetime
    target_time: datetime
    actual_value: Optional[Decimal] = None
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = {}

class SentimentAnalysis(BaseModel):
    id: str
    symbol: str
    source: SentimentSource
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    text_content: str
    processed_at: datetime
    source_id: Optional[str] = None
    author: Optional[str] = None
    engagement_metrics: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class AnomalyDetection(BaseModel):
    id: str
    symbol: str
    anomaly_type: AnomalyType
    severity: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_at: datetime
    description: str
    affected_metrics: List[str]
    historical_context: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class MarketSignal(BaseModel):
    id: str
    symbol: str
    signal_type: str
    direction: str  # buy, sell, hold
    strength: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    generated_at: datetime
    expires_at: datetime
    source_models: List[str]
    supporting_evidence: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}

class ModelPerformance(BaseModel):
    model_id: str
    model_type: ModelType
    symbol: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    last_updated: datetime
    training_samples: int
    validation_samples: int
    metadata: Dict[str, Any] = {}

class ReinforcementLearningState(BaseModel):
    state_id: str
    user_id: str
    portfolio_state: Dict[str, Any]
    market_state: Dict[str, Any]
    action_taken: str
    reward: float
    next_state: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any] = {}

class ModelTrainingJob(BaseModel):
    job_id: str
    model_type: ModelType
    symbol: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(0.0, ge=0.0, le=1.0)
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, Any] = {}
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}

class FeatureVector(BaseModel):
    id: str
    symbol: str
    timestamp: datetime
    features: Dict[str, Union[float, int, str]]
    target_value: Optional[float] = None
    metadata: Dict[str, Any] = {}

class Embedding(BaseModel):
    id: str
    content_type: str  # text, pattern, trade
    content_id: str
    embedding_vector: List[float]
    created_at: datetime
    metadata: Dict[str, Any] = {}
