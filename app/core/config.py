from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Trading Bot"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    DATABASE_URL: str = "mongodb://localhost:27017/dexter"
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None
    SUPABASE_DB_PASSWORD: Optional[str] = None
    
    SECRET_KEY: str = "dev-secret-key-change-in-production-12345"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    TELEGRAM_BOT_TOKEN: str = "dev-token"
    
    REDIS_URL: str = "redis://localhost:6379"
    
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET: Optional[str] = None
    KUCOIN_API_KEY: Optional[str] = None
    KUCOIN_SECRET: Optional[str] = None
    COINBASE_API_KEY: Optional[str] = None
    COINBASE_SECRET: Optional[str] = None
    
    MODEL_CACHE_DIR: str = "./models"
    GPU_ENABLED: bool = True
    CUDA_VISIBLE_DEVICES: str = "0"
    
    ETHEREUM_RPC_URL: Optional[str] = None
    BSC_RPC_URL: Optional[str] = None
    POLYGON_RPC_URL: Optional[str] = None
    
    WALLET_CONNECT_PROJECT_ID: Optional[str] = None
    
    GRAFANA_PASSWORD: str = "admin123"
    
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    RATE_LIMIT_PER_MINUTE: int = 100
    
    MAX_DAILY_LOSS: float = 0.05
    DEFAULT_POSITION_SIZE: float = 0.1
    RISK_FREE_RATE: float = 0.02
    
    TWITTER_BEARER_TOKEN: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra environment variables

settings = Settings()
