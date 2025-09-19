"""
Development Configuration
Default values for development environment
"""

# Development settings with safe defaults
DEV_SETTINGS = {
    "DATABASE_URL": "mongodb://localhost:27017/dexter",
    "SECRET_KEY": "dev-secret-key-change-in-production-12345",
    "JWT_ALGORITHM": "HS256",
    "ACCESS_TOKEN_EXPIRE_MINUTES": 30,
    "TELEGRAM_BOT_TOKEN": "dev-token",
    "REDIS_URL": "redis://localhost:6379",
    "MODEL_CACHE_DIR": "./models",
    "GPU_ENABLED": False,
    "CUDA_VISIBLE_DEVICES": "0",
    "GRAFANA_PASSWORD": "admin123",
    "CORS_ORIGINS": ["http://localhost:3000"],
    "RATE_LIMIT_PER_MINUTE": 100,
    "MAX_DAILY_LOSS": 0.05,
    "DEFAULT_POSITION_SIZE": 0.1,
    "RISK_FREE_RATE": 0.02,
}

def get_dev_setting(key: str, default=None):
    """Get development setting with fallback to environment variable"""
    import os
    return os.getenv(key, DEV_SETTINGS.get(key, default))
