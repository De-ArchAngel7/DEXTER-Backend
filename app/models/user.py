from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
from enum import Enum
import re

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class UserRole(str, Enum):
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"

class UserBase(BaseModel):
    email: str = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    
    @validator('email')
    def validate_email(cls, v):
        if not v:
            raise ValueError('Email cannot be empty')
        # Basic email regex validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=100)
    confirm_password: str

class UserLogin(BaseModel):
    email: str = Field(..., description="User email address")
    password: str
    
    @validator('email')
    def validate_email(cls, v):
        if not v:
            raise ValueError('Email cannot be empty')
        # Basic email regex validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()

class UserUpdate(BaseModel):
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    first_name: Optional[str] = Field(None, max_length=50)
    last_name: Optional[str] = Field(None, max_length=50)
    avatar_url: Optional[str] = None

class UserSettings(BaseModel):
    theme: str = "dark"
    notifications_enabled: bool = True
    trading_alerts: bool = True
    risk_tolerance: str = "medium"
    preferred_exchanges: List[str] = ["binance", "kucoin"]
    api_keys: dict = {}

class User(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    is_active: bool = True
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    settings: UserSettings = UserSettings()

    class Config:
        from_attributes = True

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    avatar_url: Optional[str] = None
    role: UserRole
    status: UserStatus
    created_at: datetime
    last_login: Optional[datetime] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    user_id: str
    email: str
    username: str

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    confirm_new_password: str

class APIKeyCreate(BaseModel):
    exchange: str = Field(..., description="Exchange name (e.g., binance, kucoin)")
    api_key: str = Field(..., description="API key from exchange")
    secret_key: str = Field(..., description="Secret key from exchange")
    passphrase: Optional[str] = Field(None, description="Passphrase for exchanges that require it")
    is_testnet: bool = False
    permissions: List[str] = Field(default=["read", "trade"], description="API key permissions")

class APIKeyResponse(BaseModel):
    id: str
    exchange: str
    is_testnet: bool
    permissions: List[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True

class UserProfile(BaseModel):
    user: UserResponse
    portfolio_summary: dict = {}
    trading_stats: dict = {}
    preferences: dict = {}
