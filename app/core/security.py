from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Request, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import structlog
from app.core.config import settings

logger = structlog.get_logger()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

redis_client = redis.from_url(settings.REDIS_URL)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user from JWT token"""
    try:
        token = credentials.credentials
        payload = verify_token(token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # TODO: Fetch actual user from database
        # For now, return a mock user object
        from app.models.user import User
        mock_user = User(
            id=user_id,
            email=f"user_{user_id}@example.com",
            username=f"user_{user_id}",
            is_active=True,
            created_at=datetime.utcnow()
        )
        
        return mock_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Authentication error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def rate_limit_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Rate limiting middleware using Redis"""
    try:
        client_ip = request.client.host
        rate_limit_key = f"rate_limit:{client_ip}"
        
        # Get current request count
        current_count = redis_client.get(rate_limit_key)
        
        if current_count is None:
            # First request from this IP
            redis_client.setex(rate_limit_key, 60, 1)  # 1 minute window
        else:
            current_count = int(current_count)
            if current_count >= 100:  # 100 requests per minute
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
            redis_client.incr(rate_limit_key)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Rate limiting error", error=str(e))
        # Continue without rate limiting if Redis fails
    
    response = await call_next(request)
    return response

def encrypt_api_key(api_key: str) -> str:
    return pwd_context.hash(api_key)

def verify_api_key(api_key: str, hashed_key: str) -> bool:
    return pwd_context.verify(api_key, hashed_key)
