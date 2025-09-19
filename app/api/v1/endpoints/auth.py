from fastapi import APIRouter, HTTPException, status
from datetime import timedelta

from app.core.security import (
    verify_password, 
    get_password_hash, 
    create_access_token,
    verify_token
)
from app.core.config import settings
from app.models.user import UserCreate, UserLogin, TokenResponse

router = APIRouter()

# Mock user database for development
# TODO: Replace with real MongoDB integration
mock_users_db = {
    "test@dexter.com": {
        "id": "user_001",
        "email": "test@dexter.com",
        "username": "test_user",
        "hashed_password": "$2b$12$NomOYMaxpC4.cEA1DGKSCegztBgLKDyLfwKLfvj/Jn7mGO2/2tHnS",  # "password123"
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z"
    }
}

@router.post("/login", response_model=TokenResponse)
async def login(user_credentials: UserLogin):
    """User login endpoint"""
    try:
        # Check if user exists
        if user_credentials.email not in mock_users_db:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        user_data = mock_users_db[user_credentials.email]
        
        # Verify password
        if not verify_password(user_credentials.password, user_data["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if user is active
        if not user_data["is_active"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is deactivated"
            )
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data["id"]}, 
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_data["id"],
            email=user_data["email"],
            username=user_data["username"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.post("/register", response_model=TokenResponse)
async def register(user_data: UserCreate):
    """User registration endpoint"""
    try:
        # Check if user already exists
        if user_data.email in mock_users_db:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user_id = f"user_{len(mock_users_db) + 1:03d}"
        hashed_password = get_password_hash(user_data.password)
        
        new_user = {
            "id": user_id,
            "email": user_data.email,
            "username": user_data.username,
            "hashed_password": hashed_password,
            "is_active": True,
            "created_at": "2024-01-01T00:00:00Z"
        }
        
        # Add to mock database
        mock_users_db[user_data.email] = new_user
        
        # Create access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id}, 
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_id,
            email=user_data.email,
            username=user_data.username
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        # Verify refresh token
        payload = verify_token(refresh_token)
        if payload is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        # Find user
        user_data = None
        for user in mock_users_db.values():
            if user["id"] == user_id:
                user_data = user
                break
        
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Create new access token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_id}, 
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user_id,
            email=user_data["email"],
            username=user_data["username"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token refresh failed: {str(e)}"
        )

@router.post("/logout")
async def logout():
    """User logout endpoint"""
    # In a real implementation, you might want to blacklist the token
    return {"message": "Successfully logged out"}

@router.get("/me")
async def get_current_user_info():
    """Get current user information (placeholder for now)"""
    # TODO: Implement proper user authentication
    return {
        "id": "demo_user",
        "email": "demo@dexter.com",
        "username": "demo_user",
        "is_active": True,
        "created_at": "2024-01-01T00:00:00Z"
    }

@router.post("/demo-login")
async def demo_login():
    """Demo login for testing purposes"""
    try:
        # Create demo user token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": "demo_user"}, 
            expires_delta=access_token_expires
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id="demo_user",
            email="demo@dexter.com",
            username="demo_user"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Demo login failed: {str(e)}"
        )
