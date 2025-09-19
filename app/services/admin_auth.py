import os
import httpx
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

logger = structlog.get_logger(__name__)

class AdminAuthService:
    def __init__(self):
        self.google_client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
        self.google_client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
        self.google_redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
        self.admin_emails = [email.strip() for email in os.getenv("ADMIN_EMAILS", "").split(",") if email.strip()]
        
        if not all([self.google_client_id, self.google_client_secret, self.google_redirect_uri]):
            logger.warning("Google OAuth credentials not configured - admin panel disabled")
        else:
            logger.info(f"Admin panel configured for {len(self.admin_emails)} admin emails")
    
    async def verify_google_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Google OAuth token and return user info"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://www.googleapis.com/oauth2/v1/userinfo?access_token={token}"
                )
                if response.status_code == 200:
                    user_info = response.json()
                    if user_info.get("email") in self.admin_emails:
                        return user_info
                    else:
                        logger.warning(f"Unauthorized admin access attempt: {user_info.get('email')}")
                        return None
                else:
                    logger.error(f"Google token verification failed: {response.status_code}")
                    return None
        except Exception as e:
            logger.error(f"Error verifying Google token: {e}")
            return None
    
    def is_admin_email(self, email: str) -> bool:
        """Check if email is in admin list"""
        return email in self.admin_emails

# Dependency for admin authentication
security = HTTPBearer()

async def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency to get authenticated admin user"""
    admin_auth = AdminAuthService()
    user_info = await admin_auth.verify_google_token(credentials.credentials)
    if not user_info:
        raise HTTPException(status_code=401, detail="Invalid or unauthorized admin token")
    return user_info
