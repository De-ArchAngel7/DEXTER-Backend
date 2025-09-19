from fastapi import APIRouter
from app.api.v1.endpoints import auth, users, trading, ai, portfolio, dexscreener, chat, learning, advanced_learning, admin

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
api_router.include_router(ai.router, prefix="/ai", tags=["ai"])
api_router.include_router(portfolio.router, prefix="/portfolio", tags=["portfolio"])
api_router.include_router(dexscreener.router, prefix="/dexscreener", tags=["dexscreener"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(learning.router, prefix="/learning", tags=["learning"])
api_router.include_router(advanced_learning.router, prefix="/advanced-learning", tags=["advanced-learning"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
