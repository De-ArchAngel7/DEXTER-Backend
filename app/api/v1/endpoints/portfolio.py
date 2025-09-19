from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user
from app.models.user import User
from app.services.data_service import data_service
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/overview")
async def get_portfolio_overview(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get portfolio overview with real-time data from DexScreener
    """
    try:
        # Popular token addresses for demo portfolio
        demo_tokens = [
            "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            "0xA0b86a33E6441b8c4C8D8e4C8C8C8C8C8C8C8C8",  # USDC (placeholder)
            "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
            "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
        ]
        
        # Get real-time data from DexScreener
        portfolio_data = await data_service.get_portfolio_data(demo_tokens)
        
        if portfolio_data.get("success"):
            # Transform to match expected frontend format
            assets = []
            for token in portfolio_data.get("tokens", []):
                if token.get("success"):
                    assets.append({
                        "symbol": token.get("symbol", "Unknown"),
                        "value": token.get("price_usd", 0),
                        "change": token.get("price_change_24h", 0),
                        "color": _get_asset_color(token.get("symbol", "Unknown"))
                    })
            
            return {
                "total_value": portfolio_data.get("total_value", 0),
                "change_24h": portfolio_data.get("total_change_24h", 0),
                "change_percent": portfolio_data.get("change_percent", 0),
                "assets": assets,
                "performance": {
                    "1D": portfolio_data.get("change_percent", 0),
                    "1W": portfolio_data.get("change_percent", 0) * 7,  # Estimate
                    "1M": portfolio_data.get("change_percent", 0) * 30,  # Estimate
                    "3M": portfolio_data.get("change_percent", 0) * 90   # Estimate
                },
                "provider": portfolio_data.get("provider", "dexscreener"),
                "timestamp": portfolio_data.get("timestamp", 0)
            }
        else:
            # Fallback to mock data if DexScreener fails
            logger.warning("DexScreener failed, using mock data")
            return _get_mock_portfolio_data()
            
    except Exception as e:
        logger.error(f"Error getting portfolio overview: {str(e)}")
        # Return mock data as fallback
        return _get_mock_portfolio_data()

@router.get("/tokens/{token_address}")
async def get_token_data(
    token_address: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get specific token data from DexScreener
    """
    try:
        token_data = await data_service.get_token_data(token_address)
        return token_data
    except Exception as e:
        logger.error(f"Error getting token data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching token data: {str(e)}")

@router.get("/search")
async def search_tokens(
    query: str,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    Search for tokens using DexScreener
    """
    try:
        results = await data_service.search_tokens(query)
        return results
    except Exception as e:
        logger.error(f"Error searching tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching tokens: {str(e)}")

@router.get("/market-overview")
async def get_market_overview(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get market overview from DexScreener
    """
    try:
        market_data = await data_service.get_market_overview()
        return market_data
    except Exception as e:
        logger.error(f"Error getting market overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

def _get_mock_portfolio_data() -> Dict[str, Any]:
    """Fallback mock portfolio data"""
    return {
        "total_value": 125000.00,
        "change_24h": 2500.00,
        "change_percent": 2.04,
        "assets": [
            {"symbol": "WETH", "value": 45000.00, "change": 2.5, "color": "bg-blue-500"},
            {"symbol": "USDC", "value": 30000.00, "change": 0.0, "color": "bg-blue-300"},
            {"symbol": "USDT", "value": 25000.00, "change": 0.0, "color": "bg-green-500"},
            {"symbol": "WBTC", "value": 25000.00, "change": 1.8, "color": "bg-orange-500"}
        ],
        "performance": {
            "1D": 2.04,
            "1W": 8.5,
            "1M": 15.2,
            "3M": 28.7
        },
        "provider": "mock",
        "timestamp": 0
    }

def _get_asset_color(symbol: str) -> str:
    """Get color for asset symbol"""
    colors = {
        "WETH": "bg-blue-500",
        "ETH": "bg-blue-500",
        "USDC": "bg-blue-300",
        "USDT": "bg-green-500",
        "WBTC": "bg-orange-500",
        "BTC": "bg-orange-500",
        "BNB": "bg-yellow-500",
        "ADA": "bg-blue-600",
        "SOL": "bg-purple-500",
        "DOT": "bg-pink-500",
        "LINK": "bg-blue-400",
        "MATIC": "bg-purple-600"
    }
    return colors.get(symbol, "bg-gray-500")
