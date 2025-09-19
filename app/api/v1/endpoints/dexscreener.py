from fastapi import APIRouter, Depends, HTTPException, Query
from app.core.security import get_current_user
from app.models.user import User
from app.services.data_service import data_service
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/token/{token_address}")
async def get_token_info(
    token_address: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed token information from DexScreener
    
    Args:
        token_address: Token contract address (e.g., 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2 for WETH)
    """
    try:
        token_data = await data_service.get_token_data(token_address)
        if token_data.get("success"):
            return {
                "success": True,
                "data": token_data,
                "message": f"Token data retrieved successfully from {token_data.get('provider', 'unknown')}"
            }
        else:
            return {
                "success": False,
                "error": token_data.get("error", "Unknown error"),
                "provider": token_data.get("provider", "unknown")
            }
    except Exception as e:
        logger.error(f"Error getting token info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching token data: {str(e)}")

@router.get("/search")
async def search_tokens(
    query: str = Query(..., description="Search query (token name, symbol, or address)"),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search for tokens using DexScreener
    
    Args:
        query: Search query (e.g., "Ethereum", "ETH", "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2")
    """
    try:
        results = await data_service.search_tokens(query)
        return {
            "success": True,
            "query": query,
            "results": results,
            "count": len(results),
            "provider": "dexscreener"
        }
    except Exception as e:
        logger.error(f"Error searching tokens: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error searching tokens: {str(e)}")

@router.get("/pairs/{token_address}")
async def get_token_pairs(
    token_address: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get all trading pairs for a specific token
    
    Args:
        token_address: Token contract address
    """
    try:
        pairs = await data_service.get_token_pairs(token_address)
        return {
            "success": True,
            "token_address": token_address,
            "pairs": pairs,
            "count": len(pairs),
            "provider": "dexscreener"
        }
    except Exception as e:
        logger.error(f"Error getting token pairs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching token pairs: {str(e)}")

@router.get("/market-overview")
async def get_market_overview(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get market overview with popular tokens
    """
    try:
        market_data = await data_service.get_market_overview()
        if market_data.get("success"):
            return {
                "success": True,
                "data": market_data,
                "provider": market_data.get("provider", "dexscreener")
            }
        else:
            return {
                "success": False,
                "error": market_data.get("error", "Unknown error"),
                "provider": market_data.get("provider", "dexscreener")
            }
    except Exception as e:
        logger.error(f"Error getting market overview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching market data: {str(e)}")

@router.get("/provider/status")
async def get_provider_status(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current data provider status and information
    """
    try:
        current_provider = data_service.get_current_provider()
        return {
            "success": True,
            "current_provider": current_provider,
            "available_providers": ["dexscreener", "binance", "kucoin"],
            "default_provider": "dexscreener",
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting provider status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting provider status: {str(e)}")

@router.post("/provider/switch")
async def switch_provider(
    provider: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Switch data provider
    
    Args:
        provider: New provider name (dexscreener, binance, kucoin)
    """
    try:
        success = data_service.set_provider(provider)
        if success:
            return {
                "success": True,
                "message": f"Switched to provider: {provider}",
                "current_provider": provider
            }
        else:
            return {
                "success": False,
                "error": f"Invalid provider: {provider}",
                "available_providers": ["dexscreener", "binance", "kucoin"]
            }
    except Exception as e:
        logger.error(f"Error switching provider: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error switching provider: {str(e)}")

@router.get("/popular-tokens")
async def get_popular_tokens(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get list of popular token addresses for easy access
    """
    popular_tokens = {
        "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "USDC": "0xA0b86a33E6441b8c4C8D8e4C8C8C8C8C8C8C8C8",  # Placeholder
        "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
        "CRV": "0xD533a949740bb3306d119CC777fa900bA034cd52"
    }
    
    return {
        "success": True,
        "tokens": popular_tokens,
        "description": "Popular ERC-20 token addresses for easy access",
        "note": "USDC address is a placeholder - use actual address in production"
    }
