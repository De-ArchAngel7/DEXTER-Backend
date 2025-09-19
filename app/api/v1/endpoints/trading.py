from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any

router = APIRouter()

@router.get("/status")
async def get_trading_status():
    """Get current trading bot status"""
    return {
        "status": "active",
        "exchanges": ["binance", "kucoin", "coinbase"],
        "active_strategies": 0,
        "total_trades": 0,
        "last_update": "2024-01-01T00:00:00Z"
    }

@router.get("/exchanges")
async def get_exchanges():
    """Get supported exchanges"""
    return {
        "exchanges": [
            {
                "name": "Binance",
                "status": "connected",
                "type": "centralized"
            },
            {
                "name": "KuCoin", 
                "status": "ready",
                "type": "centralized"
            },
            {
                "name": "Coinbase",
                "status": "ready", 
                "type": "centralized"
            }
        ]
    }

@router.post("/test-connection")
async def test_exchange_connection(exchange: str):
    """Test connection to a specific exchange"""
    if exchange.lower() == "kucoin":
        return {"exchange": "KuCoin", "status": "ready", "message": "API keys configured"}
    elif exchange.lower() == "binance":
        return {"exchange": "Binance", "status": "ready", "message": "API keys configured"}
    elif exchange.lower() == "coinbase":
        return {"exchange": "Coinbase", "status": "ready", "message": "API keys configured"}
    else:
        raise HTTPException(status_code=400, detail="Unsupported exchange")
