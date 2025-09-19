from typing import Dict, List, Any, Optional
from enum import Enum
import logging
from app.services.dexscreener_service import dexscreener_service
from app.core.config import settings

logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Available data providers"""
    DEXSCREENER = "dexscreener"
    BINANCE = "binance"
    KUCOIN = "kucoin"

class DataService:
    """Unified data service that can switch between providers"""
    
    def __init__(self, default_provider: str = "dexscreener"):
        self.default_provider = default_provider
        self.current_provider = DataProvider(default_provider)
        logger.info(f"DataService initialized with provider: {self.current_provider.value}")
    
    def set_provider(self, provider: str) -> bool:
        """Switch data provider"""
        try:
            self.current_provider = DataProvider(provider)
            logger.info(f"Switched to provider: {provider}")
            return True
        except ValueError:
            logger.error(f"Invalid provider: {provider}")
            return False
    
    def get_current_provider(self) -> str:
        """Get current provider name"""
        return self.current_provider.value
    
    async def get_token_data(self, token_address: str, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get token data from specified or current provider
        
        Args:
            token_address: Token contract address
            provider: Optional provider override
            
        Returns:
            Token data from selected provider
        """
        if provider:
            self.set_provider(provider)
        
        try:
            if self.current_provider == DataProvider.DEXSCREENER:
                return await dexscreener_service.get_token_data(token_address)
            elif self.current_provider == DataProvider.BINANCE:
                logger.warning("Binance provider not yet implemented")
                return self._create_fallback_response("Binance provider not yet implemented")
            elif self.current_provider == DataProvider.KUCOIN:
                logger.warning("KuCoin provider not yet implemented")
                return self._create_fallback_response("KuCoin provider not yet implemented")
            else:
                logger.error(f"Unknown provider: {self.current_provider}")
                return self._create_fallback_response(f"Unknown provider: {self.current_provider}")
                
        except Exception as e:
            logger.error(f"Error getting token data: {str(e)}")
            return self._create_fallback_response(f"Service error: {str(e)}")
    
    async def get_token_pairs(self, token_address: str, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all trading pairs for a token"""
        if provider:
            self.set_provider(provider)
        
        try:
            if self.current_provider == DataProvider.DEXSCREENER:
                return await dexscreener_service.get_token_pairs(token_address)
            elif self.current_provider == DataProvider.BINANCE:
                logger.warning("Binance pairs not yet implemented")
                return []
            elif self.current_provider == DataProvider.KUCOIN:
                logger.warning("KuCoin pairs not yet implemented")
                return []
            else:
                logger.error(f"Unknown provider: {self.current_provider}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting token pairs: {str(e)}")
            return []
    
    async def search_tokens(self, query: str, provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for tokens"""
        if provider:
            self.set_provider(provider)
        
        try:
            if self.current_provider == DataProvider.DEXSCREENER:
                return await dexscreener_service.search_tokens(query)
            elif self.current_provider == DataProvider.BINANCE:
                logger.warning("Binance search not yet implemented")
                return []
            elif self.current_provider == DataProvider.KUCOIN:
                logger.warning("KuCoin search not yet implemented")
                return []
            else:
                logger.error(f"Unknown provider: {self.current_provider}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching tokens: {str(e)}")
            return []
    
    async def get_portfolio_data(self, token_addresses: List[str], provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get portfolio data for multiple tokens
        
        Args:
            token_addresses: List of token addresses
            provider: Optional provider override
            
        Returns:
            Portfolio summary with all token data
        """
        if provider:
            self.set_provider(provider)
        
        try:
            portfolio_data = []
            total_value = 0.0
            total_change_24h = 0.0
            
            for token_address in token_addresses:
                token_data = await self.get_token_data(token_address)
                if token_data.get("success"):
                    portfolio_data.append(token_data)
                    total_value += token_data.get("price_usd", 0)
                    total_change_24h += token_data.get("price_change_24h", 0)
            
            return {
                "success": True,
                "provider": self.current_provider.value,
                "total_value": total_value,
                "total_change_24h": total_change_24h,
                "change_percent": (total_change_24h / total_value * 100) if total_value > 0 else 0,
                "tokens": portfolio_data,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio data: {str(e)}")
            return self._create_fallback_response(f"Portfolio error: {str(e)}")
    
    async def get_market_overview(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Get market overview data
        
        Args:
            provider: Optional provider override
            
        Returns:
            Market overview data
        """
        if provider:
            self.set_provider(provider)
        
        try:
            # Popular token addresses for market overview
            popular_tokens = [
                "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
                "0xA0b86a33E6441b8c4C8D8e4C8C8C8C8C8C8C8C8",  # USDC
                "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
                "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
            ]
            
            market_data = []
            for token_address in popular_tokens:
                token_data = await self.get_token_data(token_address)
                if token_data.get("success"):
                    market_data.append({
                        "symbol": token_data.get("symbol", "Unknown"),
                        "price": token_data.get("price_usd", 0),
                        "change_24h": token_data.get("price_change_24h", 0),
                        "volume": token_data.get("volume_24h", 0)
                    })
            
            return {
                "success": True,
                "provider": self.current_provider.value,
                "market_data": market_data,
                "timestamp": int(time.time())
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {str(e)}")
            return self._create_fallback_response(f"Market overview error: {str(e)}")
    
    def _create_fallback_response(self, error_message: str) -> Dict[str, Any]:
        """Create fallback response when provider fails"""
        return {
            "success": False,
            "error": error_message,
            "provider": self.current_provider.value,
            "fallback": True
        }
    
    async def close(self):
        """Close all provider connections"""
        try:
            if self.current_provider == DataProvider.DEXSCREENER:
                await dexscreener_service.close()
            # TODO: Close other providers when implemented
            logger.info("DataService connections closed")
        except Exception as e:
            logger.error(f"Error closing DataService: {str(e)}")

# Global instance with DexScreener as default
data_service = DataService(default_provider="dexscreener")

# Import time for timestamp
import time
