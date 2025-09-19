import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class DexScreenerService:
    """Service for fetching token data from DexScreener API"""
    
    BASE_URL = "https://api.dexscreener.com/latest/dex"
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
                headers={
                    "User-Agent": "DEXTER-Trading-Bot/1.0",
                    "Accept": "application/json"
                }
            )
        return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_token_data(self, token_address: str) -> Dict[str, Any]:
        """
        Fetch token data from DexScreener
        
        Args:
            token_address: Token contract address
            
        Returns:
            Normalized token data
        """
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/tokens/{token_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._normalize_response(data, token_address)
                else:
                    logger.error(f"DexScreener API error: {response.status}")
                    return self._create_error_response(f"API error: {response.status}")
                    
        except Exception as e:
            logger.error(f"Error fetching DexScreener data: {str(e)}")
            return self._create_error_response(f"Network error: {str(e)}")
    
    async def get_token_pairs(self, token_address: str) -> List[Dict[str, Any]]:
        """
        Get all trading pairs for a token
        
        Args:
            token_address: Token contract address
            
        Returns:
            List of trading pairs
        """
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/tokens/{token_address}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._extract_pairs(data)
                else:
                    logger.error(f"DexScreener pairs API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching DexScreener pairs: {str(e)}")
            return []
    
    async def search_tokens(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for tokens by name or symbol
        
        Args:
            query: Search query
            
        Returns:
            List of matching tokens
        """
        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/search?q={query}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._extract_search_results(data)
                else:
                    logger.error(f"DexScreener search API error: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching DexScreener: {str(e)}")
            return []
    
    def _normalize_response(self, data: Dict[str, Any], token_address: str) -> Dict[str, Any]:
        """Normalize DexScreener response to match our format"""
        try:
            pairs = data.get("pairs", [])
            if not pairs:
                return self._create_error_response("No trading pairs found")
            
            # Get the most liquid pair (highest volume)
            main_pair = max(pairs, key=lambda x: float(x.get("volume", {}).get("h24", 0) or 0))
            
            return {
                "success": True,
                "token_address": token_address,
                "name": main_pair.get("baseToken", {}).get("name", "Unknown"),
                "symbol": main_pair.get("baseToken", {}).get("symbol", "Unknown"),
                "price_usd": float(main_pair.get("priceUsd", 0)),
                "price_change_24h": float(main_pair.get("priceChange", {}).get("h24", 0)),
                "volume_24h": float(main_pair.get("volume", {}).get("h24", 0)),
                "liquidity_usd": float(main_pair.get("liquidity", {}).get("usd", 0)),
                "market_cap": float(main_pair.get("marketCap", 0)),
                "dex_id": main_pair.get("dexId", "Unknown"),
                "pair_address": main_pair.get("pairAddress", ""),
                "chain_id": main_pair.get("chainId", ""),
                "exchange": main_pair.get("dexId", "Unknown"),
                "timestamp": main_pair.get("priceTimestamp", 0),
                "raw_data": main_pair  # Keep original data for debugging
            }
            
        except Exception as e:
            logger.error(f"Error normalizing DexScreener response: {str(e)}")
            return self._create_error_response(f"Data parsing error: {str(e)}")
    
    def _extract_pairs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and normalize trading pairs"""
        pairs = data.get("pairs", [])
        normalized_pairs = []
        
        for pair in pairs:
            try:
                normalized_pair = {
                    "dex_id": pair.get("dexId", "Unknown"),
                    "pair_address": pair.get("pairAddress", ""),
                    "base_token": {
                        "address": pair.get("baseToken", {}).get("address", ""),
                        "name": pair.get("baseToken", {}).get("name", "Unknown"),
                        "symbol": pair.get("baseToken", {}).get("symbol", "Unknown")
                    },
                    "quote_token": {
                        "address": pair.get("quoteToken", {}).get("address", ""),
                        "name": pair.get("quoteToken", {}).get("name", "Unknown"),
                        "symbol": pair.get("quoteToken", {}).get("symbol", "Unknown")
                    },
                    "price_usd": float(pair.get("priceUsd", 0)),
                    "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0)),
                    "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
                    "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                    "chain_id": pair.get("chainId", ""),
                    "exchange": pair.get("dexId", "Unknown")
                }
                normalized_pairs.append(normalized_pair)
            except Exception as e:
                logger.warning(f"Error normalizing pair: {str(e)}")
                continue
        
        return normalized_pairs
    
    def _extract_search_results(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search results"""
        pairs = data.get("pairs", [])
        results = []
        
        for pair in pairs[:10]:  # Limit to top 10 results
            try:
                result = {
                    "token_address": pair.get("baseToken", {}).get("address", ""),
                    "name": pair.get("baseToken", {}).get("name", "Unknown"),
                    "symbol": pair.get("baseToken", {}).get("symbol", "Unknown"),
                    "price_usd": float(pair.get("priceUsd", 0)),
                    "dex_id": pair.get("dexId", "Unknown"),
                    "chain_id": pair.get("chainId", ""),
                    "exchange": pair.get("dexId", "Unknown")
                }
                results.append(result)
            except Exception as e:
                logger.warning(f"Error extracting search result: {str(e)}")
                continue
        
        return results
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "token_address": "",
            "name": "",
            "symbol": "",
            "price_usd": 0.0,
            "price_change_24h": 0.0,
            "volume_24h": 0.0,
            "liquidity_usd": 0.0,
            "market_cap": 0.0,
            "dex_id": "",
            "pair_address": "",
            "chain_id": "",
            "exchange": "",
            "timestamp": 0,
            "raw_data": {}
        }

# Global instance
dexscreener_service = DexScreenerService()
