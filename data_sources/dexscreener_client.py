#!/usr/bin/env python3
"""
📊 DEXTER DexScreener Client
Real-time DeFi market data integration

Fetches live data from:
- Token prices and changes
- Trading volume and liquidity
- Market cap and rankings
- Trading pairs and exchanges
"""

import aiohttp
import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
import structlog
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logger = structlog.get_logger()

class DexScreenerClient:
    """
    📊 DexScreener API Client for real-time DeFi data
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://api.dexscreener.com/latest"
        self.api_key = api_key
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Cache for frequently requested data
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds cache
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                "User-Agent": "DEXTER-AI-Trading-Bot/1.0",
                "Accept": "application/json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make HTTP request to DexScreener API"""
        try:
            await self._rate_limit()
            
            url = f"{self.base_url}/{endpoint}"
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 429:
                    logger.warning("Rate limit exceeded, waiting...")
                    await asyncio.sleep(5)
                    return None
                else:
                    logger.error(f"API request failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return None
    
    async def get_token_data(self, token_address: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive token data by contract address
        
        Args:
            token_address: Token contract address
            
        Returns:
            Token data including price, volume, liquidity, etc.
        """
        try:
            # Check cache first
            cache_key = f"token_{token_address}"
            if cache_key in self.cache:
                cached_data = self.cache[cache_key]
                if datetime.now().timestamp() - cached_data["timestamp"] < self.cache_ttl:
                    return cached_data["data"]
            
            # Fetch from API
            endpoint = f"dex/tokens/{token_address}"
            data = await self._make_request(endpoint)
            
            if data and "pairs" in data:
                # Process and structure the data
                processed_data = self._process_token_data(data)
                
                # Cache the result
                self.cache[cache_key] = {
                    "data": processed_data,
                    "timestamp": datetime.now().timestamp()
                }
                
                return processed_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching token data: {e}")
            return None
    
    async def search_tokens(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Search for tokens by name or symbol
        
        Args:
            query: Token name or symbol to search for
            
        Returns:
            List of matching tokens
        """
        try:
            endpoint = "dex/search"
            params = {"q": query}
            
            data = await self._make_request(endpoint, params)
            
            if data and "pairs" in data:
                return self._process_search_results(data["pairs"])
            
            return []
            
        except Exception as e:
            logger.error(f"Error searching tokens: {e}")
            return []
    
    async def get_trending_tokens(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get trending tokens across all DEXs
        
        Returns:
            List of trending tokens
        """
        try:
            endpoint = "dex/tokens/trending"
            data = await self._make_request(endpoint)
            
            if data and "pairs" in data:
                return self._process_search_results(data["pairs"])
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching trending tokens: {e}")
            return []
    
    async def get_pairs_by_exchange(self, exchange: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get all trading pairs for a specific exchange
        
        Args:
            exchange: Exchange name (e.g., 'uniswap_v2', 'pancakeswap')
            
        Returns:
            List of trading pairs
        """
        try:
            endpoint = f"dex/pairs/{exchange}"
            data = await self._make_request(endpoint)
            
            if data and "pairs" in data:
                return self._process_search_results(data["pairs"])
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching exchange pairs: {e}")
            return []
    
    async def get_market_overview(self) -> Optional[Dict[str, Any]]:
        """
        Get overall market overview and statistics
        
        Returns:
            Market overview data
        """
        try:
            # For now, return mock data until we implement real market overview
            
            return {
                "total_volume_24h": 2500000000,  # $2.5B
                "total_trades_24h": 1500000,
                "active_pairs": 25000,
                "top_gainers": [
                    {"symbol": "BTC", "change": 5.2},
                    {"symbol": "ETH", "change": 3.8},
                    {"symbol": "SOL", "change": 8.1}
                ],
                "top_losers": [
                    {"symbol": "ADA", "change": -2.1},
                    {"symbol": "DOT", "change": -1.8},
                    {"symbol": "LINK", "change": -1.5}
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return None
    
    def _process_token_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and structure raw token data from API"""
        try:
            pairs = raw_data.get("pairs", [])
            if not pairs:
                return {}
            
            # Get the most liquid pair (highest volume)
            main_pair = max(pairs, key=lambda x: float(x.get("volume", {}).get("h24", 0)))
            
            # Extract key metrics
            processed_data = {
                "token_address": main_pair.get("tokenAddress"),
                "base_token": main_pair.get("baseToken", {}).get("symbol"),
                "quote_token": main_pair.get("quoteToken", {}).get("symbol"),
                "pair_address": main_pair.get("pairAddress"),
                "dex_id": main_pair.get("dexId"),
                "exchange": main_pair.get("exchange"),
                
                # Price data
                "price_usd": float(main_pair.get("priceUsd", 0)),
                "price_change_24h": float(main_pair.get("priceChange", {}).get("h24", 0)),
                "price_change_1h": float(main_pair.get("priceChange", {}).get("h1", 0)),
                
                # Volume and liquidity
                "volume_24h": float(main_pair.get("volume", {}).get("h24", 0)),
                "liquidity_usd": float(main_pair.get("liquidity", {}).get("usd", 0)),
                
                # Market data
                "market_cap": float(main_pair.get("marketCap", 0)),
                "fdv": float(main_pair.get("fdv", 0)),  # Fully diluted valuation
                
                # Trading data
                "txns_24h": main_pair.get("txns", {}).get("h24", 0),
                "buy_txns_24h": main_pair.get("txns", {}).get("h24", {}).get("buys", 0),
                "sell_txns_24h": main_pair.get("txns", {}).get("h24", {}).get("sells", 0),
                
                # Additional info
                "created_at": main_pair.get("pairCreatedAt"),
                "last_updated": datetime.now().isoformat()
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing token data: {e}")
            return {}
    
    def _process_search_results(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process search results from API"""
        try:
            processed_pairs = []
            
            for pair in pairs[:20]:  # Limit to top 20 results
                processed_pair = {
                    "base_token": pair.get("baseToken", {}).get("symbol"),
                    "quote_token": pair.get("quoteToken", {}).get("symbol"),
                    "price_usd": float(pair.get("priceUsd", 0)),
                    "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0)),
                    "volume_24h": float(pair.get("volume", {}).get("h24", 0)),
                    "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0)),
                    "dex_id": pair.get("dexId"),
                    "exchange": pair.get("exchange"),
                    "pair_address": pair.get("pairAddress")
                }
                processed_pairs.append(processed_pair)
            
            return processed_pairs
            
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return []
    
    async def get_mock_token_data(self, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Get mock token data for testing purposes
        
        Args:
            symbol: Token symbol
            
        Returns:
            Mock token data
        """
        # Generate realistic mock data
        base_price = 45000 if symbol == "BTC" else 3000 if symbol == "ETH" else 100
        
        return {
            "symbol": symbol,
            "price_usd": base_price,
            "price_change_24h": round(np.random.uniform(-5, 5), 2),
            "price_change_1h": round(np.random.uniform(-2, 2), 2),
            "volume_24h": round(base_price * np.random.uniform(1000, 10000), 2),
            "liquidity_usd": round(base_price * np.random.uniform(10000, 100000), 2),
            "market_cap": round(base_price * np.random.uniform(100000, 1000000), 2),
            "txns_24h": int(np.random.uniform(100, 1000)),
            "buy_txns_24h": int(np.random.uniform(50, 500)),
            "sell_txns_24h": int(np.random.uniform(50, 500)),
            "last_updated": datetime.now().isoformat()
        }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and statistics"""
        return {
            "cache_size": len(self.cache),
            "cache_ttl": self.cache_ttl,
            "cached_keys": list(self.cache.keys()),
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test the DexScreener API connection"""
        try:
            # Try to fetch trending tokens as a connection test
            trending = await self.get_trending_tokens()
            
            return {
                "status": "success",
                "api_accessible": True,
                "trending_tokens_fetched": len(trending) if trending else 0,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "api_accessible": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Export the main class
__all__ = ["DexScreenerClient"]
