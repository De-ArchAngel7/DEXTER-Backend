import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import structlog
from decimal import Decimal
import asyncio
import aiohttp
import hmac
import hashlib
import time
import json
import base64

logger = structlog.get_logger()

class KuCoinExchange:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        
        if testnet:
            self.exchange = ccxt.kucoin({
                'apiKey': api_key,
                'secret': api_secret,
                'password': passphrase,
                'sandbox': True,
                'enableRateLimit': True
            })
        else:
            self.exchange = ccxt.kucoin({
                'apiKey': api_key,
                'secret': api_secret,
                'password': passphrase,
                'enableRateLimit': True
            })
            
        self.base_url = "https://sandbox-api.kucoin.com" if testnet else "https://api.kucoin.com"
        
    def _generate_signature(self, timestamp: str, method: str, endpoint: str, body: str = '') -> Dict[str, str]:
        """Generate KuCoin API signature"""
        message = timestamp + method + endpoint + body
        signature = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()
        
        passphrase = base64.b64encode(
            hmac.new(
                self.api_secret.encode('utf-8'),
                self.passphrase.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode()
        
        return {
            'KC-API-SIGN': signature,
            'KC-API-TIMESTAMP': timestamp,
            'KC-API-KEY': self.api_key,
            'KC-API-PASSPHRASE': passphrase,
            'KC-API-KEY-VERSION': '2'
        }
        
    async def get_account_info(self) -> Dict[str, Any]:
        """Get KuCoin account information"""
        try:
            timestamp = str(int(time.time() * 1000))
            endpoint = '/api/v1/accounts'
            method = 'GET'
            
            headers = self._generate_signature(timestamp, method, endpoint)
            headers['Content-Type'] = 'application/json'
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'accounts': data.get('data', []),
                            'success': data.get('code') == '200000'
                        }
                    else:
                        logger.error(f"Failed to get account info: {response.status}")
                        return {'success': False, 'error': f"HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {'success': False, 'error': str(e)}
            
    async def get_balance(self, currency: str) -> Dict[str, Any]:
        """Get balance for specific currency"""
        try:
            account_info = await self.get_account_info()
            if not account_info.get('success'):
                return {'currency': currency, 'available': 0.0, 'hold': 0.0, 'total': 0.0}
                
            for account in account_info.get('accounts', []):
                if account.get('currency') == currency:
                    return {
                        'currency': account.get('currency'),
                        'available': float(account.get('available', 0)),
                        'hold': float(account.get('hold', 0)),
                        'total': float(account.get('available', 0)) + float(account.get('hold', 0))
                    }
            return {'currency': currency, 'available': 0.0, 'hold': 0.0, 'total': 0.0}
        except Exception as e:
            logger.error(f"Error getting balance for {currency}: {e}")
            return {'currency': currency, 'available': 0.0, 'hold': 0.0, 'total': 0.0}
            
    async def get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol"""
        try:
            url = f"{self.base_url}/api/v1/market/orderbook/level1"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            return float(data['data']['price'])
            return None
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return None
            
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get detailed ticker information"""
        try:
            url = f"{self.base_url}/api/v1/market/orderbook/level1"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            ticker_data = data['data']
                            return {
                                'symbol': symbol,
                                'price': float(ticker_data['price']),
                                'best_bid': float(ticker_data['bestBid']),
                                'best_ask': float(ticker_data['bestAsk']),
                                'size': float(ticker_data['size']),
                                'timestamp': int(ticker_data['time'])
                            }
            return {}
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return {}
            
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """Get order book for a symbol"""
        try:
            url = f"{self.base_url}/api/v1/market/orderbook/level2_{depth}"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            return {
                                'symbol': symbol,
                                'bids': data['data']['bids'],
                                'asks': data['data']['asks'],
                                'timestamp': int(data['data']['time'])
                            }
            return {}
        except Exception as e:
            logger.error(f"Error getting order book for {symbol}: {e}")
            return {}
            
    async def place_order(self, symbol: str, side: str, order_type: str, 
                         size: float, price: Optional[float] = None, 
                         client_oid: Optional[str] = None) -> Dict[str, Any]:
        """Place a new order"""
        try:
            timestamp = str(int(time.time() * 1000))
            endpoint = '/api/v1/orders'
            method = 'POST'
            
            body_data = {
                'clientOid': client_oid or f"bot_{int(time.time())}",
                'symbol': symbol,
                'side': side.lower(),
                'type': order_type.lower(),
                'size': str(size)
            }
            
            if price and order_type.lower() == 'limit':
                body_data['price'] = str(price)
                
            body = json.dumps(body_data)
            headers = self._generate_signature(timestamp, method, endpoint, body)
            headers['Content-Type'] = 'application/json'
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=body) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            return {
                                'success': True,
                                'order_id': data['data']['orderId'],
                                'client_oid': body_data['clientOid']
                            }
                        else:
                            return {
                                'success': False,
                                'error': data.get('msg', 'Unknown error')
                            }
                    else:
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}"
                        }
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'success': False, 'error': str(e)}
            
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order"""
        try:
            timestamp = str(int(time.time() * 1000))
            endpoint = f'/api/v1/orders/{order_id}'
            method = 'DELETE'
            
            headers = self._generate_signature(timestamp, method, endpoint)
            headers['Content-Type'] = 'application/json'
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            return {'success': True, 'order_id': order_id}
                        else:
                            return {
                                'success': False,
                                'error': data.get('msg', 'Unknown error')
                            }
                    else:
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}"
                        }
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status and details"""
        try:
            timestamp = str(int(time.time() * 1000))
            endpoint = f'/api/v1/orders/{order_id}'
            method = 'GET'
            
            headers = self._generate_signature(timestamp, method, endpoint)
            headers['Content-Type'] = 'application/json'
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            order_data = data['data']
                            return {
                                'order_id': order_id,
                                'symbol': order_data['symbol'],
                                'side': order_data['side'],
                                'type': order_data['type'],
                                'size': float(order_data['size']),
                                'price': float(order_data['price']) if order_data.get('price') else None,
                                'status': order_data['status'],
                                'filled_size': float(order_data['dealSize']),
                                'created_at': order_data['createdAt']
                            }
                        else:
                            return {
                                'success': False,
                                'error': data.get('msg', 'Unknown error')
                            }
                    else:
                        return {
                            'success': False,
                            'error': f"HTTP {response.status}"
                        }
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def get_trading_pairs(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs"""
        try:
            url = f"{self.base_url}/api/v1/symbols"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            return data['data']
            return []
        except Exception as e:
            logger.error(f"Error getting trading pairs: {e}")
            return []
            
    async def get_24hr_stats(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour trading statistics"""
        try:
            url = f"{self.base_url}/api/v1/market/stats"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            stats = data['data']
                            return {
                                'symbol': symbol,
                                'change_rate': float(stats['changeRate']),
                                'change_price': float(stats['changePrice']),
                                'high': float(stats['high']),
                                'low': float(stats['low']),
                                'volume': float(stats['vol']),
                                'turnover': float(stats['volValue'])
                            }
            return {}
        except Exception as e:
            logger.error(f"Error getting 24hr stats for {symbol}: {e}")
            return {}
            
    async def get_klines(self, symbol: str, interval: str = '1min', 
                        start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get historical kline/candlestick data"""
        try:
            url = f"{self.base_url}/api/v1/market/candles"
            params = {
                'symbol': symbol,
                'type': interval
            }
            
            if start_time:
                params['startAt'] = start_time
            if end_time:
                params['endAt'] = end_time
                
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('code') == '200000':
                            klines = []
                            for kline in data['data']:
                                klines.append({
                                    'timestamp': int(kline[0]),
                                    'open': float(kline[1]),
                                    'close': float(kline[2]),
                                    'high': float(kline[3]),
                                    'low': float(kline[4]),
                                    'volume': float(kline[5]),
                                    'turnover': float(kline[6])
                                })
                            return klines
            return []
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return []

    async def test_connection(self) -> Dict[str, Any]:
        """Test API connection and permissions"""
        try:
            # Test account access
            account_info = await self.get_account_info()
            if not account_info.get('success'):
                return {
                    'connected': False,
                    'error': 'Failed to access account information'
                }
                
            # Test market data access
            pairs = await self.get_trading_pairs()
            if not pairs:
                return {
                    'connected': False,
                    'error': 'Failed to fetch trading pairs'
                }
                
            return {
                'connected': True,
                'account_type': 'Trading Account',
                'available_pairs': len(pairs),
                'permissions': 'Full access confirmed'
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }
