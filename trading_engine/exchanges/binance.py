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

logger = structlog.get_logger()

class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        if testnet:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': True,
                'enableRateLimit': True
            })
        else:
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True
            })
            
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
        
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
    async def get_account_info(self) -> Dict[str, Any]:
        try:
            timestamp = int(time.time() * 1000)
            params = {
                'timestamp': timestamp
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/account"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'balances': data.get('balances', []),
                            'permissions': data.get('permissions', []),
                            'maker_commission': data.get('makerCommission'),
                            'taker_commission': data.get('takerCommission')
                        }
                    else:
                        logger.error(f"Failed to get account info: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
            
    async def get_balance(self, asset: str) -> Dict[str, Any]:
        try:
            account_info = await self.get_account_info()
            for balance in account_info.get('balances', []):
                if balance['asset'] == asset:
                    return {
                        'asset': balance['asset'],
                        'free': float(balance['free']),
                        'locked': float(balance['locked']),
                        'total': float(balance['free']) + float(balance['locked'])
                    }
            return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}
        except Exception as e:
            logger.error(f"Error getting balance for {asset}: {e}")
            return {'asset': asset, 'free': 0.0, 'locked': 0.0, 'total': 0.0}
            
    async def get_market_price(self, symbol: str) -> Optional[float]:
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['price'])
                    else:
                        logger.error(f"Failed to get market price: {response.status}")
                        return None
        except Exception as e:
            logger.error(f"Error getting market price for {symbol}: {e}")
            return None
            
    async def get_klines(self, symbol: str, interval: str = '1h', limit: int = 100) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        df = pd.DataFrame(data, columns=[
                            'timestamp', 'open', 'high', 'low', 'close', 'volume',
                            'close_time', 'quote_asset_volume', 'number_of_trades',
                            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                        ])
                        
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col])
                            
                        return df
                    else:
                        logger.error(f"Failed to get klines: {response.status}")
                        return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error getting klines for {symbol}: {e}")
            return pd.DataFrame()
            
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, 
                         price: Optional[float] = None, stop_price: Optional[float] = None) -> Dict[str, Any]:
        try:
            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol,
                'side': side.upper(),
                'type': order_type.upper(),
                'quantity': quantity,
                'timestamp': timestamp
            }
            
            if price:
                params['price'] = price
            if stop_price:
                params['stopPrice'] = stop_price
                
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/order"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'order_id': data['orderId'],
                            'symbol': data['symbol'],
                            'side': data['side'],
                            'type': data['type'],
                            'quantity': float(data['origQty']),
                            'price': float(data['price']) if data['price'] != '0' else None,
                            'status': data['status'],
                            'timestamp': datetime.fromtimestamp(data['time'] / 1000)
                        }
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to place order: {error_data}")
                        return {'error': error_data}
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
            
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': timestamp
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/order"
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'order_id': data['orderId'],
                            'symbol': data['symbol'],
                            'status': 'CANCELLED'
                        }
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to cancel order: {error_data}")
                        return {'error': error_data}
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return {'error': str(e)}
            
    async def get_order_status(self, symbol: str, order_id: str) -> Dict[str, Any]:
        try:
            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol,
                'orderId': order_id,
                'timestamp': timestamp
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/order"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'order_id': data['orderId'],
                            'symbol': data['symbol'],
                            'side': data['side'],
                            'type': data['type'],
                            'quantity': float(data['origQty']),
                            'executed_quantity': float(data['executedQty']),
                            'price': float(data['price']) if data['price'] != '0' else None,
                            'status': data['status'],
                            'timestamp': datetime.fromtimestamp(data['time'] / 1000)
                        }
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to get order status: {error_data}")
                        return {'error': error_data}
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {'error': str(e)}
            
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            timestamp = int(time.time() * 1000)
            params = {'timestamp': timestamp}
            
            if symbol:
                params['symbol'] = symbol
                
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/openOrders"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        orders = []
                        for order in data:
                            orders.append({
                                'order_id': order['orderId'],
                                'symbol': order['symbol'],
                                'side': order['side'],
                                'type': order['type'],
                                'quantity': float(order['origQty']),
                                'executed_quantity': float(order['executedQty']),
                                'price': float(order['price']) if order['price'] != '0' else None,
                                'status': order['status'],
                                'timestamp': datetime.fromtimestamp(order['time'] / 1000)
                            })
                        return orders
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to get open orders: {error_data}")
                        return []
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
            
    async def get_trade_history(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            timestamp = int(time.time() * 1000)
            params = {
                'symbol': symbol,
                'limit': limit,
                'timestamp': timestamp
            }
            
            signature = self._generate_signature(params)
            params['signature'] = signature
            
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            url = f"{self.base_url}/api/v3/myTrades"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        trades = []
                        for trade in data:
                            trades.append({
                                'trade_id': trade['id'],
                                'order_id': trade['orderId'],
                                'symbol': trade['symbol'],
                                'side': trade['isBuyerMaker'] and 'SELL' or 'BUY',
                                'quantity': float(trade['qty']),
                                'price': float(trade['price']),
                                'commission': float(trade['commission']),
                                'commission_asset': trade['commissionAsset'],
                                'timestamp': datetime.fromtimestamp(trade['time'] / 1000)
                            })
                        return trades
                    else:
                        error_data = await response.json()
                        logger.error(f"Failed to get trade history: {error_data}")
                        return []
        except Exception as e:
            logger.error(f"Error getting trade history: {e}")
            return []
