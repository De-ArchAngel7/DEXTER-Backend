import os
import asyncio
import time
from typing import Dict, Any
from datetime import datetime, timedelta
import httpx
import structlog
from motor.motor_asyncio import AsyncIOMotorClient

logger = structlog.get_logger(__name__)

class MonitoringDashboardService:
    def __init__(self):
        self.grafana_url = os.getenv("GRAFANA_URL", "http://localhost:3000")
        self.grafana_api_key = os.getenv("GRAFANA_API_KEY")
        self.grafana_username = os.getenv("GRAFANA_USERNAME", "admin")
        self.grafana_password = os.getenv("GRAFANA_PASSWORD")
        self.mongodb_url = os.getenv("DATABASE_URL")
        
        self.metrics_cache = {}
        self.last_update = None
        
        if not self.grafana_api_key and not self.grafana_password:
            logger.warning("Grafana credentials not configured - monitoring dashboard disabled")
    
    async def get_trading_metrics(self) -> Dict[str, Any]:
        """Get comprehensive trading metrics for dashboard"""
        try:
            # Get data from MongoDB
            client = AsyncIOMotorClient(self.mongodb_url)
            db = client.dexter
            
            # Calculate time ranges
            now = datetime.utcnow()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # Get trading performance metrics
            trades_collection = db.trades
            conversations_collection = db.conversations
            learning_collection = db.learning_data
            
            # Parallel queries for better performance
            tasks = [
                self._get_trade_metrics(trades_collection, today, week_ago, month_ago),
                self._get_ai_metrics(conversations_collection, today, week_ago, month_ago),
                self._get_learning_metrics(learning_collection, today, week_ago, month_ago),
                self._get_system_metrics()
            ]
            
            trade_metrics, ai_metrics, learning_metrics, system_metrics = await asyncio.gather(*tasks)
            
            # Combine all metrics
            metrics = {
                "timestamp": now.isoformat(),
                "trading": trade_metrics,
                "ai": ai_metrics,
                "learning": learning_metrics,
                "system": system_metrics
            }
            
            # Cache metrics
            self.metrics_cache = metrics
            self.last_update = now
            
            client.close()
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting trading metrics: {e}")
            return {"error": str(e)}
    
    async def _get_trade_metrics(self, collection, today, week_ago, month_ago):
        """Get trading performance metrics"""
        try:
            # Total trades
            total_trades = await collection.count_documents({})
            today_trades = await collection.count_documents({"timestamp": {"$gte": today}})
            week_trades = await collection.count_documents({"timestamp": {"$gte": week_ago}})
            month_trades = await collection.count_documents({"timestamp": {"$gte": month_ago}})
            
            # Profit/Loss calculations
            pipeline = [
                {"$group": {
                    "_id": None,
                    "total_profit": {"$sum": "$profit_loss"},
                    "avg_profit": {"$avg": "$profit_loss"},
                    "max_profit": {"$max": "$profit_loss"},
                    "max_loss": {"$min": "$profit_loss"},
                    "win_rate": {
                        "$avg": {"$cond": [{"$gt": ["$profit_loss", 0]}, 1, 0]}
                    }
                }}
            ]
            
            result = await collection.aggregate(pipeline).to_list(1)
            profit_metrics = result[0] if result else {}
            
            return {
                "total_trades": total_trades,
                "today_trades": today_trades,
                "week_trades": week_trades,
                "month_trades": month_trades,
                "total_profit": profit_metrics.get("total_profit", 0),
                "avg_profit": profit_metrics.get("avg_profit", 0),
                "max_profit": profit_metrics.get("max_profit", 0),
                "max_loss": profit_metrics.get("max_loss", 0),
                "win_rate": profit_metrics.get("win_rate", 0) * 100
            }
        except Exception as e:
            logger.error(f"Error getting trade metrics: {e}")
            return {"error": str(e)}
    
    async def _get_ai_metrics(self, collection, today, week_ago, month_ago):
        """Get AI model performance metrics"""
        try:
            # Total conversations
            total_conversations = await collection.count_documents({})
            today_conversations = await collection.count_documents({"timestamp": {"$gte": today}})
            week_conversations = await collection.count_documents({"timestamp": {"$gte": week_ago}})
            
            # Model usage statistics
            pipeline = [
                {"$group": {
                    "_id": "$model_used",
                    "count": {"$sum": 1},
                    "avg_response_time": {"$avg": "$response_time"}
                }}
            ]
            
            model_stats = await collection.aggregate(pipeline).to_list(10)
            
            # DialoGPT vs OpenAI usage
            dialoGPT_count = sum(stat["count"] for stat in model_stats if stat["_id"] == "dialoGPT")
            openai_count = sum(stat["count"] for stat in model_stats if stat["_id"] == "openai")
            
            return {
                "total_conversations": total_conversations,
                "today_conversations": today_conversations,
                "week_conversations": week_conversations,
                "dialoGPT_usage": dialoGPT_count,
                "openai_usage": openai_count,
                "model_stats": model_stats
            }
        except Exception as e:
            logger.error(f"Error getting AI metrics: {e}")
            return {"error": str(e)}
    
    async def _get_learning_metrics(self, collection, today, week_ago, month_ago):
        """Get learning system metrics"""
        try:
            # Learning data points
            total_learning_data = await collection.count_documents({})
            today_learning = await collection.count_documents({"timestamp": {"$gte": today}})
            week_learning = await collection.count_documents({"timestamp": {"$gte": week_ago}})
            
            # Learning accuracy
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_accuracy": {"$avg": "$accuracy"},
                    "total_predictions": {"$sum": 1},
                    "correct_predictions": {"$sum": {"$cond": [{"$eq": ["$correct", True]}, 1, 0]}}
                }}
            ]
            
            result = await collection.aggregate(pipeline).to_list(1)
            accuracy_metrics = result[0] if result else {}
            
            return {
                "total_learning_data": total_learning_data,
                "today_learning": today_learning,
                "week_learning": week_learning,
                "avg_accuracy": accuracy_metrics.get("avg_accuracy", 0) * 100,
                "total_predictions": accuracy_metrics.get("total_predictions", 0),
                "correct_predictions": accuracy_metrics.get("correct_predictions", 0)
            }
        except Exception as e:
            logger.error(f"Error getting learning metrics: {e}")
            return {"error": str(e)}
    
    async def _get_system_metrics(self):
        """Get system performance metrics"""
        try:
            import psutil
            
            # System resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "memory_total_gb": memory.total / (1024**3),
                "disk_percent": disk.percent,
                "disk_used_gb": disk.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "process_memory_mb": process_memory.rss / (1024**2),
                "uptime_seconds": time.time() - process.create_time()
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {"error": str(e)}
    
    async def send_metrics_to_grafana(self, metrics: Dict[str, Any]):
        """Send metrics to Grafana for visualization"""
        if not self.grafana_api_key:
            return
        
        try:
            # Prepare metrics for Grafana
            grafana_metrics = []
            timestamp = int(time.time() * 1000)  # Grafana expects milliseconds
            
            # Trading metrics
            for key, value in metrics.get("trading", {}).items():
                if isinstance(value, (int, float)):
                    grafana_metrics.append({
                        "name": f"dexter.trading.{key}",
                        "value": value,
                        "timestamp": timestamp
                    })
            
            # AI metrics
            for key, value in metrics.get("ai", {}).items():
                if isinstance(value, (int, float)):
                    grafana_metrics.append({
                        "name": f"dexter.ai.{key}",
                        "value": value,
                        "timestamp": timestamp
                    })
            
            # System metrics
            for key, value in metrics.get("system", {}).items():
                if isinstance(value, (int, float)):
                    grafana_metrics.append({
                        "name": f"dexter.system.{key}",
                        "value": value,
                        "timestamp": timestamp
                    })
            
            # Send to Grafana
            async with httpx.AsyncClient() as client:
                headers = {"Authorization": f"Bearer {self.grafana_api_key}"}
                response = await client.post(
                    f"{self.grafana_url}/api/metrics",
                    json=grafana_metrics,
                    headers=headers
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Sent {len(grafana_metrics)} metrics to Grafana")
                else:
                    logger.error(f"Failed to send metrics to Grafana: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error sending metrics to Grafana: {e}")

# Global instance
monitoring_dashboard = MonitoringDashboardService()
