#!/usr/bin/env python3
"""
👥 DEXTER COLLABORATIVE LEARNING SYSTEM
============================================================
Learns from community wisdom and shared successful strategies
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
import pandas as pd
import numpy as np
from motor.motor_asyncio import AsyncIOMotorClient
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

logger = structlog.get_logger()

class CollaborativeLearningSystem:
    """
    👥 Collaborative Learning System for DEXTER
    
    Features:
    1. Learn from community trading strategies
    2. Share anonymized successful patterns
    3. Learn from collective performance
    4. Identify top performers
    5. Adapt strategies based on community wisdom
    6. Collaborative filtering for recommendations
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Collaborative learning parameters
        self.min_community_size = 10
        self.anonymization_threshold = 5
        self.similarity_threshold = 0.7
        self.performance_threshold = 0.6
        
        # Community data
        self.community_strategies = {}
        self.user_similarities = {}
        self.collective_insights = {}
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.collaborative_learning
            
            # Create indexes
            await self.collection.create_index("timestamp")
            await self.collection.create_index("strategy_hash")
            await self.collection.create_index("performance_score")
            await self.collection.create_index("anonymized_user_id")
            
            logger.info("✅ Collaborative Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Collaborative Learning System: {e}")
            raise
    
    async def share_successful_strategy(self, 
                                      user_id: str,
                                      strategy_data: Dict[str, Any],
                                      performance_metrics: Dict[str, Any]) -> str:
        """Share a successful strategy with the community"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Anonymize user data
            anonymized_user_id = self._anonymize_user_id(user_id)
            
            # Create strategy hash for deduplication
            strategy_hash = self._create_strategy_hash(strategy_data)
            
            # Check if strategy already exists
            existing_strategy = await self.collection.find_one({"strategy_hash": strategy_hash})
            if existing_strategy:
                # Update performance if better
                if performance_metrics.get("success_rate", 0) > existing_strategy.get("performance_score", 0):
                    await self.collection.update_one(
                        {"strategy_hash": strategy_hash},
                        {
                            "$set": {
                                "performance_score": performance_metrics.get("success_rate", 0),
                                "performance_metrics": performance_metrics,
                                "last_updated": datetime.utcnow()
                            }
                        }
                    )
                return existing_strategy["strategy_hash"]
            
            # Store new strategy
            strategy_doc = {
                "strategy_hash": strategy_hash,
                "anonymized_user_id": anonymized_user_id,
                "strategy_data": strategy_data,
                "performance_metrics": performance_metrics,
                "performance_score": performance_metrics.get("success_rate", 0),
                "timestamp": datetime.utcnow(),
                "type": "shared_strategy",
                "community_rating": 0.0,
                "usage_count": 0
            }
            
            result = await self.collection.insert_one(strategy_doc)
            
            logger.info(f"👥 Shared successful strategy: {strategy_hash[:8]}...")
            
            return strategy_hash
            
        except Exception as e:
            logger.error(f"Error sharing successful strategy: {e}")
            return ""
    
    def _anonymize_user_id(self, user_id: str) -> str:
        """Anonymize user ID for privacy"""
        try:
            # Create hash of user ID
            hash_object = hashlib.sha256(user_id.encode())
            return f"user_{hash_object.hexdigest()[:8]}"
        except Exception as e:
            logger.error(f"Error anonymizing user ID: {e}")
            return f"user_{hash(user_id) % 1000000}"
    
    def _create_strategy_hash(self, strategy_data: Dict[str, Any]) -> str:
        """Create hash for strategy deduplication"""
        try:
            # Sort and stringify strategy data
            strategy_string = json.dumps(strategy_data, sort_keys=True)
            hash_object = hashlib.sha256(strategy_string.encode())
            return hash_object.hexdigest()
        except Exception as e:
            logger.error(f"Error creating strategy hash: {e}")
            return hashlib.sha256(str(strategy_data).encode()).hexdigest()
    
    async def learn_from_community(self, user_id: str, user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn from community strategies"""
        try:
            if not self.collection:
                await self.initialize()
            
            logger.info(f"👥 Learning from community for user {user_id[:8]}...")
            
            # Get top performing strategies
            top_strategies = await self._get_top_community_strategies()
            
            # Find similar users
            similar_users = await self._find_similar_users(user_id, user_preferences)
            
            # Get personalized recommendations
            recommendations = await self._get_personalized_recommendations(
                user_id, top_strategies, similar_users
            )
            
            # Learn from collective performance
            collective_insights = await self._analyze_collective_performance()
            
            return {
                "top_community_strategies": top_strategies,
                "similar_users": similar_users,
                "personalized_recommendations": recommendations,
                "collective_insights": collective_insights,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error learning from community: {e}")
            return {"error": str(e)}
    
    async def _get_top_community_strategies(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top performing community strategies"""
        try:
            # Get strategies with high performance and community rating
            top_strategies = await self.collection.find({
                "type": "shared_strategy",
                "performance_score": {"$gte": self.performance_threshold},
                "community_rating": {"$gte": 0.5}
            }).sort("performance_score", -1).limit(limit).to_list(length=limit)
            
            # Anonymize and clean data
            for strategy in top_strategies:
                # Remove sensitive information
                if "strategy_data" in strategy:
                    strategy["strategy_data"] = self._clean_strategy_data(strategy["strategy_data"])
                
                # Remove user identification
                del strategy["anonymized_user_id"]
            
            return top_strategies
            
        except Exception as e:
            logger.error(f"Error getting top community strategies: {e}")
            return []
    
    def _clean_strategy_data(self, strategy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean strategy data to remove sensitive information"""
        try:
            cleaned = {}
            
            # Keep only non-sensitive strategy information
            allowed_keys = [
                "symbol", "timeframe", "indicators", "entry_conditions", 
                "exit_conditions", "risk_management", "strategy_type"
            ]
            
            for key in allowed_keys:
                if key in strategy_data:
                    cleaned[key] = strategy_data[key]
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error cleaning strategy data: {e}")
            return {}
    
    async def _find_similar_users(self, user_id: str, user_preferences: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find users with similar trading patterns"""
        try:
            # Get user's trading history
            user_trades = await self.collection.find({
                "anonymized_user_id": self._anonymize_user_id(user_id),
                "type": "user_trading_pattern"
            }).to_list(length=100)
            
            if not user_trades:
                return []
            
            # Get all users' trading patterns
            all_patterns = await self.collection.find({
                "type": "user_trading_pattern"
            }).to_list(length=1000)
            
            if len(all_patterns) < self.min_community_size:
                return []
            
            # Calculate user similarities
            similarities = self._calculate_user_similarities(user_trades, all_patterns)
            
            # Return top similar users
            similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return [
                {"user_id": user_id, "similarity": similarity}
                for user_id, similarity in similar_users
                if similarity >= self.similarity_threshold
            ]
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def _calculate_user_similarities(self, 
                                   user_trades: List[Dict[str, Any]], 
                                   all_patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate similarities between users"""
        try:
            similarities = {}
            
            # Create user feature vectors
            user_features = self._create_user_feature_vector(user_trades)
            
            # Group patterns by user
            user_patterns = {}
            for pattern in all_patterns:
                user_id = pattern.get("anonymized_user_id")
                if user_id not in user_patterns:
                    user_patterns[user_id] = []
                user_patterns[user_id].append(pattern)
            
            # Calculate similarities
            for other_user_id, other_trades in user_patterns.items():
                if other_user_id == self._anonymize_user_id(user_trades[0].get("anonymized_user_id", "")):
                    continue
                
                other_features = self._create_user_feature_vector(other_trades)
                similarity = self._calculate_feature_similarity(user_features, other_features)
                similarities[other_user_id] = similarity
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating user similarities: {e}")
            return {}
    
    def _create_user_feature_vector(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """Create feature vector for user"""
        try:
            if not trades:
                return np.zeros(10)
            
            # Extract features
            features = []
            
            # Trading frequency
            features.append(len(trades))
            
            # Symbol preferences
            symbols = [t.get("symbol", "BTC") for t in trades]
            unique_symbols = len(set(symbols))
            features.append(unique_symbols)
            
            # Timeframe preferences
            timeframes = [t.get("timeframe", "1h") for t in trades]
            timeframe_counts = {}
            for tf in timeframes:
                timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
            most_common_timeframe = max(timeframe_counts, key=timeframe_counts.get) if timeframe_counts else "1h"
            features.append(hash(most_common_timeframe) % 100)
            
            # Risk tolerance
            position_sizes = [t.get("position_size", 0.02) for t in trades]
            avg_position_size = np.mean(position_sizes) if position_sizes else 0.02
            features.append(avg_position_size * 100)
            
            # Success rate
            successes = [t.get("success", False) for t in trades]
            success_rate = np.mean(successes) if successes else 0.5
            features.append(success_rate * 100)
            
            # Average holding time
            holding_times = [t.get("holding_time", 1) for t in trades]
            avg_holding_time = np.mean(holding_times) if holding_times else 1
            features.append(avg_holding_time)
            
            # Volatility preference
            volatilities = [t.get("market_volatility", 0.05) for t in trades]
            avg_volatility = np.mean(volatilities) if volatilities else 0.05
            features.append(avg_volatility * 100)
            
            # Profit/loss ratio
            profits = [t.get("profit_loss", 0) for t in trades if t.get("profit_loss", 0) > 0]
            losses = [abs(t.get("profit_loss", 0)) for t in trades if t.get("profit_loss", 0) < 0]
            profit_loss_ratio = np.mean(profits) / np.mean(losses) if profits and losses else 1.0
            features.append(profit_loss_ratio)
            
            # Strategy complexity
            strategies = [t.get("strategy_complexity", 1) for t in trades]
            avg_complexity = np.mean(strategies) if strategies else 1
            features.append(avg_complexity)
            
            # Market timing
            timings = [t.get("market_timing", 0.5) for t in trades]
            avg_timing = np.mean(timings) if timings else 0.5
            features.append(avg_timing * 100)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"Error creating user feature vector: {e}")
            return np.zeros(10)
    
    def _calculate_feature_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate similarity between feature vectors"""
        try:
            # Normalize features
            features1_norm = features1 / (np.linalg.norm(features1) + 1e-8)
            features2_norm = features2 / (np.linalg.norm(features2) + 1e-8)
            
            # Calculate cosine similarity
            similarity = np.dot(features1_norm, features2_norm)
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating feature similarity: {e}")
            return 0.0
    
    async def _get_personalized_recommendations(self, 
                                             user_id: str, 
                                             top_strategies: List[Dict[str, Any]], 
                                             similar_users: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get personalized strategy recommendations"""
        try:
            recommendations = []
            
            # Get strategies used by similar users
            similar_user_ids = [u["user_id"] for u in similar_users]
            
            for user_id_similar in similar_user_ids:
                user_strategies = await self.collection.find({
                    "anonymized_user_id": user_id_similar,
                    "type": "shared_strategy",
                    "performance_score": {"$gte": self.performance_threshold}
                }).to_list(length=5)
                
                for strategy in user_strategies:
                    # Calculate recommendation score
                    similarity = next((u["similarity"] for u in similar_users if u["user_id"] == user_id_similar), 0)
                    recommendation_score = similarity * strategy.get("performance_score", 0)
                    
                    if recommendation_score > 0.3:  # Minimum threshold
                        recommendations.append({
                            "strategy": strategy,
                            "recommendation_score": recommendation_score,
                            "source": "similar_user"
                        })
            
            # Add top community strategies
            for strategy in top_strategies[:5]:
                recommendations.append({
                    "strategy": strategy,
                    "recommendation_score": strategy.get("performance_score", 0),
                    "source": "community_top"
                })
            
            # Sort by recommendation score
            recommendations.sort(key=lambda x: x["recommendation_score"], reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return []
    
    async def _analyze_collective_performance(self) -> Dict[str, Any]:
        """Analyze collective performance of the community"""
        try:
            # Get all shared strategies
            all_strategies = await self.collection.find({
                "type": "shared_strategy"
            }).to_list(length=1000)
            
            if not all_strategies:
                return {"message": "No community strategies available"}
            
            # Analyze performance
            performance_scores = [s.get("performance_score", 0) for s in all_strategies]
            
            collective_insights = {
                "total_strategies": len(all_strategies),
                "avg_performance": np.mean(performance_scores),
                "top_performance": np.max(performance_scores),
                "performance_distribution": {
                    "excellent": len([s for s in performance_scores if s >= 0.8]),
                    "good": len([s for s in performance_scores if 0.6 <= s < 0.8]),
                    "average": len([s for s in performance_scores if 0.4 <= s < 0.6]),
                    "poor": len([s for s in performance_scores if s < 0.4])
                },
                "most_common_strategies": self._get_most_common_strategies(all_strategies),
                "emerging_trends": self._identify_emerging_trends(all_strategies)
            }
            
            return collective_insights
            
        except Exception as e:
            logger.error(f"Error analyzing collective performance: {e}")
            return {"error": str(e)}
    
    def _get_most_common_strategies(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get most common strategy types"""
        try:
            strategy_types = {}
            
            for strategy in strategies:
                strategy_data = strategy.get("strategy_data", {})
                strategy_type = strategy_data.get("strategy_type", "unknown")
                strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
            
            # Sort by frequency
            common_strategies = sorted(strategy_types.items(), key=lambda x: x[1], reverse=True)
            
            return [
                {"strategy_type": stype, "count": count}
                for stype, count in common_strategies[:5]
            ]
            
        except Exception as e:
            logger.error(f"Error getting most common strategies: {e}")
            return []
    
    def _identify_emerging_trends(self, strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify emerging trends in strategies"""
        try:
            # Group strategies by time
            recent_strategies = [s for s in strategies if s.get("timestamp", datetime.min) > datetime.utcnow() - timedelta(days=30)]
            older_strategies = [s for s in strategies if s.get("timestamp", datetime.min) <= datetime.utcnow() - timedelta(days=30)]
            
            # Compare strategy types
            recent_types = {}
            older_types = {}
            
            for strategy in recent_strategies:
                stype = strategy.get("strategy_data", {}).get("strategy_type", "unknown")
                recent_types[stype] = recent_types.get(stype, 0) + 1
            
            for strategy in older_strategies:
                stype = strategy.get("strategy_data", {}).get("strategy_type", "unknown")
                older_types[stype] = older_types.get(stype, 0) + 1
            
            # Find emerging trends
            emerging_trends = []
            for stype in recent_types:
                recent_count = recent_types[stype]
                older_count = older_types.get(stype, 0)
                
                if older_count > 0:
                    growth_rate = (recent_count - older_count) / older_count
                    if growth_rate > 0.5:  # 50% growth
                        emerging_trends.append({
                            "strategy_type": stype,
                            "growth_rate": growth_rate,
                            "recent_count": recent_count,
                            "older_count": older_count
                        })
            
            return sorted(emerging_trends, key=lambda x: x["growth_rate"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error identifying emerging trends: {e}")
            return []
    
    async def rate_community_strategy(self, 
                                    user_id: str, 
                                    strategy_hash: str, 
                                    rating: float,
                                    feedback: str = None) -> bool:
        """Rate a community strategy"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Validate rating
            if not (0.0 <= rating <= 1.0):
                return False
            
            # Update strategy rating
            result = await self.collection.update_one(
                {"strategy_hash": strategy_hash},
                {
                    "$inc": {"usage_count": 1},
                    "$push": {
                        "ratings": {
                            "user_id": self._anonymize_user_id(user_id),
                            "rating": rating,
                            "feedback": feedback,
                            "timestamp": datetime.utcnow()
                        }
                    }
                }
            )
            
            if result.modified_count > 0:
                # Recalculate community rating
                await self._recalculate_community_rating(strategy_hash)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error rating community strategy: {e}")
            return False
    
    async def _recalculate_community_rating(self, strategy_hash: str):
        """Recalculate community rating for a strategy"""
        try:
            strategy = await self.collection.find_one({"strategy_hash": strategy_hash})
            if not strategy or "ratings" not in strategy:
                return
            
            ratings = strategy["ratings"]
            if not ratings:
                return
            
            # Calculate average rating
            avg_rating = np.mean([r["rating"] for r in ratings])
            
            # Update strategy
            await self.collection.update_one(
                {"strategy_hash": strategy_hash},
                {"$set": {"community_rating": avg_rating}}
            )
            
        except Exception as e:
            logger.error(f"Error recalculating community rating: {e}")
    
    async def get_collaborative_insights(self) -> Dict[str, Any]:
        """Get collaborative learning insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get community statistics
            total_strategies = await self.collection.count_documents({"type": "shared_strategy"})
            total_users = len(await self.collection.distinct("anonymized_user_id"))
            
            # Get recent activity
            recent_strategies = await self.collection.count_documents({
                "type": "shared_strategy",
                "timestamp": {"$gte": datetime.utcnow() - timedelta(days=7)}
            })
            
            # Get top performers
            top_strategies = await self.collection.find({
                "type": "shared_strategy"
            }).sort("performance_score", -1).limit(5).to_list(length=5)
            
            return {
                "community_size": total_users,
                "total_strategies": total_strategies,
                "recent_activity": recent_strategies,
                "top_strategies": top_strategies,
                "collaboration_health": self._calculate_collaboration_health(total_strategies, total_users),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting collaborative insights: {e}")
            return {"error": str(e)}
    
    def _calculate_collaboration_health(self, total_strategies: int, total_users: int) -> str:
        """Calculate collaboration health score"""
        try:
            if total_users == 0:
                return "poor"
            
            strategies_per_user = total_strategies / total_users
            
            if strategies_per_user >= 2:
                return "excellent"
            elif strategies_per_user >= 1:
                return "good"
            elif strategies_per_user >= 0.5:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error calculating collaboration health: {e}")
            return "unknown"

# Global instance
collaborative_learning_system = CollaborativeLearningSystem()
