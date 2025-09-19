#!/usr/bin/env python3
"""
🎯 DEXTER PERSONALIZED LEARNING SYSTEM
============================================================
Learns individual user preferences and adapts strategies accordingly
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
from sklearn.ensemble import RandomForestClassifier
import pickle

logger = structlog.get_logger()

class PersonalizedLearningSystem:
    """
    🎯 Personalized Learning System for DEXTER
    
    Features:
    1. Learn individual user preferences
    2. Adapt strategies to user's trading style
    3. Personalize risk recommendations
    4. Learn user's time preferences
    5. Adapt to user's skill level
    6. Personalized learning paths
    """
    
    def __init__(self, mongodb_url: str = None):
        self.mongodb_url = mongodb_url or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db = None
        self.collection = None
        
        # Personalization parameters
        self.min_interactions = 10
        self.learning_rate = 0.1
        self.decay_factor = 0.95
        
        # User profiles
        self.user_profiles = {}
        self.user_preferences = {}
        self.user_learning_paths = {}
        
    async def initialize(self):
        """Initialize MongoDB connection"""
        try:
            client = AsyncIOMotorClient(self.mongodb_url)
            self.db = client.dexter
            self.collection = self.db.personalized_learning
            
            # Create indexes
            await self.collection.create_index("user_id")
            await self.collection.create_index("timestamp")
            await self.collection.create_index("interaction_type")
            
            logger.info("✅ Personalized Learning System initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Personalized Learning System: {e}")
            raise
    
    async def learn_from_user_interaction(self, 
                                        user_id: str,
                                        interaction_type: str,
                                        interaction_data: Dict[str, Any],
                                        user_feedback: Dict[str, Any] = None) -> bool:
        """Learn from user interactions"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Store interaction
            interaction_doc = {
                "user_id": user_id,
                "interaction_type": interaction_type,
                "interaction_data": interaction_data,
                "user_feedback": user_feedback or {},
                "timestamp": datetime.utcnow(),
                "type": "user_interaction"
            }
            
            await self.collection.insert_one(interaction_doc)
            
            # Learn from this interaction
            await self._learn_from_interaction(user_id, interaction_doc)
            
            logger.info(f"🎯 Learned from user interaction: {interaction_type}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error learning from user interaction: {e}")
            return False
    
    async def _learn_from_interaction(self, user_id: str, interaction_doc: Dict[str, Any]):
        """Learn from individual interaction"""
        try:
            # Get user's interaction history
            user_interactions = await self.collection.find({
                "user_id": user_id,
                "type": "user_interaction"
            }).to_list(length=100)
            
            if len(user_interactions) < self.min_interactions:
                return
            
            # Analyze user patterns
            user_patterns = await self._analyze_user_patterns(user_interactions)
            
            # Update user profile
            await self._update_user_profile(user_id, user_patterns)
            
            # Update user preferences
            await self._update_user_preferences(user_id, user_interactions)
            
            # Update learning path
            await self._update_learning_path(user_id, user_patterns)
            
        except Exception as e:
            logger.error(f"Error learning from interaction: {e}")
    
    async def _analyze_user_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user interaction patterns"""
        try:
            if not interactions:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(interactions)
            
            # Analyze interaction types
            interaction_types = df["interaction_type"].value_counts().to_dict()
            
            # Analyze timing patterns
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
            
            timing_patterns = {
                "preferred_hours": df["hour"].mode().tolist(),
                "preferred_days": df["day_of_week"].mode().tolist(),
                "activity_level": len(interactions) / 30  # interactions per day
            }
            
            # Analyze feedback patterns
            feedback_analysis = self._analyze_feedback_patterns(interactions)
            
            # Analyze trading preferences
            trading_preferences = self._analyze_trading_preferences(interactions)
            
            return {
                "interaction_types": interaction_types,
                "timing_patterns": timing_patterns,
                "feedback_analysis": feedback_analysis,
                "trading_preferences": trading_preferences,
                "total_interactions": len(interactions)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing user patterns: {e}")
            return {}
    
    def _analyze_feedback_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user feedback patterns"""
        try:
            feedback_data = []
            
            for interaction in interactions:
                feedback = interaction.get("user_feedback", {})
                if feedback:
                    feedback_data.append(feedback)
            
            if not feedback_data:
                return {"avg_satisfaction": 0.5, "feedback_count": 0}
            
            # Calculate average satisfaction
            satisfactions = [f.get("satisfaction", 0.5) for f in feedback_data]
            avg_satisfaction = np.mean(satisfactions)
            
            # Analyze feedback themes
            feedback_themes = {}
            for feedback in feedback_data:
                theme = feedback.get("theme", "general")
                feedback_themes[theme] = feedback_themes.get(theme, 0) + 1
            
            return {
                "avg_satisfaction": avg_satisfaction,
                "feedback_count": len(feedback_data),
                "feedback_themes": feedback_themes,
                "satisfaction_trend": self._calculate_satisfaction_trend(feedback_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feedback patterns: {e}")
            return {"avg_satisfaction": 0.5, "feedback_count": 0}
    
    def _calculate_satisfaction_trend(self, feedback_data: List[Dict[str, Any]]) -> str:
        """Calculate satisfaction trend"""
        try:
            if len(feedback_data) < 3:
                return "stable"
            
            # Sort by timestamp
            feedback_data.sort(key=lambda x: x.get("timestamp", datetime.min))
            
            # Calculate trend
            satisfactions = [f.get("satisfaction", 0.5) for f in feedback_data]
            
            # Simple linear trend
            x = np.arange(len(satisfactions))
            y = np.array(satisfactions)
            
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                if slope > 0.01:
                    return "improving"
                elif slope < -0.01:
                    return "declining"
                else:
                    return "stable"
            
            return "stable"
            
        except Exception as e:
            logger.error(f"Error calculating satisfaction trend: {e}")
            return "stable"
    
    def _analyze_trading_preferences(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's trading preferences"""
        try:
            trading_interactions = [i for i in interactions if i.get("interaction_type") in ["trade", "prediction", "analysis"]]
            
            if not trading_interactions:
                return {}
            
            preferences = {
                "preferred_symbols": [],
                "preferred_timeframes": [],
                "risk_tolerance": "medium",
                "trading_frequency": "medium",
                "strategy_preference": "balanced"
            }
            
            # Analyze symbols
            symbols = []
            for interaction in trading_interactions:
                symbol = interaction.get("interaction_data", {}).get("symbol")
                if symbol:
                    symbols.append(symbol)
            
            if symbols:
                symbol_counts = pd.Series(symbols).value_counts()
                preferences["preferred_symbols"] = symbol_counts.head(3).index.tolist()
            
            # Analyze timeframes
            timeframes = []
            for interaction in trading_interactions:
                timeframe = interaction.get("interaction_data", {}).get("timeframe")
                if timeframe:
                    timeframes.append(timeframe)
            
            if timeframes:
                timeframe_counts = pd.Series(timeframes).value_counts()
                preferences["preferred_timeframes"] = timeframe_counts.head(3).index.tolist()
            
            # Analyze risk tolerance
            risk_indicators = []
            for interaction in trading_interactions:
                position_size = interaction.get("interaction_data", {}).get("position_size")
                if position_size:
                    risk_indicators.append(position_size)
            
            if risk_indicators:
                avg_position_size = np.mean(risk_indicators)
                if avg_position_size < 0.015:
                    preferences["risk_tolerance"] = "low"
                elif avg_position_size > 0.025:
                    preferences["risk_tolerance"] = "high"
                else:
                    preferences["risk_tolerance"] = "medium"
            
            # Analyze trading frequency
            trading_frequency = len(trading_interactions) / 30  # trades per day
            if trading_frequency > 2:
                preferences["trading_frequency"] = "high"
            elif trading_frequency < 0.5:
                preferences["trading_frequency"] = "low"
            else:
                preferences["trading_frequency"] = "medium"
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error analyzing trading preferences: {e}")
            return {}
    
    async def _update_user_profile(self, user_id: str, patterns: Dict[str, Any]):
        """Update user profile based on patterns"""
        try:
            # Create or update user profile
            profile = {
                "user_id": user_id,
                "patterns": patterns,
                "last_updated": datetime.utcnow().isoformat(),
                "profile_version": 1
            }
            
            self.user_profiles[user_id] = profile
            
            # Store in database
            await self.collection.update_one(
                {"user_id": user_id, "type": "user_profile"},
                {"$set": profile},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
    
    async def _update_user_preferences(self, user_id: str, interactions: List[Dict[str, Any]]):
        """Update user preferences"""
        try:
            # Analyze preferences from interactions
            preferences = self._extract_preferences_from_interactions(interactions)
            
            # Update preferences with learning rate
            if user_id in self.user_preferences:
                old_preferences = self.user_preferences[user_id]
                for key, value in preferences.items():
                    if key in old_preferences:
                        # Exponential moving average
                        old_preferences[key] = (1 - self.learning_rate) * old_preferences[key] + self.learning_rate * value
                    else:
                        old_preferences[key] = value
                preferences = old_preferences
            
            self.user_preferences[user_id] = preferences
            
            # Store in database
            await self.collection.update_one(
                {"user_id": user_id, "type": "user_preferences"},
                {"$set": {
                    "user_id": user_id,
                    "preferences": preferences,
                    "last_updated": datetime.utcnow().isoformat(),
                    "type": "user_preferences"
                }},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
    
    def _extract_preferences_from_interactions(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract preferences from interactions"""
        try:
            preferences = {}
            
            # Extract from feedback
            for interaction in interactions:
                feedback = interaction.get("user_feedback", {})
                if feedback:
                    # Extract preference indicators
                    if "preferred_style" in feedback:
                        preferences["communication_style"] = feedback["preferred_style"]
                    
                    if "preferred_detail_level" in feedback:
                        preferences["detail_level"] = feedback["preferred_detail_level"]
                    
                    if "preferred_time" in feedback:
                        preferences["preferred_time"] = feedback["preferred_time"]
            
            # Extract from interaction data
            for interaction in interactions:
                data = interaction.get("interaction_data", {})
                if data:
                    # Extract trading preferences
                    if "symbol" in data:
                        if "preferred_symbols" not in preferences:
                            preferences["preferred_symbols"] = []
                        preferences["preferred_symbols"].append(data["symbol"])
                    
                    if "timeframe" in data:
                        if "preferred_timeframes" not in preferences:
                            preferences["preferred_timeframes"] = []
                        preferences["preferred_timeframes"].append(data["timeframe"])
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error extracting preferences: {e}")
            return {}
    
    async def _update_learning_path(self, user_id: str, patterns: Dict[str, Any]):
        """Update user's learning path"""
        try:
            # Determine user's skill level
            skill_level = self._determine_skill_level(patterns)
            
            # Create personalized learning path
            learning_path = self._create_learning_path(skill_level, patterns)
            
            self.user_learning_paths[user_id] = learning_path
            
            # Store in database
            await self.collection.update_one(
                {"user_id": user_id, "type": "learning_path"},
                {"$set": {
                    "user_id": user_id,
                    "skill_level": skill_level,
                    "learning_path": learning_path,
                    "last_updated": datetime.utcnow().isoformat(),
                    "type": "learning_path"
                }},
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error updating learning path: {e}")
    
    def _determine_skill_level(self, patterns: Dict[str, Any]) -> str:
        """Determine user's skill level"""
        try:
            total_interactions = patterns.get("total_interactions", 0)
            trading_preferences = patterns.get("trading_preferences", {})
            feedback_analysis = patterns.get("feedback_analysis", {})
            
            # Calculate skill score
            skill_score = 0
            
            # Experience factor
            if total_interactions > 100:
                skill_score += 3
            elif total_interactions > 50:
                skill_score += 2
            elif total_interactions > 20:
                skill_score += 1
            
            # Trading sophistication
            if trading_preferences.get("risk_tolerance") == "high":
                skill_score += 1
            if len(trading_preferences.get("preferred_symbols", [])) > 3:
                skill_score += 1
            if len(trading_preferences.get("preferred_timeframes", [])) > 2:
                skill_score += 1
            
            # Feedback quality
            avg_satisfaction = feedback_analysis.get("avg_satisfaction", 0.5)
            if avg_satisfaction > 0.7:
                skill_score += 1
            
            # Determine level
            if skill_score >= 6:
                return "expert"
            elif skill_score >= 4:
                return "advanced"
            elif skill_score >= 2:
                return "intermediate"
            else:
                return "beginner"
                
        except Exception as e:
            logger.error(f"Error determining skill level: {e}")
            return "beginner"
    
    def _create_learning_path(self, skill_level: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Create personalized learning path"""
        try:
            learning_paths = {
                "beginner": {
                    "next_topics": ["basic_analysis", "risk_management", "chart_reading"],
                    "difficulty_level": "basic",
                    "recommended_frequency": "daily",
                    "focus_areas": ["education", "practice", "risk_awareness"]
                },
                "intermediate": {
                    "next_topics": ["advanced_analysis", "strategy_development", "market_timing"],
                    "difficulty_level": "intermediate",
                    "recommended_frequency": "weekly",
                    "focus_areas": ["strategy_optimization", "performance_analysis", "market_adaptation"]
                },
                "advanced": {
                    "next_topics": ["quantitative_analysis", "algorithmic_trading", "portfolio_optimization"],
                    "difficulty_level": "advanced",
                    "recommended_frequency": "bi-weekly",
                    "focus_areas": ["advanced_strategies", "market_microstructure", "risk_engineering"]
                },
                "expert": {
                    "next_topics": ["research_methods", "model_development", "market_making"],
                    "difficulty_level": "expert",
                    "recommended_frequency": "monthly",
                    "focus_areas": ["innovation", "research", "mentoring"]
                }
            }
            
            base_path = learning_paths.get(skill_level, learning_paths["beginner"])
            
            # Personalize based on patterns
            personalized_path = base_path.copy()
            
            # Adjust based on user preferences
            trading_preferences = patterns.get("trading_preferences", {})
            if trading_preferences.get("risk_tolerance") == "low":
                personalized_path["focus_areas"].insert(0, "conservative_strategies")
            elif trading_preferences.get("risk_tolerance") == "high":
                personalized_path["focus_areas"].insert(0, "aggressive_strategies")
            
            # Adjust based on timing patterns
            timing_patterns = patterns.get("timing_patterns", {})
            if timing_patterns.get("activity_level", 0) > 1:
                personalized_path["recommended_frequency"] = "daily"
            
            return personalized_path
            
        except Exception as e:
            logger.error(f"Error creating learning path: {e}")
            return {"next_topics": ["basic_analysis"], "difficulty_level": "basic"}
    
    async def get_personalized_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized recommendations for user"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get user profile
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                return {"message": "No user profile available"}
            
            # Get user preferences
            user_preferences = await self._get_user_preferences(user_id)
            
            # Get learning path
            learning_path = await self._get_learning_path(user_id)
            
            # Generate personalized recommendations
            recommendations = await self._generate_personalized_recommendations(
                user_profile, user_preferences, learning_path
            )
            
            return {
                "user_profile": user_profile,
                "user_preferences": user_preferences,
                "learning_path": learning_path,
                "recommendations": recommendations,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return {"error": str(e)}
    
    async def _get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        try:
            if user_id in self.user_profiles:
                return self.user_profiles[user_id]
            
            # Load from database
            profile = await self.collection.find_one({
                "user_id": user_id,
                "type": "user_profile"
            })
            
            if profile:
                self.user_profiles[user_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return None
    
    async def _get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences"""
        try:
            if user_id in self.user_preferences:
                return self.user_preferences[user_id]
            
            # Load from database
            preferences = await self.collection.find_one({
                "user_id": user_id,
                "type": "user_preferences"
            })
            
            if preferences:
                self.user_preferences[user_id] = preferences.get("preferences", {})
                return preferences.get("preferences", {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            return {}
    
    async def _get_learning_path(self, user_id: str) -> Dict[str, Any]:
        """Get user's learning path"""
        try:
            if user_id in self.user_learning_paths:
                return self.user_learning_paths[user_id]
            
            # Load from database
            learning_path = await self.collection.find_one({
                "user_id": user_id,
                "type": "learning_path"
            })
            
            if learning_path:
                self.user_learning_paths[user_id] = learning_path.get("learning_path", {})
                return learning_path.get("learning_path", {})
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting learning path: {e}")
            return {}
    
    async def _generate_personalized_recommendations(self, 
                                                   user_profile: Dict[str, Any],
                                                   user_preferences: Dict[str, Any],
                                                   learning_path: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized recommendations"""
        try:
            recommendations = {
                "trading_strategies": [],
                "learning_materials": [],
                "risk_recommendations": [],
                "timing_suggestions": [],
                "skill_improvements": []
            }
            
            # Trading strategy recommendations
            trading_preferences = user_profile.get("patterns", {}).get("trading_preferences", {})
            if trading_preferences:
                recommendations["trading_strategies"] = self._recommend_trading_strategies(trading_preferences)
            
            # Learning material recommendations
            if learning_path:
                recommendations["learning_materials"] = self._recommend_learning_materials(learning_path)
            
            # Risk recommendations
            risk_tolerance = trading_preferences.get("risk_tolerance", "medium")
            recommendations["risk_recommendations"] = self._recommend_risk_strategies(risk_tolerance)
            
            # Timing suggestions
            timing_patterns = user_profile.get("patterns", {}).get("timing_patterns", {})
            if timing_patterns:
                recommendations["timing_suggestions"] = self._recommend_timing_strategies(timing_patterns)
            
            # Skill improvement recommendations
            skill_level = learning_path.get("skill_level", "beginner")
            recommendations["skill_improvements"] = self._recommend_skill_improvements(skill_level)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating personalized recommendations: {e}")
            return {}
    
    def _recommend_trading_strategies(self, trading_preferences: Dict[str, Any]) -> List[str]:
        """Recommend trading strategies based on preferences"""
        try:
            strategies = []
            
            risk_tolerance = trading_preferences.get("risk_tolerance", "medium")
            trading_frequency = trading_preferences.get("trading_frequency", "medium")
            
            if risk_tolerance == "low":
                strategies.extend(["DCA Strategy", "Conservative Trend Following", "Dividend Investing"])
            elif risk_tolerance == "high":
                strategies.extend(["Momentum Trading", "Swing Trading", "Scalping"])
            else:
                strategies.extend(["Balanced Portfolio", "Mean Reversion", "Breakout Trading"])
            
            if trading_frequency == "high":
                strategies.extend(["Day Trading", "Scalping", "High-Frequency Strategies"])
            elif trading_frequency == "low":
                strategies.extend(["Long-term Investing", "Position Trading", "Buy and Hold"])
            
            return strategies[:5]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error recommending trading strategies: {e}")
            return []
    
    def _recommend_learning_materials(self, learning_path: Dict[str, Any]) -> List[str]:
        """Recommend learning materials based on learning path"""
        try:
            materials = []
            
            skill_level = learning_path.get("skill_level", "beginner")
            next_topics = learning_path.get("next_topics", [])
            
            if skill_level == "beginner":
                materials.extend(["Trading Basics", "Risk Management 101", "Chart Reading Guide"])
            elif skill_level == "intermediate":
                materials.extend(["Advanced Technical Analysis", "Strategy Development", "Market Psychology"])
            elif skill_level == "advanced":
                materials.extend(["Quantitative Analysis", "Algorithmic Trading", "Portfolio Theory"])
            else:
                materials.extend(["Research Methods", "Model Development", "Market Microstructure"])
            
            # Add topic-specific materials
            for topic in next_topics:
                if topic == "basic_analysis":
                    materials.append("Technical Analysis Fundamentals")
                elif topic == "risk_management":
                    materials.append("Risk Management Strategies")
                elif topic == "strategy_development":
                    materials.append("Strategy Development Guide")
            
            return materials[:5]  # Top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error recommending learning materials: {e}")
            return []
    
    def _recommend_risk_strategies(self, risk_tolerance: str) -> List[str]:
        """Recommend risk strategies based on tolerance"""
        try:
            if risk_tolerance == "low":
                return ["2% Rule", "Stop Loss at 1%", "Diversification", "Conservative Position Sizing"]
            elif risk_tolerance == "high":
                return ["5% Rule", "Trailing Stop Loss", "Concentrated Positions", "Aggressive Position Sizing"]
            else:
                return ["3% Rule", "Stop Loss at 2%", "Balanced Portfolio", "Moderate Position Sizing"]
                
        except Exception as e:
            logger.error(f"Error recommending risk strategies: {e}")
            return []
    
    def _recommend_timing_strategies(self, timing_patterns: Dict[str, Any]) -> List[str]:
        """Recommend timing strategies based on patterns"""
        try:
            suggestions = []
            
            preferred_hours = timing_patterns.get("preferred_hours", [])
            activity_level = timing_patterns.get("activity_level", 1)
            
            if activity_level > 1:
                suggestions.append("High-frequency trading strategies")
            else:
                suggestions.append("Long-term position strategies")
            
            if preferred_hours:
                suggestions.append(f"Focus trading during hours: {preferred_hours}")
            
            suggestions.extend(["Market open strategies", "End-of-day strategies", "News-based timing"])
            
            return suggestions[:5]
            
        except Exception as e:
            logger.error(f"Error recommending timing strategies: {e}")
            return []
    
    def _recommend_skill_improvements(self, skill_level: str) -> List[str]:
        """Recommend skill improvements based on level"""
        try:
            if skill_level == "beginner":
                return ["Learn basic chart patterns", "Understand risk management", "Practice with paper trading"]
            elif skill_level == "intermediate":
                return ["Develop trading strategies", "Learn advanced indicators", "Improve market timing"]
            elif skill_level == "advanced":
                return ["Quantitative analysis", "Algorithm development", "Portfolio optimization"]
            else:
                return ["Research methodologies", "Model development", "Mentoring others"]
                
        except Exception as e:
            logger.error(f"Error recommending skill improvements: {e}")
            return []
    
    async def get_personalization_insights(self) -> Dict[str, Any]:
        """Get personalization insights"""
        try:
            if not self.collection:
                await self.initialize()
            
            # Get user statistics
            total_users = await self.collection.count_documents({"type": "user_profile"})
            total_interactions = await self.collection.count_documents({"type": "user_interaction"})
            
            # Get skill level distribution
            skill_levels = await self.collection.find({"type": "learning_path"}).to_list(length=1000)
            skill_distribution = {}
            for path in skill_levels:
                skill_level = path.get("skill_level", "beginner")
                skill_distribution[skill_level] = skill_distribution.get(skill_level, 0) + 1
            
            # Get preference patterns
            preferences = await self.collection.find({"type": "user_preferences"}).to_list(length=1000)
            common_preferences = {}
            for pref in preferences:
                user_prefs = pref.get("preferences", {})
                for key, value in user_prefs.items():
                    if key not in common_preferences:
                        common_preferences[key] = {}
                    if isinstance(value, list):
                        for item in value:
                            common_preferences[key][item] = common_preferences[key].get(item, 0) + 1
                    else:
                        common_preferences[key][value] = common_preferences[key].get(value, 0) + 1
            
            return {
                "total_users": total_users,
                "total_interactions": total_interactions,
                "skill_distribution": skill_distribution,
                "common_preferences": common_preferences,
                "personalization_health": self._calculate_personalization_health(total_users, total_interactions),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting personalization insights: {e}")
            return {"error": str(e)}
    
    def _calculate_personalization_health(self, total_users: int, total_interactions: int) -> str:
        """Calculate personalization health score"""
        try:
            if total_users == 0:
                return "poor"
            
            interactions_per_user = total_interactions / total_users
            
            if interactions_per_user >= 50:
                return "excellent"
            elif interactions_per_user >= 20:
                return "good"
            elif interactions_per_user >= 10:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            logger.error(f"Error calculating personalization health: {e}")
            return "unknown"

# Global instance
personalized_learning_system = PersonalizedLearningSystem()
