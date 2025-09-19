#!/usr/bin/env python3
"""
📝 DEXTER CONVERSATION LOGGER
============================================================
MongoDB logging service for conversations
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import structlog
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure
import os

logger = structlog.get_logger()

class ConversationLogger:
    """
    MongoDB logging service for conversations
    """
    
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self.connected = False
        
        # MongoDB connection settings
        self.mongodb_url = os.getenv("MONGODB_URL", "mongodb+srv://heylelyaka:q8YsHRxgCBPb3fvb@cluster0.oedhlx5.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
        self.database_name = os.getenv("DATABASE_NAME", "dexter")
        self.collection_name = "conversations"
        
        # Initialize connection lazily
        self._initialized = False
    
    async def _initialize_connection(self):
        """Initialize MongoDB connection"""
        try:
            self.client = AsyncIOMotorClient(self.mongodb_url)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test connection
            await self.client.admin.command('ping')
            self.connected = True
            
            logger.info(f"Connected to MongoDB: {self.database_name}")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            self.connected = False
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {e}")
            self.connected = False
    
    async def _ensure_connected(self):
        """Ensure MongoDB connection is initialized"""
        if not self._initialized:
            await self._initialize_connection()
            self._initialized = True

    async def log_conversation(
        self,
        user_id: str,
        message: str,
        reply: str,
        model_used: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a conversation to MongoDB
        
        Args:
            user_id: User identifier
            message: User message
            reply: AI response
            model_used: Which model was used
            source: Source of the message
            metadata: Additional metadata
        """
        await self._ensure_connected()
        
        if not self.connected:
            logger.warning("MongoDB not connected, skipping conversation log")
            return
        
        try:
            conversation_doc = {
                "user_id": user_id,
                "timestamp": datetime.now(),
                "message": message,
                "reply": reply,
                "model_used": model_used,
                "source": source,
                "metadata": metadata or {},
                "created_at": datetime.now()
            }
            
            result = await self.collection.insert_one(conversation_doc)
            logger.info(f"Logged conversation for user {user_id} with model {model_used}")
            
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    async def get_user_conversations(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user
        
        Args:
            user_id: User identifier
            limit: Maximum number of conversations to return
            skip: Number of conversations to skip
        
        Returns:
            List of conversation documents
        """
        if not self.connected:
            logger.warning("MongoDB not connected, returning empty list")
            return []
        
        try:
            cursor = self.collection.find(
                {"user_id": user_id}
            ).sort("timestamp", -1).skip(skip).limit(limit)
            
            conversations = []
            async for doc in cursor:
                # Convert ObjectId to string for JSON serialization
                doc["_id"] = str(doc["_id"])
                conversations.append(doc)
            
            return conversations
            
        except Exception as e:
            logger.error(f"Error getting user conversations: {e}")
            return []
    
    async def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics
        
        Returns:
            Dictionary with conversation statistics
        """
        if not self.connected:
            return {"error": "MongoDB not connected"}
        
        try:
            # Total conversations
            total_conversations = await self.collection.count_documents({})
            
            # Unique users
            unique_users = len(await self.collection.distinct("user_id"))
            
            # Model usage stats
            model_stats = {}
            async for doc in self.collection.aggregate([
                {"$group": {"_id": "$model_used", "count": {"$sum": 1}}}
            ]):
                model_stats[doc["_id"]] = doc["count"]
            
            # Source stats
            source_stats = {}
            async for doc in self.collection.aggregate([
                {"$group": {"_id": "$source", "count": {"$sum": 1}}}
            ]):
                source_stats[doc["_id"]] = doc["count"]
            
            return {
                "total_conversations": total_conversations,
                "unique_users": unique_users,
                "model_usage": model_stats,
                "source_usage": source_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_conversations(self, days: int = 30):
        """
        Clean up conversations older than specified days
        
        Args:
            days: Number of days to keep conversations
        """
        if not self.connected:
            logger.warning("MongoDB not connected, skipping cleanup")
            return
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            result = await self.collection.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            logger.info(f"Cleaned up {result.deleted_count} old conversations")
            
        except Exception as e:
            logger.error(f"Error cleaning up old conversations: {e}")
    
    async def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("MongoDB connection closed")

# Global instance
conversation_logger = ConversationLogger()
