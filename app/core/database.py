import motor.motor_asyncio
from app.core.config import settings
import structlog

logger = structlog.get_logger()

# Database client
client = None
database = None

async def connect_to_mongo():
    """Connect to MongoDB"""
    global client, database
    try:
        # For development, try to connect but don't fail if MongoDB is not available
        client = motor.motor_asyncio.AsyncIOMotorClient(settings.DATABASE_URL)
        database = client.dexter
        await client.admin.command('ping')
        logger.info("Connected to MongoDB")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {e}")
        logger.info("Running in development mode without database")
        # Create mock database objects for development
        client = None
        database = None

async def close_mongo_connection():
    """Close MongoDB connection"""
    global client
    if client:
        client.close()
        logger.info("Closed MongoDB connection")

def get_database():
    """Get database instance"""
    return database

def get_collection(collection_name: str):
    """Get collection from database"""
    if database:
        return database[collection_name]
    else:
        # Return mock collection for development
        return MockCollection()

class MockCollection:
    """Mock collection for development when MongoDB is not available"""
    
    async def find_one(self, *args, **kwargs):
        return None
    
    async def find(self, *args, **kwargs):
        return []
    
    async def insert_one(self, *args, **kwargs):
        return MockInsertResult()
    
    async def update_one(self, *args, **kwargs):
        return MockUpdateResult()
    
    async def delete_one(self, *args, **kwargs):
        return MockDeleteResult()

class MockInsertResult:
    inserted_id = "mock_id"

class MockUpdateResult:
    modified_count = 1
    matched_count = 1

class MockDeleteResult:
    deleted_count = 1
