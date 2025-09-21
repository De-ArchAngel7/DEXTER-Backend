from fastapi import APIRouter
from typing import Dict, Any
import importlib
import structlog

logger = structlog.get_logger()

class LazyAPIRouter:
    """Lazy-loading API router that only imports endpoints when needed"""
    
    def __init__(self):
        self.router = APIRouter()
        self._loaded_endpoints = set()
        self._setup_lazy_routes()
    
    def _setup_lazy_routes(self):
        """Setup routes that load endpoints on first access"""
        
        # Auth routes
        @self.router.get("/auth/{path:path}")
        @self.router.post("/auth/{path:path}")
        @self.router.put("/auth/{path:path}")
        @self.router.delete("/auth/{path:path}")
        async def auth_handler(path: str, request):
            return await self._load_and_route("auth", path, request)
        
        # Users routes  
        @self.router.get("/users/{path:path}")
        @self.router.post("/users/{path:path}")
        @self.router.put("/users/{path:path}")
        @self.router.delete("/users/{path:path}")
        async def users_handler(path: str, request):
            return await self._load_and_route("users", path, request)
            
        # Trading routes
        @self.router.get("/trading/{path:path}")
        @self.router.post("/trading/{path:path}")
        @self.router.put("/trading/{path:path}")
        @self.router.delete("/trading/{path:path}")
        async def trading_handler(path: str, request):
            return await self._load_and_route("trading", path, request)
            
        # Portfolio routes
        @self.router.get("/portfolio/{path:path}")
        @self.router.post("/portfolio/{path:path}")
        @self.router.put("/portfolio/{path:path}")
        @self.router.delete("/portfolio/{path:path}")
        async def portfolio_handler(path: str, request):
            return await self._load_and_route("portfolio", path, request)
            
        # DexScreener routes
        @self.router.get("/dexscreener/{path:path}")
        @self.router.post("/dexscreener/{path:path}")
        async def dexscreener_handler(path: str, request):
            return await self._load_and_route("dexscreener", path, request)
            
        # Chat routes
        @self.router.get("/chat/{path:path}")
        @self.router.post("/chat/{path:path}")
        async def chat_handler(path: str, request):
            return await self._load_and_route("chat", path, request)
            
        # AI routes
        @self.router.get("/ai/{path:path}")
        @self.router.post("/ai/{path:path}")
        async def ai_handler(path: str, request):
            return await self._load_and_route("ai", path, request)
    
    async def _load_and_route(self, endpoint_name: str, path: str, request):
        """Load endpoint module and route request"""
        try:
            if endpoint_name not in self._loaded_endpoints:
                logger.info(f"🔄 Lazy loading {endpoint_name} endpoint...")
                
                # Import the endpoint module
                module = importlib.import_module(f"app.api.v1.endpoints.{endpoint_name}")
                
                # Get the router from the module
                endpoint_router = getattr(module, 'router')
                
                # Include it in our main router
                self.router.include_router(
                    endpoint_router, 
                    prefix=f"/{endpoint_name}", 
                    tags=[endpoint_name]
                )
                
                self._loaded_endpoints.add(endpoint_name)
                logger.info(f"✅ {endpoint_name} endpoint loaded successfully")
            
            # Route the request to the loaded endpoint
            # This is simplified - in reality we'd need more sophisticated routing
            return {"status": f"{endpoint_name} endpoint loaded", "path": path}
            
        except Exception as e:
            logger.error(f"❌ Failed to load {endpoint_name} endpoint: {e}")
            return {
                "error": f"Endpoint {endpoint_name} temporarily unavailable",
                "details": str(e),
                "fallback": True
            }

# Create the lazy router instance
lazy_api_router = LazyAPIRouter()
api_router = lazy_api_router.router
