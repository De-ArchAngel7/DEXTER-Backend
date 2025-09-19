from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
import os

# Import services
from app.services.error_monitoring import error_monitor
from app.services.monitoring_dashboard import monitoring_dashboard

# Import API routes
from app.api.v1.api import api_router

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DEXTER AI Trading Bot",
    description="Advanced AI-powered cryptocurrency trading platform with unified conversation engine",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure this properly for production
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log request with timing
    logger.info(
        "Request processed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Capture exception in Sentry
    error_monitor.capture_exception(
        exc,
        message_type="global_exception",
        url=str(request.url),
        method=request.method
    )
    
    logger.error(
        "Unhandled exception",
        exception=str(exc),
        url=str(request.url),
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check system health
        system_metrics = await monitoring_dashboard._get_system_metrics()
        
        # Determine health status
        health_status = "healthy"
        if system_metrics.get("cpu_percent", 0) > 90:
            health_status = "critical"
        elif system_metrics.get("memory_percent", 0) > 85:
            health_status = "warning"
        
        return {
            "status": health_status,
            "timestamp": time.time(),
            "version": "2.0.0",
            "system": system_metrics
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "DEXTER AI Trading Bot API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Unified Conversation Engine",
            "DialoGPT AI Integration",
            "Advanced Learning Systems",
            "Real-time Trading",
            "Admin Panel",
            "Error Monitoring",
            "Performance Dashboard"
        ]
    }

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting DEXTER AI Trading Bot...")
    
    # Initialize error monitoring
    if error_monitor.sentry_dsn:
        logger.info("✅ Error monitoring initialized")
    else:
        logger.warning("⚠️ Error monitoring not configured")
    
    # Initialize monitoring dashboard
    logger.info("✅ Monitoring dashboard initialized")
    
    # Log startup completion
    logger.info("🎯 DEXTER AI Trading Bot started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down DEXTER AI Trading Bot...")
    
    # Close any open connections
    # Add cleanup logic here if needed
    
    logger.info("✅ DEXTER AI Trading Bot shutdown complete")

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )