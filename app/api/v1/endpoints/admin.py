from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any
import structlog
from app.services.admin_auth import get_admin_user
from app.services.monitoring_dashboard import monitoring_dashboard
from app.services.error_monitoring import error_monitor

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/dashboard/metrics")
async def get_dashboard_metrics(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Get comprehensive trading dashboard metrics (Admin only)"""
    try:
        metrics = await monitoring_dashboard.get_trading_metrics()
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error getting dashboard metrics: {e}")
        error_monitor.capture_exception(e, message_type="admin_dashboard")
        raise HTTPException(status_code=500, detail="Failed to get dashboard metrics")

@router.post("/dashboard/refresh")
async def refresh_dashboard_metrics(
    background_tasks: BackgroundTasks,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Force refresh dashboard metrics (Admin only)"""
    try:
        # Clear cache and refresh
        monitoring_dashboard.metrics_cache = {}
        monitoring_dashboard.last_update = None
        
        # Get fresh metrics
        metrics = await monitoring_dashboard.get_trading_metrics()
        
        # Send to Grafana in background
        background_tasks.add_task(monitoring_dashboard.send_metrics_to_grafana, metrics)
        
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "message": "Dashboard metrics refreshed",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error refreshing dashboard metrics: {e}")
        error_monitor.capture_exception(e, message_type="admin_refresh")
        raise HTTPException(status_code=500, detail="Failed to refresh dashboard metrics")

@router.get("/system/health")
async def get_system_health(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Get system health status (Admin only)"""
    try:
        # Get system metrics
        system_metrics = await monitoring_dashboard._get_system_metrics()
        
        # Determine health status
        health_status = "healthy"
        issues = []
        
        cpu_percent = system_metrics.get("cpu_percent", 0)
        memory_percent = system_metrics.get("memory_percent", 0)
        disk_percent = system_metrics.get("disk_percent", 0)
        
        if isinstance(cpu_percent, (int, float)) and cpu_percent > 80:
            health_status = "warning"
            issues.append("High CPU usage")
        
        if isinstance(memory_percent, (int, float)) and memory_percent > 85:
            health_status = "warning"
            issues.append("High memory usage")
        
        if isinstance(disk_percent, (int, float)) and disk_percent > 90:
            health_status = "critical"
            issues.append("Low disk space")
        
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "health": {
                "status": health_status,
                "issues": issues,
                "metrics": system_metrics
            }
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        error_monitor.capture_exception(e, message_type="admin_health")
        raise HTTPException(status_code=500, detail="Failed to get system health")

@router.get("/ai/models/status")
async def get_ai_models_status(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Get AI models status and performance (Admin only)"""
    try:
        # Import AI fusion engine
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        from ai_module.ai_fusion_engine import ai_fusion_engine
        
        # Get AI system status
        ai_status = await ai_fusion_engine.get_ai_system_status()
        
        # Get model performance metrics
        metrics = await monitoring_dashboard.get_trading_metrics()
        ai_metrics = metrics.get("ai", {})
        
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "ai_status": ai_status,
            "performance": ai_metrics
        }
    except Exception as e:
        logger.error(f"Error getting AI models status: {e}")
        error_monitor.capture_exception(e, message_type="admin_ai_status")
        raise HTTPException(status_code=500, detail="Failed to get AI models status")

@router.post("/ai/models/retrain")
async def trigger_ai_retraining(
    background_tasks: BackgroundTasks,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Trigger AI model retraining (Admin only)"""
    try:
        # Import model retraining system
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
        from ai_module.model_retraining import model_retraining_system
        
        # Start retraining in background
        background_tasks.add_task(model_retraining_system.retrain_models)
        
        # Log admin action
        error_monitor.capture_message(
            f"Admin {admin_user.get('email')} triggered AI model retraining",
            level="info",
            message_type="admin_action"
        )
        
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "message": "AI model retraining started in background"
        }
    except Exception as e:
        logger.error(f"Error triggering AI retraining: {e}")
        error_monitor.capture_exception(e, message_type="admin_retrain")
        raise HTTPException(status_code=500, detail="Failed to trigger AI retraining")

@router.get("/trading/performance")
async def get_trading_performance(admin_user: Dict[str, Any] = Depends(get_admin_user)):
    """Get detailed trading performance analysis (Admin only)"""
    try:
        metrics = await monitoring_dashboard.get_trading_metrics()
        trading_metrics = metrics.get("trading", {})
        
        # Calculate additional performance metrics
        total_profit = trading_metrics.get("total_profit", 0)
        total_trades = trading_metrics.get("total_trades", 0)
        win_rate = trading_metrics.get("win_rate", 0)
        
        # Performance analysis
        performance_analysis = {
            "profitability": "profitable" if total_profit > 0 else "loss",
            "efficiency": "high" if win_rate > 60 else "medium" if win_rate > 40 else "low",
            "activity": "high" if total_trades > 100 else "medium" if total_trades > 50 else "low",
            "recommendations": []
        }
        
        # Generate recommendations
        if win_rate < 40:
            performance_analysis["recommendations"].append("Consider adjusting AI parameters for better accuracy")
        if total_trades < 50:
            performance_analysis["recommendations"].append("Increase trading activity for better data collection")
        if total_profit < 0:
            performance_analysis["recommendations"].append("Review risk management settings")
        
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "trading_metrics": trading_metrics,
            "analysis": performance_analysis
        }
    except Exception as e:
        logger.error(f"Error getting trading performance: {e}")
        error_monitor.capture_exception(e, message_type="admin_trading_performance")
        raise HTTPException(status_code=500, detail="Failed to get trading performance")

@router.get("/logs/recent")
async def get_recent_logs(
    limit: int = 100,
    admin_user: Dict[str, Any] = Depends(get_admin_user)
):
    """Get recent system logs (Admin only)"""
    try:
        # This would typically connect to your logging system
        # For now, return a placeholder
        return {
            "status": "success",
            "admin": admin_user.get("email"),
            "message": "Log retrieval not implemented yet",
            "limit": limit
        }
    except Exception as e:
        logger.error(f"Error getting recent logs: {e}")
        error_monitor.capture_exception(e, message_type="admin_logs")
        raise HTTPException(status_code=500, detail="Failed to get recent logs")
