import os
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
import structlog

logger = structlog.get_logger(__name__)

class ErrorMonitoringService:
    def __init__(self):
        self.sentry_dsn = os.getenv("SENTRY_DSN")
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.release = os.getenv("RELEASE_VERSION", "1.0.0")
        
        if self.sentry_dsn:
            self._initialize_sentry()
        else:
            logger.warning("Sentry DSN not configured - error monitoring disabled")
    
    def _initialize_sentry(self):
        """Initialize Sentry for error monitoring"""
        try:
            sentry_sdk.init(
                dsn=self.sentry_dsn,
                environment=self.environment,
                release=self.release,
                integrations=[
                    FastApiIntegration(auto_enabling_instrumentations=True),
                    StarletteIntegration(auto_enabling_instrumentations=True),
                    HttpxIntegration(auto_enabling_instrumentations=True),
                ],
                traces_sample_rate=0.1,  # 10% of transactions
                profiles_sample_rate=0.1,  # 10% of profiles
                send_default_pii=False,  # Don't send personal info
                attach_stacktrace=True,
                before_send=self._before_send,
            )
            logger.info("✅ Sentry error monitoring initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Sentry: {e}")
    
    def _before_send(self, event, hint):
        """Filter and modify events before sending to Sentry"""
        # Add custom tags for trading context
        if event.get("tags"):
            event["tags"]["service"] = "dexter-trading-bot"
            event["tags"]["component"] = "ai-trading"
        
        # Add user context if available
        if "user" in event:
            # Don't send sensitive trading data
            if "api_key" in str(event["user"]):
                return None
        
        return event
    
    def capture_exception(self, exception: Exception, **kwargs):
        """Capture an exception with trading context"""
        if self.sentry_dsn:
            with sentry_sdk.push_scope() as scope:
                # Add trading-specific context
                scope.set_tag("trading_bot", True)
                scope.set_tag("ai_model", kwargs.get("ai_model", "unknown"))
                scope.set_tag("exchange", kwargs.get("exchange", "unknown"))
                scope.set_tag("trading_pair", kwargs.get("trading_pair", "unknown"))
                
                # Add custom data
                if "trade_data" in kwargs:
                    scope.set_extra("trade_data", kwargs["trade_data"])
                if "ai_prediction" in kwargs:
                    scope.set_extra("ai_prediction", kwargs["ai_prediction"])
                
                sentry_sdk.capture_exception(exception)
    
    def capture_message(self, message: str, level: str = "info", **kwargs):
        """Capture a custom message with trading context"""
        if self.sentry_dsn:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("trading_bot", True)
                scope.set_tag("message_type", kwargs.get("message_type", "general"))
                
                if "trading_pair" in kwargs:
                    scope.set_tag("trading_pair", kwargs["trading_pair"])
                if "profit_loss" in kwargs:
                    scope.set_extra("profit_loss", kwargs["profit_loss"])
                
                sentry_sdk.capture_message(message, level=level)
    
    def add_breadcrumb(self, message: str, category: str = "trading", **kwargs):
        """Add a breadcrumb for debugging"""
        if self.sentry_dsn:
            sentry_sdk.add_breadcrumb(
                message=message,
                category=category,
                level="info",
                data=kwargs
            )

# Global instance
error_monitor = ErrorMonitoringService()
