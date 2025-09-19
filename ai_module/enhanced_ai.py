import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import structlog
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = structlog.get_logger()

class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "BULL_MARKET"
    BEAR_MARKET = "BEAR_MARKET"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING = "TRENDING"
    MEAN_REVERTING = "MEAN_REVERTING"

class SentimentSource(Enum):
    """Sentiment analysis sources"""
    NEWS = "NEWS"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    REDDIT = "REDDIT"
    TWITTER = "TWITTER"
    TELEGRAM = "TELEGRAM"
    FORUMS = "FORUMS"

@dataclass
class MarketRegimeInfo:
    """Information about current market regime"""
    regime: MarketRegime
    confidence: float
    duration: timedelta
    volatility: float
    trend_strength: float
    mean_reversion_strength: float
    timestamp: datetime

@dataclass
class EnhancedSentiment:
    """Enhanced sentiment analysis result"""
    overall_sentiment: float  # -1 to 1 (negative to positive)
    confidence: float
    sources: Dict[str, float]
    keywords: List[str]
    timestamp: datetime
    market_impact: float

class MarketRegimeDetector:
    """Detects and analyzes market regimes"""
    
    def __init__(self):
        self.regime_history: List[MarketRegimeInfo] = []
        self.volatility_threshold = 0.05  # 5% volatility threshold
        self.trend_threshold = 0.1  # 10% trend strength threshold
        
    def detect_regime(self, price_data: pd.Series) -> MarketRegimeInfo:
        """Detect current market regime from price data"""
        
        try:
            # Calculate basic metrics
            returns = price_data.pct_change().dropna()
            volatility = float(returns.std())
            
            # Calculate trend strength
            price_range = price_data.max() - price_data.min()
            avg_price = price_data.mean()
            trend_strength = float(price_range / avg_price if avg_price > 0 else 0)
            
            # Calculate mean reversion strength
            autocorr = returns.autocorr() if len(returns) > 1 else 0
            mean_reversion_strength = float(abs(autocorr) if not pd.isna(autocorr) else 0)
            
            # Determine regime
            if volatility > self.volatility_threshold:
                if trend_strength > self.trend_threshold:
                    regime = MarketRegime.TRENDING
                else:
                    regime = MarketRegime.HIGH_VOLATILITY
            elif trend_strength > self.trend_threshold:
                if price_data.iloc[-1] > price_data.iloc[0]:
                    regime = MarketRegime.BULL_MARKET
                else:
                    regime = MarketRegime.BEAR_MARKET
            elif mean_reversion_strength > 0.3:
                regime = MarketRegime.MEAN_REVERTING
            else:
                regime = MarketRegime.SIDEWAYS
                
            # Calculate confidence based on regime clarity
            confidence = min(100, (volatility + trend_strength + mean_reversion_strength) * 100)
            
            # Create regime info
            regime_info = MarketRegimeInfo(
                regime=regime,
                confidence=confidence,
                duration=timedelta(days=len(price_data)),
                volatility=volatility,
                trend_strength=trend_strength,
                mean_reversion_strength=mean_reversion_strength,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.regime_history.append(regime_info)
            
            logger.info(f"🔍 Market regime detected: {regime.value} (confidence: {confidence:.1f}%)")
            
            return regime_info
            
        except Exception as e:
            logger.error(f"❌ Error detecting market regime: {e}")
            # Return default regime
            return MarketRegimeInfo(
                regime=MarketRegime.SIDEWAYS,
                confidence=0.0,
                duration=timedelta(days=1),
                volatility=0.0,
                trend_strength=0.0,
                mean_reversion_strength=0.0,
                timestamp=datetime.now()
            )
            
    def get_regime_transitions(self, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Get recent regime transitions"""
        if len(self.regime_history) < 2:
            return []
            
        transitions = []
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        
        for i in range(1, len(self.regime_history)):
            prev_regime = self.regime_history[i-1]
            curr_regime = self.regime_history[i]
            
            if curr_regime.timestamp >= cutoff_time and prev_regime.regime != curr_regime.regime:
                transitions.append({
                    'from_regime': prev_regime.regime.value,
                    'to_regime': curr_regime.regime.value,
                    'timestamp': curr_regime.timestamp.isoformat(),
                    'confidence': curr_regime.confidence
                })
                
        return transitions

class EnhancedAIFusionEngine:
    """Enhanced AI fusion engine with advanced capabilities"""
    
    def __init__(self):
        self.regime_detector = MarketRegimeDetector()
        
    async def generate_enhanced_insight(self, symbol: str, 
                                      price_data: pd.Series) -> Dict[str, Any]:
        """Generate enhanced trading insight with market regime detection"""
        
        try:
            # Detect market regime
            regime_info = self.regime_detector.detect_regime(price_data)
            
            # Get regime transitions
            regime_transitions = self.regime_detector.get_regime_transitions()
            
            # Create enhanced insight
            enhanced_insight = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_regime': {
                    'current_regime': regime_info.regime.value,
                    'confidence': regime_info.confidence,
                    'volatility': regime_info.volatility,
                    'trend_strength': regime_info.trend_strength,
                    'mean_reversion_strength': regime_info.mean_reversion_strength
                },
                'regime_transitions': regime_transitions,
                'enhanced_features': {
                    'regime_aware': True,
                    'regime_transition_tracking': True
                }
            }
            
            logger.info(f"🚀 Enhanced AI insight generated for {symbol}")
            
            return enhanced_insight
            
        except Exception as e:
            logger.error(f"❌ Error generating enhanced insight: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
    def get_ai_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of AI capabilities"""
        return {
            'market_regime_detection': {
                'regimes_detected': len(self.regime_detector.regime_history),
                'current_regime': self.regime_detector.regime_history[-1].regime.value if self.regime_detector.regime_history else 'UNKNOWN'
            },
            'enhanced_features': [
                'Market Regime Detection',
                'Regime Transition Tracking',
                'Volatility Analysis',
                'Trend Strength Analysis',
                'Mean Reversion Detection'
            ]
        }
