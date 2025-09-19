import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import structlog
from dataclasses import dataclass
import json
import os

logger = structlog.get_logger()

@dataclass
class DashboardMetric:
    """Dashboard metric data"""
    name: str
    value: float
    unit: str
    change: float
    change_percent: float
    trend: str  # 'up', 'down', 'stable'
    timestamp: datetime

@dataclass
class PerformanceChart:
    """Performance chart data"""
    title: str
    chart_type: str  # 'line', 'bar', 'candlestick', 'heatmap'
    data: Dict[str, Any]
    layout: Dict[str, Any]

class PerformanceAnalyzer:
    """Analyzes trading performance and generates insights"""
    
    def __init__(self):
        self.performance_history: List[Dict[str, Any]] = []
        
    def add_performance_data(self, data: Dict[str, Any]):
        """Add performance data point"""
        self.performance_history.append(data)
        
    def calculate_metrics(self) -> Dict[str, DashboardMetric]:
        """Calculate key performance metrics"""
        if not self.performance_history:
            return {}
            
        try:
            # Get latest data
            latest = self.performance_history[-1]
            previous = self.performance_history[-2] if len(self.performance_history) > 1 else latest
            
            # Calculate metrics
            total_value = latest.get('total_value', 0)
            total_pnl = latest.get('total_pnl', 0)
            total_pnl_percent = latest.get('total_pnl_percentage', 0)
            
            # Calculate changes
            value_change = total_value - previous.get('total_value', total_value)
            value_change_percent = (value_change / previous.get('total_value', total_value)) * 100 if previous.get('total_value', 0) > 0 else 0
            
            pnl_change = total_pnl - previous.get('total_pnl', 0)
            pnl_change_percent = (pnl_change / previous.get('total_pnl', 1)) * 100 if previous.get('total_pnl', 0) != 0 else 0
            
            # Determine trends
            value_trend = 'up' if value_change > 0 else 'down' if value_change < 0 else 'stable'
            pnl_trend = 'up' if pnl_change > 0 else 'down' if pnl_change < 0 else 'stable'
            
            metrics = {
                'portfolio_value': DashboardMetric(
                    name="Portfolio Value",
                    value=total_value,
                    unit="USD",
                    change=value_change,
                    change_percent=value_change_percent,
                    trend=value_trend,
                    timestamp=datetime.now()
                ),
                'total_pnl': DashboardMetric(
                    name="Total PnL",
                    value=total_pnl,
                    unit="USD",
                    change=pnl_change,
                    change_percent=pnl_change_percent,
                    trend=pnl_trend,
                    timestamp=datetime.now()
                ),
                'pnl_percentage': DashboardMetric(
                    name="PnL %",
                    value=total_pnl_percent,
                    unit="%",
                    change=pnl_change_percent,
                    change_percent=pnl_change_percent,
                    trend=pnl_trend,
                    timestamp=datetime.now()
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            return {}
            
    def generate_performance_chart(self, chart_type: str = "portfolio_value") -> PerformanceChart:
        """Generate performance chart"""
        try:
            if not self.performance_history:
                return PerformanceChart(
                    title="No Data Available",
                    chart_type="line",
                    data={},
                    layout={}
                )
                
            # Extract data for chart
            timestamps = [pd.to_datetime(data.get('timestamp', '')) for data in self.performance_history]
            
            if chart_type == "portfolio_value":
                values = [data.get('total_value', 0) for data in self.performance_history]
                title = "Portfolio Value Over Time"
                y_label = "Portfolio Value (USD)"
                
            elif chart_type == "pnl":
                values = [data.get('total_pnl', 0) for data in self.performance_history]
                title = "Total PnL Over Time"
                y_label = "PnL (USD)"
                
            elif chart_type == "pnl_percentage":
                values = [data.get('total_pnl_percentage', 0) for data in self.performance_history]
                title = "PnL Percentage Over Time"
                y_label = "PnL (%)"
                
            else:
                values = [data.get('total_value', 0) for data in self.performance_history]
                title = "Portfolio Performance"
                y_label = "Value"
            
            # Create chart data
            chart_data = {
                'x': timestamps,
                'y': values,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': title
            }
            
            # Create chart layout
            chart_layout = {
                'title': title,
                'xaxis': {'title': 'Time'},
                'yaxis': {'title': y_label},
                'hovermode': 'x unified'
            }
            
            return PerformanceChart(
                title=title,
                chart_type="line",
                data=chart_data,
                layout=chart_layout
            )
            
        except Exception as e:
            logger.error(f"❌ Error generating performance chart: {e}")
            return PerformanceChart(
                title="Error Generating Chart",
                chart_type="line",
                data={},
                layout={}
            )

class RiskAnalyzer:
    """Analyzes portfolio risk and generates risk metrics"""
    
    def __init__(self):
        self.risk_history: List[Dict[str, Any]] = []
        
    def add_risk_data(self, data: Dict[str, Any]):
        """Add risk data point"""
        self.risk_history.append(data)
        
    def calculate_risk_metrics(self) -> Dict[str, DashboardMetric]:
        """Calculate risk metrics"""
        if not self.risk_history:
            return {}
            
        try:
            latest = self.risk_history[-1]
            previous = self.risk_history[-2] if len(self.risk_history) > 1 else latest
            
            # Extract risk metrics
            volatility = latest.get('volatility', 0)
            sharpe_ratio = latest.get('sharpe_ratio', 0)
            max_drawdown = latest.get('max_drawdown', 0)
            
            # Calculate changes
            vol_change = volatility - previous.get('volatility', volatility)
            sharpe_change = sharpe_ratio - previous.get('sharpe_ratio', sharpe_ratio)
            drawdown_change = max_drawdown - previous.get('max_drawdown', max_drawdown)
            
            # Determine trends
            vol_trend = 'up' if vol_change > 0 else 'down' if vol_change < 0 else 'stable'
            sharpe_trend = 'up' if sharpe_change > 0 else 'down' if sharpe_change < 0 else 'stable'
            drawdown_trend = 'up' if drawdown_change > 0 else 'down' if drawdown_change < 0 else 'stable'
            
            metrics = {
                'volatility': DashboardMetric(
                    name="Portfolio Volatility",
                    value=volatility,
                    unit="%",
                    change=vol_change,
                    change_percent=(vol_change / max(previous.get('volatility', 1), 0.01)) * 100,
                    trend=vol_trend,
                    timestamp=datetime.now()
                ),
                'sharpe_ratio': DashboardMetric(
                    name="Sharpe Ratio",
                    value=sharpe_ratio,
                    unit="",
                    change=sharpe_change,
                    change_percent=(sharpe_change / max(abs(previous.get('sharpe_ratio', 1)), 0.01)) * 100,
                    trend=sharpe_trend,
                    timestamp=datetime.now()
                ),
                'max_drawdown': DashboardMetric(
                    name="Max Drawdown",
                    value=max_drawdown,
                    unit="%",
                    change=drawdown_change,
                    change_percent=(drawdown_change / max(previous.get('max_drawdown', 1), 0.01)) * 100,
                    trend=drawdown_trend,
                    timestamp=datetime.now()
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating risk metrics: {e}")
            return {}

class TradingAnalyzer:
    """Analyzes trading performance and generates trading insights"""
    
    def __init__(self):
        self.trading_history: List[Dict[str, Any]] = []
        
    def add_trading_data(self, data: Dict[str, Any]):
        """Add trading data point"""
        self.trading_history.append(data)
        
    def calculate_trading_metrics(self) -> Dict[str, DashboardMetric]:
        """Calculate trading metrics"""
        if not self.trading_history:
            return {}
            
        try:
            latest = self.trading_history[-1]
            previous = self.trading_history[-2] if len(self.trading_history) > 1 else latest
            
            # Extract trading metrics
            total_trades = latest.get('total_trades', 0)
            win_rate = latest.get('win_rate', 0)
            avg_trade_pnl = latest.get('average_trade_pnl', 0)
            
            # Calculate changes
            trades_change = total_trades - previous.get('total_trades', total_trades)
            win_rate_change = win_rate - previous.get('win_rate', win_rate)
            pnl_change = avg_trade_pnl - previous.get('average_trade_pnl', avg_trade_pnl)
            
            # Determine trends
            trades_trend = 'up' if trades_change > 0 else 'down' if trades_change < 0 else 'stable'
            win_rate_trend = 'up' if win_rate_change > 0 else 'down' if win_rate_change < 0 else 'stable'
            pnl_trend = 'up' if pnl_change > 0 else 'down' if pnl_change < 0 else 'stable'
            
            metrics = {
                'total_trades': DashboardMetric(
                    name="Total Trades",
                    value=total_trades,
                    unit="",
                    change=trades_change,
                    change_percent=(trades_change / max(previous.get('total_trades', 1), 1)) * 100,
                    trend=trades_trend,
                    timestamp=datetime.now()
                ),
                'win_rate': DashboardMetric(
                    name="Win Rate",
                    value=win_rate,
                    unit="%",
                    change=win_rate_change,
                    change_percent=win_rate_change,
                    trend=win_rate_trend,
                    timestamp=datetime.now()
                ),
                'avg_trade_pnl': DashboardMetric(
                    name="Avg Trade PnL",
                    value=avg_trade_pnl,
                    unit="USD",
                    change=pnl_change,
                    change_percent=(pnl_change / max(abs(previous.get('average_trade_pnl', 1)), 0.01)) * 100,
                    trend=pnl_trend,
                    timestamp=datetime.now()
                )
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating trading metrics: {e}")
            return {}

class ProfessionalDashboard:
    """Professional dashboard for comprehensive trading analysis"""
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
        self.trading_analyzer = TradingAnalyzer()
        self.dashboard_data: Dict[str, Any] = {}
        
    def update_dashboard(self, performance_data: Dict[str, Any], 
                        risk_data: Dict[str, Any] = None,
                        trading_data: Dict[str, Any] = None):
        """Update dashboard with new data"""
        try:
            # Update analyzers
            self.performance_analyzer.add_performance_data(performance_data)
            
            if risk_data:
                self.risk_analyzer.add_risk_data(risk_data)
                
            if trading_data:
                self.trading_analyzer.add_trading_data(trading_data)
                
            # Calculate all metrics
            self.dashboard_data = {
                'performance_metrics': self.performance_analyzer.calculate_metrics(),
                'risk_metrics': self.risk_analyzer.calculate_risk_metrics(),
                'trading_metrics': self.trading_analyzer.calculate_trading_metrics(),
                'last_updated': datetime.now().isoformat()
            }
            
            logger.info("✅ Dashboard updated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error updating dashboard: {e}")
            
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive dashboard summary"""
        return {
            'dashboard_data': self.dashboard_data,
            'performance_charts': {
                'portfolio_value': self.performance_analyzer.generate_performance_chart("portfolio_value"),
                'pnl': self.performance_analyzer.generate_performance_chart("pnl"),
                'pnl_percentage': self.performance_analyzer.generate_performance_chart("pnl_percentage")
            },
            'summary_stats': {
                'total_metrics': len(self.dashboard_data.get('performance_metrics', {})) + 
                               len(self.dashboard_data.get('risk_metrics', {})) + 
                               len(self.dashboard_data.get('trading_metrics', {})),
                'data_points': len(self.performance_analyzer.performance_history),
                'last_update': self.dashboard_data.get('last_updated', 'Never')
            }
        }
        
    def export_dashboard_data(self, filename: str = "dashboard_export.json"):
        """Export dashboard data to file"""
        try:
            dashboard_summary = self.get_dashboard_summary()
            
            with open(filename, 'w') as f:
                json.dump(dashboard_summary, f, indent=2, default=str)
                
            logger.info(f"💾 Dashboard data exported to {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting dashboard data: {e}")
            
    def generate_html_report(self, filename: str = "dashboard_report.html"):
        """Generate HTML dashboard report"""
        try:
            dashboard_summary = self.get_dashboard_summary()
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>DEXTER Professional Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .metric {{ border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px; }}
                    .metric.up {{ border-left: 5px solid #28a745; }}
                    .metric.down {{ border-left: 5px solid #dc3545; }}
                    .metric.stable {{ border-left: 5px solid #ffc107; }}
                    .metric-name {{ font-weight: bold; font-size: 18px; }}
                    .metric-value {{ font-size: 24px; color: #007bff; }}
                    .metric-change {{ font-size: 14px; }}
                    .change.positive {{ color: #28a745; }}
                    .change.negative {{ color: #dc3545; }}
                    .change.neutral {{ color: #6c757d; }}
                    .section {{ margin: 20px 0; }}
                    .section-title {{ font-size: 24px; color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
                </style>
            </head>
            <body>
                <h1>🚀 DEXTER Professional Dashboard</h1>
                <p><strong>Last Updated:</strong> {dashboard_summary['summary_stats']['last_update']}</p>
                
                <div class="section">
                    <h2 class="section-title">📊 Performance Metrics</h2>
            """
            
            # Add performance metrics
            for metric_id, metric in dashboard_summary['dashboard_data'].get('performance_metrics', {}).items():
                change_class = 'positive' if metric.change > 0 else 'negative' if metric.change < 0 else 'neutral'
                change_symbol = '+' if metric.change > 0 else ''
                
                html_content += f"""
                    <div class="metric {metric.trend}">
                        <div class="metric-name">{metric.name}</div>
                        <div class="metric-value">{metric.value:,.2f} {metric.unit}</div>
                        <div class="metric-change">
                            <span class="change {change_class}">
                                {change_symbol}{metric.change:,.2f} ({change_symbol}{metric.change_percent:.2f}%)
                            </span>
                        </div>
                    </div>
                """
            
            # Add risk metrics
            html_content += """
                </div>
                <div class="section">
                    <h2 class="section-title">⚠️ Risk Metrics</h2>
            """
            
            for metric_id, metric in dashboard_summary['dashboard_data'].get('risk_metrics', {}).items():
                change_class = 'positive' if metric.change > 0 else 'negative' if metric.change < 0 else 'neutral'
                change_symbol = '+' if metric.change > 0 else ''
                
                html_content += f"""
                    <div class="metric {metric.trend}">
                        <div class="metric-name">{metric.name}</div>
                        <div class="metric-value">{metric.value:,.4f} {metric.unit}</div>
                        <div class="metric-change">
                            <span class="change {change_class}">
                                {change_symbol}{metric.change:,.4f} ({change_symbol}{metric.change_percent:.2f}%)
                            </span>
                        </div>
                    </div>
                """
            
            # Add trading metrics
            html_content += """
                </div>
                <div class="section">
                    <h2 class="section-title">📈 Trading Metrics</h2>
            """
            
            for metric_id, metric in dashboard_summary['dashboard_data'].get('trading_metrics', {}).items():
                change_class = 'positive' if metric.change > 0 else 'negative' if metric.change < 0 else 'neutral'
                change_symbol = '+' if metric.change > 0 else ''
                
                html_content += f"""
                    <div class="metric {metric.trend}">
                        <div class="metric-name">{metric.name}</div>
                        <div class="metric-value">{metric.value:,.2f} {metric.unit}</div>
                        <div class="metric-change">
                            <span class="change {change_class}">
                                {change_symbol}{metric.change:,.2f} ({change_symbol}{metric.change_percent:.2f}%)
                            </span>
                        </div>
                    </div>
                """
            
            html_content += """
                </div>
                <div class="section">
                    <h2 class="section-title">📋 Summary Statistics</h2>
                    <p><strong>Total Metrics:</strong> """ + str(dashboard_summary['summary_stats']['total_metrics']) + """</p>
                    <p><strong>Data Points:</strong> """ + str(dashboard_summary['summary_stats']['data_points']) + """</p>
                </div>
            </body>
            </html>
            """
            
            # Save HTML file
            with open(filename, 'w') as f:
                f.write(html_content)
                
            logger.info(f"🌐 HTML dashboard report generated: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Error generating HTML report: {e}")
