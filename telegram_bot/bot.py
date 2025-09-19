import asyncio
import logging
import sys
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from typing import Dict, List, Any, Optional
import structlog
from datetime import datetime, timedelta
import json
import aiohttp
from decimal import Decimal

# Add backend to path for unified conversation engine
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_module.unified_conversation_engine import conversation_engine

logger = structlog.get_logger()

class TradingBot:
    def __init__(self, token: str, backend_url: str):
        self.token = token
        self.backend_url = backend_url
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        
    def setup_handlers(self):
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("chat", self.chat_command))
        self.application.add_handler(CommandHandler("trade", self.trade_command))
        self.application.add_handler(CommandHandler("portfolio", self.portfolio_command))
        self.application.add_handler(CommandHandler("signals", self.signals_command))
        self.application.add_handler(CommandHandler("balance", self.balance_command))
        self.application.add_handler(CommandHandler("orders", self.orders_command))
        self.application.add_handler(CommandHandler("settings", self.settings_command))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Add message handler for natural language chat
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_natural_chat))
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        welcome_message = f"""
🤖 Welcome to AI Trading Bot, {user.first_name}!

I'm your AI-powered cryptocurrency trading assistant. Here's what I can do:

📊 **Portfolio Management**
• View your portfolio and PnL
• Check balances across exchanges
• Monitor open orders

📈 **Trading Commands**
• Execute trades with /trade
• Get AI-generated signals with /signals
• View market analysis

💬 **AI Chat**
• Chat naturally with /chat
• Ask trading questions directly
• Get personalized advice

⚙️ **Settings & Configuration**
• Configure risk parameters
• Set up exchange API keys
• Customize notifications

Use /help to see all available commands.
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Portfolio", callback_data="portfolio")],
            [InlineKeyboardButton("📈 Trade", callback_data="trade")],
            [InlineKeyboardButton("🔔 Signals", callback_data="signals")],
            [InlineKeyboardButton("⚙️ Settings", callback_data="settings")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(welcome_message, reply_markup=reply_markup)
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        help_text = """
📚 **Available Commands:**

🔹 **Core Commands**
• /start - Initialize the bot
• /help - Show this help message
• /chat - Chat with AI trading advisor

🔹 **Trading Commands**
• /trade - Execute trades
• /portfolio - View portfolio and PnL
• /balance - Check balances
• /orders - View open orders
• /signals - Get AI trading signals

🔹 **Configuration**
• /settings - Configure bot settings
• /connect - Connect exchange accounts

🔹 **Information**
• /status - Check bot status
• /version - Show bot version

💡 **Quick Tips:**
• Use inline buttons for faster navigation
• Set up stop-loss orders for risk management
• Monitor AI signals for optimal entry/exit points
        """
        await update.message.reply_text(help_text)
    
    async def chat_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chat command for AI conversation"""
        user_id = str(update.effective_user.id)
        
        if not context.args:
            await update.message.reply_text(
                "💬 **AI Chat Mode**\n\n"
                "You can now chat with me naturally! Just send me a message and I'll respond using my AI models.\n\n"
                "**Examples:**\n"
                "• \"What's your analysis of Bitcoin?\"\n"
                "• \"Should I buy Ethereum now?\"\n"
                "• \"How do I manage risk in trading?\"\n"
                "• \"Explain RSI indicator\"\n\n"
                "Or just start typing your question! 🤖",
                parse_mode='Markdown'
            )
            return
        
        # Join all arguments as the message
        message = " ".join(context.args)
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Use unified conversation engine
            response = await conversation_engine.chat(
                user_id=user_id,
                message=message,
                source="telegram"
            )
            
            # Send response
            await update.message.reply_text(
                f"{response['reply']}\n\n*Model: {response['model_used']}*",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in chat command: {e}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error. Please try again later.",
                parse_mode='Markdown'
            )
    
    async def handle_natural_chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle natural language messages"""
        user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        try:
            # Use unified conversation engine
            response = await conversation_engine.chat(
                user_id=user_id,
                message=message_text,
                source="telegram"
            )
            
            # Send response
            await update.message.reply_text(
                f"{response['reply']}\n\n*Model: {response['model_used']}*",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in natural chat: {e}")
            await update.message.reply_text(
                "❌ Sorry, I encountered an error. Please try again later.",
                parse_mode='Markdown'
            )
        
    async def trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("💰 Market Buy", callback_data="trade_market_buy")],
            [InlineKeyboardButton("💰 Market Sell", callback_data="trade_market_sell")],
            [InlineKeyboardButton("📊 Limit Buy", callback_data="trade_limit_buy")],
            [InlineKeyboardButton("📊 Limit Sell", callback_data="trade_limit_sell")],
            [InlineKeyboardButton("🛑 Stop Loss", callback_data="trade_stop_loss")],
            [InlineKeyboardButton("🎯 Take Profit", callback_data="trade_take_profit")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "🚀 **Trading Interface**\n\nSelect the type of trade you want to execute:",
            reply_markup=reply_markup
        )
        
    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = update.effective_user.id
            portfolio_data = await self.get_portfolio_data(user_id)
            
            if portfolio_data:
                message = self.format_portfolio_message(portfolio_data)
            else:
                message = "❌ Unable to fetch portfolio data. Please try again later."
                
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="portfolio_refresh")],
                [InlineKeyboardButton("📊 Detailed View", callback_data="portfolio_detailed")],
                [InlineKeyboardButton("📈 Performance", callback_data="portfolio_performance")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in portfolio command: {e}")
            await update.message.reply_text("❌ Error fetching portfolio data. Please try again.")
            
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            signals = await self.get_ai_signals()
            
            if signals:
                message = self.format_signals_message(signals)
            else:
                message = "🔍 No AI signals available at the moment. Check back later!"
                
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh Signals", callback_data="signals_refresh")],
                [InlineKeyboardButton("📊 Market Analysis", callback_data="market_analysis")],
                [InlineKeyboardButton("🤖 AI Insights", callback_data="ai_insights")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in signals command: {e}")
            await update.message.reply_text("❌ Error fetching AI signals. Please try again.")
            
    async def balance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = update.effective_user.id
            balances = await self.get_balances(user_id)
            
            if balances:
                message = self.format_balances_message(balances)
            else:
                message = "❌ Unable to fetch balance data. Please try again later."
                
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="balance_refresh")],
                [InlineKeyboardButton("💱 Convert", callback_data="balance_convert")],
                [InlineKeyboardButton("📊 Assets", callback_data="balance_assets")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in balance command: {e}")
            await update.message.reply_text("❌ Error fetching balance data. Please try again.")
            
    async def orders_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        try:
            user_id = update.effective_user.id
            orders = await self.get_open_orders(user_id)
            
            if orders:
                message = self.format_orders_message(orders)
            else:
                message = "📋 No open orders found."
                
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="orders_refresh")],
                [InlineKeyboardButton("❌ Cancel All", callback_data="orders_cancel_all")],
                [InlineKeyboardButton("📊 Order History", callback_data="orders_history")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(message, reply_markup=reply_markup)
            
        except Exception as e:
            logger.error(f"Error in orders command: {e}")
            await update.message.reply_text("❌ Error fetching orders. Please try again.")
            
    async def settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = [
            [InlineKeyboardButton("⚙️ Risk Settings", callback_data="settings_risk")],
            [InlineKeyboardButton("🔑 API Keys", callback_data="settings_api")],
            [InlineKeyboardButton("🔔 Notifications", callback_data="settings_notifications")],
            [InlineKeyboardButton("🌍 Exchange Settings", callback_data="settings_exchange")],
            [InlineKeyboardButton("🤖 AI Preferences", callback_data="settings_ai")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "⚙️ **Bot Settings**\n\nConfigure your trading preferences and bot behavior:",
            reply_markup=reply_markup
        )
        
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data.startswith("trade_"):
            await self.handle_trade_callback(query, data)
        elif data.startswith("portfolio_"):
            await self.handle_portfolio_callback(query, data)
        elif data.startswith("signals_"):
            await self.handle_signals_callback(query, data)
        elif data.startswith("balance_"):
            await self.handle_balance_callback(query, data)
        elif data.startswith("orders_"):
            await self.handle_orders_callback(query, data)
        elif data.startswith("settings_"):
            await self.handle_settings_callback(query, data)
        else:
            await query.edit_message_text("❌ Unknown callback data")
            
    async def handle_trade_callback(self, query, data):
        if data == "trade_market_buy":
            await query.edit_message_text(
                "💰 **Market Buy**\n\nPlease enter the trading pair and amount:\n\n"
                "Format: SYMBOL AMOUNT\nExample: BTCUSDT 0.001"
            )
        elif data == "trade_market_sell":
            await query.edit_message_text(
                "💰 **Market Sell**\n\nPlease enter the trading pair and amount:\n\n"
                "Format: SYMBOL AMOUNT\nExample: BTCUSDT 0.001"
            )
        # Add more trade type handlers
        
    async def handle_portfolio_callback(self, query, data):
        if data == "portfolio_refresh":
            user_id = query.from_user.id
            portfolio_data = await self.get_portfolio_data(user_id)
            if portfolio_data:
                message = self.format_portfolio_message(portfolio_data)
                await query.edit_message_text(message)
            else:
                await query.edit_message_text("❌ Unable to refresh portfolio data.")
                
    async def get_portfolio_data(self, user_id: int) -> Optional[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.backend_url}/api/v1/users/{user_id}/portfolio"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching portfolio data: {e}")
            return None
            
    async def get_ai_signals(self) -> Optional[List[Dict[str, Any]]]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.backend_url}/api/v1/ai/signals"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching AI signals: {e}")
            return None
            
    async def get_balances(self, user_id: int) -> Optional[List[Dict[str, Any]]]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.backend_url}/api/v1/users/{user_id}/balances"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching balances: {e}")
            return None
            
    async def get_open_orders(self, user_id: int) -> Optional[List[Dict[str, Any]]]:
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.backend_url}/api/v1/users/{user_id}/orders/open"
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.json()
                    return None
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return None
            
    def format_portfolio_message(self, portfolio: Dict[str, Any]) -> str:
        total_balance = portfolio.get('total_balance', 0)
        total_pnl = portfolio.get('total_pnl', 0)
        daily_pnl = portfolio.get('daily_pnl', 0)
        win_rate = portfolio.get('win_rate', 0)
        
        pnl_color = "🟢" if total_pnl >= 0 else "🔴"
        daily_color = "🟢" if daily_pnl >= 0 else "🔴"
        
        return f"""
📊 **Portfolio Overview**

💰 **Total Balance:** ${total_balance:,.2f}
{pnl_color} **Total PnL:** ${total_pnl:,.2f}
{daily_color} **Daily PnL:** ${daily_pnl:,.2f}
📈 **Win Rate:** {win_rate:.1f}%

🔄 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
    def format_signals_message(self, signals: List[Dict[str, Any]]) -> str:
        message = "🔔 **AI Trading Signals**\n\n"
        
        for signal in signals[:5]:
            symbol = signal.get('symbol', 'Unknown')
            direction = signal.get('direction', 'Unknown')
            strength = signal.get('strength', 0)
            confidence = signal.get('confidence', 0)
            
            direction_emoji = "🟢" if direction == "buy" else "🔴" if direction == "sell" else "🟡"
            strength_bar = "█" * int(strength * 10)
            
            message += f"{direction_emoji} **{symbol}** ({direction.upper()})\n"
            message += f"💪 Strength: {strength_bar} ({strength:.1%})\n"
            message += f"🎯 Confidence: {confidence:.1%}\n\n"
            
        return message
        
    def format_balances_message(self, balances: List[Dict[str, Any]]) -> str:
        message = "💰 **Account Balances**\n\n"
        
        for balance in balances:
            asset = balance.get('asset', 'Unknown')
            free = balance.get('free', 0)
            locked = balance.get('locked', 0)
            total = balance.get('total', 0)
            
            if total > 0:
                message += f"🔸 **{asset}**\n"
                message += f"   Available: {free:.8f}\n"
                message += f"   Locked: {locked:.8f}\n"
                message += f"   Total: {total:.8f}\n\n"
                
        return message
        
    def format_orders_message(self, orders: List[Dict[str, Any]]) -> str:
        if not orders:
            return "📋 No open orders found."
            
        message = "📋 **Open Orders**\n\n"
        
        for order in orders[:5]:
            symbol = order.get('symbol', 'Unknown')
            side = order.get('side', 'Unknown')
            quantity = order.get('quantity', 0)
            price = order.get('price', 0)
            status = order.get('status', 'Unknown')
            
            side_emoji = "🟢" if side == "BUY" else "🔴"
            
            message += f"{side_emoji} **{symbol}** ({side})\n"
            message += f"   Qty: {quantity:.8f}\n"
            message += f"   Price: ${price:.2f}\n"
            message += f"   Status: {status}\n\n"
            
        return message
        
    async def send_notification(self, user_id: int, message: str, priority: str = "normal"):
        try:
            await self.application.bot.send_message(
                chat_id=user_id,
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending notification to {user_id}: {e}")
            
    def run(self):
        self.application.run_polling()
        
    async def start_webhook(self, webhook_url: str):
        await self.application.bot.set_webhook(url=webhook_url)
        await self.application.start_webhook(
            listen="0.0.0.0",
            port=8443,
            webhook_url=webhook_url
        )

# Main execution block
if __name__ == "__main__":
    import asyncio
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get bot token from environment
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not bot_token:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not found in environment variables!")
        print("Please set your bot token in the .env file")
        exit(1)
    
    print("🤖 Starting DEXTER Telegram Bot...")
    print(f"📱 Bot Token: {bot_token[:10]}...")
    print("🚀 Connecting to Telegram...")
    
    try:
        # Create and start the bot
        bot = TradingBot(bot_token, "http://localhost:8000")
        print("✅ Bot created successfully!")
        print("🔄 Starting bot polling...")
        print("📱 Bot is now running! Send /start to test")
        
        # Start the bot
        bot.run()
        
    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        print("Please check your bot token and try again")
