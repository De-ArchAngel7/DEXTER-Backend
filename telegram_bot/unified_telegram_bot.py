#!/usr/bin/env python3
"""
🤖 DEXTER UNIFIED TELEGRAM BOT
============================================================
Telegram bot that connects to the unified conversation engine
Supports both slash commands and natural language chat
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    CallbackQueryHandler,
    filters, 
    ContextTypes
)

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai_module.unified_conversation_engine import conversation_engine
from app.services.conversation_logger import conversation_logger

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class DexterTelegramBot:
    """
    DEXTER Telegram Bot with unified conversation engine
    """
    
    def __init__(self, token: str):
        self.token = token
        self.application = Application.builder().token(token).build()
        self.setup_handlers()
        
    def setup_handlers(self):
        """Setup command and message handlers"""
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("predict", self.predict_command))
        self.application.add_handler(CommandHandler("analyze", self.analyze_command))
        self.application.add_handler(CommandHandler("status", self.status_command))
        self.application.add_handler(CommandHandler("clear", self.clear_command))
        self.application.add_handler(CommandHandler("history", self.history_command))
        
        # Callback query handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        # Message handler for natural language chat
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        
        # Error handler
        self.application.add_error_handler(self.error_handler)
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        user_id = str(update.effective_user.id)
        username = update.effective_user.username or "User"
        
        welcome_message = f"""
🤖 **Welcome to DEXTER AI Trading Bot!**

Hello {username}! I'm DEXTER, your AI trading assistant powered by advanced machine learning models.

**What I can do:**
• 📊 Analyze market data and provide trading insights
• 💰 Give personalized trading recommendations  
• 📈 Explain technical indicators in plain English
• 🎯 Help with risk management and position sizing
• 💬 Chat naturally about trading topics

**Quick Commands:**
• `/predict BTC` - Get price prediction for a token
• `/analyze ETH` - Get detailed market analysis
• `/status` - Check my AI model status
• `/help` - Show all available commands

**Just start chatting!** Ask me anything about trading, markets, or specific tokens.

*Powered by your custom-trained DialoGPT AI model* 🧠
        """
        
        # Create inline keyboard with quick actions
        keyboard = [
            [
                InlineKeyboardButton("📊 Market Analysis", callback_data="market_analysis"),
                InlineKeyboardButton("💰 Trading Tips", callback_data="trading_tips")
            ],
            [
                InlineKeyboardButton("📈 Predict BTC", callback_data="predict_btc"),
                InlineKeyboardButton("📈 Predict ETH", callback_data="predict_eth")
            ],
            [
                InlineKeyboardButton("❓ Help", callback_data="help"),
                InlineKeyboardButton("📊 Status", callback_data="status")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
        # Log the start command
        await conversation_logger.log_conversation(
            user_id=user_id,
            message="/start command",
            reply="Welcome message sent",
            model_used="system",
            source="telegram",
            metadata={"command": "start", "username": username}
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
🤖 **DEXTER AI Trading Bot - Help**

**Available Commands:**
• `/start` - Welcome message and quick actions
• `/help` - Show this help message
• `/predict <TOKEN>` - Get price prediction (e.g., `/predict BTC`)
• `/analyze <TOKEN>` - Get detailed market analysis
• `/status` - Check AI model status and health
• `/clear` - Clear your conversation history
• `/history` - View your recent conversation history

**Natural Language Chat:**
You can also just chat with me naturally! Try:
• "What's your analysis of Bitcoin?"
• "Should I buy Ethereum now?"
• "How do I manage risk in trading?"
• "Explain RSI indicator"
• "What's the market sentiment today?"

**Features:**
• 🧠 Powered by your custom-trained DialoGPT AI
• 📊 Real-time market data integration
• 💬 Conversation history and context
• 🔄 Automatic fallback to OpenAI if needed
• 📝 All conversations logged for analysis

**Tips:**
• Be specific with your questions for better answers
• Use token symbols (BTC, ETH, etc.) for price analysis
• Ask follow-up questions to dive deeper into topics

*Need more help? Just ask me anything!* 😊
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def predict_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predict command"""
        user_id = str(update.effective_user.id)
        
        if not context.args:
            await update.message.reply_text(
                "Please specify a token symbol. Example: `/predict BTC`",
                parse_mode='Markdown'
            )
            return
        
        token = context.args[0].upper()
        message = f"Get price prediction for {token}"
        
        # Get prediction from unified conversation engine
        response = await conversation_engine.chat(
            user_id=user_id,
            message=f"Can you provide a price prediction for {token}? Include technical analysis and key levels to watch.",
            source="telegram"
        )
        
        # Format the response
        reply_text = f"📈 **{token} Price Prediction**\n\n{response['reply']}\n\n*Model: {response['model_used']}*"
        
        await update.message.reply_text(reply_text, parse_mode='Markdown')
    
    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /analyze command"""
        user_id = str(update.effective_user.id)
        
        if not context.args:
            await update.message.reply_text(
                "Please specify a token symbol. Example: `/analyze ETH`",
                parse_mode='Markdown'
            )
            return
        
        token = context.args[0].upper()
        
        # Get analysis from unified conversation engine
        response = await conversation_engine.chat(
            user_id=user_id,
            message=f"Provide a comprehensive market analysis for {token}. Include technical indicators, market sentiment, trading recommendations, and risk assessment.",
            source="telegram"
        )
        
        # Format the response
        reply_text = f"📊 **{token} Market Analysis**\n\n{response['reply']}\n\n*Model: {response['model_used']}*"
        
        await update.message.reply_text(reply_text, parse_mode='Markdown')
    
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        user_id = str(update.effective_user.id)
        
        # Get engine status
        status = conversation_engine.get_engine_status()
        
        # Get conversation stats
        stats = await conversation_logger.get_conversation_stats()
        
        status_message = f"""
🤖 **DEXTER AI Status**

**AI Models:**
• DialoGPT: {'✅ Active' if status['dialoGPT_loaded'] else '❌ Inactive'}
• OpenAI Fallback: {'✅ Available' if status['openai_available'] else '❌ Unavailable'}

**System:**
• Active Conversations: {status['active_conversations']}
• Total Conversations: {stats.get('total_conversations', 'N/A')}
• Unique Users: {stats.get('unique_users', 'N/A')}

**Model Details:**
• Parameters: {status['dialoGPT_status']['parameters']:,} (DialoGPT)
• Device: {status['dialoGPT_status']['device']}
• Quantization: {'Enabled' if status['dialoGPT_status']['quantization'] else 'Disabled'}

*All systems operational!* ✅
        """
        
        await update.message.reply_text(status_message, parse_mode='Markdown')
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clear command"""
        user_id = str(update.effective_user.id)
        
        # Clear conversation history
        conversation_engine.clear_conversation(user_id)
        
        await update.message.reply_text(
            "🗑️ Your conversation history has been cleared!",
            parse_mode='Markdown'
        )
    
    async def history_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /history command"""
        user_id = str(update.effective_user.id)
        
        # Get conversation history
        history = conversation_engine.get_conversation_history(user_id)
        
        if not history:
            await update.message.reply_text("📝 No conversation history found.")
            return
        
        # Format history (show last 5 messages)
        recent_history = history[-5:] if len(history) > 5 else history
        
        history_text = "📝 **Recent Conversation History:**\n\n"
        
        for msg in recent_history:
            role = "👤 You" if msg['role'] == 'user' else "🤖 DEXTER"
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            model = f" ({msg.get('model_used', 'unknown')})" if msg['role'] == 'assistant' else ""
            
            history_text += f"{role}{model}: {content}\n\n"
        
        await update.message.reply_text(history_text, parse_mode='Markdown')
    
    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle natural language messages"""
        user_id = str(update.effective_user.id)
        message_text = update.message.text
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Get response from unified conversation engine
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
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle inline button callbacks"""
        query = update.callback_query
        await query.answer()
        
        user_id = str(update.effective_user.id)
        
        if query.data == "market_analysis":
            response = await conversation_engine.chat(
                user_id=user_id,
                message="Provide a general market analysis overview",
                source="telegram"
            )
            await query.edit_message_text(
                f"📊 **Market Analysis**\n\n{response['reply']}\n\n*Model: {response['model_used']}*",
                parse_mode='Markdown'
            )
            
        elif query.data == "trading_tips":
            response = await conversation_engine.chat(
                user_id=user_id,
                message="Give me some general trading tips and best practices",
                source="telegram"
            )
            await query.edit_message_text(
                f"💰 **Trading Tips**\n\n{response['reply']}\n\n*Model: {response['model_used']}*",
                parse_mode='Markdown'
            )
            
        elif query.data == "predict_btc":
            response = await conversation_engine.chat(
                user_id=user_id,
                message="Provide a price prediction for Bitcoin (BTC)",
                source="telegram"
            )
            await query.edit_message_text(
                f"📈 **BTC Prediction**\n\n{response['reply']}\n\n*Model: {response['model_used']}*",
                parse_mode='Markdown'
            )
            
        elif query.data == "predict_eth":
            response = await conversation_engine.chat(
                user_id=user_id,
                message="Provide a price prediction for Ethereum (ETH)",
                source="telegram"
            )
            await query.edit_message_text(
                f"📈 **ETH Prediction**\n\n{response['reply']}\n\n*Model: {response['model_used']}*",
                parse_mode='Markdown'
            )
            
        elif query.data == "help":
            await self.help_command(update, context)
            
        elif query.data == "status":
            await self.status_command(update, context)
    
    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors"""
        logger.error(f"Update {update} caused error {context.error}")
        
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "❌ Sorry, I encountered an error. Please try again later.",
                parse_mode='Markdown'
            )
    
    def run(self):
        """Start the bot"""
        logger.info("Starting DEXTER Telegram Bot...")
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)

def main():
    """Main function to run the bot"""
    # Get bot token from environment
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
        return
    
    # Create and run bot
    bot = DexterTelegramBot(token)
    bot.run()

if __name__ == "__main__":
    main()
