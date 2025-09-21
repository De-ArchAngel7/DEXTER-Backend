#!/usr/bin/env python3
"""
Setup Telegram Webhook for DEXTER Bot
"""
import requests
import os

# Your bot token and webhook URL
BOT_TOKEN = "8019778994:AAFOefRbsI6O6ynCHfpozIh9EajENzAP0qg"
WEBHOOK_URL = "https://dexter-backend-dqx1.onrender.com/telegram/webhook"

def setup_webhook():
    """Set up Telegram webhook"""
    print(f"🤖 Setting up Telegram webhook for DEXTER...")
    print(f"📡 Bot Token: {BOT_TOKEN[:20]}...")
    print(f"🌐 Webhook URL: {WEBHOOK_URL}")
    
    # Set webhook
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/setWebhook"
    data = {"url": WEBHOOK_URL}
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        
        if result.get("ok"):
            print("✅ Webhook set successfully!")
            print(f"📋 Response: {result}")
        else:
            print(f"❌ Failed to set webhook: {result}")
            
    except Exception as e:
        print(f"❌ Error setting webhook: {e}")

def get_webhook_info():
    """Get current webhook info"""
    print("\n🔍 Checking current webhook info...")
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getWebhookInfo"
    
    try:
        response = requests.get(url)
        result = response.json()
        
        if result.get("ok"):
            info = result.get("result", {})
            print("📋 Current webhook info:")
            print(f"   URL: {info.get('url', 'Not set')}")
            print(f"   Pending updates: {info.get('pending_update_count', 0)}")
            print(f"   Last error: {info.get('last_error_message', 'None')}")
        else:
            print(f"❌ Failed to get webhook info: {result}")
            
    except Exception as e:
        print(f"❌ Error getting webhook info: {e}")

def get_bot_info():
    """Get bot information"""
    print("\n🤖 Getting bot information...")
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    
    try:
        response = requests.get(url)
        result = response.json()
        
        if result.get("ok"):
            bot = result.get("result", {})
            print("🤖 Bot info:")
            print(f"   Name: {bot.get('first_name', 'Unknown')}")
            print(f"   Username: @{bot.get('username', 'Unknown')}")
            print(f"   ID: {bot.get('id', 'Unknown')}")
        else:
            print(f"❌ Failed to get bot info: {result}")
            
    except Exception as e:
        print(f"❌ Error getting bot info: {e}")

if __name__ == "__main__":
    print("🚀 DEXTER Telegram Bot Setup")
    print("=" * 50)
    
    get_bot_info()
    get_webhook_info()
    setup_webhook()
    
    print("\n✅ Setup complete!")
    print("💬 Try sending a message to your bot in Telegram!")
