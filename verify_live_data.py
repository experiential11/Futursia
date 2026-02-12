#!/usr/bin/env python
"""Quick verification that APIs are configured for REAL live data (not mock)."""

import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load config
import yaml

config_path = project_root / "configs" / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("LIVE DATA VERIFICATION - Massive 40-Minute Forecaster")
print("=" * 70)
print()

# Check Stock API
massive_config = config.get('massive_api', {})
stock_api_key = massive_config.get('api_key')
print("1. STOCK MARKET API (Massive Trading)")
print("-" * 70)
if stock_api_key:
    print(f"   ✅ API Key Configured: {stock_api_key[:10]}***{stock_api_key[-5:]}")
    print(f"   ✅ Base URL: {massive_config.get('base_url')}")
    print(f"   ✅ Auth Header: {massive_config.get('auth_header_name')}")
    print(f"   ✅ Rate Limit: {massive_config.get('rate_limit', {}).get('calls_per_second')} calls/sec")
    print()
else:
    print("   ❌ NO API KEY FOUND - Will use MOCK data")
    print()

# Check News API
news_config = config.get('news', {})
news_api_key = news_config.get('api_key')
news_provider = news_config.get('provider')
news_enabled = news_config.get('enabled')
print("2. NEWS API (NewsAPI.org)")
print("-" * 70)
if news_enabled and news_provider == "newsapi" and news_api_key:
    print(f"   ✅ Provider: {news_provider}")
    print(f"   ✅ API Key Configured: {news_api_key[:10]}***{news_api_key[-5:]}")
    print(f"   ✅ Base URL: https://newsapi.org/v2")
    print(f"   ✅ Headline Limit: {news_config.get('headline_limit')}")
    print()
else:
    if not news_enabled:
        print("   ⚠️  News disabled in config")
    elif news_provider != "newsapi":
        print(f"   ⚠️  Provider is '{news_provider}' (not 'newsapi') - Will use MOCK data")
    elif not news_api_key:
        print("   ❌ NO API KEY FOUND - Will use MOCK data")
    print()

# Test API connectivity
print("3. API CONNECTIVITY TEST")
print("-" * 70)

try:
    # Test Stock API
    from core.massive_client import MassiveClient
    client = MassiveClient(config)
    
    print(f"   Stock API Client:")
    print(f"      Mock Mode: {client.mock_mode}")
    if not client.mock_mode:
        print(f"      ✅ REAL API - Will fetch actual market data")
        print(f"      Auth Headers Present: {bool(client._get_headers())}")
    else:
        print(f"      ❌ MOCK MODE - Will use demo data")
    print()
except Exception as e:
    print(f"   ❌ Error testing Stock API: {e}")
    print()

try:
    # Test News API
    from core.news_client import NewsClient
    news = NewsClient(config)
    
    print(f"   News API Client:")
    print(f"      Provider: {news.provider}")
    print(f"      Enabled: {news.enabled}")
    print(f"      API Key Present: {bool(news.api_key)}")
    
    if news.provider == "newsapi" and news.api_key:
        print(f"      ✅ REAL API - Will fetch actual news headlines")
    else:
        print(f"      ⚠️  MOCK MODE - Will use demo headlines")
    print()
except Exception as e:
    print(f"   ❌ Error testing News API: {e}")
    print()

# Summary
print("4. LIVE DATA STATUS")
print("-" * 70)

has_stock_api = stock_api_key is not None
has_news_api = news_enabled and news_provider == "newsapi" and news_api_key

if has_stock_api and has_news_api:
    print("   ✅ ✅ ✅ ALL REAL LIVE DATA CONFIGURED ✅ ✅ ✅")
    print()
    print("   The application is configured for REAL market data:")
    print("      • Real-time quotes and bars from Massive Trading API")
    print("      • Live news headlines from NewsAPI.org")
    print("      • No mock/demo data will be used")
    print()
elif has_stock_api:
    print("   ⚠️  PARTIAL REAL DATA")
    print("      ✅ Stock data is REAL (Massive Trading API)")
    print("      ❌ News data is MOCK (NewsAPI.org not configured)")
    print()
elif has_news_api:
    print("   ⚠️  PARTIAL REAL DATA")
    print("      ❌ Stock data is MOCK (Massive Trading API not configured)")
    print("      ✅ News data is REAL (NewsAPI.org)")
    print()
else:
    print("   ❌ ❌ ❌ NO REAL DATA CONFIGURED ❌ ❌ ❌")
    print("      Both APIs are missing or in mock mode!")
    print()

print("=" * 70)
print(f"Database: {config.get('storage', {}).get('database')}")
print(f"State File: {config.get('storage', {}).get('state_file')}")
print("=" * 70)
print()
print("To run the app:")
print("   python launcher.py")
print()
