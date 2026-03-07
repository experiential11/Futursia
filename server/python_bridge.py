#!/usr/bin/env python3
"""
Python bridge that interfaces with the existing forecasting logic.
Reads JSON requests from Node.js and returns JSON responses.
"""

import sys
import json
import os
from pathlib import Path
import yaml
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.client_factory import get_market_client
from core.forecasting import ForecasterEngine
from core.news_client import NewsClient
from core.logging_setup import setup_logging, get_logger
from core.top10 import get_top_movers

setup_logging()
logger = get_logger()

config_path = Path(__file__).parent.parent / 'configs' / 'config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

market_client = get_market_client(config)
news_client = NewsClient(config)
forecaster = ForecasterEngine(config, market_client, news_client)

print("ready", flush=True)

def get_top_movers_handler():
    """Get top market movers."""
    try:
        movers = market_client.get_top_movers()
        return {"movers": movers or []}
    except Exception as e:
        logger.error(f"Error getting top movers: {e}")
        return {"movers": [], "error": str(e)}

def get_quote_handler(symbol):
    """Get quote for a symbol."""
    try:
        quote = market_client.get_quote(symbol) or {}
        return quote
    except Exception as e:
        logger.error(f"Error getting quote for {symbol}: {e}")
        return {"error": str(e)}

def get_forecast_handler(symbol):
    """Get forecast for a symbol."""
    try:
        forecast = forecaster.generate_forecast(symbol)
        if forecast:
            return {
                "direction": forecast.get("direction", "FLAT"),
                "confidence": float(forecast.get("confidence", 50)),
                "predicted_return": float(forecast.get("prediction_return", 0)),
                "model_status": forecast.get("model_status", "OK")
            }
        return {"direction": "FLAT", "confidence": 0, "predicted_return": 0}
    except Exception as e:
        logger.error(f"Error getting forecast for {symbol}: {e}")
        return {"direction": "FLAT", "confidence": 0, "predicted_return": 0, "error": str(e)}

def get_bars_handler(symbol, limit=240):
    """Get price bars for a symbol."""
    try:
        bars = market_client.get_bars(symbol, limit=limit)
        if isinstance(bars, list):
            return {"bars": bars}
        elif hasattr(bars, 'to_dict'):
            bars_list = bars.to_dict('records')
            return {"bars": bars_list}
        return {"bars": []}
    except Exception as e:
        logger.error(f"Error getting bars for {symbol}: {e}")
        return {"bars": [], "error": str(e)}

def get_news_handler(symbol, limit=10):
    """Get news headlines for a symbol."""
    try:
        headlines = news_client.get_headlines(symbol, limit=limit)
        return {"headlines": headlines or []}
    except Exception as e:
        logger.error(f"Error getting news for {symbol}: {e}")
        return {"headlines": [], "error": str(e)}

def get_config_handler():
    """Get system configuration."""
    try:
        provider = config.get('market_api_provider', 'yfinance').lower()
        info = f"""
Futursia Forecasting V1.0 - Diagnostics

Configuration:
  Market Provider: {provider}
  Forecast Horizon: {config.get('forecast', {}).get('horizon_minutes', 40)} minutes
  Model: {config.get('forecast', {}).get('primary_model', 'ridge')}

Features:
  - Real-time stock quotes
  - 40-minute forecasts
  - Forecast path chart
  - Financial news headlines
  - 1-second auto-refresh
  - Non-blocking UI updates
"""
        return {
            "provider": provider,
            "horizon_minutes": config.get('forecast', {}).get('horizon_minutes', 40),
            "primary_model": config.get('forecast', {}).get('primary_model', 'ridge'),
            "info": info
        }
    except Exception as e:
        logger.error(f"Error getting config: {e}")
        return {"error": str(e)}

handlers = {
    "get_top_movers": lambda params: get_top_movers_handler(),
    "get_quote": lambda params: get_quote_handler(params.get("symbol")),
    "get_forecast": lambda params: get_forecast_handler(params.get("symbol")),
    "get_bars": lambda params: get_bars_handler(params.get("symbol"), params.get("limit", 240)),
    "get_news": lambda params: get_news_handler(params.get("symbol"), params.get("limit", 10)),
    "get_config": lambda params: get_config_handler(),
}

def process_request(request):
    """Process a single request from Node.js."""
    try:
        method = request.get("method")
        params = request.get("params", {})

        if method not in handlers:
            return {"error": f"Unknown method: {method}"}

        result = handlers[method](params)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        while True:
            line = sys.stdin.readline()
            if not line:
                break

            try:
                request = json.loads(line.strip())
                response = process_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON: {e}")
                sys.stdout.write(json.dumps({"error": "Invalid JSON"}) + "\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        logger.info("Bridge shutting down")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Bridge error: {e}")
        sys.exit(1)
