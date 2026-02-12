"""
Finnhub API Client
Provides real-time stock data via Finnhub.io API
"""

import requests
import time
import logging
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class FinnhubClient:
    """Client for Finnhub stock data API"""
    
    def __init__(self, config: Dict):
        """
        Initialize Finnhub client
        
        Args:
            config: Configuration dictionary with 'finnhub_api' section
        """
        self.config = config.get('finnhub_api', {})
        self.base_url = self.config.get('base_url', 'https://finnhub.io/api/v1')
        self.api_key = os.getenv('FINNHUB_API_KEY') or self.config.get('api_key', '')
        self.endpoints = self.config.get('endpoints', {})
        
        # Rate limiting
        rate_config = self.config.get('rate_limit', {})
        self.calls_per_minute = rate_config.get('calls_per_minute', 60)
        self.min_call_interval = 60.0 / self.calls_per_minute
        self.last_call_time = 0
        
        # Session with SSL verification disabled
        self.session = requests.Session()
        self.session.verify = False
        
        logger.info("FinnhubClient initialized (key present=%s)", bool(self.api_key))
    
    def _rate_limit(self):
        """Apply rate limiting"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get current quote for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Quote data dict with keys: c (current), h (high), l (low), o (open), pc (prev close), t (timestamp)
        """
        try:
            self._rate_limit()
            url = f"{self.base_url}{self.endpoints.get('quote', '/quote')}"
            params = {
                'symbol': symbol.upper(),
                'token': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Retrieved quote for {symbol}: {data.get('c', 'N/A')}")
            return data
            
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return None
    
    def get_bars(self, symbol: str, resolution: str = '1', from_timestamp: Optional[int] = None, 
                 to_timestamp: Optional[int] = None) -> Optional[Dict]:
        """
        Get OHLC bars for a symbol
        Note: Candle endpoint requires premium. Generating synthetic bars from quote data.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            resolution: Bar resolution ('1' for 1-minute, '5' for 5-minute, etc.)
            from_timestamp: Start timestamp (Unix seconds)
            to_timestamp: End timestamp (Unix seconds)
            
        Returns:
            Bars data dict with keys: o, h, l, c, v, t (lists), s (status)
        """
        try:
            # Get current quote to generate synthetic bars
            quote = self.get_quote(symbol)
            if not quote:
                return None
            
            # Generate synthetic bars based on current price
            current_price = float(quote.get('c', 100))
            prev_close = float(quote.get('pc', current_price))
            
            # Create 100 synthetic bars going back 100 minutes
            bars = {
                'o': [],
                'h': [],
                'l': [],
                'c': [],
                'v': [],
                't': [],
                's': 'ok'
            }
            
            now = int(time.time())
            price = prev_close
            
            for i in range(100, 0, -1):
                ts = now - (i * 60)
                # Gradual movement towards current price
                move_fraction = (100 - i) / 100.0
                trend = (current_price - prev_close) * move_fraction
                
                # Add some volatility
                volatility = abs(current_price - prev_close) * 0.002
                noise = np.random.normal(0, volatility)
                
                bar_close = prev_close + trend + noise
                bar_open = bar_close - (volatility * 0.5)
                bar_high = max(bar_open, bar_close) + (volatility * 0.5)
                bar_low = min(bar_open, bar_close) - (volatility * 0.5)
                
                bars['t'].append(ts)
                bars['o'].append(round(bar_open, 2))
                bars['h'].append(round(bar_high, 2))
                bars['l'].append(round(bar_low, 2))
                bars['c'].append(round(bar_close, 2))
                bars['v'].append(np.random.uniform(1e6, 5e6))
                
                price = bar_close
            
            logger.info(f"Generated synthetic {len(bars.get('c', []))} bars for {symbol}")
            return bars
                
        except Exception as e:
            logger.error(f"Error getting bars for {symbol}: {str(e)}")
            return None
    
    def get_news(self, symbol: str, limit: int = 20) -> Optional[List[Dict]]:
        """
        Get company news for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            limit: Number of news items to retrieve
            
        Returns:
            List of news items with keys: id, headline, image, source, summary, url, category, datetime
        """
        try:
            self._rate_limit()
            
            # Set time range (last 30 days)
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            
            url = f"{self.base_url}{self.endpoints.get('news', '/company-news')}"
            params = {
                'symbol': symbol.upper(),
                'from': from_date.strftime('%Y-%m-%d'),
                'to': to_date.strftime('%Y-%m-%d'),
                'token': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, list):
                logger.info(f"Retrieved {len(data)} news items for {symbol}")
                return data[:limit]
            else:
                logger.warning(f"Unexpected news format for {symbol}: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting news for {symbol}: {str(e)}")
            return None
    
    def get_top_movers(self) -> Optional[List[Dict]]:
        """
        Get top moving stocks (market movers)
        For Finnhub, we'll return a mock list since there's no direct top movers endpoint
        
        Returns:
            List of top moving stocks
        """
        # Popular stocks to monitor
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        movers = []
        
        try:
            for symbol in symbols:
                quote = self.get_quote(symbol)
                if quote and 'c' in quote:
                    movers.append({
                        'symbol': symbol,
                        'price': quote.get('c'),
                        'change': quote.get('c', 0) - quote.get('pc', quote.get('c', 0)),
                        'percent_change': ((quote.get('c', 0) - quote.get('pc', quote.get('c', 0))) / quote.get('pc', 1)) * 100 if quote.get('pc') else 0,
                        'high': quote.get('h'),
                        'low': quote.get('l'),
                        'open': quote.get('o'),
                        'prev_close': quote.get('pc')
                    })
            
            # Sort by absolute percent change
            movers.sort(key=lambda x: abs(x.get('percent_change', 0)), reverse=True)
            logger.info(f"Retrieved {len(movers)} top movers")
            return movers[:10]
            
        except Exception as e:
            logger.error(f"Error getting top movers: {str(e)}")
            return None


def format_bars_for_forecast(bars_data: Optional[Dict]) -> Optional[Dict]:
    """
    Convert Finnhub bars format to standard DataFrame-compatible format
    
    Args:
        bars_data: Raw bars response from Finnhub
        
    Returns:
        Dictionary with lists: open, high, low, close, volume, timestamp
    """
    if not bars_data or bars_data.get('s') != 'ok':
        return None
    
    try:
        formatted = {
            'timestamp': bars_data.get('t', []),
            'open': bars_data.get('o', []),
            'high': bars_data.get('h', []),
            'low': bars_data.get('l', []),
            'close': bars_data.get('c', []),
            'volume': bars_data.get('v', [])
        }
        
        # Ensure all lists have the same length
        min_len = min(len(v) for v in formatted.values() if isinstance(v, list))
        for key in formatted:
            if isinstance(formatted[key], list):
                formatted[key] = formatted[key][:min_len]
        
        return formatted
    except Exception as e:
        logger.error(f"Error formatting bars: {str(e)}")
        return None
