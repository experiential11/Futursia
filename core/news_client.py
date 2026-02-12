"""News client for fetching market news and headlines."""

import logging
import os
import requests
from datetime import datetime
from core.logging_setup import get_logger


logger = get_logger()


class NewsClient:
    """Client for news data (supports mock and NewsAPI provider)."""
    
    def __init__(self, config):
        """Initialize news client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('news', {})
        self.enabled = self.config.get('enabled', True)
        self.provider = self.config.get('provider', 'mock')
        self.api_key = os.getenv('NEWS_API_KEY') or self.config.get('api_key')
        self.headline_limit = self.config.get('headline_limit', 10)
        self.base_url = "https://newsapi.org/v2"
    
    def get_headlines(self, symbol=None, limit=None):
        """Get latest headlines for a symbol or market.
        
        Args:
            symbol: Stock symbol (optional)
            limit: Number of headlines (uses config default if not provided)
        
        Returns:
            list: List of headline dicts
        """
        if not self.enabled:
            return []
        
        if limit is None:
            limit = self.headline_limit
        
        # If using real NewsAPI and API key is available
        if self.provider == "newsapi" and self.api_key:
            return self._get_headlines_newsapi(symbol, limit)
        else:
            # Fall back to mock implementation
            return self._generate_mock_headlines(symbol, limit)
    def _generate_mock_headlines(self, symbol=None, limit=10):
        """Generate mock headlines for demo.
        
        Args:
            symbol: Stock symbol
            limit: Number to generate
        
        Returns:
            list: Mock headlines
        """
        from datetime import timedelta
        import random
        
        if not symbol:
            symbol = 'AAPL'
        
        headlines_templates = [
            f"{symbol} shares rise on strong earnings beat",
            f"{symbol} announces new partnership with major tech firm",
            f"{symbol} faces regulatory pressure on data privacy",
            f"Analyst upgrades {symbol} to strong buy",
            f"{symbol} stock slides after market concerns",
            f"Innovation in {symbol} product line drives investor confidence",
            f"{symbol} Q3 guidance exceeds expectations",
            f"Market sentiment shifts for {symbol} amid competition",
            f"{symbol} CEO comments spark rally",
            f"Technical pullback expected for {symbol} stock"
        ]
        
        headlines = []
        now = datetime.utcnow()
        
        for i in range(min(limit, len(headlines_templates))):
            ts = now - timedelta(minutes=i*5)
            headlines.append({
                'title': headlines_templates[i],
                'text': headlines_templates[i],
                'source': random.choice(['Reuters', 'Bloomberg', 'CNBC', 'Financial Times', 'MarketWatch']),
                'timestamp': ts.isoformat(),
                'url': f'https://example.com/news/{i}'
            })
        
        return headlines
    
    def _generate_mock_market_headlines(self, limit=10):
        """Generate mock market-wide headlines for demo.
        
        Args:
            limit: Number to generate
        
        Returns:
            list: Mock market headlines
        """
        from datetime import timedelta
        import random
        
        headlines_templates = [
            "US stock market reaches record highs amid strong economic data",
            "Federal Reserve maintains interest rates, signals caution",
            "Tech sector leads rally on AI enthusiasm",
            "Oil prices surge on geopolitical tensions",
            "Bond yields decline as inflation expectations ease",
            "Major indices post weekly gains despite volatility",
            "Retail sales beat expectations, consumer resilience continues",
            "Unemployment rate holds steady at historical lows",
            "Earnings season kicks off with strong corporate results",
            "Market volatility index (VIX) declines as risk sentiment improves",
            "Dollar strengthens against major currencies",
            "Gold prices stable amid economic uncertainty",
            "Market futures point to positive open after overnight gains",
            "Sector rotation favors financials and energy stocks",
            "Global equity markets rise on China stimulus hopes"
        ]
        
        headlines = []
        now = datetime.utcnow()
        
        for i in range(min(limit, len(headlines_templates))):
            ts = now - timedelta(minutes=i*10)
            headlines.append({
                'title': headlines_templates[i],
                'text': headlines_templates[i],
                'source': random.choice(['Reuters', 'Bloomberg', 'CNBC', 'Financial Times', 'MarketWatch', 'Yahoo Finance']),
                'timestamp': ts.isoformat(),
                'url': f'https://example.com/market-news/{i}'
            })
        
        return headlines
    
    def get_market_headlines(self, limit=None):
        """Get market-wide headlines.
        
        Args:
            limit: Number of headlines
        
        Returns:
            list: Market headlines
        """
        if not self.enabled:
            return []
        
        if limit is None:
            limit = self.headline_limit
        
        # If using real NewsAPI and API key is available
        if self.provider == "newsapi" and self.api_key:
            return self._get_headlines_newsapi(None, limit)
        else:
            # Fall back to mock implementation
            return self._generate_mock_market_headlines(limit)
    
    def _get_headlines_newsapi(self, symbol=None, limit=10):
        """Fetch headlines from NewsAPI.
        
        Args:
            symbol: Stock symbol (optional)
            limit: Number of headlines
        
        Returns:
            list: Headlines from NewsAPI
        """
        try:
            # Build search query
            if symbol:
                query = f"{symbol} stock"
            else:
                query = "stock market"
            
            params = {
                'q': query,
                'apiKey': self.api_key,
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'language': 'en'
            }
            
            response = requests.get(f"{self.base_url}/everything", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                headlines = []
                for article in articles[:limit]:
                    headlines.append({
                        'title': article.get('title', ''),
                        'text': article.get('description', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'timestamp': article.get('publishedAt', ''),
                        'url': article.get('url', ''),
                        'image': article.get('urlToImage', '')
                    })
                
                logger.info(f"Fetched {len(headlines)} headlines for {symbol or 'market'}")
                return headlines
            else:
                logger.warning(f"NewsAPI error: HTTP {response.status_code}")
                return self._generate_mock_headlines(symbol, limit)
        
        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return self._generate_mock_headlines(symbol, limit)
