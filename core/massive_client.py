"""Client for Massive Trading API."""

import requests
import time
import json
import logging
import os
from datetime import datetime, timedelta
import pandas as pd
from core.caching import Cache, APIThrottler
from core.logging_setup import get_logger
from core.db import MarketDB


logger = get_logger()


class MassiveClient:
    """Client for Massive Trading API with rate limiting and caching."""
    
    def __init__(self, config, api_key=None):
        """Initialize Massive client.
        
        Args:
            config: Configuration dictionary
            api_key: API key (optional, will use env var or config if not provided)
        """
        self.config = config
        # Get API key from: parameter > env var > config file
        self.api_key = api_key or os.getenv('MASSIVE_API_KEY') or config.get('massive_api', {}).get('api_key')
        self.base_url = config.get('massive_api', {}).get('base_url', 'https://api.massive.ai/v1')
        
        # Auth configuration
        massive_config = config.get('massive_api', {})
        self.auth_header_name = massive_config.get('auth_header_name', 'Authorization')
        self.auth_header_format = massive_config.get('auth_header_format', 'Bearer {key}')
        
        # Rate limiting
        rate_config = massive_config.get('rate_limit', {})
        calls_per_sec = rate_config.get('calls_per_second', 10)
        self.throttler = APIThrottler(calls_per_second=calls_per_sec)
        
        # Caching
        self.cache = Cache()
        self.max_retries = rate_config.get('max_retries', 3)
        
        # Database
        self.db = MarketDB()
        
        # Request session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MassiveForecaster/1.0'
        })
        # Disable SSL verification for self-signed certificates
        self.session.verify = False
        
        # Import urllib3 to suppress SSL warnings
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except:
            pass
        
        # Mock mode flag - only true if no API key found from any source
        self.mock_mode = self.api_key is None
    
    def _get_headers(self):
        """Get request headers with auth.
        
        Returns:
            dict: Headers dictionary
        """
        headers = {}
        if self.api_key and not self.mock_mode:
            auth_value = self.auth_header_format.format(key=self.api_key)
            headers[self.auth_header_name] = auth_value
        return headers
    
    def _make_request(self, method, endpoint, params=None, timeout=10):
        """Make HTTP request with retries and throttling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            params: Query parameters
            timeout: Request timeout in seconds
        
        Returns:
            dict: Response data or None
        """
        # Mock mode - return dummy data
        if self.mock_mode:
            logger.debug(f"Mock mode: {method} {endpoint}")
            return self._generate_mock_response(endpoint, params)
        
        # Throttle if needed
        wait_time = self.throttler.get_wait_time(endpoint)
        if wait_time > 0:
            logger.debug(f"Throttling {endpoint}, waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        # Attempt with retries
        for attempt in range(self.max_retries):
            try:
                if method == 'GET':
                    response = self.session.get(url, params=params, headers=headers, timeout=timeout)
                elif method == 'POST':
                    response = self.session.post(url, params=params, headers=headers, timeout=timeout)
                else:
                    response = self.session.request(method, url, params=params, headers=headers, timeout=timeout)
                
                # Record in database
                status_code = response.status_code
                self.db.log_api_call(endpoint, symbol=params.get('symbol') if params else None,
                                     status_code=status_code)
                
                # Handle rate limit
                if status_code == 429:
                    backoff = 2 ** attempt
                    self.throttler.record_rate_limit(endpoint, backoff_seconds=backoff)
                    logger.warning(f"Rate limited on {endpoint}, backoff {backoff}s")
                    if attempt < self.max_retries - 1:
                        time.sleep(backoff)
                        continue
                
                # Handle errors
                if status_code >= 500:
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    error_msg = f"Server error {status_code} on {endpoint}"
                    logger.error(error_msg)
                    self.db.log_api_call(endpoint, symbol=params.get('symbol') if params else None,
                                        status_code=status_code, error_message=error_msg)
                    return None
                
                # Success
                if status_code == 200:
                    self.throttler.record_call(endpoint)
                    try:
                        data = response.json()
                        return data
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON response from {endpoint}")
                        return None
                else:
                    error_msg = f"HTTP {status_code} on {endpoint}"
                    logger.warning(error_msg)
                    return None
            
            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout on {endpoint}, retrying...")
                    time.sleep(1)
                else:
                    logger.error(f"Timeout on {endpoint} after {self.max_retries} attempts")
                    return None
            
            except requests.RequestException as e:
                logger.error(f"Request error on {endpoint}: {str(e)}")
                return None
        
        return None
    
    def get_quote(self, symbol):
        """Get latest quote for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Quote data with bid, ask, last_price, etc. or None
        """
        # Check cache
        cache_key = f"quote_{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        endpoint = self.config.get('massive_api', {}).get('endpoints', {}).get('quotes', '/quotes')
        data = self._make_request('GET', endpoint, params={'symbol': symbol})
        
        if data:
            quote = self._parse_quote_response(data)
            if quote:
                self.cache.set(cache_key, quote, ttl_seconds=5)
                return quote
        
        return None
    
    def get_bars(self, symbol, timeframe='1m', limit=1000):
        """Get intraday bars for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1m, 5m, 15m, etc.)
            limit: Maximum bars to return
        
        Returns:
            DataFrame: Bars data or empty DataFrame
        """
        # Check cache (bars cache is longer since they don't change)
        cache_key = f"bars_{symbol}_{timeframe}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Try database first
        db_bars = self.db.get_bars(symbol, limit=limit)
        if not db_bars.empty:
            self.cache.set(cache_key, db_bars, ttl_seconds=300)
            return db_bars
        
        endpoint = self.config.get('massive_api', {}).get('endpoints', {}).get('bars', '/bars')
        data = self._make_request('GET', endpoint, params={
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        })
        
        if data:
            bars = self._parse_bars_response(data)
            if not bars.empty:
                self.db.save_bars(symbol, bars)
                self.cache.set(cache_key, bars, ttl_seconds=300)
                return bars
        
        return pd.DataFrame()
    
    def get_top_movers(self, limit=10):
        """Get top moving stocks.
        
        Args:
            limit: Number of top movers to return
        
        Returns:
            list: List of symbol dicts with change, price, etc. or empty list
        """
        # Top movers cache is short (1 minute)
        cache_key = "top_movers"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        endpoint = self.config.get('massive_api', {}).get('endpoints', {}).get('top_movers', '/top_movers')
        data = self._make_request('GET', endpoint, params={'limit': limit})
        
        if data:
            movers = self._parse_top_movers_response(data)
            self.cache.set(cache_key, movers, ttl_seconds=60)
            return movers
        
        return []
    
    def get_market_status(self):
        """Get market status (open/closed).
        
        Returns:
            dict: Market status info or None
        """
        cache_key = "market_status"
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        endpoint = self.config.get('massive_api', {}).get('endpoints', {}).get('market_status', '/market/status')
        data = self._make_request('GET', endpoint)
        
        if data:
            status = self._parse_market_status_response(data)
            self.cache.set(cache_key, status, ttl_seconds=300)
            return status
        
        return None
    
    # Mock response generators for when API is unavailable
    
    def _generate_mock_response(self, endpoint, params):
        """Generate mock data for demo/testing.
        
        Args:
            endpoint: API endpoint
            params: Request parameters
        
        Returns:
            dict: Mock data
        """
        if 'quotes' in endpoint or endpoint.endswith('/quotes'):
            return self._mock_quote(params.get('symbol', 'AAPL') if params else 'AAPL')
        elif 'bars' in endpoint or endpoint.endswith('/bars'):
            return self._mock_bars(params.get('symbol', 'AAPL') if params else 'AAPL', params.get('limit', 100) if params else 100)
        elif 'top_movers' in endpoint:
            return self._mock_top_movers()
        elif 'market' in endpoint and 'status' in endpoint:
            return self._mock_market_status()
        return {}
    
    def _mock_quote(self, symbol):
        """Generate mock quote data."""
        import random
        base_price = 150.0
        return {
            'symbol': symbol,
            'bid': base_price - 0.05,
            'ask': base_price + 0.05,
            'bid_size': 1000,
            'ask_size': 1000,
            'last_price': base_price,
            'last_size': 100,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _mock_bars(self, symbol, limit):
        """Generate mock bars data."""
        import random
        bars = []
        price = 150.0
        now = datetime.utcnow()
        
        for i in range(limit):
            ts = now - timedelta(minutes=limit-i)
            change = random.uniform(-0.5, 0.5)
            price += change
            
            bars.append({
                'timestamp': ts.isoformat(),
                'open': price - 0.2,
                'high': price + 0.3,
                'low': price - 0.4,
                'close': price,
                'volume': random.randint(10000, 100000),
                'vwap': price
            })
        
        return {'bars': bars}
    
    def _mock_top_movers(self):
        """Generate mock top movers data."""
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN', 'NFLX', 'PYPL', 'CRM']
        movers = []
        
        for symbol in symbols:
            movers.append({
                'symbol': symbol,
                'price': 150.0 + (hash(symbol) % 100),
                'change': hash(symbol) % 10 - 5,
                'change_percent': (hash(symbol) % 10 - 5) / 100,
                'volume': hash(symbol) % 1000000
            })
        
        return {'movers': sorted(movers, key=lambda x: x['change_percent'], reverse=True)[:10]}
    
    def _mock_market_status(self):
        """Generate mock market status."""
        return {
            'market': 'US',
            'status': 'open',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    # Response parsers
    
    def _parse_quote_response(self, response):
        """Parse quote response from API.
        
        Args:
            response: API response dict
        
        Returns:
            dict: Normalized quote or None
        """
        try:
            # Handle different response formats
            if isinstance(response, dict):
                if 'quote' in response:
                    quote = response['quote']
                else:
                    quote = response
                
                return {
                    'bid': quote.get('bid'),
                    'ask': quote.get('ask'),
                    'bid_size': quote.get('bid_size'),
                    'ask_size': quote.get('ask_size'),
                    'last_price': quote.get('last_price') or quote.get('price'),
                    'last_size': quote.get('last_size'),
                    'timestamp': quote.get('timestamp') or datetime.utcnow().isoformat()
                }
        except Exception as e:
            logger.error(f"Error parsing quote response: {e}")
        
        return None
    
    def _parse_bars_response(self, response):
        """Parse bars response from API.
        
        Args:
            response: API response dict
        
        Returns:
            DataFrame: Bars data or empty DataFrame
        """
        try:
            if isinstance(response, dict):
                if 'bars' in response:
                    bars_list = response['bars']
                elif isinstance(response, list):
                    bars_list = response
                else:
                    bars_list = [response]
            else:
                bars_list = []
            
            if bars_list:
                df = pd.DataFrame(bars_list)
                required_cols = ['timestamp', 'open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    return df[required_cols + [c for c in df.columns if c not in required_cols]]
        
        except Exception as e:
            logger.error(f"Error parsing bars response: {e}")
        
        return pd.DataFrame()
    
    def _parse_top_movers_response(self, response):
        """Parse top movers response from API.
        
        Args:
            response: API response dict
        
        Returns:
            list: List of mover dicts
        """
        try:
            if isinstance(response, dict):
                if 'movers' in response:
                    return response['movers']
                elif 'data' in response:
                    return response['data']
            elif isinstance(response, list):
                return response
        except Exception as e:
            logger.error(f"Error parsing top movers response: {e}")
        
        return []
    
    def _parse_market_status_response(self, response):
        """Parse market status response from API.
        
        Args:
            response: API response dict
        
        Returns:
            dict: Market status info
        """
        try:
            if isinstance(response, dict):
                return {
                    'status': response.get('status', 'open'),
                    'market': response.get('market', 'US'),
                    'timestamp': response.get('timestamp', datetime.utcnow().isoformat())
                }
        except Exception as e:
            logger.error(f"Error parsing market status response: {e}")
        
        return {'status': 'open', 'market': 'US'}
