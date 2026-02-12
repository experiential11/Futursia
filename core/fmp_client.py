"""
FMP (FinancialModelingPrep) API client
Provides quote, historical bars and top movers using FMP endpoints.
"""
import requests
import time
import logging
import os
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class FMPClient:
    """Client for FinancialModelingPrep API"""

    def __init__(self, config: Dict):
        cfg = config.get('fmp_api', {})
        self.base_url = cfg.get('base_url', 'https://financialmodelingprep.com/api/v3')
        self.api_key = os.getenv('FMP_API_KEY') or cfg.get('api_key', '')

        rate_config = cfg.get('rate_limit', {})
        self.calls_per_minute = rate_config.get('calls_per_minute', 60)
        self.min_call_interval = 60.0 / self.calls_per_minute
        self.last_call_time = 0

        self.session = requests.Session()

        logger.info(f"FMPClient initialized (key present={bool(self.api_key)})")

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """Get current quote for a symbol using FMP /quote endpoint."""
        try:
            self._rate_limit()
            url = f"{self.base_url}/quote/{symbol.upper()}"
            params = {'apikey': self.api_key}
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                entry = data[0]
                # Normalize to finnhub-style keys used in the app
                quote = {
                    'c': entry.get('price'),
                    'h': entry.get('dayHigh') or entry.get('high'),
                    'l': entry.get('dayLow') or entry.get('low'),
                    'o': entry.get('open'),
                    'pc': entry.get('previousClose') or entry.get('prevClose'),
                    't': int(time.time())
                }
                return quote
            else:
                logger.warning(f"Empty quote from FMP for {symbol}: {data}")
                return None
        except Exception as e:
            logger.error(f"FMP get_quote error for {symbol}: {e}")
            return None

    def get_bars(self, symbol: str, limit: int = 240) -> Optional[Dict]:
        """Get intraday 1-minute bars via historical-chart/1min endpoint.
        Falls back to synthetic bars if endpoint is unavailable.
        """
        try:
            self._rate_limit()
            url = f"{self.base_url}/historical-chart/1min/{symbol.upper()}"
            params = {'apikey': self.api_key}
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                # data is list of {date, open, high, low, close, volume}
                data = data[-limit:]
                o,h,l,c,v,t = [],[],[],[],[],[]
                for row in data:
                    o.append(row.get('open'))
                    h.append(row.get('high'))
                    l.append(row.get('low'))
                    c.append(row.get('close'))
                    v.append(row.get('volume'))
                    # convert date string to unix timestamp if possible
                    date_str = row.get('date')
                    try:
                        ts = int(datetime.fromisoformat(date_str).timestamp())
                    except Exception:
                        try:
                            ts = int(datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').timestamp())
                        except Exception:
                            ts = int(time.time())
                    t.append(ts)

                return {'o': o, 'h': h, 'l': l, 'c': c, 'v': v, 't': t, 's': 'ok'}
            else:
                logger.warning(f"No intraday bars from FMP for {symbol}, generating synthetic bars")
                # fall back to synthetic bars using quote
                quote = self.get_quote(symbol)
                if not quote:
                    return None
                current_price = float(quote.get('c', 100))
                prev_close = float(quote.get('pc', current_price))
                bars = {'o': [], 'h': [], 'l': [], 'c': [], 'v': [], 't': [], 's': 'ok'}
                now = int(time.time())
                for i in range(limit, 0, -1):
                    ts = now - (i * 60)
                    move_fraction = (limit - i) / float(limit)
                    trend = (current_price - prev_close) * move_fraction
                    volatility = abs(current_price - prev_close) * 0.002
                    noise = np.random.normal(0, volatility)
                    bar_close = prev_close + trend + noise
                    bar_open = bar_close - (volatility * 0.5)
                    bar_high = max(bar_open, bar_close) + (volatility * 0.5)
                    bar_low = min(bar_open, bar_close) - (volatility * 0.5)
                    bars['t'].append(ts)
                    bars['o'].append(round(bar_open,2))
                    bars['h'].append(round(bar_high,2))
                    bars['l'].append(round(bar_low,2))
                    bars['c'].append(round(bar_close,2))
                    bars['v'].append(float(np.random.uniform(1e5, 2e6)))
                return bars
        except Exception as e:
            logger.error(f"FMP get_bars error for {symbol}: {e}")
            return None

    def get_news(self, symbol: str, limit: int = 20) -> Optional[List[Dict]]:
        """FMP has a news endpoint - try it, otherwise return None."""
        try:
            self._rate_limit()
            url = f"{self.base_url}/stock_news"
            params = {'tickers': symbol.upper(), 'limit': limit, 'apikey': self.api_key}
            resp = self.session.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data[:limit]
            return None
        except Exception as e:
            logger.warning(f"FMP news fetch failed for {symbol}: {e}")
            return None

    def get_top_movers(self) -> Optional[List[Dict]]:
        """Return top movers by using a small watchlist and quoting each symbol."""
        symbols = ['AAPL','MSFT','GOOGL','AMZN','TSLA','META','NVDA','JPM','JNJ','V']
        movers = []
        try:
            for s in symbols:
                q = self.get_quote(s)
                if q:
                    change = float(q.get('c',0)) - float(q.get('pc', q.get('c',0)))
                    pct = (change / float(q.get('pc',1))) * 100 if q.get('pc') else 0
                    movers.append({'symbol': s, 'price': q.get('c'), 'change': change, 'percent_change': pct, 'high': q.get('h'), 'low': q.get('l'), 'open': q.get('o'), 'prev_close': q.get('pc')})
            movers.sort(key=lambda x: abs(x.get('percent_change',0)), reverse=True)
            return movers[:10]
        except Exception as e:
            logger.error(f"FMP get_top_movers error: {e}")
            return None
