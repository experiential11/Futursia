"""Yahoo Finance (yfinance-style) client for US/Nasdaq symbols."""

from datetime import datetime
from typing import Dict, List, Optional
import logging

import pandas as pd
import requests

from core.db import MarketDB

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Market data client backed by Yahoo Finance chart endpoints."""

    def __init__(self, config: Dict):
        self.config = config.get("yfinance_api", {})
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.mock_mode = False
        self.db = MarketDB()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})

    def _fetch_chart(self, symbol: str, range_: str = "5d", interval: str = "1m") -> Optional[Dict]:
        url = f"{self.base_url}/{symbol.upper()}"
        params = {"range": range_, "interval": interval}
        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            result = payload.get("chart", {}).get("result", [])
            if not result:
                return None
            return result[0]
        except Exception as e:
            logger.error("Yahoo chart fetch error for %s: %s", symbol, e)
            return None

    def _chart_to_df(self, chart: Dict, limit: int) -> pd.DataFrame:
        timestamps = chart.get("timestamp") or []
        indicators = chart.get("indicators", {}).get("quote", [])
        if not indicators:
            return pd.DataFrame()
        quote = indicators[0]

        rows = []
        for i, ts in enumerate(timestamps):
            try:
                o = quote.get("open", [None])[i]
                h = quote.get("high", [None])[i]
                l = quote.get("low", [None])[i]
                c = quote.get("close", [None])[i]
                v = quote.get("volume", [0])[i]
            except Exception:
                continue

            if None in (o, h, l, c):
                continue
            rows.append(
                {
                    "timestamp": pd.to_datetime(int(ts), unit="s", utc=True),
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v or 0),
                    "vwap": float(c),
                }
            )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).tail(limit).reset_index(drop=True)

    def get_bars(self, symbol: str, timeframe: str = "1m", limit: int = 1000) -> pd.DataFrame:
        symbol = symbol.upper()
        interval = "1m" if timeframe in ("1m", "1min", "1") else "5m"
        range_ = "5d" if interval == "1m" else "1mo"

        chart = self._fetch_chart(symbol, range_=range_, interval=interval)
        if chart:
            df = self._chart_to_df(chart, limit=limit)
            if not df.empty:
                self.db.save_bars(symbol, df)
                return df

        # DB fallback only when live fetch fails.
        db_bars = self.db.get_bars(symbol, limit=limit)
        return db_bars if not db_bars.empty else pd.DataFrame()

    def get_quote(self, symbol: str) -> Optional[Dict]:
        symbol = symbol.upper()
        chart = self._fetch_chart(symbol, range_="1d", interval="1m")
        if not chart:
            bars = self.db.get_bars(symbol, limit=2)
            if bars.empty:
                return None
            meta = {}
            df = bars
        else:
            meta = chart.get("meta", {}) or {}
            df = self._chart_to_df(chart, limit=390)
            if df.empty:
                return None

        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last
        day_open = float(df.iloc[0]["open"])
        day_high = float(df["high"].max())
        day_low = float(df["low"].min())
        price = float(last["close"])
        prev_close = float(meta.get("previousClose") or prev["close"])

        quote = {
            "symbol": symbol,
            "bid": price - 0.01,
            "ask": price + 0.01,
            "bid_size": None,
            "ask_size": None,
            "last_price": price,
            "last_size": None,
            "timestamp": datetime.utcnow().isoformat(),
            "c": price,
            "h": day_high,
            "l": day_low,
            "o": day_open,
            "pc": float(prev_close),
            "t": int(datetime.utcnow().timestamp()),
        }
        self.db.save_quote(symbol, quote)
        return quote

    def get_top_movers(self, limit: int = 10) -> Optional[List[Dict]]:
        symbols = self.config.get("watch_symbols") or self.config.get("nasdaq_symbols") or [
            "AAPL", "MSFT", "NVDA", "7203.T", "6758.T"
        ]
        movers = []
        for s in symbols:
            q = self.get_quote(s)
            if not q:
                continue
            price = float(q.get("last_price", 0))
            prev = float(q.get("pc", price) or price)
            change = price - prev
            pct = (change / prev) * 100 if prev else 0.0
            movers.append(
                {
                    "symbol": s,
                    "price": price,
                    "change": change,
                    "change_percent": pct,
                    "percent_change": pct,
                    "high": float(q.get("h", price)),
                    "low": float(q.get("l", price)),
                    "open": float(q.get("o", price)),
                    "prev_close": prev,
                }
            )
        movers.sort(key=lambda x: abs(x.get("percent_change", 0)), reverse=True)
        return movers[:limit]

    def get_market_status(self) -> Dict:
        now = datetime.utcnow()
        weekday = now.weekday()
        is_weekday = weekday < 5
        open_time = now.replace(hour=14, minute=30, second=0, microsecond=0)
        close_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
        is_open = is_weekday and open_time <= now <= close_time
        return {"market": "US", "status": "open" if is_open else "closed", "timestamp": now.isoformat()}
