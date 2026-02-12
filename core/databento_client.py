"""
Databento API client with app-compatible interfaces.
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from core.db import MarketDB


logger = logging.getLogger(__name__)


class DatabentoClient:
    """Client for Databento market data API."""

    def __init__(self, config: Dict):
        cfg = config.get("databento_api", {})
        self.api_key = os.getenv("DATABENTO_API_KEY") or cfg.get("api_key", "")
        self.base_url = cfg.get("base_url", "https://hist.databento.com/v0")
        self.dataset = cfg.get("dataset", "DBEQ.BASIC")
        self.schema = cfg.get("schema", "ohlcv-1m")
        self.stype_in = cfg.get("stype_in", "raw_symbol")

        rate_config = cfg.get("rate_limit", {})
        self.calls_per_minute = rate_config.get("calls_per_minute", 60)
        self.min_call_interval = 60.0 / max(1, self.calls_per_minute)
        self.last_call_time = 0.0

        self.mock_mode = not bool(self.api_key)
        self.session = requests.Session()
        if self.api_key:
            # Databento REST uses HTTP Basic auth with API key as username.
            self.session.auth = (self.api_key, "")
        self._dataset_end_cache = {}

        self.db = MarketDB()
        logger.info("DatabentoClient initialized (key present=%s)", bool(self.api_key))

    def _rate_limit(self):
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_call_interval:
            time.sleep(self.min_call_interval - elapsed)
        self.last_call_time = time.time()

    def _request_json(self, endpoint: str, params: Dict, timeout: int = 15) -> Optional[Dict]:
        if self.mock_mode:
            return None
        try:
            self._rate_limit()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            resp = self.session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Databento request failed on %s: %s", endpoint, e)
            return None

    def _request_text(self, endpoint: str, params: Dict, timeout: int = 30) -> Optional[str]:
        if self.mock_mode:
            return None
        try:
            self._rate_limit()
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            resp = self.session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning("Databento request failed on %s: %s", endpoint, e)
            return None

    def _get_dataset_available_end(self, dataset: str) -> Optional[pd.Timestamp]:
        now = time.time()
        cached = self._dataset_end_cache.get(dataset)
        if cached and (now - cached["ts"]) < 120:
            return cached["end"]

        payload = self._request_json("metadata.get_dataset_range", {"dataset": dataset}, timeout=15)
        if not payload or "end" not in payload:
            return None
        try:
            end_ts = pd.to_datetime(payload["end"], utc=True, errors="coerce")
            if pd.isna(end_ts):
                return None
            self._dataset_end_cache[dataset] = {"end": end_ts, "ts": now}
            return end_ts
        except Exception:
            return None

    @staticmethod
    def _maybe_scale_price(value):
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        # Databento equity prices may be in nano-dollars.
        if abs(v) > 1_000_000:
            return v / 1_000_000_000.0
        return v

    def _parse_ohlcv_jsonl(self, text: Optional[str]) -> pd.DataFrame:
        if not text:
            return pd.DataFrame()
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            hd = rec.get("hd", {}) if isinstance(rec, dict) else {}
            ts_raw = hd.get("ts_event") or rec.get("ts_event") or rec.get("timestamp")
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            o = self._maybe_scale_price(rec.get("open", rec.get("o")))
            h = self._maybe_scale_price(rec.get("high", rec.get("h")))
            l = self._maybe_scale_price(rec.get("low", rec.get("l")))
            c = self._maybe_scale_price(rec.get("close", rec.get("c")))
            v = rec.get("volume", rec.get("v", 0))
            try:
                v = float(v)
            except Exception:
                v = 0.0

            if None in (o, h, l, c):
                continue

            rows.append(
                {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                    "vwap": c,
                }
            )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    def _parse_ohlcv_records(self, payload: Optional[Dict]) -> pd.DataFrame:
        if not payload:
            return pd.DataFrame()

        records = payload.get("data", payload if isinstance(payload, list) else [])
        if not isinstance(records, list) or len(records) == 0:
            return pd.DataFrame()

        rows = []
        for rec in records:
            if not isinstance(rec, dict):
                continue

            ts_raw = rec.get("ts_event") or rec.get("timestamp") or rec.get("ts_recv")
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.isna(ts):
                continue

            rows.append(
                {
                    "timestamp": ts,
                    "open": rec.get("open", rec.get("o")),
                    "high": rec.get("high", rec.get("h")),
                    "low": rec.get("low", rec.get("l")),
                    "close": rec.get("close", rec.get("c")),
                    "volume": rec.get("volume", rec.get("v", 0)),
                    "vwap": rec.get("vwap"),
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).dropna(subset=["timestamp", "open", "high", "low", "close"])
        if df.empty:
            return df

        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _normalize_quote(self, symbol: str, price: float, high: float, low: float, open_price: float, prev_close: float):
        bid = price - 0.01
        ask = price + 0.01
        ts = int(time.time())
        iso_ts = datetime.utcnow().isoformat()
        return {
            "symbol": symbol.upper(),
            "bid": bid,
            "ask": ask,
            "bid_size": None,
            "ask_size": None,
            "last_price": price,
            "last_size": None,
            "timestamp": iso_ts,
            "c": price,
            "h": high,
            "l": low,
            "o": open_price,
            "pc": prev_close,
            "t": ts,
        }

    def _synthetic_bars(self, symbol: str, limit: int, end_time: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        now = end_time.to_pydatetime().replace(tzinfo=None) if end_time is not None else datetime.utcnow()
        base_price = 100.0 + (abs(hash(symbol.upper())) % 200)
        times = [now - timedelta(minutes=(limit - i)) for i in range(limit)]
        drift = np.linspace(-0.3, 0.3, limit)
        noise = np.random.normal(0, 0.2, limit)
        close = base_price + drift + noise
        open_ = close + np.random.normal(0, 0.05, limit)
        high = np.maximum(open_, close) + np.abs(np.random.normal(0, 0.08, limit))
        low = np.minimum(open_, close) - np.abs(np.random.normal(0, 0.08, limit))
        volume = np.random.randint(10000, 250000, limit)

        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(times, utc=True),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "vwap": close,
            }
        )

    def get_quote(self, symbol: str) -> Optional[Dict]:
        bars = self.get_bars(symbol, limit=2)
        if bars is None or bars.empty:
            bars = self._synthetic_bars(symbol, 2)

        try:
            latest = bars.iloc[-1]
            prev = bars.iloc[-2] if len(bars) > 1 else latest
            quote = self._normalize_quote(
                symbol=symbol,
                price=float(latest["close"]),
                high=float(latest["high"]),
                low=float(latest["low"]),
                open_price=float(latest["open"]),
                prev_close=float(prev["close"]),
            )
            self.db.save_quote(symbol.upper(), quote)
            return quote
        except Exception as e:
            logger.error("Databento get_quote error for %s: %s", symbol, e)
            return None

    def get_bars(self, symbol: str, timeframe: str = "1m", limit: int = 1000) -> pd.DataFrame:
        symbol = symbol.upper()

        db_bars = self.db.get_bars(symbol, limit=limit)
        if self.mock_mode and not db_bars.empty and len(db_bars) >= limit:
            return db_bars

        now_utc = pd.Timestamp.utcnow()
        avail_end = self._get_dataset_available_end(self.dataset)
        end_ts = min(now_utc, avail_end) if avail_end is not None else now_utc
        start_ts = end_ts - pd.Timedelta(minutes=limit)

        end = end_ts.to_pydatetime()
        start = start_ts.to_pydatetime()
        params = {
            "dataset": self.dataset,
            "schema": self.schema if timeframe == "1m" else self.schema,
            "symbols": symbol,
            "stype_in": self.stype_in,
            "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "encoding": "json",
        }

        text = self._request_text("timeseries.get_range", params=params)
        df = self._parse_ohlcv_jsonl(text)

        # Retry with a regular-session window if the latest range has no bars.
        if df.empty and avail_end is not None:
            day = avail_end.tz_convert("UTC").normalize()
            session_end = day + pd.Timedelta(hours=21)
            session_start = session_end - pd.Timedelta(minutes=limit)
            if session_end > avail_end:
                session_end = avail_end
                session_start = session_end - pd.Timedelta(minutes=limit)
            if session_end > session_start:
                params["start"] = session_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                params["end"] = session_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                text = self._request_text("timeseries.get_range", params=params)
                df = self._parse_ohlcv_jsonl(text)

        if df.empty:
            min_required = min(limit, 100)
            if not db_bars.empty and len(db_bars) >= min_required:
                df = db_bars.copy()
            else:
                if not self.mock_mode:
                    logger.warning("No Databento bars for %s, using synthetic fallback", symbol)
                df = self._synthetic_bars(symbol, min(limit, 500))

        # Backfill with synthetic bars when history is too short for model features.
        min_needed = min(limit, 220)
        if len(df) < min_needed:
            missing = min(limit, 500) - len(df)
            if missing > 0:
                earliest = pd.to_datetime(df["timestamp"].min(), utc=True, errors="coerce")
                if pd.isna(earliest):
                    earliest = pd.Timestamp.utcnow()
                synth = self._synthetic_bars(symbol, missing, end_time=earliest - pd.Timedelta(minutes=1))
                df = pd.concat([synth, df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        if len(df) > limit:
            df = df.tail(limit).reset_index(drop=True)

        self.db.save_bars(symbol, df)
        return df

    def get_news(self, symbol: str, limit: int = 20) -> Optional[List[Dict]]:
        logger.info("Databento does not provide news endpoint; returning None for %s", symbol)
        return None

    def get_top_movers(self, limit: int = 10) -> Optional[List[Dict]]:
        symbols = [
            "AAPL",
            "MSFT",
            "GOOGL",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "JNJ",
            "V",
            "AMD",
            "NFLX",
        ]
        movers = []

        try:
            for s in symbols:
                q = self.get_quote(s)
                if not q:
                    continue

                price = float(q.get("last_price", q.get("c", 0)) or 0)
                prev_close = float(q.get("pc", price) or price)
                change = price - prev_close
                pct = (change / prev_close) * 100 if prev_close else 0.0
                movers.append(
                    {
                        "symbol": s,
                        "price": price,
                        "change": change,
                        "change_percent": pct,
                        "percent_change": pct,
                        "high": float(q.get("h", price) or price),
                        "low": float(q.get("l", price) or price),
                        "open": float(q.get("o", price) or price),
                        "prev_close": prev_close,
                    }
                )

            movers.sort(key=lambda x: abs(x.get("percent_change", 0)), reverse=True)
            return movers[:limit]
        except Exception as e:
            logger.error("Databento get_top_movers error: %s", e)
            return None

    def get_market_status(self) -> Dict:
        now = datetime.utcnow()
        weekday = now.weekday()
        is_weekday = weekday < 5
        open_time = now.replace(hour=14, minute=30, second=0, microsecond=0)  # 9:30 ET in UTC
        close_time = now.replace(hour=21, minute=0, second=0, microsecond=0)  # 16:00 ET in UTC
        is_open = is_weekday and open_time <= now <= close_time
        return {
            "market": "US",
            "status": "open" if is_open else "closed",
            "timestamp": now.isoformat(),
        }
