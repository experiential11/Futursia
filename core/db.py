"""Database operations for market data storage and forecast scoring."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


class MarketDB:
    """SQLite database for market data, forecasts, and realized scoring."""

    def __init__(self, db_path: str = "storage/market.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_schema()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS intraday_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                created_at TEXT,
                UNIQUE(symbol, timestamp)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bid REAL,
                ask REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                last_price REAL,
                last_size INTEGER,
                created_at TEXT,
                UNIQUE(symbol, timestamp)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                forecast_time INTEGER NOT NULL,
                horizon_minutes INTEGER,
                prediction_return REAL,
                prediction_return_raw REAL,
                prediction_return_norm REAL,
                confidence REAL,
                confidence_score REAL,
                direction TEXT,
                predicted_direction TEXT,
                direction_threshold REAL,
                volatility_40 REAL,
                interval_50_lower REAL,
                interval_50_upper REAL,
                interval_90_lower REAL,
                interval_90_upper REAL,
                model_name TEXT,
                features_version TEXT,
                model_version TEXT,
                config_hash TEXT,
                target_due_at INTEGER,
                realized_return REAL,
                realized_direction TEXT,
                is_correct_3class INTEGER,
                is_correct_binary INTEGER,
                is_scored INTEGER DEFAULT 0,
                score_status TEXT,
                model_status TEXT,
                created_at TEXT,
                scored_at TEXT,
                UNIQUE(symbol, forecast_time)
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                symbol TEXT,
                status_code INTEGER,
                timestamp INTEGER,
                error_message TEXT,
                created_at TEXT
            )
            """
        )

        self._ensure_forecast_columns(cursor)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON intraday_bars(symbol, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes(symbol, timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_time ON forecasts(symbol, forecast_time DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_due ON forecasts(target_due_at, is_scored)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_forecasts_scored ON forecasts(is_scored, forecast_time DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_time ON api_calls(timestamp DESC)")

        conn.commit()
        conn.close()

    def _ensure_forecast_columns(self, cursor):
        cursor.execute("PRAGMA table_info(forecasts)")
        existing = {row[1] for row in cursor.fetchall()}

        required = {
            "horizon_minutes": "INTEGER",
            "prediction_return_raw": "REAL",
            "prediction_return_norm": "REAL",
            "confidence_score": "REAL",
            "predicted_direction": "TEXT",
            "direction_threshold": "REAL",
            "volatility_40": "REAL",
            "model_name": "TEXT",
            "features_version": "TEXT",
            "model_version": "TEXT",
            "config_hash": "TEXT",
            "target_due_at": "INTEGER",
            "realized_return": "REAL",
            "realized_direction": "TEXT",
            "is_correct_3class": "INTEGER",
            "is_correct_binary": "INTEGER",
            "is_scored": "INTEGER DEFAULT 0",
            "score_status": "TEXT",
            "scored_at": "TEXT",
        }
        for col, col_type in required.items():
            if col not in existing:
                cursor.execute(f"ALTER TABLE forecasts ADD COLUMN {col} {col_type}")

    def save_bars(self, symbol, bars):
        """Save intraday bars to database."""
        if bars is None or bars.empty:
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()

        for _, row in bars.iterrows():
            try:
                ts = int(pd.Timestamp(row["timestamp"]).timestamp())
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO intraday_bars
                    (symbol, timestamp, open, high, low, close, volume, vwap, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(symbol).upper(),
                        ts,
                        float(row.get("open")),
                        float(row.get("high")),
                        float(row.get("low")),
                        float(row.get("close")),
                        int(float(row.get("volume", 0) or 0)),
                        float(row.get("vwap")) if row.get("vwap") is not None else None,
                        now,
                    ),
                )
            except Exception:
                continue

        conn.commit()
        conn.close()

    def get_bars(self, symbol, limit: int = 1000):
        """Get intraday bars for one symbol sorted ascending by timestamp."""
        conn = self._get_connection()
        query = """
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM intraday_bars
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(str(symbol).upper(), int(limit)))
        conn.close()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def get_bars_by_symbols(
        self,
        symbols: Iterable[str],
        limit_per_symbol: int = 1500,
        end_timestamp: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Get bars for multiple symbols keyed by symbol."""
        out: Dict[str, pd.DataFrame] = {}
        conn = self._get_connection()
        try:
            for symbol in symbols:
                sym = str(symbol).upper()
                if end_timestamp is None:
                    query = """
                        SELECT timestamp, open, high, low, close, volume, vwap
                        FROM intraday_bars
                        WHERE symbol = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    params = (sym, int(limit_per_symbol))
                else:
                    query = """
                        SELECT timestamp, open, high, low, close, volume, vwap
                        FROM intraday_bars
                        WHERE symbol = ? AND timestamp <= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """
                    params = (sym, int(end_timestamp), int(limit_per_symbol))
                df = pd.read_sql_query(query, conn, params=params)
                if df.empty:
                    out[sym] = df
                    continue
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                out[sym] = df.sort_values("timestamp").reset_index(drop=True)
        finally:
            conn.close()
        return out

    def get_symbol_counts(self, min_rows: int = 1):
        """Return symbols with stored bar counts."""
        conn = self._get_connection()
        query = """
            SELECT symbol, COUNT(*) as bar_count
            FROM intraday_bars
            GROUP BY symbol
            HAVING COUNT(*) >= ?
            ORDER BY bar_count DESC
        """
        df = pd.read_sql_query(query, conn, params=(int(min_rows),))
        conn.close()
        return df

    def save_quote(self, symbol, quote):
        """Save latest quote."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        ts = int(datetime.utcnow().timestamp())

        cursor.execute(
            """
            INSERT OR REPLACE INTO quotes
            (symbol, timestamp, bid, ask, bid_size, ask_size, last_price, last_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(symbol).upper(),
                ts,
                quote.get("bid"),
                quote.get("ask"),
                quote.get("bid_size"),
                quote.get("ask_size"),
                quote.get("last_price"),
                quote.get("last_size"),
                now,
            ),
        )

        conn.commit()
        conn.close()

    def save_forecast(self, symbol, forecast):
        """Persist forecast metadata for later realized scoring."""
        conn = self._get_connection()
        cursor = conn.cursor()
        created_at = forecast.get("created_at") or datetime.utcnow().isoformat()
        forecast_time = int(forecast.get("forecast_time") or datetime.utcnow().timestamp())
        horizon = int(forecast.get("horizon_minutes") or 40)
        target_due_at = int(forecast.get("target_due_at") or (forecast_time + (horizon * 60)))

        pred_raw = forecast.get("prediction_return_raw", forecast.get("prediction_return"))
        pred_norm = forecast.get("prediction_return_norm")
        direction = forecast.get("predicted_direction", forecast.get("direction"))
        confidence = forecast.get("confidence_score", forecast.get("confidence"))

        cursor.execute(
            """
            INSERT OR REPLACE INTO forecasts (
                symbol, forecast_time, horizon_minutes,
                prediction_return, prediction_return_raw, prediction_return_norm,
                confidence, confidence_score,
                direction, predicted_direction, direction_threshold, volatility_40,
                interval_50_lower, interval_50_upper, interval_90_lower, interval_90_upper,
                model_name, model_status, features_version, model_version, config_hash,
                target_due_at, created_at, is_scored, score_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(symbol).upper(),
                forecast_time,
                horizon,
                pred_raw,
                pred_raw,
                pred_norm,
                confidence,
                confidence,
                direction,
                direction,
                forecast.get("direction_threshold"),
                forecast.get("volatility_40"),
                forecast.get("interval_50_lower"),
                forecast.get("interval_50_upper"),
                forecast.get("interval_90_lower"),
                forecast.get("interval_90_upper"),
                forecast.get("model_name"),
                forecast.get("model_status"),
                forecast.get("features_version"),
                forecast.get("model_version"),
                forecast.get("config_hash"),
                target_due_at,
                created_at,
                0,
                "pending",
            ),
        )

        conn.commit()
        conn.close()

    def get_latest_forecast(self, symbol):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM forecasts
            WHERE symbol = ?
            ORDER BY forecast_time DESC
            LIMIT 1
            """,
            (str(symbol).upper(),),
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    @staticmethod
    def _direction_from_return(value: Optional[float], threshold: float):
        if value is None or not np.isfinite(value):
            return None
        if value > threshold:
            return "UP"
        if value < -threshold:
            return "DOWN"
        return "FLAT"

    def _nearest_close(self, cursor, symbol: str, target_ts: int, tolerance_seconds: int):
        cursor.execute(
            """
            SELECT close, timestamp
            FROM intraday_bars
            WHERE symbol = ? AND timestamp BETWEEN ? AND ?
            ORDER BY ABS(timestamp - ?) ASC
            LIMIT 1
            """,
            (
                symbol,
                int(target_ts - tolerance_seconds),
                int(target_ts + tolerance_seconds),
                int(target_ts),
            ),
        )
        row = cursor.fetchone()
        return (float(row["close"]), int(row["timestamp"])) if row else (None, None)

    def score_due_forecasts(
        self,
        max_rows: int = 500,
        default_threshold: float = 0.001,
        tolerance_seconds: int = 30 * 60,
    ):
        """Score forecasts whose target_due_at has passed and data is available."""
        conn = self._get_connection()
        cursor = conn.cursor()
        now_ts = int(datetime.utcnow().timestamp())
        now_iso = datetime.utcnow().isoformat()

        cursor.execute(
            """
            SELECT
                id, symbol, forecast_time, target_due_at,
                COALESCE(prediction_return_raw, prediction_return) AS pred_raw,
                prediction_return_norm,
                COALESCE(predicted_direction, direction) AS pred_direction,
                direction_threshold
            FROM forecasts
            WHERE target_due_at IS NOT NULL
              AND target_due_at <= ?
              AND COALESCE(is_scored, 0) = 0
              AND COALESCE(prediction_return_raw, prediction_return) IS NOT NULL
            ORDER BY target_due_at ASC
            LIMIT ?
            """,
            (now_ts, int(max_rows)),
        )
        rows = cursor.fetchall()

        scored = 0
        skipped = 0
        for row in rows:
            forecast_id = int(row["id"])
            symbol = str(row["symbol"]).upper()
            forecast_time = int(row["forecast_time"])
            due_at = int(row["target_due_at"])
            pred_raw = float(row["pred_raw"])

            # Entry close uses the latest bar at or before forecast_time.
            cursor.execute(
                """
                SELECT close, timestamp
                FROM intraday_bars
                WHERE symbol = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (symbol, forecast_time),
            )
            entry_row = cursor.fetchone()
            if not entry_row:
                skipped += 1
                continue
            entry_close = float(entry_row["close"])

            due_close, due_ts = self._nearest_close(cursor, symbol, due_at, tolerance_seconds=tolerance_seconds)
            if due_close is None:
                skipped += 1
                continue

            if entry_close <= 0:
                skipped += 1
                continue

            realized_return = (due_close - entry_close) / entry_close
            threshold = float(row["direction_threshold"] or default_threshold)
            pred_direction = row["pred_direction"] or self._direction_from_return(pred_raw, threshold)
            realized_direction = self._direction_from_return(realized_return, threshold)

            is_correct_3 = int(pred_direction == realized_direction) if realized_direction is not None else None
            if pred_direction in {"UP", "DOWN"} and realized_direction in {"UP", "DOWN"}:
                is_correct_binary = int(pred_direction == realized_direction)
            else:
                is_correct_binary = None

            cursor.execute(
                """
                UPDATE forecasts
                SET
                    realized_return = ?,
                    realized_direction = ?,
                    is_correct_3class = ?,
                    is_correct_binary = ?,
                    is_scored = 1,
                    score_status = ?,
                    scored_at = ?
                WHERE id = ?
                """,
                (
                    float(realized_return),
                    realized_direction,
                    is_correct_3,
                    is_correct_binary,
                    f"scored@{due_ts}",
                    now_iso,
                    forecast_id,
                ),
            )
            scored += 1

        conn.commit()
        conn.close()
        return {"due_rows": len(rows), "scored": scored, "skipped": skipped}

    def get_live_oos_metrics(self, lookback_hours: int = 24, symbol: Optional[str] = None):
        """Return rolling realized OOS metrics."""
        cutoff = int(datetime.utcnow().timestamp()) - int(lookback_hours * 3600)
        conn = self._get_connection()
        if symbol:
            query = """
                SELECT
                    symbol,
                    forecast_time,
                    COALESCE(prediction_return_raw, prediction_return) AS predicted_return_raw,
                    prediction_return_norm,
                    volatility_40,
                    realized_return,
                    is_correct_3class,
                    is_correct_binary
                FROM forecasts
                WHERE is_scored = 1
                  AND realized_return IS NOT NULL
                  AND forecast_time >= ?
                  AND symbol = ?
            """
            df = pd.read_sql_query(query, conn, params=(cutoff, str(symbol).upper()))
        else:
            query = """
                SELECT
                    symbol,
                    forecast_time,
                    COALESCE(prediction_return_raw, prediction_return) AS predicted_return_raw,
                    prediction_return_norm,
                    volatility_40,
                    realized_return,
                    is_correct_3class,
                    is_correct_binary
                FROM forecasts
                WHERE is_scored = 1
                  AND realized_return IS NOT NULL
                  AND forecast_time >= ?
            """
            df = pd.read_sql_query(query, conn, params=(cutoff,))
        conn.close()

        if df.empty:
            return {
                "lookback_hours": int(lookback_hours),
                "total_scored": 0,
                "accuracy_3class_pct": None,
                "accuracy_binary_excl_flat_pct": None,
                "mae_return": None,
                "mae_norm": None,
                "by_symbol": [],
            }

        for col in [
            "predicted_return_raw",
            "prediction_return_norm",
            "volatility_40",
            "realized_return",
            "is_correct_3class",
            "is_correct_binary",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        mae_return = float((df["predicted_return_raw"] - df["realized_return"]).abs().mean())
        if "prediction_return_norm" in df.columns:
            denom = df["volatility_40"].replace(0, np.nan)
            realized_norm = df["realized_return"] / denom
            mae_norm = float((df["prediction_return_norm"] - realized_norm).abs().dropna().mean())
            if not np.isfinite(mae_norm):
                mae_norm = None
        else:
            mae_norm = None

        acc3 = float(df["is_correct_3class"].dropna().mean() * 100.0)
        bin_series = df["is_correct_binary"].dropna()
        acc_bin = float(bin_series.mean() * 100.0) if not bin_series.empty else None

        by_symbol = []
        grouped = df.groupby("symbol", dropna=True)
        for sym, g in grouped:
            g_bin = g["is_correct_binary"].dropna()
            by_symbol.append(
                {
                    "symbol": str(sym),
                    "total_scored": int(len(g)),
                    "accuracy_3class_pct": float(g["is_correct_3class"].dropna().mean() * 100.0),
                    "accuracy_binary_excl_flat_pct": float(g_bin.mean() * 100.0) if not g_bin.empty else None,
                    "mae_return": float((g["predicted_return_raw"] - g["realized_return"]).abs().mean()),
                }
            )

        by_symbol.sort(key=lambda x: x["total_scored"], reverse=True)
        return {
            "lookback_hours": int(lookback_hours),
            "total_scored": int(len(df)),
            "accuracy_3class_pct": acc3,
            "accuracy_binary_excl_flat_pct": acc_bin,
            "mae_return": mae_return,
            "mae_norm": mae_norm,
            "by_symbol": by_symbol,
        }

    def log_api_call(self, endpoint, symbol=None, status_code=None, error_message=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        ts = int(datetime.utcnow().timestamp())
        cursor.execute(
            """
            INSERT INTO api_calls
            (endpoint, symbol, status_code, timestamp, error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (endpoint, symbol, status_code, ts, error_message, now),
        )
        conn.commit()
        conn.close()

    def get_api_call_stats(self, minutes: int = 5):
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff_time = int(datetime.utcnow().timestamp()) - (int(minutes) * 60)
        cursor.execute(
            """
            SELECT COUNT(*) as total, status_code,
                   COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as errors
            FROM api_calls
            WHERE timestamp > ?
            GROUP BY status_code
            """,
            (cutoff_time,),
        )
        rows = cursor.fetchall()
        conn.close()

        stats = {"total": 0, "errors": 0, "by_status": {}}
        for row in rows:
            stats["total"] += row["total"]
            stats["errors"] += row["errors"]
            if row["status_code"] is not None:
                stats["by_status"][row["status_code"]] = row["total"]
        return stats

    def cleanup_old_data(self, days: int = 30):
        """Remove old intraday/quote/api rows."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff = int(datetime.utcnow().timestamp()) - (int(days) * 86400)
        cursor.execute("DELETE FROM intraday_bars WHERE timestamp < ?", (cutoff,))
        cursor.execute("DELETE FROM quotes WHERE timestamp < ?", (cutoff,))
        cursor.execute("DELETE FROM api_calls WHERE timestamp < ?", (cutoff,))
        conn.commit()
        conn.close()
