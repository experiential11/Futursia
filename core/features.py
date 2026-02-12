"""Feature engineering for the forecasting model."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Extract leakage-safe features from symbol and market bar data."""

    def __init__(self, config: dict):
        """Initialize feature configuration."""
        features_cfg = config.get("features", {})
        market_cfg = config.get("market_context", {})

        self.return_lags = self._as_int_list(
            features_cfg.get("return_lags", [1, 2, 3, 5, 10, 20, 40]),
            default=[1, 2, 3, 5, 10, 20, 40],
        )
        self.volatility_periods = self._as_int_list(
            features_cfg.get("volatility_periods", [10, 20, 40, 80]),
            default=[10, 20, 40, 80],
        )
        self.ma_periods = self._as_int_list(
            features_cfg.get("ma_periods", [10, 20, 40, 80]),
            default=[10, 20, 40, 80],
        )
        self.trend_windows = self._as_int_list(
            features_cfg.get("trend_windows", [20, 40, 80, 120]),
            default=[20, 40, 80, 120],
        )
        self.volume_windows = self._as_int_list(
            features_cfg.get("volume_spike_windows", [10, 20, 40, 80]),
            default=[10, 20, 40, 80],
        )
        self.vwap_windows = self._as_int_list(
            features_cfg.get("vwap_windows", [10, 20, 40]),
            default=[10, 20, 40],
        )
        self.beta_windows = self._as_int_list(
            market_cfg.get("beta_windows", [20, 40, 80]),
            default=[20, 40, 80],
        )
        self.rsi_period = int(features_cfg.get("rsi_period", 14))
        self.include_time_features = bool(features_cfg.get("include_time_features", True))
        self.include_market_regime = bool(features_cfg.get("include_market_regime", True))
        self.regime_window = int(features_cfg.get("regime_window", 200))

    @staticmethod
    def _as_int_list(values, default):
        try:
            parsed = sorted({max(1, int(v)) for v in values})
            return parsed or list(default)
        except Exception:
            return list(default)

    def compute_features(
        self,
        bars_df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        symbol: Optional[str] = None,
    ) -> pd.DataFrame:
        """Compute all model features.

        Args:
            bars_df: DataFrame with symbol OHLCV bars (must include close + timestamp)
            market_df: Optional market proxy bars aligned with timestamp (e.g. SPY/QQQ)
            symbol: Optional symbol used for session-aware time features

        Returns:
            DataFrame with original columns and feature columns.
        """
        if bars_df is None or bars_df.empty:
            return bars_df if bars_df is not None else pd.DataFrame()

        df = bars_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        if df.empty:
            return df

        for col in ("open", "high", "low", "close", "volume", "vwap"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "open" not in df.columns:
            df["open"] = df["close"]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)
        df["close"] = df["close"].ffill()

        log_close = np.log(df["close"].clip(lower=1e-9))
        df["log_return_1m"] = log_close.diff()

        # Return lags
        for lag in self.return_lags:
            df[f"return_{lag}m"] = df["close"].pct_change(periods=lag)
            df[f"log_return_{lag}m"] = log_close.diff(lag)

        # Volatility regime and dynamics
        for period in self.volatility_periods:
            min_periods = min(period, max(2, period // 4))
            vol = df["log_return_1m"].rolling(window=period, min_periods=min_periods).std()
            df[f"volatility_{period}m"] = vol
            df[f"volatility_change_{period}m"] = vol.pct_change(periods=period)

        # Trend and mean-reversion style features
        for period in self.ma_periods:
            min_periods = min(period, max(2, period // 2))
            ma = df["close"].rolling(window=period, min_periods=min_periods).mean()
            rolling_std = df["close"].rolling(window=period, min_periods=min_periods).std()
            df[f"ma_{period}"] = ma
            df[f"ma_gap_{period}"] = (df["close"] / (ma + 1e-9)) - 1.0
            df[f"zscore_{period}m"] = (df["close"] - ma) / (rolling_std + 1e-9)
            df[f"momentum_{period}m"] = df["close"].pct_change(periods=period)

        for window in self.trend_windows:
            min_periods = min(window, max(2, window // 2))
            df[f"growth_{window}m"] = (df["close"] / (df["close"].shift(window) + 1e-9)) - 1.0
            df[f"log_growth_{window}m"] = log_close - log_close.shift(window)
            df[f"trend_slope_{window}m"] = self._rolling_slope(log_close, window, min_periods=min_periods)
            rolling_high = df["high"].rolling(window=window, min_periods=min_periods).max()
            rolling_low = df["low"].rolling(window=window, min_periods=min_periods).min()
            df[f"range_ratio_{window}m"] = (rolling_high - rolling_low) / (df["close"] + 1e-9)
            df[f"drawdown_{window}m"] = (df["close"] / (rolling_high + 1e-9)) - 1.0

        # RSI
        df["rsi"] = self._compute_rsi(df["close"], period=self.rsi_period)

        # Volume and dollar-volume features
        df["dollar_volume"] = df["close"] * df["volume"]
        for window in self.volume_windows:
            min_periods = min(window, max(2, window // 2))
            vol_mean = df["volume"].rolling(window=window, min_periods=min_periods).mean()
            vol_std = df["volume"].rolling(window=window, min_periods=min_periods).std()
            dv_mean = df["dollar_volume"].rolling(window=window, min_periods=min_periods).mean()
            dv_std = df["dollar_volume"].rolling(window=window, min_periods=min_periods).std()
            df[f"volume_spike_{window}"] = df["volume"] / (vol_mean + 1e-9)
            df[f"volume_zscore_{window}"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)
            df[f"dollar_volume_zscore_{window}"] = (df["dollar_volume"] - dv_mean) / (dv_std + 1e-9)

        # VWAP approximation from typical price and volume
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        for window in self.vwap_windows:
            min_periods = min(window, max(2, window // 2))
            weighted = (typical_price * df["volume"]).rolling(window=window, min_periods=min_periods).sum()
            volume_roll = df["volume"].rolling(window=window, min_periods=min_periods).sum()
            approx_vwap = weighted / (volume_roll + 1e-9)
            fallback = typical_price.rolling(window=window, min_periods=min_periods).mean()
            df[f"vwap_{window}m"] = approx_vwap.where(volume_roll > 0, fallback)
            df[f"vwap_gap_{window}m"] = (df["close"] / (df[f"vwap_{window}m"] + 1e-9)) - 1.0

        # Time features
        if self.include_time_features:
            minute_of_day = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
            df["minute_of_day"] = minute_of_day
            df["minute_of_day_sin"] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
            df["minute_of_day_cos"] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

            minute_of_session = self._minute_of_session(df["timestamp"], symbol=symbol)
            df["minute_of_session"] = minute_of_session
            session_len = np.where(str(symbol or "").upper().endswith(".T"), 300.0, 390.0)
            df["minute_of_session_sin"] = np.sin(2 * np.pi * minute_of_session / session_len)
            df["minute_of_session_cos"] = np.cos(2 * np.pi * minute_of_session / session_len)

        # Market-context features (aligned by timestamp; backward-only to prevent lookahead)
        if market_df is not None and not market_df.empty:
            mkt = self._prepare_market_features(market_df)
            if not mkt.empty:
                left = df.sort_values("timestamp")
                right = mkt.sort_values("timestamp")
                aligned = pd.merge_asof(
                    left,
                    right,
                    on="timestamp",
                    direction="backward",
                    allow_exact_matches=True,
                )
                df = aligned
                if "market_return_1m" in df.columns:
                    df["relative_strength_1m"] = df["log_return_1m"] - df["market_return_1m"]
                    for window in self.beta_windows:
                        min_periods = min(window, max(2, window // 2))
                        cov = df["log_return_1m"].rolling(window=window, min_periods=min_periods).cov(
                            df["market_return_1m"]
                        )
                        mkt_var = df["market_return_1m"].rolling(window=window, min_periods=min_periods).var()
                        df[f"corr_market_{window}m"] = df["log_return_1m"].rolling(
                            window=window, min_periods=min_periods
                        ).corr(df["market_return_1m"])
                        df[f"beta_market_{window}m"] = cov / (mkt_var + 1e-9)
            else:
                df["relative_strength_1m"] = np.nan
        else:
            df["relative_strength_1m"] = np.nan

        # Leakage-safe volatility regime (rolling quantiles only)
        if self.include_market_regime and "volatility_40m" in df.columns:
            min_periods = min(self.regime_window, max(2, self.regime_window // 4))
            q50 = df["volatility_40m"].rolling(window=self.regime_window, min_periods=min_periods).quantile(0.50)
            q75 = df["volatility_40m"].rolling(window=self.regime_window, min_periods=min_periods).quantile(0.75)
            low_mask = df["volatility_40m"] <= q50
            mid_mask = (df["volatility_40m"] > q50) & (df["volatility_40m"] <= q75)
            regime = np.select([low_mask, mid_mask], [0, 1], default=2)
            regime = np.where(np.isfinite(df["volatility_40m"]), regime, 1)
            df["volatility_regime"] = regime.astype(int)

        return df

    def _prepare_market_features(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare market proxy features from market bars."""
        if market_df is None or market_df.empty or "timestamp" not in market_df.columns:
            return pd.DataFrame()

        mkt = market_df.copy()
        mkt["timestamp"] = pd.to_datetime(mkt["timestamp"], utc=True, errors="coerce")
        if "close" not in mkt.columns:
            return pd.DataFrame()
        mkt["close"] = pd.to_numeric(mkt["close"], errors="coerce")
        mkt = mkt.dropna(subset=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        if mkt.empty:
            return pd.DataFrame()

        log_close = np.log(mkt["close"].clip(lower=1e-9))
        mkt["market_return_1m"] = log_close.diff()
        mkt["market_return_5m"] = log_close.diff(5)
        mkt["market_return_20m"] = log_close.diff(20)
        mkt["market_volatility_20m"] = mkt["market_return_1m"].rolling(window=20, min_periods=5).std()
        mkt["market_volatility_40m"] = mkt["market_return_1m"].rolling(window=40, min_periods=10).std()

        keep_cols = [
            "timestamp",
            "market_return_1m",
            "market_return_5m",
            "market_return_20m",
            "market_volatility_20m",
            "market_volatility_40m",
        ]
        return mkt[keep_cols]

    def _rolling_slope(self, series: pd.Series, window: int, min_periods: int) -> pd.Series:
        """Rolling OLS slope against an integer time index."""
        if window <= 1:
            return pd.Series(index=series.index, dtype=float)

        def _slope(values):
            arr = np.asarray(values, dtype=float)
            if len(arr) < 2:
                return np.nan
            if np.isnan(arr).any():
                return np.nan
            x = np.arange(len(arr), dtype=float)
            x_mean = x.mean()
            x_var = ((x - x_mean) ** 2).sum() + 1e-12
            y_mean = arr.mean()
            cov = ((x - x_mean) * (arr - y_mean)).sum()
            return cov / x_var

        return series.rolling(window=window, min_periods=min_periods).apply(_slope, raw=True)

    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI momentum indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / max(1, period), min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / max(1, period), min_periods=period, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-9)
        return 100.0 - (100.0 / (1.0 + rs))

    def _minute_of_session(self, ts: pd.Series, symbol: Optional[str]) -> pd.Series:
        """Encode minute from local market session start for each timestamp."""
        symbol_u = str(symbol or "").upper()
        if symbol_u.endswith(".T"):
            local = ts.dt.tz_convert("Asia/Tokyo")
            # TSE sessions: 09:00-11:30 and 12:30-15:30.
            m = local.dt.hour * 60 + local.dt.minute
            first = 9 * 60
            second = 12 * 60 + 30
            out = np.where(
                (m >= first) & (m < 11 * 60 + 30),
                m - first,
                np.where((m >= second) & (m < 15 * 60 + 30), (m - second) + 150, np.nan),
            )
            return pd.Series(out, index=ts.index).fillna(0.0)

        local = ts.dt.tz_convert("US/Eastern")
        m = local.dt.hour * 60 + local.dt.minute
        start = 9 * 60 + 30
        out = np.where((m >= start) & (m < 16 * 60), m - start, np.nan)
        return pd.Series(out, index=ts.index).fillna(0.0)

    def get_feature_names(self):
        """Return configured feature names."""
        features = []

        for lag in self.return_lags:
            features.append(f"return_{lag}m")
            features.append(f"log_return_{lag}m")

        for period in self.volatility_periods:
            features.append(f"volatility_{period}m")
            features.append(f"volatility_change_{period}m")

        for period in self.ma_periods:
            features.extend(
                [
                    f"ma_{period}",
                    f"ma_gap_{period}",
                    f"zscore_{period}m",
                    f"momentum_{period}m",
                ]
            )

        for window in self.trend_windows:
            features.extend(
                [
                    f"growth_{window}m",
                    f"log_growth_{window}m",
                    f"trend_slope_{window}m",
                    f"range_ratio_{window}m",
                    f"drawdown_{window}m",
                ]
            )

        features.append("rsi")
        features.append("dollar_volume")

        for window in self.volume_windows:
            features.extend(
                [
                    f"volume_spike_{window}",
                    f"volume_zscore_{window}",
                    f"dollar_volume_zscore_{window}",
                ]
            )

        for window in self.vwap_windows:
            features.extend([f"vwap_{window}m", f"vwap_gap_{window}m"])

        if self.include_time_features:
            features.extend(
                [
                    "minute_of_day_sin",
                    "minute_of_day_cos",
                    "minute_of_session_sin",
                    "minute_of_session_cos",
                ]
            )

        features.extend(
            [
                "market_return_1m",
                "market_return_5m",
                "market_return_20m",
                "market_volatility_20m",
                "market_volatility_40m",
                "relative_strength_1m",
            ]
        )
        for window in self.beta_windows:
            features.extend([f"corr_market_{window}m", f"beta_market_{window}m"])

        if self.include_market_regime:
            features.append("volatility_regime")

        # Keep deterministic ordering without duplicates.
        deduped = []
        seen = set()
        for f in features:
            if f not in seen:
                deduped.append(f)
                seen.add(f)
        return deduped
