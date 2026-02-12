"""Feature engineering for the forecasting model."""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Extract and compute features from market data."""
    
    def __init__(self, config):
        """Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('features', {})
        self.momentum_periods = self.config.get('momentum_periods', [1, 5, 10, 20, 40])
        self.ma_periods = self.config.get('ma_periods', [10, 20, 40])
        self.volatility_periods = self.config.get('volatility_periods', [20, 40])
        self.trend_windows = self.config.get('trend_windows', [20, 60, 120])
        self.rsi_period = self.config.get('rsi_period', 14)
        self.volume_windows = self.config.get('volume_spike_windows', [5, 40])
        self.include_time_features = self.config.get('include_time_features', True)
        self.include_market_regime = self.config.get('include_market_regime', True)

    def compute_features(self, bars_df):
        """Compute all features from bars data.
        
        Args:
            bars_df: DataFrame with OHLCV data, must have 'timestamp' and 'close' columns
        
        Returns:
            DataFrame: Original data with feature columns added
        """
        if bars_df.empty:
            return bars_df
        
        df = bars_df.copy()
        
        # Returns (momentum)
        for period in self.momentum_periods:
            df[f'return_{period}m'] = df['close'].pct_change(periods=period)
        
        # Moving averages
        for period in self.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_gap_{period}'] = (df['close'] / (df[f'ma_{period}'] + 1e-9)) - 1.0

        # Volatility (rolling std of returns)
        for period in self.volatility_periods:
            returns = df['close'].pct_change()
            df[f'volatility_{period}m'] = returns.rolling(window=period).std()
            df[f'volatility_change_{period}m'] = (
                df[f'volatility_{period}m'] / (df[f'volatility_{period}m'].shift(period) + 1e-9)
            ) - 1.0

        # Trend and growth patterns
        log_close = np.log(df['close'].clip(lower=1e-9))
        for window in self.trend_windows:
            df[f'growth_{window}m'] = (df['close'] / (df['close'].shift(window) + 1e-9)) - 1.0
            df[f'log_growth_{window}m'] = log_close - log_close.shift(window)
            df[f'trend_slope_{window}m'] = self._rolling_slope(log_close, window)
            rolling_high = df['high'].rolling(window=window).max() if 'high' in df.columns else df['close'].rolling(window=window).max()
            rolling_low = df['low'].rolling(window=window).min() if 'low' in df.columns else df['close'].rolling(window=window).min()
            df[f'range_ratio_{window}m'] = (rolling_high - rolling_low) / (df['close'] + 1e-9)
            df[f'drawdown_{window}m'] = (df['close'] / (rolling_high + 1e-9)) - 1.0

        # RSI-like momentum (rate of change)
        df['rsi'] = self._compute_rsi(df['close'], self.rsi_period)
        
        # Volume features
        if 'volume' in df.columns:
            for window in self.volume_windows:
                recent_volume = df['volume'].rolling(window=window).sum()
                overall_volume = df['volume'].rolling(window=window * 2).sum()
                df[f'volume_spike_{window}'] = recent_volume / (overall_volume + 1e-9)
        
        # Time-of-day features (minute of day with sin/cos encoding)
        if self.include_time_features and 'timestamp' in df.columns:
            df['minute_of_day'] = pd.to_datetime(df['timestamp']).dt.hour * 60 + pd.to_datetime(df['timestamp']).dt.minute
            df['minute_of_day_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / (24 * 60))
            df['minute_of_day_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / (24 * 60))
        
        # Market regime (volatility bucket)
        if self.include_market_regime:
            if 'volatility_40m' in df.columns:
                vol_col = 'volatility_40m'
            elif 'volatility_20m' in df.columns:
                vol_col = 'volatility_20m'
            else:
                vol_col = None
            
            if vol_col:
                vol_median = df[vol_col].median()
                vol_q75 = df[vol_col].quantile(0.75)
                regime = pd.cut(
                    df[vol_col],
                    bins=[0, vol_median, vol_q75, float('inf')],
                    labels=[0, 1, 2],
                    include_lowest=True
                )
                # Rolling vol produces NaNs at the beginning; default those to mid regime.
                df['volatility_regime'] = pd.to_numeric(regime, errors='coerce').fillna(1).astype(int)
        
        # Spread proxy (if bid/ask available)
        if 'bid' in df.columns and 'ask' in df.columns:
            df['spread'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)

        return df

    def _rolling_slope(self, series, window):
        """Rolling slope of a series using OLS against time index."""
        if window <= 1:
            return pd.Series(index=series.index, dtype=float)
        x = np.arange(window, dtype=float)
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum() + 1e-12

        def _slope(values):
            arr = np.asarray(values, dtype=float)
            if np.isnan(arr).any():
                return np.nan
            y_mean = arr.mean()
            cov = ((x - x_mean) * (arr - y_mean)).sum()
            return cov / x_var

        return series.rolling(window=window).apply(_slope, raw=True)
    
    def _compute_rsi(self, prices, period=14):
        """Compute RSI indicator.
        
        Args:
            prices: Series of closing prices
            period: RSI period
        
        Returns:
            Series: RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_feature_names(self):
        """Get list of feature column names.
        
        Returns:
            list: Feature column names
        """
        features = []
        
        # Returns
        for period in self.momentum_periods:
            features.append(f'return_{period}m')
        
        # Moving averages
        for period in self.ma_periods:
            features.append(f'ma_{period}')
            features.append(f'ma_gap_{period}')

        # Volatility
        for period in self.volatility_periods:
            features.append(f'volatility_{period}m')
            features.append(f'volatility_change_{period}m')

        # Trend and growth
        for window in self.trend_windows:
            features.extend([
                f'growth_{window}m',
                f'log_growth_{window}m',
                f'trend_slope_{window}m',
                f'range_ratio_{window}m',
                f'drawdown_{window}m',
            ])
        
        # RSI
        features.append('rsi')
        
        # Volume
        for window in self.volume_windows:
            features.append(f'volume_spike_{window}')
        
        # Time features
        if self.include_time_features:
            features.extend(['minute_of_day_sin', 'minute_of_day_cos'])
        
        # Regime
        if self.include_market_regime:
            features.append('volatility_regime')
        
        return features
