"""Main forecasting engine."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from core.features import FeatureEngineer
from core.models import get_model
from core.sentiment import SimpleSentimentAnalyzer
from core.logging_setup import get_logger


logger = get_logger()

try:
    import jpholiday
except Exception:
    jpholiday = None


class ForecasterEngine:
    """Main forecasting engine combining data, features, and models."""
    
    def __init__(self, config, api_client, news_client):
        """Initialize forecaster.
        
        Args:
            config: Configuration dictionary
            api_client: MassiveClient instance
            news_client: NewsClient instance
        """
        self.config = config
        self.api = api_client
        self.news = news_client
        self.feature_engineer = FeatureEngineer(config)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()
        
        forecast_config = config.get('forecast', {})
        self.horizon_minutes = forecast_config.get('horizon_minutes', 40)
        self.min_bars = forecast_config.get('min_bars_for_forecast', 100)
        self.model_name = forecast_config.get('primary_model', 'ridge')
        self.include_ensemble = forecast_config.get('include_ensemble', False)
        self.trend_blend_weight = float(forecast_config.get('trend_blend_weight', 0.2))
        self.max_abs_prediction = float(forecast_config.get('max_abs_prediction', 0.25))
        self.validation_fraction = float(forecast_config.get('validation_fraction', 0.2))
        self.respect_market_hours = bool(forecast_config.get('respect_market_hours', True))
        self.stale_bar_minutes = int(forecast_config.get('stale_bar_minutes', 20))
        
        # Model cache per symbol
        self._models = {}
    
    def generate_forecast(self, symbol):
        """Generate 40-minute forecast for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Forecast with prediction, confidence, intervals, status
        """
        try:
            # Get and clean historical bars
            bars = self.api.get_bars(symbol, limit=800)
            bars = self._prepare_bars(bars)

            if bars.empty or len(bars) < self.min_bars:
                return self._create_status_forecast('Low data')

            last_bar_ts = bars['timestamp'].iloc[-1] if 'timestamp' in bars.columns else None
            market_ctx = self._get_market_context(symbol, last_bar_ts)
            if self.respect_market_hours:
                bar_age = market_ctx.get('last_bar_age_minutes')
                is_open = market_ctx.get('is_open', False)
                if (not is_open) and (bar_age is None or bar_age >= self.stale_bar_minutes):
                    status = 'Market closed'
                    if market_ctx.get('next_open_local'):
                        status = f"Market closed (next open {market_ctx['next_open_local']})"
                    return self._create_status_forecast(status, market_context=market_ctx)

            # Compute features
            bars_with_features = self.feature_engineer.compute_features(bars)

            # Create target variable (40-minute forward return)
            bars_with_features['target'] = self._compute_target(bars_with_features)

            # Get feature columns
            feature_cols = self.feature_engineer.get_feature_names()
            feature_cols = [col for col in feature_cols if col in bars_with_features.columns]

            if not feature_cols:
                return self._create_status_forecast('No features')

            # Prepare training data
            train_df = bars_with_features.dropna(subset=feature_cols + ['target']).copy()
            train_df = train_df[np.isfinite(train_df[feature_cols + ['target']]).all(axis=1)]

            min_samples = max(30, int(self.min_bars * 0.4))
            if len(train_df) < min_samples:
                return self._create_status_forecast('Insufficient data')

            X = train_df[feature_cols].values.astype(float)
            y_raw = train_df['target'].values.astype(float)
            y = self._winsorize_array(y_raw, lower_q=0.01, upper_q=0.99)

            # Time-based split for validation
            val_size = max(20, int(len(train_df) * self.validation_fraction))
            val_size = min(val_size, max(0, len(train_df) - 20))

            if val_size > 0:
                split_idx = len(train_df) - val_size
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
            else:
                X_train, y_train = X, y
                X_val = np.empty((0, X.shape[1]))
                y_val = np.empty((0,))

            train_weights = np.linspace(0.5, 1.5, len(X_train))
            full_weights = np.linspace(0.5, 1.5, len(X))

            # Train primary model
            primary_key = f"{symbol}:{self.model_name}"
            if primary_key not in self._models:
                self._models[primary_key] = get_model(self.model_name, self.config)
            primary_model = self._models[primary_key]

            if not primary_model.train(X_train, y_train, sample_weight=train_weights):
                return self._create_status_forecast('Model training failed')

            # Validation score (RMSE)
            target_std = float(np.std(y_train) + 1e-9)
            rmse_primary = None
            if len(X_val) > 0:
                pred_val = primary_model.predict(X_val)
                pred_val = np.asarray(pred_val, dtype=float).reshape(-1)
                rmse_primary = float(np.sqrt(np.mean((pred_val - y_val) ** 2)))

            # Retrain on full history for final inference
            primary_model.train(X, y, sample_weight=full_weights)

            # Get latest features for prediction
            latest_features = bars_with_features[feature_cols].iloc[-1].values

            pred_result = primary_model.predict_intervals(latest_features)
            model_name_used = self.model_name
            model_agreement = 100.0

            # Optional ensemble with complementary model
            if self.include_ensemble:
                secondary_name = 'xgboost' if self.model_name != 'xgboost' else 'ridge'
                secondary_key = f"{symbol}:{secondary_name}"
                if secondary_key not in self._models:
                    self._models[secondary_key] = get_model(secondary_name, self.config)
                secondary_model = self._models[secondary_key]

                if secondary_model.train(X_train, y_train, sample_weight=train_weights):
                    rmse_secondary = None
                    if len(X_val) > 0:
                        pred_val_2 = np.asarray(secondary_model.predict(X_val), dtype=float).reshape(-1)
                        rmse_secondary = float(np.sqrt(np.mean((pred_val_2 - y_val) ** 2)))

                    secondary_model.train(X, y, sample_weight=full_weights)
                    pred_2 = secondary_model.predict_intervals(latest_features)

                    w1, w2 = 0.5, 0.5
                    if rmse_primary and rmse_secondary:
                        inv1 = 1.0 / (rmse_primary + 1e-9)
                        inv2 = 1.0 / (rmse_secondary + 1e-9)
                        denom = inv1 + inv2
                        w1, w2 = inv1 / denom, inv2 / denom

                    pred_result = self._combine_predictions(pred_result, pred_2, w1, w2)
                    model_name_used = f"ensemble({self.model_name}+{secondary_name})"

                    denom = max(target_std, 1e-6)
                    model_agreement = max(
                        0.0,
                        100.0 - min(100.0, (abs(pred_result['mean'] - pred_2['mean']) / denom) * 100.0)
                    )

            # Blend with simple recent-trend baseline for stability
            trend_pred = self._trend_baseline_prediction(bars_with_features)
            blend_w = min(max(self.trend_blend_weight, 0.0), 0.5)
            pred_result = self._blend_with_trend(pred_result, trend_pred, blend_w)
            pred_result = self._clip_prediction_dict(pred_result, self.max_abs_prediction)

            # Time-of-day adjustment: if close to session end, reduce expected move.
            near_close_adjusted = False
            if self.respect_market_hours and market_ctx.get('is_open', False):
                minutes_to_close = market_ctx.get('minutes_to_close')
                if minutes_to_close is not None and minutes_to_close < self.horizon_minutes:
                    factor = max(0.15, float(minutes_to_close) / float(max(1, self.horizon_minutes)))
                    pred_result = self._scale_prediction_dict(pred_result, factor)
                    near_close_adjusted = True

            # Compute confidence
            confidence = self._compute_confidence(
                bars_with_features,
                feature_cols,
                val_rmse=rmse_primary,
                target_std=target_std,
                model_agreement=model_agreement,
            )

            # Determine direction
            direction = self._determine_direction(pred_result['mean'])

            # Get sentiment if available
            sentiment_impact = self._get_sentiment_impact(symbol)

            # Check volatility status
            recent_vol = bars_with_features['volatility_40m'].iloc[-1] if 'volatility_40m' in bars_with_features.columns else None
            vol_is_high = bool(
                recent_vol is not None
                and np.isfinite(recent_vol)
                and 'volatility_40m' in bars_with_features.columns
                and recent_vol > bars_with_features['volatility_40m'].dropna().quantile(0.75)
            )
            status_parts = []
            if near_close_adjusted:
                status_parts.append(f"Near close ({int(market_ctx.get('minutes_to_close', 0))}m left)")
            if vol_is_high:
                status_parts.append('High volatility')
            status = ' | '.join(status_parts) if status_parts else 'OK'

            forecast = {
                'symbol': symbol,
                'prediction_return': pred_result['mean'],
                'direction': direction,
                'confidence': confidence,
                'interval_50_lower': pred_result['lower_50'],
                'interval_50_upper': pred_result['upper_50'],
                'interval_90_lower': pred_result['lower_90'],
                'interval_90_upper': pred_result['upper_90'],
                'sentiment_impact': sentiment_impact,
                'model_status': status,
                'timestamp': datetime.utcnow().isoformat(),
                'data_points_used': len(train_df),
                'model_name': model_name_used,
                'market_context': market_ctx,
            }

            logger.info(f"Forecast {symbol}: return={pred_result['mean']:.4f}, "
                       f"direction={direction}, confidence={confidence:.1f}%")

            return forecast

        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {e}")
            return self._create_status_forecast('Error')
    
    def _compute_target(self, df):
        """Compute target variable (40-minute forward return).
        
        Args:
            df: DataFrame with close prices
        
        Returns:
            Series: Forward return over 40 minutes
        """
        forward_close = df['close'].shift(-self.horizon_minutes)
        return (forward_close - df['close']) / df['close']

    def _get_market_schedule(self, symbol):
        """Return timezone and session windows for a symbol."""
        sym = str(symbol).upper()
        if sym.endswith('.T'):
            # Tokyo Stock Exchange cash equities: 09:00-11:30 and 12:30-15:30 JST.
            return 'Asia/Tokyo', [(9, 0, 11, 30), (12, 30, 15, 30)]
        # Default US equities (NYSE/Nasdaq): 09:30-16:00 ET.
        return 'US/Eastern', [(9, 30, 16, 0)]

    def _is_exchange_holiday(self, symbol, date_local):
        """Exchange holiday check (focused on TSE symbols)."""
        sym = str(symbol).upper()
        # Weekends are non-trading for all supported equity exchanges.
        if date_local.weekday() >= 5:
            return True

        if sym.endswith('.T'):
            # JPX/TSE holidays: national holidays + Jan 1-3 + Dec 31.
            if (date_local.month, date_local.day) in {(1, 1), (1, 2), (1, 3), (12, 31)}:
                return True
            if jpholiday is not None:
                try:
                    return bool(jpholiday.is_holiday(date_local))
                except Exception:
                    pass
        return False

    def _next_session_open(self, symbol, now_local, sessions):
        """Compute next session open datetime in local exchange timezone."""
        first_h, first_m, _, _ = sessions[0]
        candidate = now_local

        # If today still has a future opening (before first session), keep same day.
        same_day_first = now_local.replace(hour=first_h, minute=first_m, second=0, microsecond=0)
        if (not self._is_exchange_holiday(symbol, now_local.date())) and now_local < same_day_first:
            candidate = same_day_first
        else:
            candidate = (now_local + timedelta(days=1)).replace(
                hour=first_h, minute=first_m, second=0, microsecond=0
            )
            while self._is_exchange_holiday(symbol, candidate.date()):
                candidate = (candidate + timedelta(days=1)).replace(
                    hour=first_h, minute=first_m, second=0, microsecond=0
                )
        return candidate

    def _get_market_context(self, symbol, last_bar_ts=None):
        """Determine exchange session state for symbol and latest bar freshness."""
        tz_name, sessions = self._get_market_schedule(symbol)
        tz = pytz.timezone(tz_name)
        now_local = datetime.now(tz)
        is_trading_day = not self._is_exchange_holiday(symbol, now_local.date())

        is_open = False
        minutes_to_close = None
        next_open_local = None

        session_bounds = []
        for sh, sm, eh, em in sessions:
            start = now_local.replace(hour=sh, minute=sm, second=0, microsecond=0)
            end = now_local.replace(hour=eh, minute=em, second=0, microsecond=0)
            session_bounds.append((start, end))

        if is_trading_day:
            for start, end in session_bounds:
                if start <= now_local <= end:
                    is_open = True
                    minutes_to_close = max(0.0, (end - now_local).total_seconds() / 60.0)
                    break

            if not is_open:
                # Between two sessions (e.g., Tokyo lunch break)
                for i in range(len(session_bounds) - 1):
                    end_i = session_bounds[i][1]
                    next_start = session_bounds[i + 1][0]
                    if end_i < now_local < next_start:
                        next_open_local = next_start
                        break

                if next_open_local is None:
                    # Before first session or after last session
                    first_start = session_bounds[0][0]
                    if now_local < first_start:
                        next_open_local = first_start
                    else:
                        next_open_local = self._next_session_open(symbol, now_local, sessions)
        else:
            next_open_local = self._next_session_open(symbol, now_local, sessions)

        last_bar_age_minutes = None
        last_bar_local = None
        if last_bar_ts is not None:
            ts = pd.to_datetime(last_bar_ts, utc=True, errors='coerce')
            if pd.notna(ts):
                ts_local = ts.tz_convert(tz)
                last_bar_local = ts_local.strftime('%Y-%m-%d %H:%M %Z')
                try:
                    delta_min = (now_local - ts_local.to_pydatetime()).total_seconds() / 60.0
                    last_bar_age_minutes = max(0.0, float(delta_min))
                except Exception:
                    last_bar_age_minutes = None

        return {
            'timezone': tz_name,
            'is_open': is_open,
            'minutes_to_close': minutes_to_close,
            'next_open_local': next_open_local.strftime('%Y-%m-%d %H:%M %Z') if next_open_local else None,
            'last_bar_local': last_bar_local,
            'last_bar_age_minutes': last_bar_age_minutes,
        }

    def _prepare_bars(self, bars):
        """Normalize and clean bars dataframe for modeling."""
        if bars is None or bars.empty:
            return pd.DataFrame()
        df = bars.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values('timestamp')
        df = df.drop_duplicates(subset=['timestamp'], keep='last') if 'timestamp' in df.columns else df

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Basic sanity filters to reduce bad ticks.
        if 'close' in df.columns:
            df = df[df['close'] > 0]
            ret = df['close'].pct_change()
            spike_mask = ret.abs() < 0.35  # remove obvious 1-min outliers (>35%)
            df = df[spike_mask.fillna(True)]

        return df.reset_index(drop=True)

    def _winsorize_array(self, values, lower_q=0.01, upper_q=0.99):
        """Clip extreme target values for training stability."""
        arr = np.asarray(values, dtype=float)
        if len(arr) < 10:
            return arr
        lo = np.nanquantile(arr, lower_q)
        hi = np.nanquantile(arr, upper_q)
        return np.clip(arr, lo, hi)

    def _combine_predictions(self, pred_a, pred_b, w_a=0.5, w_b=0.5):
        """Weighted combination of two prediction interval dictionaries."""
        keys = ['mean', 'lower_50', 'upper_50', 'lower_90', 'upper_90']
        out = {}
        for k in keys:
            a = float(pred_a.get(k, 0.0))
            b = float(pred_b.get(k, 0.0))
            out[k] = (w_a * a) + (w_b * b)
        return out

    def _trend_baseline_prediction(self, df):
        """Simple trend/growth baseline over recent history."""
        try:
            closes = pd.to_numeric(df['close'], errors='coerce').dropna()
            if len(closes) < 20:
                return 0.0
            recent_ret = closes.pct_change().dropna().tail(90)
            if recent_ret.empty:
                return 0.0
            ewma_mean = recent_ret.ewm(span=12, adjust=False).mean().iloc[-1]
            baseline = float(ewma_mean * self.horizon_minutes)
            return float(np.clip(baseline, -self.max_abs_prediction, self.max_abs_prediction))
        except Exception:
            return 0.0

    def _blend_with_trend(self, pred_result, trend_pred, blend_w):
        """Blend model prediction with trend baseline."""
        out = dict(pred_result)
        model_mean = float(out.get('mean', 0.0))
        blended = (1.0 - blend_w) * model_mean + (blend_w * float(trend_pred))
        delta = blended - model_mean
        out['mean'] = blended
        for k in ['lower_50', 'upper_50', 'lower_90', 'upper_90']:
            out[k] = float(out.get(k, blended)) + delta
        return out

    def _clip_prediction_dict(self, pred_result, max_abs):
        """Clamp prediction/interval values to avoid unrealistic extremes."""
        out = dict(pred_result)
        for k in ['mean', 'lower_50', 'upper_50', 'lower_90', 'upper_90']:
            out[k] = float(np.clip(float(out.get(k, 0.0)), -max_abs, max_abs))
        # Ensure interval ordering after clipping.
        out['lower_50'], out['upper_50'] = sorted([out['lower_50'], out['upper_50']])
        out['lower_90'], out['upper_90'] = sorted([out['lower_90'], out['upper_90']])
        out['lower_90'] = min(out['lower_90'], out['lower_50'])
        out['upper_90'] = max(out['upper_90'], out['upper_50'])
        return out

    def _scale_prediction_dict(self, pred_result, factor):
        """Scale prediction mean and intervals by a factor (used near session close)."""
        out = dict(pred_result)
        for k in ['mean', 'lower_50', 'upper_50', 'lower_90', 'upper_90']:
            out[k] = float(out.get(k, 0.0)) * float(factor)
        return out
    
    def _compute_confidence(self, df, feature_cols, val_rmse=None, target_std=None, model_agreement=50.0):
        """Compute confidence score 0-100.
        
        Args:
            df: DataFrame with features
            feature_cols: Feature column names
            val_rmse: Validation RMSE (optional)
            target_std: Target standard deviation (optional)
            model_agreement: 0-100 score for agreement across models
        
        Returns:
            float: Confidence score 0-100
        """
        try:
            # Factor 1: Data coverage (how much complete data we have)
            complete_rows = df[feature_cols].notna().all(axis=1).sum()
            data_coverage = min(100, (complete_rows / len(df)) * 100)
            
            # Factor 2: Volatility stability (is volatility consistent?)
            if 'volatility_40m' in df.columns:
                vol_series = df['volatility_40m'].dropna()
                if len(vol_series) > 10:
                    vol_cv = vol_series.std() / (vol_series.mean() + 1e-9)
                    volatility_stability = max(0, 100 - vol_cv * 100)
                else:
                    volatility_stability = 50
            else:
                volatility_stability = 50

            # Factor 3: Predictive quality from validation error
            if val_rmse is not None and target_std is not None and target_std > 0:
                predictive_quality = max(0.0, 100.0 * (1.0 - min(1.0, val_rmse / (target_std * 1.5 + 1e-9))))
            else:
                predictive_quality = 50.0

            # Factor 4: Agreement score (for ensemble or fallback single-model default)
            agreement_score = max(0.0, min(100.0, model_agreement))
            
            # Factor 4: News sentiment
            if self.news.enabled:
                news_factor = 20
            else:
                news_factor = 0
            
            # Weighted average (simplified)
            weights = {
                'data_coverage': 0.4,
                'volatility_stability': 0.3,
                'predictive_quality': 0.2,
                'agreement_score': 0.1,
                'news_available': 0.1 if news_factor > 0 else 0
            }
            
            confidence = (
                weights['data_coverage'] * data_coverage +
                weights['volatility_stability'] * volatility_stability +
                weights['predictive_quality'] * predictive_quality +
                weights['agreement_score'] * agreement_score
            ) / sum(weights.values())
            
            return float(max(0, min(100, confidence)))
        
        except Exception as e:
            logger.warning(f"Error computing confidence: {e}")
            return 50.0
    
    def _determine_direction(self, predicted_return):
        """Determine direction (UP/DOWN/FLAT) from predicted return.
        
        Args:
            predicted_return: Predicted return value
        
        Returns:
            str: 'UP', 'DOWN', or 'FLAT'
        """
        threshold = 0.001  # 0.1% threshold
        
        if predicted_return > threshold:
            return 'UP'
        elif predicted_return < -threshold:
            return 'DOWN'
        else:
            return 'FLAT'
    
    def _get_sentiment_impact(self, symbol):
        """Get sentiment impact from news.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Sentiment metrics
        """
        if not self.news.enabled:
            return {'available': False, 'overall_sentiment': 0, 'score': 0.0}
        
        try:
            headlines = self.news.get_headlines(symbol, limit=10)
            sentiment = self.sentiment_analyzer.aggregate_sentiment(headlines)
            
            return {
                'available': True,
                'overall_sentiment': sentiment['overall_sentiment'],
                'score': sentiment['score'],
                'positive_count': sentiment['positive_count'],
                'negative_count': sentiment['negative_count'],
                'headline_count': sentiment['headline_count']
            }
        except Exception as e:
            logger.warning(f"Error getting sentiment for {symbol}: {e}")
            return {'available': False, 'overall_sentiment': 0, 'score': 0.0}
    
    def _create_status_forecast(self, status, **extra):
        """Create a status-only forecast when actual forecast not possible.
        
        Args:
            status: Status message
        
        Returns:
            dict: Minimal forecast with status
        """
        payload = {
            'prediction_return': None,
            'direction': None,
            'confidence': 0,
            'interval_50_lower': None,
            'interval_50_upper': None,
            'interval_90_lower': None,
            'interval_90_upper': None,
            'model_status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'sentiment_impact': {'available': False}
        }
        payload.update(extra)
        return payload
