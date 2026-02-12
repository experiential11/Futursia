"""Main forecasting engine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

from core.db import MarketDB
from core.features import FeatureEngineer
from core.logging_setup import get_logger
from core.models import get_model
from core.sentiment import SimpleSentimentAnalyzer

logger = get_logger()

try:
    import jpholiday
except Exception:
    jpholiday = None


class ForecasterEngine:
    """Forecasting engine with pooled cross-symbol training and realized scoring."""

    FEATURES_VERSION = "v2_market_patterns"
    MODEL_VERSION = "v2_pooled_ensemble"

    def __init__(self, config, api_client, news_client):
        self.config = config
        self.api = api_client
        self.news = news_client
        self.db = getattr(api_client, "db", None) or MarketDB(config.get("storage", {}).get("database", "storage/market.db"))
        self.feature_engineer = FeatureEngineer(config)
        self.sentiment_analyzer = SimpleSentimentAnalyzer()

        forecast_cfg = config.get("forecast", {})
        pool_cfg = forecast_cfg.get("pooled_training", {})
        market_cfg = config.get("market_context", {})

        self.horizon_minutes = int(forecast_cfg.get("horizon_minutes", 40))
        self.min_bars = int(forecast_cfg.get("min_bars_for_forecast", 100))
        self.model_name = str(forecast_cfg.get("primary_model", "ridge")).lower()
        self.include_ensemble = bool(forecast_cfg.get("include_ensemble", False))
        self.trend_blend_weight = float(forecast_cfg.get("trend_blend_weight", 0.2))
        self.max_abs_prediction = float(forecast_cfg.get("max_abs_prediction", 0.25))
        self.validation_fraction = float(forecast_cfg.get("validation_fraction", 0.2))
        self.respect_market_hours = bool(forecast_cfg.get("respect_market_hours", True))
        self.stale_bar_minutes = int(forecast_cfg.get("stale_bar_minutes", 20))
        self.min_train_rows = int(pool_cfg.get("min_train_rows", max(150, self.min_bars)))
        self.max_symbols = int(pool_cfg.get("max_symbols", 20))
        self.training_bars_per_symbol = int(pool_cfg.get("bars_per_symbol", 2000))
        self.include_symbol_one_hot = bool(pool_cfg.get("include_symbol_one_hot", True))
        self.max_feature_nan_ratio = float(pool_cfg.get("max_feature_nan_ratio", 0.9))

        self.predict_normalized_target = bool(forecast_cfg.get("predict_normalized_target", False))
        self.target_vol_window = int(forecast_cfg.get("target_vol_window", 40))
        self.target_vol_floor = float(forecast_cfg.get("target_vol_floor", 1e-4))
        self.flat_threshold_mode = str(forecast_cfg.get("flat_threshold_mode", "volatility")).lower()
        self.flat_threshold_multiplier = float(forecast_cfg.get("flat_threshold_multiplier", 0.35))
        self.flat_threshold_min = float(forecast_cfg.get("flat_threshold_min", 0.001))
        self.enable_abstain = bool(forecast_cfg.get("enable_abstain", True))
        self.enable_persistence = bool(forecast_cfg.get("persist_predictions", True))
        self.scoring_lookback_hours = int(forecast_cfg.get("scoring_lookback_hours", 24))

        self.market_proxy_symbols = list(market_cfg.get("proxy_symbols", ["SPY", "QQQ"]))
        self.market_proxy_limit = int(market_cfg.get("proxy_bars_limit", self.training_bars_per_symbol))

        self.walk_cfg = forecast_cfg.get("walk_forward", {})
        self._config_hash = self._compute_config_hash()

    def generate_forecast(self, symbol):
        """Generate a 40-minute return forecast."""
        symbol = str(symbol).upper()
        try:
            bars = self.api.get_bars(symbol, limit=max(800, self.training_bars_per_symbol))
            bars = self._prepare_bars(bars)
            if bars.empty or len(bars) < self.min_bars:
                payload = self._create_status_forecast("Low data")
                self._persist_forecast(symbol, payload)
                return payload

            # Ensure the latest symbol history is in DB before pooled training.
            self.db.save_bars(symbol, bars)

            last_bar_ts = bars["timestamp"].iloc[-1]
            market_ctx = self._get_market_context(symbol, last_bar_ts)
            if self.respect_market_hours:
                bar_age = market_ctx.get("last_bar_age_minutes")
                is_open = market_ctx.get("is_open", False)
                if (not is_open) and (bar_age is None or bar_age >= self.stale_bar_minutes):
                    status = "Market closed"
                    if market_ctx.get("next_open_local"):
                        status = f"Market closed (next open {market_ctx['next_open_local']})"
                    payload = self._create_status_forecast(status, market_context=market_ctx)
                    self._persist_forecast(symbol, payload)
                    return payload

            cutoff_ts = int(pd.Timestamp(last_bar_ts).timestamp())
            pooled = self._build_pooled_dataset(current_symbol=symbol, cutoff_ts=cutoff_ts)
            if pooled is None:
                payload = self._create_status_forecast("No pooled data", market_context=market_ctx)
                self._persist_forecast(symbol, payload)
                return payload

            train_df = pooled["train_df"]
            feature_cols = pooled["feature_cols"]
            symbol_levels = pooled["symbol_levels"]
            prediction_row = pooled["prediction_row"]
            symbol_features = pooled["symbol_features"]

            if train_df.empty or len(train_df) < self.min_train_rows:
                payload = self._create_status_forecast("Insufficient pooled data", market_context=market_ctx)
                self._persist_forecast(symbol, payload)
                return payload
            if prediction_row is None or prediction_row.empty:
                payload = self._create_status_forecast("No latest features", market_context=market_ctx)
                self._persist_forecast(symbol, payload)
                return payload

            model_bundle = self._train_models(train_df, feature_cols, symbol_levels)
            if model_bundle is None:
                payload = self._create_status_forecast("Model training failed", market_context=market_ctx)
                self._persist_forecast(symbol, payload)
                return payload

            pred_model = self._predict_one_row(model_bundle, prediction_row)
            if pred_model is None:
                payload = self._create_status_forecast("Prediction failed", market_context=market_ctx)
                self._persist_forecast(symbol, payload)
                return payload

            latest_vol = float(prediction_row["target_vol"].iloc[-1]) if "target_vol" in prediction_row.columns else self.flat_threshold_min
            latest_vol = max(latest_vol, self.target_vol_floor)

            if self.predict_normalized_target:
                prediction_return_norm = float(pred_model["mean"])
                pred_raw = {k: float(v) * latest_vol for k, v in pred_model.items()}
            else:
                prediction_return_norm = None
                pred_raw = {k: float(v) for k, v in pred_model.items()}

            # Blend with trend baseline in raw-return space.
            trend_pred = self._trend_baseline_prediction(symbol_features)
            blend_w = min(max(self.trend_blend_weight, 0.0), 0.5)
            pred_raw = self._blend_with_trend(pred_raw, trend_pred, blend_w)
            pred_raw = self._clip_prediction_dict(pred_raw, self.max_abs_prediction)

            near_close_adjusted = False
            if self.respect_market_hours and market_ctx.get("is_open", False):
                minutes_to_close = market_ctx.get("minutes_to_close")
                if minutes_to_close is not None and minutes_to_close < self.horizon_minutes:
                    factor = max(0.15, float(minutes_to_close) / float(max(1, self.horizon_minutes)))
                    pred_raw = self._scale_prediction_dict(pred_raw, factor)
                    near_close_adjusted = True

            direction_threshold = self._compute_direction_threshold(latest_vol)
            direction = self._determine_direction(pred_raw["mean"], threshold=direction_threshold)

            confidence = self._compute_confidence(
                train_df=train_df,
                feature_cols=feature_cols,
                val_rmse=model_bundle.get("val_rmse"),
                target_std=model_bundle.get("target_std"),
                model_agreement=model_bundle.get("model_agreement", 50.0),
            )

            sentiment_impact = self._get_sentiment_impact(symbol)

            vol_is_high = False
            if "volatility_40m" in symbol_features.columns:
                vol_series = pd.to_numeric(symbol_features["volatility_40m"], errors="coerce").dropna()
                if not vol_series.empty:
                    recent_vol = float(vol_series.iloc[-1])
                    vol_is_high = recent_vol > float(vol_series.quantile(0.75))

            status_parts = []
            if near_close_adjusted:
                status_parts.append(f"Near close ({int(market_ctx.get('minutes_to_close', 0))}m left)")
            if vol_is_high:
                status_parts.append("High volatility")
            if pooled.get("market_proxy_symbol") is None:
                status_parts.append("No market proxy")
            status = " | ".join(status_parts) if status_parts else "OK"

            created_at = datetime.utcnow()
            forecast = {
                "symbol": symbol,
                "prediction_return": float(pred_raw["mean"]),
                "prediction_return_raw": float(pred_raw["mean"]),
                "prediction_return_norm": prediction_return_norm,
                "direction": direction,
                "predicted_direction": direction,
                "direction_threshold": direction_threshold,
                "volatility_40": latest_vol,
                "confidence": confidence,
                "confidence_score": confidence,
                "interval_50_lower": float(pred_raw["lower_50"]),
                "interval_50_upper": float(pred_raw["upper_50"]),
                "interval_90_lower": float(pred_raw["lower_90"]),
                "interval_90_upper": float(pred_raw["upper_90"]),
                "sentiment_impact": sentiment_impact,
                "model_status": status,
                "timestamp": created_at.isoformat(),
                "forecast_time": int(created_at.timestamp()),
                "horizon_minutes": self.horizon_minutes,
                "target_due_at": int((created_at + timedelta(minutes=self.horizon_minutes)).timestamp()),
                "data_points_used": int(len(train_df)),
                "model_name": model_bundle.get("model_name", self.model_name),
                "features_version": self.FEATURES_VERSION,
                "model_version": self.MODEL_VERSION,
                "config_hash": self._config_hash,
                "market_context": market_ctx,
            }

            self._persist_forecast(symbol, forecast)
            logger.info(
                "Forecast %s: return=%.4f direction=%s conf=%.1f%% model=%s",
                symbol,
                forecast["prediction_return"],
                forecast["direction"],
                forecast["confidence"],
                forecast["model_name"],
            )
            return forecast

        except Exception as exc:
            logger.error("Error generating forecast for %s: %s", symbol, exc)
            payload = self._create_status_forecast("Error")
            self._persist_forecast(symbol, payload)
            return payload

    def run_walk_forward_validation(
        self,
        symbols: Optional[Iterable[str]] = None,
        save_json_path: Optional[str] = None,
    ):
        """Run leakage-safe walk-forward validation and return a report dict."""
        symbol_list = [str(s).upper() for s in (symbols or self._resolve_training_symbols())]
        if not symbol_list:
            return {"status": "no_symbols"}

        cutoff_ts = self._get_latest_bar_ts(symbol_list)
        if cutoff_ts is None:
            return {"status": "no_bars"}

        pooled = self._build_pooled_dataset(current_symbol=symbol_list[0], cutoff_ts=cutoff_ts, symbols=symbol_list)
        if pooled is None:
            return {"status": "no_dataset"}

        all_df = pooled["all_features"].copy()
        feature_cols = pooled["feature_cols"]
        symbol_levels = pooled["symbol_levels"]
        labeled = all_df.dropna(subset=["target_model", "target_raw"]).copy()
        labeled = labeled[np.isfinite(labeled["target_model"])]
        labeled = labeled.sort_values("timestamp").reset_index(drop=True)
        if len(labeled) < 300:
            return {"status": "insufficient_rows", "rows": int(len(labeled))}

        mode = str(self.walk_cfg.get("mode", "expanding")).lower()
        n_splits = int(self.walk_cfg.get("n_splits", 5))
        min_train_rows = int(self.walk_cfg.get("min_train_rows", 400))
        val_rows = int(self.walk_cfg.get("val_rows", 200))
        rolling_window_rows = int(self.walk_cfg.get("rolling_window_rows", 2500))

        max_start = len(labeled) - val_rows
        if max_start <= min_train_rows:
            return {"status": "insufficient_window", "rows": int(len(labeled))}

        n_splits = max(1, n_splits)
        step = max(1, (max_start - min_train_rows) // n_splits)

        fold_reports = []
        for split_idx in range(n_splits):
            train_end = min_train_rows + split_idx * step
            if train_end >= max_start:
                break
            test_start = train_end
            test_end = min(len(labeled), test_start + val_rows)
            if test_end - test_start < 20:
                continue

            if mode == "rolling":
                train_start = max(0, train_end - rolling_window_rows)
            else:
                train_start = 0

            train_fold = labeled.iloc[train_start:train_end].copy()
            test_fold = labeled.iloc[test_start:test_end].copy()
            if len(train_fold) < 120 or len(test_fold) < 20:
                continue

            model_bundle = self._train_models(train_fold, feature_cols, symbol_levels)
            if model_bundle is None:
                continue

            pred_raw, pred_norm = self._predict_frame(model_bundle, test_fold)
            fold_metrics = self._compute_eval_metrics(test_fold, pred_raw, pred_norm)
            fold_metrics["fold"] = int(split_idx)
            fold_metrics["train_rows"] = int(len(train_fold))
            fold_metrics["test_rows"] = int(len(test_fold))
            fold_metrics["train_start"] = str(train_fold["timestamp"].iloc[0])
            fold_metrics["train_end"] = str(train_fold["timestamp"].iloc[-1])
            fold_metrics["test_start"] = str(test_fold["timestamp"].iloc[0])
            fold_metrics["test_end"] = str(test_fold["timestamp"].iloc[-1])
            fold_reports.append(fold_metrics)

        if not fold_reports:
            return {"status": "no_folds"}

        report = self._aggregate_walk_report(
            fold_reports=fold_reports,
            mode=mode,
            n_splits=n_splits,
            symbol_count=len(symbol_levels),
            total_rows=len(labeled),
        )

        logger.info("Walk-forward report: 3class=%.2f%% binary=%.2f%% mae=%.6f folds=%d",
                    report["pooled"]["accuracy_3class_pct"] or 0.0,
                    report["pooled"]["accuracy_binary_excl_flat_pct"] or 0.0,
                    report["pooled"]["mae_return"] or 0.0,
                    len(report["folds"]))

        if save_json_path:
            out_path = Path(save_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Walk-forward report saved to %s", out_path)

        return report

    def score_due_forecasts(self, max_rows: int = 500):
        """Score forecasts whose horizon has elapsed."""
        return self.db.score_due_forecasts(
            max_rows=max_rows,
            default_threshold=self.flat_threshold_min,
        )

    def get_live_oos_metrics(self, lookback_hours: Optional[int] = None, symbol: Optional[str] = None):
        """Return realized live OOS metrics from scored forecasts."""
        hours = int(lookback_hours or self.scoring_lookback_hours)
        return self.db.get_live_oos_metrics(lookback_hours=hours, symbol=symbol)

    def _build_pooled_dataset(
        self,
        current_symbol: str,
        cutoff_ts: int,
        symbols: Optional[Iterable[str]] = None,
    ):
        symbol_list = [str(s).upper() for s in (symbols or self._resolve_training_symbols(current_symbol))]
        if current_symbol not in symbol_list:
            symbol_list.insert(0, current_symbol)

        bars_by_symbol = self.db.get_bars_by_symbols(
            symbols=symbol_list,
            limit_per_symbol=self.training_bars_per_symbol,
            end_timestamp=cutoff_ts,
        )

        market_symbol, market_df = self._get_market_proxy_bars(cutoff_ts=cutoff_ts)

        all_features = []
        latest_per_symbol = {}
        for sym in symbol_list:
            bars = self._prepare_bars(bars_by_symbol.get(sym, pd.DataFrame()))
            if bars.empty or len(bars) < max(40, int(self.min_bars * 0.6)):
                continue

            features = self.feature_engineer.compute_features(bars, market_df=market_df, symbol=sym)
            features["symbol"] = sym
            features["target_raw"] = self._compute_target(features)

            vol_col = f"volatility_{self.target_vol_window}m"
            if vol_col in features.columns:
                target_vol = pd.to_numeric(features[vol_col], errors="coerce")
            else:
                target_vol = pd.to_numeric(features.get("volatility_40m"), errors="coerce")
            if target_vol is None:
                target_vol = pd.Series(index=features.index, dtype=float)
            features["target_vol"] = target_vol.clip(lower=self.target_vol_floor)

            if self.predict_normalized_target:
                features["target_model"] = features["target_raw"] / (features["target_vol"] + 1e-9)
            else:
                features["target_model"] = features["target_raw"]

            features["target_due_at"] = features["timestamp"] + pd.to_timedelta(self.horizon_minutes, unit="m")
            all_features.append(features)

            latest = features[features["timestamp"] <= pd.to_datetime(cutoff_ts, unit="s", utc=True)].tail(1)
            if not latest.empty:
                latest_per_symbol[sym] = latest

        if not all_features:
            return None

        pooled_df = pd.concat(all_features, axis=0, ignore_index=True)
        pooled_df = pooled_df.sort_values("timestamp").reset_index(drop=True)

        feature_cols = [c for c in self.feature_engineer.get_feature_names() if c in pooled_df.columns]
        if not feature_cols:
            return None

        # Drop features that are mostly missing across the pooled dataset.
        nan_ratio = pooled_df[feature_cols].replace([np.inf, -np.inf], np.nan).isna().mean()
        feature_cols = [c for c in feature_cols if float(nan_ratio.get(c, 1.0)) <= self.max_feature_nan_ratio]
        if not feature_cols:
            return None

        # Training rows only include samples with target available by cutoff.
        train_df = pooled_df.dropna(subset=["target_model", "target_raw"]).copy()
        train_df = train_df[np.isfinite(train_df["target_model"])]
        train_df = train_df.sort_values("timestamp").reset_index(drop=True)

        symbol_levels = sorted(train_df["symbol"].dropna().astype(str).unique().tolist())
        if current_symbol not in symbol_levels:
            symbol_levels.append(current_symbol)

        prediction_row = latest_per_symbol.get(current_symbol)
        symbol_features = pooled_df[pooled_df["symbol"] == current_symbol].copy()

        return {
            "train_df": train_df,
            "all_features": pooled_df,
            "feature_cols": feature_cols,
            "symbol_levels": symbol_levels,
            "prediction_row": prediction_row,
            "symbol_features": symbol_features,
            "market_proxy_symbol": market_symbol,
        }

    def _train_models(self, train_df: pd.DataFrame, feature_cols: List[str], symbol_levels: List[str]):
        if train_df is None or train_df.empty or len(train_df) < self.min_train_rows:
            return None

        train_df = train_df.sort_values("timestamp").reset_index(drop=True)
        val_size = max(20, int(len(train_df) * self.validation_fraction))
        val_size = min(val_size, max(0, len(train_df) - 40))

        if val_size > 0:
            split_idx = len(train_df) - val_size
            fit_df = train_df.iloc[:split_idx].copy()
            val_df = train_df.iloc[split_idx:].copy()
        else:
            fit_df = train_df.copy()
            val_df = train_df.iloc[0:0].copy()

        fit_aug, model_feature_cols = self._augment_with_symbol_dummies(fit_df, symbol_levels, feature_cols)
        val_aug, _ = self._augment_with_symbol_dummies(val_df, symbol_levels, feature_cols)
        full_aug, _ = self._augment_with_symbol_dummies(train_df, symbol_levels, feature_cols)

        medians = fit_aug[model_feature_cols].replace([np.inf, -np.inf], np.nan).median(numeric_only=True).to_dict()
        X_fit = self._to_matrix(fit_aug, model_feature_cols, medians)
        y_fit = fit_aug["target_model"].values.astype(float)
        X_full = self._to_matrix(full_aug, model_feature_cols, medians)
        y_full = full_aug["target_model"].values.astype(float)

        if len(X_fit) < 40 or len(X_full) < 40:
            return None

        if len(val_aug) > 0:
            X_val = self._to_matrix(val_aug, model_feature_cols, medians)
            y_val = val_aug["target_model"].values.astype(float)
        else:
            X_val = np.empty((0, len(model_feature_cols)))
            y_val = np.empty((0,))

        fit_weights = np.linspace(0.5, 1.5, len(X_fit))
        full_weights = np.linspace(0.5, 1.5, len(X_full))
        val_weights = np.linspace(0.8, 1.2, len(X_val)) if len(X_val) > 0 else None

        primary_model = get_model(self.model_name, self.config)
        ok = primary_model.train(
            X_fit,
            y_fit,
            sample_weight=fit_weights,
            eval_set=(X_val, y_val) if len(X_val) > 0 else None,
            eval_sample_weight=val_weights if len(X_val) > 0 else None,
        )
        if not ok:
            return None

        if len(X_val) > 0:
            pred_val_primary = np.asarray(primary_model.predict(X_val), dtype=float).reshape(-1)
            rmse_primary = float(np.sqrt(np.mean((pred_val_primary - y_val) ** 2)))
        else:
            pred_val_primary = np.array([])
            rmse_primary = None

        secondary_model = None
        rmse_secondary = None
        w_primary, w_secondary = 1.0, 0.0
        model_name_used = self.model_name
        model_agreement = 100.0
        val_rmse = rmse_primary

        if self.include_ensemble:
            secondary_name = "xgboost" if self.model_name != "xgboost" else "ridge"
            secondary_model = get_model(secondary_name, self.config)
            ok_secondary = secondary_model.train(
                X_fit,
                y_fit,
                sample_weight=fit_weights,
                eval_set=(X_val, y_val) if len(X_val) > 0 else None,
                eval_sample_weight=val_weights if len(X_val) > 0 else None,
            )
            if ok_secondary and len(X_val) > 0:
                pred_val_secondary = np.asarray(secondary_model.predict(X_val), dtype=float).reshape(-1)
                rmse_secondary = float(np.sqrt(np.mean((pred_val_secondary - y_val) ** 2)))
                inv1 = 1.0 / (rmse_primary + 1e-9)
                inv2 = 1.0 / (rmse_secondary + 1e-9)
                w_primary = inv1 / (inv1 + inv2)
                w_secondary = inv2 / (inv1 + inv2)
                pred_val_ensemble = (w_primary * pred_val_primary) + (w_secondary * pred_val_secondary)
                val_rmse = float(np.sqrt(np.mean((pred_val_ensemble - y_val) ** 2)))
                target_std = max(float(np.std(y_fit) + 1e-9), 1e-6)
                mean_gap = float(np.mean(np.abs(pred_val_primary - pred_val_secondary)))
                model_agreement = max(0.0, 100.0 - min(100.0, (mean_gap / target_std) * 100.0))
                model_name_used = f"ensemble({self.model_name}+{secondary_name})"

        # Retrain models on full available training history.
        primary_model.train(
            X_full,
            y_full,
            sample_weight=full_weights,
            eval_set=(X_val, y_val) if len(X_val) > 0 else None,
            eval_sample_weight=val_weights if len(X_val) > 0 else None,
        )
        if secondary_model is not None:
            secondary_model.train(
                X_full,
                y_full,
                sample_weight=full_weights,
                eval_set=(X_val, y_val) if len(X_val) > 0 else None,
                eval_sample_weight=val_weights if len(X_val) > 0 else None,
            )

        target_std = float(np.std(y_fit) + 1e-9)
        return {
            "primary_model": primary_model,
            "secondary_model": secondary_model,
            "w_primary": float(w_primary),
            "w_secondary": float(w_secondary),
            "feature_cols": feature_cols,
            "model_feature_cols": model_feature_cols,
            "symbol_levels": symbol_levels,
            "feature_medians": medians,
            "target_std": target_std,
            "val_rmse": val_rmse,
            "model_agreement": model_agreement,
            "model_name": model_name_used,
        }

    def _predict_one_row(self, model_bundle: dict, row_df: pd.DataFrame):
        if row_df is None or row_df.empty:
            return None

        aug_row, _ = self._augment_with_symbol_dummies(
            row_df.copy(),
            model_bundle["symbol_levels"],
            model_bundle["feature_cols"],
        )
        X = self._to_matrix(aug_row, model_bundle["model_feature_cols"], model_bundle["feature_medians"])
        if len(X) == 0:
            return None
        x_vec = X[0]

        primary = model_bundle["primary_model"].predict_intervals(x_vec)
        secondary_model = model_bundle.get("secondary_model")
        if secondary_model is None:
            return primary

        secondary = secondary_model.predict_intervals(x_vec)
        return self._combine_predictions(
            primary,
            secondary,
            model_bundle.get("w_primary", 0.5),
            model_bundle.get("w_secondary", 0.5),
        )

    def _predict_frame(self, model_bundle: dict, frame: pd.DataFrame):
        aug_df, _ = self._augment_with_symbol_dummies(
            frame.copy(),
            model_bundle["symbol_levels"],
            model_bundle["feature_cols"],
        )
        X = self._to_matrix(aug_df, model_bundle["model_feature_cols"], model_bundle["feature_medians"])
        if len(X) == 0:
            return np.array([]), None

        pred_primary = np.asarray(model_bundle["primary_model"].predict(X), dtype=float).reshape(-1)
        secondary_model = model_bundle.get("secondary_model")
        if secondary_model is not None:
            pred_secondary = np.asarray(secondary_model.predict(X), dtype=float).reshape(-1)
            pred_model = (model_bundle.get("w_primary", 0.5) * pred_primary) + (
                model_bundle.get("w_secondary", 0.5) * pred_secondary
            )
        else:
            pred_model = pred_primary

        if self.predict_normalized_target:
            vol = pd.to_numeric(frame["target_vol"], errors="coerce").fillna(self.target_vol_floor).clip(lower=self.target_vol_floor)
            pred_raw = pred_model * vol.values
            return pred_raw, pred_model
        return pred_model, None

    def _augment_with_symbol_dummies(self, df: pd.DataFrame, symbol_levels: List[str], feature_cols: List[str]):
        out = df.copy()
        model_feature_cols = list(feature_cols)
        if self.include_symbol_one_hot:
            for sym in symbol_levels:
                col = self._symbol_col(sym)
                out[col] = (out["symbol"].astype(str).str.upper() == str(sym).upper()).astype(float)
                model_feature_cols.append(col)
        return out, model_feature_cols

    def _to_matrix(self, df: pd.DataFrame, cols: List[str], medians: Dict[str, float]):
        if df.empty:
            return np.empty((0, len(cols)))
        xdf = df.reindex(columns=cols).copy()
        for col in cols:
            xdf[col] = pd.to_numeric(xdf[col], errors="coerce")
            if col in medians and np.isfinite(medians[col]):
                xdf[col] = xdf[col].fillna(float(medians[col]))
            else:
                xdf[col] = xdf[col].fillna(0.0)
        xdf = xdf.replace([np.inf, -np.inf], 0.0)
        return xdf.values.astype(float)

    def _resolve_training_symbols(self, current_symbol: Optional[str] = None):
        counts = self.db.get_symbol_counts(min_rows=max(60, int(self.min_bars * 0.6)))
        if counts.empty:
            return [current_symbol] if current_symbol else []

        symbols = counts["symbol"].astype(str).str.upper().tolist()
        proxies = {str(s).upper() for s in self.market_proxy_symbols}
        symbols = [s for s in symbols if s not in proxies]
        if current_symbol and current_symbol.upper() not in symbols:
            symbols.insert(0, current_symbol.upper())
        return symbols[: self.max_symbols]

    def _get_market_proxy_bars(self, cutoff_ts: int):
        """Return first available market proxy bars and symbol."""
        for proxy in self.market_proxy_symbols:
            sym = str(proxy).upper()
            bars = self.db.get_bars(sym, limit=self.market_proxy_limit)
            if bars.empty:
                try:
                    live = self.api.get_bars(sym, limit=self.market_proxy_limit)
                    live = self._prepare_bars(live)
                    if not live.empty:
                        self.db.save_bars(sym, live)
                        bars = live
                except Exception:
                    bars = pd.DataFrame()
            bars = self._prepare_bars(bars)
            if bars.empty:
                continue
            bars = bars[bars["timestamp"] <= pd.to_datetime(cutoff_ts, unit="s", utc=True)].copy()
            if len(bars) >= 80:
                return sym, bars
        return None, pd.DataFrame()

    def _get_latest_bar_ts(self, symbols: Iterable[str]):
        latest = None
        for sym in symbols:
            bars = self.db.get_bars(sym, limit=1)
            if bars.empty:
                continue
            ts = int(pd.Timestamp(bars["timestamp"].iloc[-1]).timestamp())
            latest = ts if latest is None else max(latest, ts)
        return latest

    def _compute_target(self, df):
        """Forward return target over configured horizon."""
        forward_close = pd.to_numeric(df["close"], errors="coerce").shift(-self.horizon_minutes)
        close_now = pd.to_numeric(df["close"], errors="coerce")
        return (forward_close - close_now) / (close_now + 1e-9)

    @staticmethod
    def _symbol_col(symbol: str):
        clean = "".join(ch if ch.isalnum() else "_" for ch in str(symbol).upper())
        return f"sym_{clean}"

    def _compute_direction_threshold(self, vol_40: Optional[float]):
        if self.flat_threshold_mode == "volatility":
            vol = float(vol_40 or 0.0)
            return float(max(self.flat_threshold_min, self.flat_threshold_multiplier * vol))
        return float(self.flat_threshold_min)

    def _determine_direction(self, predicted_return: float, threshold: Optional[float] = None):
        thresh = float(self.flat_threshold_min if threshold is None else threshold)
        if predicted_return > thresh:
            return "UP"
        if predicted_return < -thresh:
            return "DOWN"
        return "FLAT"

    def _compute_eval_metrics(self, test_df: pd.DataFrame, pred_raw: np.ndarray, pred_norm: Optional[np.ndarray]):
        if len(test_df) == 0 or len(pred_raw) == 0:
            return {
                "accuracy_3class_pct": None,
                "accuracy_binary_excl_flat_pct": None,
                "mae_return": None,
                "mae_norm": None,
                "sample_count": 0,
                "by_symbol": [],
            }

        test_local = test_df.reset_index(drop=True).copy()
        true_raw = pd.to_numeric(test_local["target_raw"], errors="coerce").values
        vol = pd.to_numeric(test_local["target_vol"], errors="coerce").fillna(self.target_vol_floor).values
        if self.flat_threshold_mode == "volatility":
            thresholds = np.maximum(self.flat_threshold_min, self.flat_threshold_multiplier * vol)
        else:
            thresholds = np.full_like(vol, self.flat_threshold_min, dtype=float)

        pred_dir = np.where(pred_raw > thresholds, "UP", np.where(pred_raw < -thresholds, "DOWN", "FLAT"))
        true_dir = np.where(true_raw > thresholds, "UP", np.where(true_raw < -thresholds, "DOWN", "FLAT"))

        acc3 = float(np.mean(pred_dir == true_dir) * 100.0)
        mask_bin = (pred_dir != "FLAT") & (true_dir != "FLAT")
        acc_bin = float(np.mean(pred_dir[mask_bin] == true_dir[mask_bin]) * 100.0) if np.any(mask_bin) else None
        mae_return = float(np.mean(np.abs(pred_raw - true_raw)))

        if pred_norm is not None:
            true_norm = pd.to_numeric(test_local["target_model"], errors="coerce").values
            mae_norm = float(np.nanmean(np.abs(pred_norm - true_norm)))
        else:
            mae_norm = None

        by_symbol = []
        for sym, grp in test_local.groupby("symbol"):
            idx = grp.index.values
            sym_pred = pred_raw[idx]
            sym_true = true_raw[idx]
            sym_vol = vol[idx]
            if self.flat_threshold_mode == "volatility":
                sym_thr = np.maximum(self.flat_threshold_min, self.flat_threshold_multiplier * sym_vol)
            else:
                sym_thr = np.full_like(sym_vol, self.flat_threshold_min, dtype=float)
            s_pred_dir = np.where(sym_pred > sym_thr, "UP", np.where(sym_pred < -sym_thr, "DOWN", "FLAT"))
            s_true_dir = np.where(sym_true > sym_thr, "UP", np.where(sym_true < -sym_thr, "DOWN", "FLAT"))
            s_acc3 = float(np.mean(s_pred_dir == s_true_dir) * 100.0)
            s_mask_bin = (s_pred_dir != "FLAT") & (s_true_dir != "FLAT")
            s_acc_bin = float(np.mean(s_pred_dir[s_mask_bin] == s_true_dir[s_mask_bin]) * 100.0) if np.any(s_mask_bin) else None
            by_symbol.append(
                {
                    "symbol": str(sym),
                    "sample_count": int(len(grp)),
                    "accuracy_3class_pct": s_acc3,
                    "accuracy_binary_excl_flat_pct": s_acc_bin,
                    "mae_return": float(np.mean(np.abs(sym_pred - sym_true))),
                }
            )

        return {
            "accuracy_3class_pct": acc3,
            "accuracy_binary_excl_flat_pct": acc_bin,
            "mae_return": mae_return,
            "mae_norm": mae_norm,
            "sample_count": int(len(test_local)),
            "by_symbol": by_symbol,
        }

    def _aggregate_walk_report(self, fold_reports, mode, n_splits, symbol_count, total_rows):
        weights = np.array([f["sample_count"] for f in fold_reports], dtype=float)
        weights = np.where(weights <= 0, 1.0, weights)

        def wavg(key):
            vals = np.array([f[key] if f.get(key) is not None else np.nan for f in fold_reports], dtype=float)
            mask = np.isfinite(vals)
            if not np.any(mask):
                return None
            return float(np.average(vals[mask], weights=weights[mask]))

        pooled = {
            "accuracy_3class_pct": wavg("accuracy_3class_pct"),
            "accuracy_binary_excl_flat_pct": wavg("accuracy_binary_excl_flat_pct"),
            "mae_return": wavg("mae_return"),
            "mae_norm": wavg("mae_norm"),
            "sample_count": int(np.sum(weights)),
        }

        symbol_agg = {}
        for fold in fold_reports:
            for row in fold.get("by_symbol", []):
                sym = row["symbol"]
                symbol_agg.setdefault(sym, {"rows": 0, "acc3_sum": 0.0, "mae_sum": 0.0, "acc3_w": 0.0, "mae_w": 0.0, "acc_bin_vals": []})
                n = float(row["sample_count"])
                symbol_agg[sym]["rows"] += int(n)
                if row.get("accuracy_3class_pct") is not None:
                    symbol_agg[sym]["acc3_sum"] += float(row["accuracy_3class_pct"]) * n
                    symbol_agg[sym]["acc3_w"] += n
                if row.get("mae_return") is not None:
                    symbol_agg[sym]["mae_sum"] += float(row["mae_return"]) * n
                    symbol_agg[sym]["mae_w"] += n
                if row.get("accuracy_binary_excl_flat_pct") is not None:
                    symbol_agg[sym]["acc_bin_vals"].append(float(row["accuracy_binary_excl_flat_pct"]))

        per_symbol = []
        for sym, agg in symbol_agg.items():
            per_symbol.append(
                {
                    "symbol": sym,
                    "sample_count": int(agg["rows"]),
                    "accuracy_3class_pct": float(agg["acc3_sum"] / agg["acc3_w"]) if agg["acc3_w"] > 0 else None,
                    "accuracy_binary_excl_flat_pct": float(np.mean(agg["acc_bin_vals"])) if agg["acc_bin_vals"] else None,
                    "mae_return": float(agg["mae_sum"] / agg["mae_w"]) if agg["mae_w"] > 0 else None,
                }
            )
        per_symbol.sort(key=lambda x: x["sample_count"], reverse=True)

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": mode,
            "requested_splits": int(n_splits),
            "actual_folds": int(len(fold_reports)),
            "symbol_count": int(symbol_count),
            "total_rows": int(total_rows),
            "pooled": pooled,
            "per_symbol": per_symbol,
            "folds": fold_reports,
        }

    def _compute_confidence(self, train_df, feature_cols, val_rmse=None, target_std=None, model_agreement=50.0):
        try:
            complete_rows = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).sum()
            data_coverage = min(100.0, (complete_rows / max(1, len(train_df))) * 100.0)

            if "volatility_40m" in train_df.columns:
                vol_series = pd.to_numeric(train_df["volatility_40m"], errors="coerce").dropna()
                if len(vol_series) > 20 and float(vol_series.mean()) != 0:
                    vol_cv = float(vol_series.std() / (vol_series.mean() + 1e-9))
                    volatility_stability = max(0.0, 100.0 - (vol_cv * 100.0))
                else:
                    volatility_stability = 50.0
            else:
                volatility_stability = 50.0

            if val_rmse is not None and target_std is not None and target_std > 0:
                predictive_quality = max(0.0, 100.0 * (1.0 - min(1.0, val_rmse / (target_std * 1.5 + 1e-9))))
            else:
                predictive_quality = 50.0

            agreement_score = max(0.0, min(100.0, float(model_agreement)))
            news_available = 20.0 if getattr(self.news, "enabled", False) else 0.0

            weights = {
                "data_coverage": 0.35,
                "volatility_stability": 0.25,
                "predictive_quality": 0.25,
                "agreement_score": 0.1,
                "news_available": 0.05,
            }
            confidence = (
                weights["data_coverage"] * data_coverage
                + weights["volatility_stability"] * volatility_stability
                + weights["predictive_quality"] * predictive_quality
                + weights["agreement_score"] * agreement_score
                + weights["news_available"] * news_available
            ) / sum(weights.values())
            return float(max(0.0, min(100.0, confidence)))
        except Exception as exc:
            logger.warning("Error computing confidence: %s", exc)
            return 50.0

    def _trend_baseline_prediction(self, df):
        try:
            closes = pd.to_numeric(df["close"], errors="coerce").dropna()
            if len(closes) < 20:
                return 0.0
            recent_ret = closes.pct_change().dropna().tail(120)
            if recent_ret.empty:
                return 0.0
            ewma_mean = recent_ret.ewm(span=12, adjust=False).mean().iloc[-1]
            baseline = float(ewma_mean * self.horizon_minutes)
            return float(np.clip(baseline, -self.max_abs_prediction, self.max_abs_prediction))
        except Exception:
            return 0.0

    def _blend_with_trend(self, pred_result, trend_pred, blend_w):
        out = dict(pred_result)
        model_mean = float(out.get("mean", 0.0))
        blended = ((1.0 - float(blend_w)) * model_mean) + (float(blend_w) * float(trend_pred))
        delta = blended - model_mean
        out["mean"] = blended
        for k in ["lower_50", "upper_50", "lower_90", "upper_90"]:
            out[k] = float(out.get(k, blended)) + delta
        return out

    def _combine_predictions(self, pred_a, pred_b, w_a=0.5, w_b=0.5):
        keys = ["mean", "lower_50", "upper_50", "lower_90", "upper_90"]
        out = {}
        for k in keys:
            a = float(pred_a.get(k, 0.0))
            b = float(pred_b.get(k, 0.0))
            out[k] = (float(w_a) * a) + (float(w_b) * b)
        return out

    def _clip_prediction_dict(self, pred_result, max_abs):
        out = dict(pred_result)
        for k in ["mean", "lower_50", "upper_50", "lower_90", "upper_90"]:
            out[k] = float(np.clip(float(out.get(k, 0.0)), -max_abs, max_abs))
        out["lower_50"], out["upper_50"] = sorted([out["lower_50"], out["upper_50"]])
        out["lower_90"], out["upper_90"] = sorted([out["lower_90"], out["upper_90"]])
        out["lower_90"] = min(out["lower_90"], out["lower_50"])
        out["upper_90"] = max(out["upper_90"], out["upper_50"])
        return out

    def _scale_prediction_dict(self, pred_result, factor):
        out = dict(pred_result)
        for k in ["mean", "lower_50", "upper_50", "lower_90", "upper_90"]:
            out[k] = float(out.get(k, 0.0)) * float(factor)
        return out

    def _prepare_bars(self, bars):
        if bars is None or getattr(bars, "empty", True):
            return pd.DataFrame()

        df = bars.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
            df = df.drop_duplicates(subset=["timestamp"], keep="last")

        for col in ["open", "high", "low", "close", "volume", "vwap"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "close" in df.columns:
            df = df[df["close"] > 0]
            ret = df["close"].pct_change()
            spike_mask = ret.abs() < 0.35
            df = df[spike_mask.fillna(True)]

        return df.reset_index(drop=True)

    def _persist_forecast(self, symbol: str, forecast: dict):
        if not self.enable_persistence:
            return
        try:
            payload = dict(forecast)
            payload.setdefault("horizon_minutes", self.horizon_minutes)
            payload.setdefault("features_version", self.FEATURES_VERSION)
            payload.setdefault("model_version", self.MODEL_VERSION)
            payload.setdefault("config_hash", self._config_hash)
            self.db.save_forecast(symbol, payload)
        except Exception as exc:
            logger.warning("Failed to persist forecast for %s: %s", symbol, exc)

    def _compute_config_hash(self):
        try:
            serialized = json.dumps(self.config, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "unknown"

    def _get_sentiment_impact(self, symbol):
        if not getattr(self.news, "enabled", False):
            return {"available": False, "overall_sentiment": 0, "score": 0.0}
        try:
            headlines = self.news.get_headlines(symbol, limit=10)
            sentiment = self.sentiment_analyzer.aggregate_sentiment(headlines)
            return {
                "available": True,
                "overall_sentiment": sentiment["overall_sentiment"],
                "score": sentiment["score"],
                "positive_count": sentiment["positive_count"],
                "negative_count": sentiment["negative_count"],
                "headline_count": sentiment["headline_count"],
            }
        except Exception as exc:
            logger.warning("Error getting sentiment for %s: %s", symbol, exc)
            return {"available": False, "overall_sentiment": 0, "score": 0.0}

    def _create_status_forecast(self, status, **extra):
        payload = {
            "prediction_return": None,
            "prediction_return_raw": None,
            "prediction_return_norm": None,
            "direction": None,
            "predicted_direction": None,
            "confidence": 0.0,
            "confidence_score": 0.0,
            "interval_50_lower": None,
            "interval_50_upper": None,
            "interval_90_lower": None,
            "interval_90_upper": None,
            "model_status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "horizon_minutes": self.horizon_minutes,
            "features_version": self.FEATURES_VERSION,
            "model_version": self.MODEL_VERSION,
            "config_hash": self._config_hash,
            "sentiment_impact": {"available": False},
        }
        payload.update(extra)
        return payload

    def _get_market_schedule(self, symbol):
        sym = str(symbol).upper()
        if sym.endswith(".T"):
            return "Asia/Tokyo", [(9, 0, 11, 30), (12, 30, 15, 30)]
        return "US/Eastern", [(9, 30, 16, 0)]

    def _is_exchange_holiday(self, symbol, date_local):
        if date_local.weekday() >= 5:
            return True
        sym = str(symbol).upper()
        if sym.endswith(".T"):
            if (date_local.month, date_local.day) in {(1, 1), (1, 2), (1, 3), (12, 31)}:
                return True
            if jpholiday is not None:
                try:
                    return bool(jpholiday.is_holiday(date_local))
                except Exception:
                    pass
        return False

    def _next_session_open(self, symbol, now_local, sessions):
        first_h, first_m, _, _ = sessions[0]
        same_day_first = now_local.replace(hour=first_h, minute=first_m, second=0, microsecond=0)
        if (not self._is_exchange_holiday(symbol, now_local.date())) and now_local < same_day_first:
            return same_day_first
        candidate = (now_local + timedelta(days=1)).replace(hour=first_h, minute=first_m, second=0, microsecond=0)
        while self._is_exchange_holiday(symbol, candidate.date()):
            candidate = (candidate + timedelta(days=1)).replace(hour=first_h, minute=first_m, second=0, microsecond=0)
        return candidate

    def _get_market_context(self, symbol, last_bar_ts=None):
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
                for i in range(len(session_bounds) - 1):
                    end_i = session_bounds[i][1]
                    next_start = session_bounds[i + 1][0]
                    if end_i < now_local < next_start:
                        next_open_local = next_start
                        break
                if next_open_local is None:
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
            ts = pd.to_datetime(last_bar_ts, utc=True, errors="coerce")
            if pd.notna(ts):
                ts_local = ts.tz_convert(tz)
                last_bar_local = ts_local.strftime("%Y-%m-%d %H:%M %Z")
                try:
                    delta_min = (now_local - ts_local.to_pydatetime()).total_seconds() / 60.0
                    last_bar_age_minutes = max(0.0, float(delta_min))
                except Exception:
                    last_bar_age_minutes = None

        return {
            "timezone": tz_name,
            "is_open": is_open,
            "minutes_to_close": minutes_to_close,
            "next_open_local": next_open_local.strftime("%Y-%m-%d %H:%M %Z") if next_open_local else None,
            "last_bar_local": last_bar_local,
            "last_bar_age_minutes": last_bar_age_minutes,
        }
