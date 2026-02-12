"""Forecasting models and prediction generation."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ForecastModel:
    """Base class for forecast models."""

    def __init__(self, config):
        self.config = config.get("forecast", {})
        self.horizon_minutes = self.config.get("horizon_minutes", 40)
        self.min_bars = self.config.get("min_bars_for_forecast", 100)

    def train(
        self,
        X,
        y,
        sample_weight=None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        eval_sample_weight=None,
    ):
        """Train the model."""
        raise NotImplementedError

    def predict(self, X):
        """Make predictions."""
        raise NotImplementedError

    def predict_intervals(self, X):
        """Predict with confidence intervals."""
        raise NotImplementedError


class RidgeForecaster(ForecastModel):
    """Ridge regression forecaster."""

    def __init__(self, config):
        super().__init__(config)
        ridge_cfg = self.config.get("ridge", {})
        alpha = float(ridge_cfg.get("alpha", 1.0))
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.residuals = []
        self._is_trained = False

    def train(
        self,
        X,
        y,
        sample_weight=None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        eval_sample_weight=None,
    ):
        if X.shape[0] < 10:
            return False

        try:
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            if len(X_arr) < 10:
                return False

            X_scaled = self.scaler.fit_transform(X_arr)
            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=float)[mask]
                self.model.fit(X_scaled, y_arr, sample_weight=sw)
            else:
                self.model.fit(X_scaled, y_arr)
            self.residuals = y_arr - self.model.predict(X_scaled)
            self._is_trained = True
            return True
        except Exception:
            return False

    def predict(self, X):
        if not self._is_trained:
            return 0.0

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict(X_scaled)

    def predict_intervals(self, X):
        pred = self.predict(X)
        pred_scalar = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        if not self._is_trained or len(self.residuals) < 10:
            return {
                "mean": pred_scalar,
                "lower_50": pred_scalar * 0.98,
                "upper_50": pred_scalar * 1.02,
                "lower_90": pred_scalar * 0.95,
                "upper_90": pred_scalar * 1.05,
            }

        residual_std = float(np.std(self.residuals))
        z_50 = 0.6745
        z_90 = 1.645
        return {
            "mean": pred_scalar,
            "lower_50": pred_scalar - z_50 * residual_std,
            "upper_50": pred_scalar + z_50 * residual_std,
            "lower_90": pred_scalar - z_90 * residual_std,
            "upper_90": pred_scalar + z_90 * residual_std,
        }


class XGBoostForecaster(ForecastModel):
    """XGBoost forecaster with optional early stopping."""

    def __init__(self, config):
        super().__init__(config)
        xgb_cfg = self.config.get("xgboost", {})
        self.early_stopping_rounds = int(xgb_cfg.get("early_stopping_rounds", 40))
        self.eval_metric = xgb_cfg.get("eval_metric", "rmse")

        try:
            import xgboost as xgb

            self.model = xgb.XGBRegressor(
                n_estimators=int(xgb_cfg.get("n_estimators", 400)),
                max_depth=int(xgb_cfg.get("max_depth", 4)),
                learning_rate=float(xgb_cfg.get("learning_rate", 0.04)),
                min_child_weight=float(xgb_cfg.get("min_child_weight", 2.0)),
                subsample=float(xgb_cfg.get("subsample", 0.9)),
                colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
                reg_alpha=float(xgb_cfg.get("reg_alpha", 0.0)),
                reg_lambda=float(xgb_cfg.get("reg_lambda", 1.0)),
                random_state=int(xgb_cfg.get("random_state", 42)),
                objective=xgb_cfg.get("objective", "reg:squarederror"),
                tree_method=xgb_cfg.get("tree_method", "hist"),
                n_jobs=int(xgb_cfg.get("n_jobs", 1)),
                verbosity=0,
            )
            self.available = True
        except Exception:
            self.model = None
            self.available = False
            self.fallback = RidgeForecaster(config)

        self.residuals = []
        self._is_trained = False

    def train(
        self,
        X,
        y,
        sample_weight=None,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        eval_sample_weight=None,
    ):
        if not self.available:
            return self.fallback.train(
                X,
                y,
                sample_weight=sample_weight,
                eval_set=eval_set,
                eval_sample_weight=eval_sample_weight,
            )

        if X.shape[0] < 30:
            return False

        try:
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            if len(X_arr) < 30:
                return False

            fit_kwargs = {
                "verbose": False,
                "eval_metric": self.eval_metric,
            }
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=float)[mask]

            if eval_set is not None:
                X_eval, y_eval = eval_set
                X_eval = np.asarray(X_eval, dtype=float)
                y_eval = np.asarray(y_eval, dtype=float)
                eval_mask = np.isfinite(X_eval).all(axis=1) & np.isfinite(y_eval)
                X_eval = X_eval[eval_mask]
                y_eval = y_eval[eval_mask]
                if len(X_eval) >= 20:
                    fit_kwargs["eval_set"] = [(X_eval, y_eval)]
                    fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds
                    if eval_sample_weight is not None:
                        eval_sw = np.asarray(eval_sample_weight, dtype=float)
                        fit_kwargs["sample_weight_eval_set"] = [eval_sw[eval_mask]]

            self.model.fit(X_arr, y_arr, **fit_kwargs)
            self.residuals = y_arr - self.model.predict(X_arr)
            self._is_trained = True
            return True
        except Exception:
            return False

    def predict(self, X):
        if not self.available:
            return self.fallback.predict(X)
        if not self._is_trained:
            return 0.0

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return self.model.predict(X_arr)

    def predict_intervals(self, X):
        if not self.available:
            return self.fallback.predict_intervals(X)
        pred = self.predict(X)
        pred_scalar = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        if len(self.residuals) < 10:
            return {
                "mean": pred_scalar,
                "lower_50": pred_scalar * 0.98,
                "upper_50": pred_scalar * 1.02,
                "lower_90": pred_scalar * 0.95,
                "upper_90": pred_scalar * 1.05,
            }

        residual_std = float(np.std(self.residuals))
        z_50 = 0.6745
        z_90 = 1.645
        return {
            "mean": pred_scalar,
            "lower_50": pred_scalar - z_50 * residual_std,
            "upper_50": pred_scalar + z_50 * residual_std,
            "lower_90": pred_scalar - z_90 * residual_std,
            "upper_90": pred_scalar + z_90 * residual_std,
        }


def get_model(model_name, config):
    """Return a configured model implementation."""
    if str(model_name).lower() == "xgboost":
        return XGBoostForecaster(config)
    return RidgeForecaster(config)
