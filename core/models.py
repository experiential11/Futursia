"""Forecasting models and prediction generation."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class ForecastModel:
    """Base class for forecast models."""
    
    def __init__(self, config):
        """Initialize model.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get('forecast', {})
        self.horizon_minutes = self.config.get('horizon_minutes', 40)
        self.min_bars = self.config.get('min_bars_for_forecast', 100)
    
    def train(self, X, y, sample_weight=None):
        """Train the model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        
        Returns:
            bool: True if training successful
        """
        raise NotImplementedError
    
    def predict(self, X):
        """Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features) or (n_features,)
        
        Returns:
            array: Predictions
        """
        raise NotImplementedError
    
    def predict_intervals(self, X):
        """Predict with confidence intervals.
        
        Args:
            X: Feature matrix
        
        Returns:
            dict: With keys 'mean', 'lower_50', 'upper_50', 'lower_90', 'upper_90'
        """
        raise NotImplementedError


class RidgeForecaster(ForecastModel):
    """Ridge regression forecaster - fast and stable."""
    
    def __init__(self, config):
        """Initialize Ridge forecaster.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.residuals = []
        self._is_trained = False
    
    def train(self, X, y, sample_weight=None):
        """Train Ridge model.
        
        Args:
            X: Feature matrix
            y: Target returns
        
        Returns:
            bool: True if successful
        """
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
        """Make prediction.
        
        Args:
            X: Feature vector or matrix
        
        Returns:
            array or float: Prediction(s)
        """
        if not self._is_trained:
            return 0.0
        
        X_array = np.array(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X_array)
        return self.model.predict(X_scaled)
    
    def predict_intervals(self, X):
        """Predict with confidence intervals using residual bootstrap.
        
        Args:
            X: Feature vector (1D array)
        
        Returns:
            dict: Prediction with intervals
        """
        if not self._is_trained or len(self.residuals) < 10:
            pred = self.predict(X)
            return {
                'mean': float(pred[0]) if hasattr(pred, '__len__') else float(pred),
                'lower_50': float(pred) * 0.98 if hasattr(pred, '__len__') else float(pred) * 0.98,
                'upper_50': float(pred) * 1.02 if hasattr(pred, '__len__') else float(pred) * 1.02,
                'lower_90': float(pred) * 0.95 if hasattr(pred, '__len__') else float(pred) * 0.95,
                'upper_90': float(pred) * 1.05 if hasattr(pred, '__len__') else float(pred) * 1.05,
            }
        
        pred = self.predict(X)
        pred_scalar = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
        
        # Use residual standard deviation for intervals
        residual_std = np.std(self.residuals)
        z_50 = 0.6745  # ~50% confidence
        z_90 = 1.645   # ~90% confidence
        
        return {
            'mean': pred_scalar,
            'lower_50': pred_scalar - z_50 * residual_std,
            'upper_50': pred_scalar + z_50 * residual_std,
            'lower_90': pred_scalar - z_90 * residual_std,
            'upper_90': pred_scalar + z_90 * residual_std,
        }


class XGBoostForecaster(ForecastModel):
    """XGBoost forecaster - gradient boosting model."""
    
    def __init__(self, config):
        """Initialize XGBoost forecaster.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
            self.available = True
        except ImportError:
            self.model = None
            self.available = False
            self.fallback = RidgeForecaster(config)
        
        self.residuals = []
        self._is_trained = False
    
    def train(self, X, y, sample_weight=None):
        """Train XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target returns
        
        Returns:
            bool: True if successful
        """
        if not self.available:
            return self.fallback.train(X, y, sample_weight=sample_weight)
        
        if X.shape[0] < 20:
            return False

        try:
            X_arr = np.asarray(X, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            mask = np.isfinite(X_arr).all(axis=1) & np.isfinite(y_arr)
            X_arr = X_arr[mask]
            y_arr = y_arr[mask]
            if len(X_arr) < 20:
                return False

            if sample_weight is not None:
                sw = np.asarray(sample_weight, dtype=float)[mask]
                self.model.fit(X_arr, y_arr, sample_weight=sw, verbose=False)
            else:
                self.model.fit(X_arr, y_arr, verbose=False)
            self.residuals = y_arr - self.model.predict(X_arr)
            self._is_trained = True
            return True
        except Exception:
            return False
    
    def predict(self, X):
        """Make prediction.
        
        Args:
            X: Feature vector or matrix
        
        Returns:
            array or float: Prediction(s)
        """
        if not self.available:
            return self.fallback.predict(X)
        if not self._is_trained:
            return 0.0
        
        X_array = np.array(X)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)
        
        return self.model.predict(X_array)
    
    def predict_intervals(self, X):
        """Predict with confidence intervals.
        
        Args:
            X: Feature vector
        
        Returns:
            dict: Prediction with intervals
        """
        if not self.available:
            return self.fallback.predict_intervals(X)
        pred = self.predict(X)
        
        pred_scalar = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
        
        if len(self.residuals) < 10:
            return {
                'mean': pred_scalar,
                'lower_50': pred_scalar * 0.98,
                'upper_50': pred_scalar * 1.02,
                'lower_90': pred_scalar * 0.95,
                'upper_90': pred_scalar * 1.05,
            }

        residual_std = np.std(self.residuals)
        z_50 = 0.6745
        z_90 = 1.645
        
        return {
            'mean': pred_scalar,
            'lower_50': pred_scalar - z_50 * residual_std,
            'upper_50': pred_scalar + z_50 * residual_std,
            'lower_90': pred_scalar - z_90 * residual_std,
            'upper_90': pred_scalar + z_90 * residual_std,
        }


def get_model(model_name, config):
    """Factory function to get appropriate model.
    
    Args:
        model_name: 'ridge' or 'xgboost'
        config: Configuration dictionary
    
    Returns:
        ForecastModel: Model instance
    """
    if model_name == 'xgboost':
        return XGBoostForecaster(config)
    else:
        return RidgeForecaster(config)
