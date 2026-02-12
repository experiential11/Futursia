import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.db import MarketDB
from core.features import FeatureEngineer
from core.forecasting import ForecasterEngine


class DummyClient:
    def __init__(self, db):
        self.db = db

    def get_bars(self, symbol, limit=1000):
        return self.db.get_bars(symbol, limit=limit)


class DummyNews:
    enabled = False

    def get_headlines(self, symbol, limit=10):
        return []


def base_config(db_path: str):
    return {
        "storage": {"database": db_path},
        "forecast": {
            "horizon_minutes": 2,
            "min_bars_for_forecast": 10,
            "primary_model": "ridge",
            "include_ensemble": False,
            "validation_fraction": 0.2,
            "predict_normalized_target": True,
            "target_vol_window": 2,
            "flat_threshold_mode": "volatility",
            "flat_threshold_multiplier": 0.5,
            "flat_threshold_min": 0.001,
            "pooled_training": {"max_symbols": 5, "bars_per_symbol": 200, "min_train_rows": 20},
            "persist_predictions": True,
        },
        "features": {
            "return_lags": [1, 2],
            "ma_periods": [2, 3],
            "volatility_periods": [2, 3],
            "trend_windows": [2, 3],
            "volume_spike_windows": [2, 3],
            "vwap_windows": [2],
            "rsi_period": 2,
            "include_time_features": True,
            "include_market_regime": True,
            "regime_window": 5,
        },
        "market_context": {"proxy_symbols": ["SPY"], "beta_windows": [2]},
    }


class ForecastingUpgradeTests(unittest.TestCase):
    def test_feature_alignment_uses_backward_market_data_only(self):
        cfg = base_config("storage/test_unused.db")
        fe = FeatureEngineer(cfg)

        ts = pd.date_range("2026-01-01 14:30:00+00:00", periods=5, freq="min", tz="UTC")
        sym_df = pd.DataFrame(
            {
                "timestamp": ts,
                "open": [100, 100.5, 101, 101.5, 102],
                "high": [100.2, 100.7, 101.2, 101.7, 102.2],
                "low": [99.8, 100.2, 100.7, 101.2, 101.7],
                "close": [100, 100.4, 100.8, 101.2, 101.6],
                "volume": [1000, 1200, 900, 1100, 1000],
            }
        )
        mkt_df = pd.DataFrame(
            {
                "timestamp": [ts[0], ts[2], ts[4]],
                "open": [100, 100, 200],
                "high": [100, 100, 200],
                "low": [100, 100, 200],
                "close": [100, 100, 200],
                "volume": [1000, 1000, 1000],
            }
        )

        out = fe.compute_features(sym_df, market_df=mkt_df, symbol="AAPL")
        row_t3 = out.loc[out["timestamp"] == ts[3]].iloc[0]
        # If lookahead existed, this row would see the huge jump from ts[4].
        self.assertLess(abs(float(row_t3.get("market_return_1m", 0.0) or 0.0)), 1e-8)

    def test_target_shift_correctness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "market.db")
            cfg = base_config(db_path)
            engine = ForecasterEngine(cfg, DummyClient(MarketDB(db_path)), DummyNews())
            df = pd.DataFrame({"close": [100.0, 102.0, 104.0, 106.0, 108.0]})
            y = engine._compute_target(df)
            self.assertAlmostEqual(float(y.iloc[0]), 0.04, places=9)  # (104-100)/100
            self.assertAlmostEqual(float(y.iloc[1]), 0.0392156862745098, places=9)  # (106-102)/102
            self.assertTrue(pd.isna(y.iloc[-1]))

    def test_forecast_persistence_and_scoring(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "market.db")
            db = MarketDB(db_path)

            now_ts = int(datetime.utcnow().timestamp())
            t0 = now_ts - 180
            t1 = now_ts - 120
            t2 = now_ts - 60

            bars = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime([t0, t1, t2], unit="s", utc=True),
                    "open": [100.0, 100.0, 110.0],
                    "high": [100.0, 100.0, 110.0],
                    "low": [100.0, 100.0, 110.0],
                    "close": [100.0, 100.0, 110.0],
                    "volume": [1000, 1000, 1000],
                }
            )
            db.save_bars("TEST", bars)

            db.save_forecast(
                "TEST",
                {
                    "forecast_time": t0,
                    "horizon_minutes": 2,
                    "target_due_at": t2,
                    "prediction_return": 0.05,
                    "prediction_return_raw": 0.05,
                    "direction": "UP",
                    "predicted_direction": "UP",
                    "direction_threshold": 0.001,
                    "confidence": 60.0,
                    "model_status": "OK",
                    "features_version": "test",
                    "model_version": "test",
                    "config_hash": "abc123",
                },
            )

            res = db.score_due_forecasts(max_rows=10, default_threshold=0.001, tolerance_seconds=3600)
            self.assertEqual(int(res["scored"]), 1)

            latest = db.get_latest_forecast("TEST")
            self.assertIsNotNone(latest)
            self.assertAlmostEqual(float(latest["realized_return"]), 0.1, places=9)  # (110-100)/100
            self.assertEqual(int(latest["is_correct_3class"]), 1)

            metrics = db.get_live_oos_metrics(lookback_hours=100000)
            self.assertEqual(int(metrics["total_scored"]), 1)
            self.assertAlmostEqual(float(metrics["accuracy_3class_pct"]), 100.0, places=9)


if __name__ == "__main__":
    unittest.main()
