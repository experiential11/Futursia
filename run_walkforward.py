"""Run walk-forward validation for the forecasting engine."""

import argparse
from pathlib import Path

import yaml

from core.client_factory import get_market_client
from core.forecasting import ForecasterEngine
from core.news_client import NewsClient


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward validation.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--out", default="", help="Optional JSON report output path")
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = get_market_client(config)
    news = NewsClient(config)
    engine = ForecasterEngine(config, client, news)
    report = engine.run_walk_forward_validation(save_json_path=args.out or None)

    pooled = report.get("pooled", {}) if isinstance(report, dict) else {}
    print("status:", report.get("status", "ok"))
    if pooled:
        print("accuracy_3class_pct:", pooled.get("accuracy_3class_pct"))
        print("accuracy_binary_excl_flat_pct:", pooled.get("accuracy_binary_excl_flat_pct"))
        print("mae_return:", pooled.get("mae_return"))
        print("folds:", len(report.get("folds", [])))
    if args.out:
        print("saved_report:", args.out)


if __name__ == "__main__":
    main()
