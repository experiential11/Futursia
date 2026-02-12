# Futursia Forecasting Desktop App

Desktop stock forecasting app with:
- Live market chart updates
- 40-minute forecast path (minute-by-minute points)
- Interactive zoom and drag pan
- News and diagnostics tabs

## Quick Start (Windows / PowerShell)

1. Clone and enter the repo:

```powershell
git clone <your-repo-url>
cd massive_40min_forecaster_exe
```

2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. (Optional) Set API keys as environment variables:

```powershell
$env:DATABENTO_API_KEY="your_key"
$env:FMP_API_KEY="your_key"
$env:FINNHUB_API_KEY="your_key"
$env:NEWS_API_KEY="your_key"
```

5. Run the app:

```powershell
python main.py
```

## Main Entry Point

- Primary entry: `main.py`
- App module: `app_desktop.py`

## Upgraded Forecasting Pipeline

The app now includes:
- Pooled cross-symbol training with symbol one-hot encoding
- Market-pattern features (market proxy returns/volatility, relative strength, rolling beta/correlation)
- Volatility-normalized target option for improved cross-symbol stability
- Dynamic volatility-aware FLAT threshold (abstain behavior)
- Walk-forward validation support with leakage-safe time splits
- Forecast persistence and realized scoring in `storage/market.db`

## Database Migration (Existing `market.db`)

No manual SQL migration is required.

When the app starts, `MarketDB` auto-migrates the `forecasts` table by adding missing columns (e.g. `horizon_minutes`, `prediction_return_raw`, `prediction_return_norm`, `target_due_at`, realized scoring fields, metadata/version fields).

## Run Walk-Forward Validation

```powershell
python run_walkforward.py --config configs/config.yaml --out storage/walkforward_report.json
```

## Run Unit Tests

```powershell
python -m unittest discover -s tests -v
```

## Provider Selection

Edit `configs/config.yaml`:
- `market_api_provider: "yfinance"` (no key required)
- or `databento`, `fmp`, `finnhub` (set matching env key)

## Notes

- No API keys are committed in this repository.
- Runtime/generated data is ignored by git via `.gitignore`.
