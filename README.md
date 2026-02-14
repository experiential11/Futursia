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
cd Futursia
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

## Provider Selection

Edit `configs/config.yaml`:
- `market_api_provider: "yfinance"` (no key required)
- or `databento`, `fmp`, `finnhub` (set matching env key)

## Notes

- No API keys are committed in this repository.
- Runtime/generated data is ignored by git via `.gitignore`.
