# Futursia Web - Quick Start Guide

## Run the Web App in 3 Steps

### Step 1: Install Dependencies
```bash
npm install
```

### Step 2: Start the Backend Server (Terminal 1)
```bash
npm run server
```

You'll see:
```
Initializing Python bridge...
ready
Server running on http://localhost:3001
```

### Step 3: Start the Frontend (Terminal 2)
```bash
npm run dev
```

You'll see:
```
  VITE v5.x.x  ready in XXX ms

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

## Open in Browser

Visit: **http://localhost:5173**

## What You'll See

### Market Tab
- Real-time top 10 movers
- Updates every second
- Shows price, change %, and high/low

### Ticker + Forecast Tab
- Select a stock (AAPL, MSFT, GOOGL, etc.)
- View current price and daily change
- See 40-minute forecast with confidence
- Interactive price chart with 2-hour history

### News Tab
- Latest financial headlines
- Symbol-specific stories
- Sources and timestamps

### Diagnostics Tab
- System configuration
- Market provider info
- Feature overview

## Troubleshooting

**Problem: "Cannot find module"**
- Run: `npm install`

**Problem: "Connection refused" error**
- Ensure backend is running in Terminal 1
- Check ports 3001 and 5173 are not in use

**Problem: "No data available"**
- Give it 2-3 seconds to fetch initial data
- Check your internet connection
- Verify API provider is set correctly in `configs/config.yaml`

## Stop the App

Press `Ctrl+C` in both terminal windows

## Next Steps

- See `WEB_SETUP.md` for full documentation
- Edit `configs/config.yaml` to change settings
- Add API keys to `.env` for Finnhub, FMP, etc.
- Deploy production build with: `npm run build`
