# Futursia Web Application

A modern, responsive web version of the Futursia stock forecasting application.

## Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+
- Dependencies from `requirements.txt`

### Installation

1. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Open two terminal windows:

**Terminal 1 - Start the backend server:**
```bash
npm run server
```

The server will start on http://localhost:3001 and initialize the Python bridge.

**Terminal 2 - Start the frontend dev server:**
```bash
npm run dev
```

The frontend will be available at http://localhost:5173

### Accessing the App

Open your browser and navigate to: **http://localhost:5173**

## Features

### Market Dashboard
- Real-time top 10 market movers
- 1-second auto-refresh
- Live price updates with percentage changes
- High/low price tracking

### Ticker + Forecast
- Select any stock symbol from the watchlist
- Real-time quote data
- 40-minute price forecast with direction and confidence
- Interactive price chart with 120-minute history
- Detailed price and forecast statistics

### News & Headlines
- Symbol-specific financial news
- Latest headlines with sources
- Timestamps in EST timezone
- Summary and detailed information

### Diagnostics
- System configuration overview
- Market provider information
- Forecast model settings
- Feature list and capabilities

## Architecture

### Frontend
- **Framework:** React 18
- **Build Tool:** Vite
- **Charts:** Recharts
- **HTTP Client:** Axios
- **Styling:** CSS3 with CSS Variables

### Backend
- **Server:** Node.js + Express
- **Port:** 3001
- **Python Bridge:** Subprocess communication via JSON over stdio

### Python Core
- Utilizes all existing forecasting logic
- Market data clients (Yahoo Finance, Finnhub, FMP, Databento)
- Machine learning models (Ridge, XGBoost)
- News sentiment analysis
- Feature engineering

## Configuration

Edit `configs/config.yaml` to:
- Change market data provider
- Adjust forecast parameters
- Configure API keys (via environment variables)
- Set refresh intervals

## Production Build

```bash
npm run build
```

This creates an optimized build in the `dist/` directory ready for deployment.

## Environment Variables

Create a `.env` file with:
```
FINNHUB_API_KEY=your_key_here
FMP_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
DATABENTO_API_KEY=your_key_here
```

## Troubleshooting

**"Cannot find module" errors:**
- Run `npm install` to ensure all dependencies are installed

**"Connection refused" on frontend:**
- Ensure backend server is running on port 3001
- Check that no other process is using ports 3001 or 5173

**Python bridge not responding:**
- Verify Python installation and path
- Check that all Python dependencies are installed
- View backend console for error messages

**Chart not rendering:**
- Clear browser cache
- Check browser console for JavaScript errors
- Ensure Recharts is properly installed

## Development

### Add New API Endpoint

1. Add handler in `server/python_bridge.py`:
```python
def new_feature_handler(params):
    # Your implementation
    return {"result": data}

handlers["new_feature"] = lambda params: new_feature_handler(params)
```

2. Add route in `server/index.js`:
```javascript
app.get('/api/new-feature/:param', async (req, res) => {
  try {
    const result = await callPython('new_feature', { ... })
    res.json(result)
  } catch (err) {
    res.status(500).json({ error: err.message })
  }
})
```

3. Call from React component using axios

### Styling Guidelines

- Use CSS variables defined in `src/styles/index.css`
- Follow mobile-first responsive design
- Maintain consistent spacing (8px grid)
- Use provided color palette
- Test on mobile, tablet, and desktop

## Performance Tips

- Charts limit data to last 120 bars
- API calls cached with 1-second refresh
- Lazy loading for news headlines
- Optimized re-renders with React hooks

## License

See LICENSE file in project root
