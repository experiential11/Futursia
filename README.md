# Futursia Web (Dark Theme)

This project is now a website-first app (Node.js + Express + browser frontend).

## Stack
- Backend: Node.js (`server/index.js`)
- Frontend: static website (`public/index.html`, `public/site.js`, `public/site.css`)
- Market data: Yahoo Finance chart endpoint
- Theme: Dark (default, full UI)

## Pull + Run Instructions (Website)

1. Pull latest code:

```powershell
git pull origin main
```

2. Install dependencies:

```powershell
npm install
```

3. Create your local env file (first run only):

```powershell
Copy-Item .env.example .env
```

4. Start the website server:

```powershell
npm run dev
```

5. Open:

- `http://localhost:5000`

## Environment Variables

- `FUTURSIA_WEB_HOST` (default `127.0.0.1`)
- `FUTURSIA_WEB_PORT` (default `5000`)
- `NEWS_API_KEY` (optional, for headlines)

## Notes

- Config file: `configs/config.yaml`
- Provider is configured as `yfinance`.
- A broad North American watchlist (US + Canada) is configured and displayed in the UI.
