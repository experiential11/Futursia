import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import dotenv from "dotenv";
import express from "express";
import yaml from "js-yaml";

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const CONFIG_PATH = path.join(ROOT, "configs", "config.yaml");

const US_EASTERN_TZ = "America/New_York";

const DEFAULT_SYMBOLS = [
  "AAPL",
  "MSFT",
  "NVDA",
  "AMZN",
  "GOOGL",
  "META",
  "TSLA",
  "AVGO",
  "AMD",
  "INTC",
  "QCOM",
  "MU",
  "NFLX",
  "ORCL",
  "CRM",
  "ADBE",
  "CSCO",
  "IBM",
  "TXN",
  "NOW",
  "PANW",
  "PLTR",
  "SNOW",
  "JPM",
  "BAC",
  "WFC",
  "C",
  "GS",
  "MS",
  "AXP",
  "BLK",
  "SCHW",
  "UNH",
  "LLY",
  "JNJ",
  "MRK",
  "ABBV",
  "PFE",
  "TMO",
  "DHR",
  "ISRG",
  "MDT",
  "XOM",
  "CVX",
  "COP",
  "SLB",
  "CAT",
  "DE",
  "GE",
  "HON",
  "RTX",
  "LMT",
  "BA",
  "UPS",
  "FDX",
  "WMT",
  "COST",
  "HD",
  "LOW",
  "TGT",
  "MCD",
  "SBUX",
  "NKE",
  "DIS",
  "KO",
  "PEP",
  "PG",
  "VZ",
  "T",
  "TMUS",
  "CMCSA",
  "RY.TO",
  "TD.TO",
  "BNS.TO",
  "BMO.TO",
  "CM.TO",
  "NA.TO",
  "ENB.TO",
  "SU.TO",
  "CNQ.TO",
  "IMO.TO",
  "CNR.TO",
  "CP.TO",
  "BCE.TO",
  "T.TO",
  "ATD.TO",
  "SHOP.TO",
  "TRI.TO",
  "BN.TO",
  "BAM.TO",
];

const DEFAULT_SYMBOL_NAMES = {
  AAPL: "Apple Inc.",
  MSFT: "Microsoft Corporation",
  NVDA: "NVIDIA Corporation",
  AMZN: "Amazon.com, Inc.",
  GOOGL: "Alphabet Inc. Class A",
  META: "Meta Platforms, Inc.",
  TSLA: "Tesla, Inc.",
  AVGO: "Broadcom Inc.",
  AMD: "Advanced Micro Devices, Inc.",
  INTC: "Intel Corporation",
  QCOM: "QUALCOMM Incorporated",
  MU: "Micron Technology, Inc.",
  NFLX: "Netflix, Inc.",
  ORCL: "Oracle Corporation",
  CRM: "Salesforce, Inc.",
  ADBE: "Adobe Inc.",
  CSCO: "Cisco Systems, Inc.",
  IBM: "International Business Machines Corporation",
  TXN: "Texas Instruments Incorporated",
  NOW: "ServiceNow, Inc.",
  PANW: "Palo Alto Networks, Inc.",
  PLTR: "Palantir Technologies Inc.",
  SNOW: "Snowflake Inc.",
  JPM: "JPMorgan Chase & Co.",
  BAC: "Bank of America Corporation",
  WFC: "Wells Fargo & Company",
  C: "Citigroup Inc.",
  GS: "Goldman Sachs Group, Inc.",
  MS: "Morgan Stanley",
  AXP: "American Express Company",
  BLK: "BlackRock, Inc.",
  SCHW: "Charles Schwab Corporation",
  UNH: "UnitedHealth Group Incorporated",
  LLY: "Eli Lilly and Company",
  JNJ: "Johnson & Johnson",
  MRK: "Merck & Co., Inc.",
  ABBV: "AbbVie Inc.",
  PFE: "Pfizer Inc.",
  TMO: "Thermo Fisher Scientific Inc.",
  DHR: "Danaher Corporation",
  ISRG: "Intuitive Surgical, Inc.",
  MDT: "Medtronic plc",
  XOM: "Exxon Mobil Corporation",
  CVX: "Chevron Corporation",
  COP: "ConocoPhillips",
  SLB: "Schlumberger N.V.",
  CAT: "Caterpillar Inc.",
  DE: "Deere & Company",
  GE: "GE Aerospace",
  HON: "Honeywell International Inc.",
  RTX: "RTX Corporation",
  LMT: "Lockheed Martin Corporation",
  BA: "Boeing Company",
  UPS: "United Parcel Service, Inc.",
  FDX: "FedEx Corporation",
  WMT: "Walmart Inc.",
  COST: "Costco Wholesale Corporation",
  HD: "Home Depot, Inc.",
  LOW: "Lowe's Companies, Inc.",
  TGT: "Target Corporation",
  MCD: "McDonald's Corporation",
  SBUX: "Starbucks Corporation",
  NKE: "NIKE, Inc.",
  DIS: "Walt Disney Company",
  KO: "Coca-Cola Company",
  PEP: "PepsiCo, Inc.",
  PG: "Procter & Gamble Company",
  VZ: "Verizon Communications Inc.",
  T: "AT&T Inc.",
  TMUS: "T-Mobile US, Inc.",
  CMCSA: "Comcast Corporation",
  "RY.TO": "Royal Bank of Canada",
  "TD.TO": "Toronto-Dominion Bank",
  "BNS.TO": "Bank of Nova Scotia",
  "BMO.TO": "Bank of Montreal",
  "CM.TO": "Canadian Imperial Bank of Commerce",
  "NA.TO": "National Bank of Canada",
  "ENB.TO": "Enbridge Inc.",
  "SU.TO": "Suncor Energy Inc.",
  "CNQ.TO": "Canadian Natural Resources",
  "IMO.TO": "Imperial Oil Limited",
  "CNR.TO": "Canadian National Railway",
  "CP.TO": "Canadian Pacific Kansas City",
  "BCE.TO": "BCE Inc.",
  "T.TO": "TELUS Corporation",
  "ATD.TO": "Alimentation Couche-Tard",
  "SHOP.TO": "Shopify Inc.",
  "TRI.TO": "Thomson Reuters Corporation",
  "BN.TO": "Brookfield Corporation",
  "BAM.TO": "Brookfield Asset Management",
};

function utcNow() {
  return new Date();
}

function toIso(value) {
  return new Date(value).toISOString();
}

function toEstString(value) {
  return new Date(value).toLocaleString("en-US", {
    timeZone: US_EASTERN_TZ,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZoneName: "short",
  }).replace(",", "");
}

function loadConfig() {
  if (!fs.existsSync(CONFIG_PATH)) {
    return {
      app: { host: "127.0.0.1", port: 5000 },
      market: { provider: "yfinance", watch_symbols: DEFAULT_SYMBOLS, symbol_names: DEFAULT_SYMBOL_NAMES },
      refresh: { quote_poll_interval: 3, forecast_recompute_interval: 15, market_status_check_interval: 300 },
      forecast: { horizon_minutes: 40, min_bars_for_forecast: 100, features_lookback_minutes: 240, primary_model: "ridge" },
      news: { enabled: true, provider: "newsapi", headline_limit: 10 },
      ui: { theme: "dark" },
    };
  }

  const text = fs.readFileSync(CONFIG_PATH, "utf8");
  const parsed = yaml.load(text);
  return parsed && typeof parsed === "object" ? parsed : {};
}

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function dedupeSymbols(symbols) {
  const out = [];
  const seen = new Set();
  for (const raw of symbols || []) {
    const symbol = String(raw || "").trim().toUpperCase();
    if (!symbol || seen.has(symbol)) continue;
    seen.add(symbol);
    out.push(symbol);
  }
  return out;
}

const config = loadConfig();
const appCfg = config.app || {};
const marketCfg = config.market || {};
const refreshCfg = config.refresh || {};
const forecastCfg = config.forecast || {};
const uiCfg = config.ui || {};
const newsCfg = config.news || {};

const symbols = dedupeSymbols(marketCfg.watch_symbols || DEFAULT_SYMBOLS);
const symbolNames = {
  ...DEFAULT_SYMBOL_NAMES,
  ...(marketCfg.symbol_names || {}),
};

const provider = "yfinance";
const quotePollInterval = clampNumber(refreshCfg.quote_poll_interval, 1, 60, 3);
const forecastRecomputeInterval = clampNumber(refreshCfg.forecast_recompute_interval, 1, 300, 15);
const barsRefreshInterval = clampNumber(refreshCfg.bars_refresh_interval, 1, 300, Math.max(5, quotePollInterval));

const barsCache = new Map();

async function fetchYahooChart(symbol, range = "5d", interval = "1m") {
  const cleanSymbol = encodeURIComponent(String(symbol || "").trim().toUpperCase());
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${cleanSymbol}?range=${encodeURIComponent(range)}&interval=${encodeURIComponent(interval)}`;

  const response = await fetch(url, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
      Accept: "application/json",
    },
  });

  if (!response.ok) {
    throw new Error(`Yahoo chart request failed (${response.status})`);
  }

  const payload = await response.json();
  const result = payload?.chart?.result?.[0];
  return result || null;
}

function chartToBars(chart, limit = 240) {
  const timestamps = Array.isArray(chart?.timestamp) ? chart.timestamp : [];
  const quote = chart?.indicators?.quote?.[0] || {};
  const open = Array.isArray(quote.open) ? quote.open : [];
  const high = Array.isArray(quote.high) ? quote.high : [];
  const low = Array.isArray(quote.low) ? quote.low : [];
  const close = Array.isArray(quote.close) ? quote.close : [];
  const volume = Array.isArray(quote.volume) ? quote.volume : [];

  const rows = [];
  for (let i = 0; i < timestamps.length; i += 1) {
    const ts = timestamps[i];
    const o = Number(open[i]);
    const h = Number(high[i]);
    const l = Number(low[i]);
    const c = Number(close[i]);
    const v = Number(volume[i] || 0);

    if (!Number.isFinite(ts) || !Number.isFinite(o) || !Number.isFinite(h) || !Number.isFinite(l) || !Number.isFinite(c)) {
      continue;
    }
    if (o <= 0 || h <= 0 || l <= 0 || c <= 0) continue;
    if (h < l || h < o || h < c || l > o || l > c) continue;

    rows.push({
      timestamp: new Date(ts * 1000).toISOString(),
      open: o,
      high: h,
      low: l,
      close: c,
      volume: Number.isFinite(v) ? v : 0,
      vwap: c,
    });
  }

  if (rows.length > limit) {
    return rows.slice(rows.length - limit);
  }
  return rows;
}

function getBarsFetchPlan(symbol, limit) {
  if (limit <= 390) {
    return [
      { range: "5d", interval: "1m" },
      { range: "1mo", interval: "5m" },
      { range: "3mo", interval: "15m" },
    ];
  }

  if (limit <= 2000) {
    return [
      { range: "1mo", interval: "5m" },
      { range: "3mo", interval: "15m" },
      { range: "6mo", interval: "1d" },
    ];
  }

  return [
    { range: "3mo", interval: "15m" },
    { range: "6mo", interval: "1d" },
    { range: "1y", interval: "1d" },
  ];
}

async function getBars(symbol, limit = 240) {
  const cleanSymbol = String(symbol || "").trim().toUpperCase();
  const now = Date.now();
  const targetLimit = Math.max(10, Number(limit) || 240);
  const cacheItem = barsCache.get(cleanSymbol);

  if (cacheItem) {
    const ageSec = (now - cacheItem.fetchedAt) / 1000;
    const cacheCanServeLimit = Number(cacheItem.maxLimit || 0) >= targetLimit;
    if (ageSec < barsRefreshInterval && cacheCanServeLimit) {
      const cachedBars = cacheItem.bars || [];
      return cachedBars.length > targetLimit ? cachedBars.slice(cachedBars.length - targetLimit) : cachedBars;
    }
  }

  const fetchPlan = getBarsFetchPlan(cleanSymbol, targetLimit);
  let bestBars = [];

  for (const step of fetchPlan) {
    try {
      const chart = await fetchYahooChart(cleanSymbol, step.range, step.interval);
      const bars = chartToBars(chart, Math.max(targetLimit, 240));
      if (!bars.length) continue;
      if (bars.length > bestBars.length) bestBars = bars;

      if (bars.length >= targetLimit || bars.length >= 240) {
        barsCache.set(cleanSymbol, { fetchedAt: now, bars, maxLimit: targetLimit });
        return bars.length > targetLimit ? bars.slice(bars.length - targetLimit) : bars;
      }
    } catch {
      // Try the next resolution step.
    }
  }

  if (bestBars.length) {
    barsCache.set(cleanSymbol, { fetchedAt: now, bars: bestBars, maxLimit: targetLimit });
    return bestBars.length > targetLimit ? bestBars.slice(bestBars.length - targetLimit) : bestBars;
  }

  if (cacheItem?.bars?.length) {
    const cachedBars = cacheItem.bars;
    return cachedBars.length > targetLimit ? cachedBars.slice(cachedBars.length - targetLimit) : cachedBars;
  }
  return [];
}

async function getQuoteRaw(symbol, fallbackBars = []) {
  const cleanSymbol = String(symbol || "").trim().toUpperCase();
  const quotePlan = [
    { range: "1d", interval: "1m", limit: 390 },
    { range: "5d", interval: "5m", limit: 390 },
    { range: "1mo", interval: "1d", limit: 120 },
  ];

  for (const step of quotePlan) {
    try {
      const chart = await fetchYahooChart(cleanSymbol, step.range, step.interval);
      const bars = chartToBars(chart, step.limit);
      if (!bars.length) continue;

      const last = bars[bars.length - 1];
      const prev = bars.length > 1 ? bars[bars.length - 2] : last;
      const dayOpen = bars[0]?.open ?? prev.close;
      const dayHigh = Math.max(...bars.map((x) => Number(x.high)).filter(Number.isFinite));
      const dayLow = Math.min(...bars.map((x) => Number(x.low)).filter(Number.isFinite));

      return {
        symbol: cleanSymbol,
        last_price: Number(last.close),
        c: Number(last.close),
        pc: Number(prev.close),
        o: Number(dayOpen),
        h: Number(dayHigh),
        l: Number(dayLow),
        timestamp: toIso(utcNow()),
        t: Math.floor(Date.now() / 1000),
      };
    } catch {
      // Try the next quote plan step.
    }
  }

  const bars = fallbackBars || [];
  if (!bars.length) {
    return null;
  }

  const last = bars[bars.length - 1];
  const prev = bars.length > 1 ? bars[bars.length - 2] : last;

  return {
    symbol: cleanSymbol,
    last_price: Number(last.close),
    c: Number(last.close),
    pc: Number(prev.close),
    o: Number(bars[0]?.open ?? prev.close),
    h: Number(last.high),
    l: Number(last.low),
    timestamp: toIso(utcNow()),
    t: Math.floor(Date.now() / 1000),
  };
}

function normalizeQuote(symbol, quote, bars) {
  const q = quote || {};
  const cleanSymbol = String(symbol || "").trim().toUpperCase();

  let price = Number(q.last_price ?? q.c ?? 0);
  let prevClose = Number(q.pc ?? price);
  let high = Number(q.h ?? price);
  let low = Number(q.l ?? price);
  let open = Number(q.o ?? prevClose);

  if ((!Number.isFinite(price) || price <= 0) && bars.length) {
    const last = bars[bars.length - 1];
    const prev = bars.length > 1 ? bars[bars.length - 2] : last;
    price = Number(last.close);
    prevClose = Number(prev.close);
    high = Number(last.high);
    low = Number(last.low);
    open = Number(bars[0]?.open ?? prev.close);
  }

  if (!Number.isFinite(price) || price <= 0) price = 0;
  if (!Number.isFinite(prevClose) || prevClose <= 0) prevClose = price;
  if (!Number.isFinite(high) || high <= 0) high = price;
  if (!Number.isFinite(low) || low <= 0) low = price;
  if (!Number.isFinite(open) || open <= 0) open = prevClose;

  const change = prevClose ? price - prevClose : 0;
  const changePct = prevClose ? (change / prevClose) * 100 : 0;
  const ts = q.timestamp ? new Date(q.timestamp) : new Date();

  return {
    symbol: cleanSymbol,
    price,
    prev_close: prevClose,
    open,
    high,
    low,
    change,
    change_pct: changePct,
    timestamp_utc: toIso(ts),
    timestamp_est: toEstString(ts),
  };
}

function buildQuickForecast(quote) {
  const price = Number(quote.price || 0);
  const prevClose = Number(quote.prev_close || price || 0);
  const high = Number(quote.high || price || 0);
  const low = Number(quote.low || price || 0);

  if (!Number.isFinite(price) || price <= 0) {
    return {
      direction: "FLAT",
      confidence: 0,
      predicted_return_pct: 0,
      model_status: "Unavailable",
      source: "none",
    };
  }

  const dailyChangePct = prevClose > 0 ? ((price - prevClose) / prevClose) * 100 : 0;
  const intradayRangePct = price > 0 ? ((high - low) / price) * 100 : 0;

  let forecastReturnPct = dailyChangePct * 0.15;
  if (dailyChangePct > 0.5) {
    forecastReturnPct -= intradayRangePct * 0.05;
  } else if (dailyChangePct < -0.5) {
    forecastReturnPct += intradayRangePct * 0.05;
  }

  const confidence = Math.max(45, Math.min(95, 50 + Math.abs(dailyChangePct) * 8));

  let direction = "FLAT";
  if (forecastReturnPct > 0.01) direction = "UP";
  if (forecastReturnPct < -0.01) direction = "DOWN";

  return {
    direction,
    confidence,
    predicted_return_pct: forecastReturnPct,
    model_status: "Quick forecast fallback",
    source: "quick",
  };
}

function generateForecastPath(symbol, bars, forecast, horizonMinutes = 40) {
  if (!bars.length) return [];

  const last = bars[bars.length - 1];
  const startPrice = Number(last.close);
  if (!Number.isFinite(startPrice) || startPrice <= 0) return [];

  const startTs = new Date(last.timestamp).getTime();
  if (!Number.isFinite(startTs)) return [];

  const retPct = Number(forecast.predicted_return_pct || 0);
  const targetPrice = Math.max(0.01, startPrice * (1 + retPct / 100));
  const steps = Math.max(1, Math.floor(horizonMinutes));

  const seed = crypto.createHash("sha256").update(`${symbol}|${startTs}|${startPrice}|${retPct}`).digest();
  let seedInt = seed.readUInt32BE(0);
  function rand() {
    seedInt = (1664525 * seedInt + 1013904223) % 4294967296;
    return seedInt / 4294967296;
  }

  const path = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const base = startPrice + (targetPrice - startPrice) * t;
    const wiggle = (rand() - 0.5) * startPrice * 0.0025;
    const price = i === 0 ? startPrice : i === steps ? targetPrice : Math.max(0.01, base + wiggle);
    path.push({
      timestamp: new Date(startTs + i * 60000).toISOString(),
      price,
    });
  }

  return path;
}

function generateForecastPathFromQuote(symbol, quote, forecast, horizonMinutes = 40) {
  const startPrice = Number(quote?.price || 0);
  if (!Number.isFinite(startPrice) || startPrice <= 0) return [];

  const quoteTs = quote?.timestamp_utc ? new Date(quote.timestamp_utc).getTime() : Date.now();
  if (!Number.isFinite(quoteTs)) return [];

  const retPct = Number(forecast?.predicted_return_pct || 0);
  const targetPrice = Math.max(0.01, startPrice * (1 + retPct / 100));
  const steps = Math.max(1, Math.floor(horizonMinutes));

  const seed = crypto.createHash("sha256").update(`${symbol}|${quoteTs}|${startPrice}|${retPct}`).digest();
  let seedInt = seed.readUInt32BE(0);
  function rand() {
    seedInt = (1664525 * seedInt + 1013904223) % 4294967296;
    return seedInt / 4294967296;
  }

  const path = [];
  for (let i = 0; i <= steps; i += 1) {
    const t = i / steps;
    const base = startPrice + (targetPrice - startPrice) * t;
    const wiggle = (rand() - 0.5) * startPrice * 0.0025;
    const price = i === 0 ? startPrice : i === steps ? targetPrice : Math.max(0.01, base + wiggle);
    path.push({
      timestamp: new Date(quoteTs + i * 60000).toISOString(),
      price,
    });
  }
  return path;
}

function getMarketStatus(now = new Date()) {
  const weekday = now.getUTCDay();
  const isWeekday = weekday >= 1 && weekday <= 5;
  const open = new Date(now);
  open.setUTCHours(14, 30, 0, 0);
  const close = new Date(now);
  close.setUTCHours(21, 0, 0, 0);

  const isOpen = isWeekday && now >= open && now <= close;
  return {
    label: isOpen ? "OPEN" : "CLOSED",
    is_open: isOpen,
    timezone: "US/Eastern",
    next_open_local: null,
    minutes_to_close: null,
    last_bar_age_minutes: null,
    status_source: "schedule",
    checked_at_utc: now.toISOString(),
    stream_quality: isOpen ? "REALTIME" : "MARKET_CLOSED",
    is_live_streaming: isOpen,
  };
}

function barsToJson(bars, limit = 120) {
  const cleaned = (bars || []).map((row) => ({
    timestamp: row.timestamp,
    open: Number(row.open),
    high: Number(row.high),
    low: Number(row.low),
    close: Number(row.close),
    volume: Number(row.volume || 0),
  }));
  const valid = cleaned.filter((x) => {
    const ts = new Date(x.timestamp).getTime();
    return (
      Number.isFinite(ts) &&
      Number.isFinite(x.open) &&
      Number.isFinite(x.high) &&
      Number.isFinite(x.low) &&
      Number.isFinite(x.close) &&
      x.open > 0 &&
      x.high > 0 &&
      x.low > 0 &&
      x.close > 0
    );
  });
  return valid.slice(-limit);
}

async function getTickerSnapshot(symbol) {
  const cleanSymbol = String(symbol || "").trim().toUpperCase();
  const requestedBars = 360;
  const chartBars = 240;
  const now = utcNow();
  const startedAt = Date.now();

  const barsFetchStart = Date.now();
  const bars = await getBars(cleanSymbol, requestedBars);
  const barsFetchMs = Date.now() - barsFetchStart;

  const quoteFetchStart = Date.now();
  const rawQuote = await getQuoteRaw(cleanSymbol, bars);
  const quoteFetchMs = Date.now() - quoteFetchStart;
  const quote = normalizeQuote(cleanSymbol, rawQuote, bars);

  const marketStatus = getMarketStatus(now);
  const forecastFetchStart = Date.now();
  const forecast = buildQuickForecast(quote);
  const forecastFetchMs = Date.now() - forecastFetchStart;

  const barsForPlot = bars.slice(-chartBars);
  const barsJson = barsToJson(barsForPlot, chartBars);

  const horizonMin = clampNumber(forecastCfg.horizon_minutes, 1, 120, 40);
  let forecastPath = generateForecastPath(cleanSymbol, barsForPlot, forecast, horizonMin);
  if (!forecastPath.length && quote?.price) {
    forecastPath = generateForecastPathFromQuote(cleanSymbol, quote, forecast, horizonMin);
  }

  let lastMarketPrintEst = null;
  let lastMarketPrintAgeSeconds = null;
  let lastMarketPrintAgeMinutes = null;
  if (barsJson.length) {
    const lastTs = new Date(barsJson[barsJson.length - 1].timestamp);
    lastMarketPrintEst = toEstString(lastTs);
    const ageSec = Math.max(0, Math.floor((now.getTime() - lastTs.getTime()) / 1000));
    lastMarketPrintAgeSeconds = ageSec;
    lastMarketPrintAgeMinutes = Math.floor(ageSec / 60);
    marketStatus.last_bar_age_seconds = ageSec;
    marketStatus.last_bar_age_minutes = lastMarketPrintAgeMinutes;
  }

  const quoteTs = new Date(quote.timestamp_utc);
  const quoteAgeSeconds = Math.max(0, Math.floor((now.getTime() - quoteTs.getTime()) / 1000));
  const totalMs = Date.now() - startedAt;

  return {
    symbol: cleanSymbol,
    symbol_name: symbolNames[cleanSymbol] || null,
    updated_at_utc: now.toISOString(),
    updated_at_est: toEstString(now),
    quote,
    market_status: marketStatus,
    bars: barsJson,
    forecast,
    forecast_path: forecastPath,
    last_market_print_est: lastMarketPrintEst,
    last_market_print_age_minutes: lastMarketPrintAgeMinutes,
    last_market_print_age_seconds: lastMarketPrintAgeSeconds,
    quote_timestamp_est: toEstString(quoteTs),
    quote_age_seconds: quoteAgeSeconds,
    backend_timings_ms: {
      bars_fetch: barsFetchMs,
      quote_fetch: quoteFetchMs,
      forecast_fetch: forecastFetchMs,
      total: totalMs,
    },
    live_oos_metrics: {
      total_scored: 0,
      accuracy_3class_pct: null,
      accuracy_binary_excl_flat_pct: null,
      mae_return: null,
    },
  };
}

async function getDashboard(limit = 10) {
  const now = utcNow();
  const picked = symbols.slice(0, Math.max(1, Math.min(30, Number(limit) || 10)));

  const movers = await Promise.all(
    picked.map(async (symbol) => {
      const bars = await getBars(symbol, 2);
      const rawQuote = await getQuoteRaw(symbol, bars);
      const quote = normalizeQuote(symbol, rawQuote, bars);

      return {
        symbol,
        name: symbolNames[symbol] || null,
        price: quote.price,
        change: quote.change,
        percent_change: quote.change_pct,
        high: quote.high,
        low: quote.low,
      };
    })
  );

  movers.sort((a, b) => Math.abs(b.percent_change || 0) - Math.abs(a.percent_change || 0));

  return {
    updated_at_utc: now.toISOString(),
    updated_at_est: toEstString(now),
    movers: movers.slice(0, limit),
  };
}

async function getNews(symbol, limit = 10) {
  const now = utcNow();
  const newsApiKey = process.env.NEWS_API_KEY || "";
  const cleanSymbol = String(symbol || "").trim().toUpperCase();

  if (!newsCfg.enabled || !newsApiKey) {
    return {
      symbol: cleanSymbol,
      updated_at_utc: now.toISOString(),
      updated_at_est: toEstString(now),
      headlines: [],
    };
  }

  const url = new URL("https://newsapi.org/v2/everything");
  url.searchParams.set("q", cleanSymbol);
  url.searchParams.set("language", "en");
  url.searchParams.set("sortBy", "publishedAt");
  url.searchParams.set("pageSize", String(Math.max(1, Math.min(30, Number(limit) || 10))));
  url.searchParams.set("apiKey", newsApiKey);

  try {
    const response = await fetch(url);
    const payload = await response.json();
    const articles = Array.isArray(payload?.articles) ? payload.articles : [];

    const headlines = articles.map((article) => {
      const ts = article?.publishedAt ? new Date(article.publishedAt) : now;
      return {
        title: String(article?.title || "").trim(),
        summary: String(article?.description || article?.content || "").trim(),
        source: String(article?.source?.name || "Unknown").trim(),
        url: String(article?.url || "").trim(),
        timestamp_utc: ts.toISOString(),
        timestamp_est: toEstString(ts),
      };
    });

    return {
      symbol: cleanSymbol,
      updated_at_utc: now.toISOString(),
      updated_at_est: toEstString(now),
      headlines,
    };
  } catch {
    return {
      symbol: cleanSymbol,
      updated_at_utc: now.toISOString(),
      updated_at_est: toEstString(now),
      headlines: [],
    };
  }
}

function getSettings() {
  return {
    provider,
    symbols,
    symbol_names: symbolNames,
    forecast: {
      horizon_minutes: clampNumber(forecastCfg.horizon_minutes, 1, 240, 40),
      features_lookback_minutes: clampNumber(forecastCfg.features_lookback_minutes, 1, 5000, 240),
      primary_model: String(forecastCfg.primary_model || "ridge"),
      min_bars_for_forecast: clampNumber(forecastCfg.min_bars_for_forecast, 10, 5000, 100),
    },
    refresh: {
      quote_poll_interval: quotePollInterval,
      forecast_recompute_interval: forecastRecomputeInterval,
      bars_refresh_interval: barsRefreshInterval,
      market_status_check_interval: clampNumber(refreshCfg.market_status_check_interval, 5, 3600, 300),
    },
    news: {
      enabled: Boolean(newsCfg.enabled),
      provider: String(newsCfg.provider || "newsapi"),
      headline_limit: clampNumber(newsCfg.headline_limit, 1, 30, 10),
    },
    ui: {
      theme: String(uiCfg.theme || "dark"),
    },
  };
}

const app = express();
app.use(express.json());
app.use(express.static(path.join(ROOT, "public"), { index: false }));

app.get("/", (_req, res) => {
  res.sendFile(path.join(ROOT, "public", "index.html"));
});

app.get("/api/health", (_req, res) => {
  const now = utcNow();
  res.json({
    status: "ok",
    time_utc: now.toISOString(),
    time_est: toEstString(now),
  });
});

app.get("/api/symbols", (_req, res) => {
  res.json({ symbols, symbol_names: symbolNames });
});

app.get("/api/settings", (_req, res) => {
  res.json(getSettings());
});

app.get("/api/dashboard", async (req, res) => {
  try {
    const limit = clampNumber(req.query.limit, 1, 50, 10);
    res.json(await getDashboard(limit));
  } catch (error) {
    res.status(500).json({ error: String(error?.message || error) });
  }
});

app.get("/api/ticker/:symbol", async (req, res) => {
  try {
    res.json(await getTickerSnapshot(req.params.symbol));
  } catch (error) {
    res.status(500).json({ error: String(error?.message || error) });
  }
});

app.get("/api/news/:symbol", async (req, res) => {
  try {
    const limit = clampNumber(req.query.limit, 1, 30, 10);
    res.json(await getNews(req.params.symbol, limit));
  } catch (error) {
    res.status(500).json({ error: String(error?.message || error) });
  }
});

const host = process.env.FUTURSIA_WEB_HOST || String(appCfg.host || "127.0.0.1");
const port = clampNumber(process.env.FUTURSIA_WEB_PORT || appCfg.port, 1, 65535, 5000);

if (!process.env.VERCEL) {
  app.listen(port, host, () => {
    // eslint-disable-next-line no-console
    console.log(`Futursia website running at http://${host}:${port}`);
  });
}

export default app;
