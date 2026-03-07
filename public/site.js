const state = {
  symbol: "AAPL",
  settings: null,
  symbolNames: {},
  symbolList: [],
  filteredSymbolList: [],
  refresh: {
    tickerPollMs: 1000,
    tickerTimerId: null,
  },
  chart: {
    liveBars: [],
    forecastBars: [],
    viewMinTs: null,
    viewMaxTs: null,
    manualView: false,
    dragging: false,
    dragStartX: 0,
    dragMinTs: 0,
    dragMaxTs: 0,
    interactionsReady: false,
  },
  requestIds: {
    ticker: 0,
    news: 0,
  },
  pending: {
    dashboard: false,
    ticker: false,
    news: false,
  },
};

function getSymbolName(symbol) {
  if (!symbol) return "";
  return state.symbolNames[String(symbol).toUpperCase()] || "";
}

function getSymbolLabel(symbol) {
  const name = getSymbolName(symbol);
  return name ? `${symbol} - ${name}` : String(symbol || "");
}

function getSymbolDisplayName(symbol) {
  if (!symbol) return "";
  const s = String(symbol).toUpperCase();
  return getSymbolName(s) || s;
}

function getSymbolProminentLabel(symbol) {
  if (!symbol) return "";
  const s = String(symbol).toUpperCase();
  const fullName = getSymbolName(s);
  return fullName ? `${fullName} (${s})` : s;
}

function updateSelectedSymbolHeader(symbol = state.symbol) {
  const el = document.getElementById("tickerSelectedSymbol");
  if (!el) return;
  el.textContent = getSymbolProminentLabel(symbol) || "-";
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  return `$${Number(value).toFixed(2)}`;
}

function formatSigned(value, digits = 2, suffix = "") {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "-";
  }
  const n = Number(value);
  const sign = n >= 0 ? "+" : "";
  return `${sign}${n.toFixed(digits)}${suffix}`;
}

function formatPercent(value, digits = 2) {
  return formatSigned(value, digits, "%");
}

function directionClass(direction) {
  const dir = String(direction || "FLAT").toUpperCase();
  if (dir === "UP") return "txt-up";
  if (dir === "DOWN") return "txt-down";
  return "txt-flat";
}

function computeTickerPollMs(market, lastBarAgeSeconds, quoteAgeSeconds) {
  const barAge = Number(lastBarAgeSeconds);
  const quoteAge = Number(quoteAgeSeconds);
  const streamQuality = String((market && market.stream_quality) || "").toUpperCase();

  if (market && market.is_open === true) {
    if (streamQuality === "REALTIME") return 1000;
    if (
      Number.isFinite(barAge) &&
      Number.isFinite(quoteAge) &&
      barAge <= 120 &&
      quoteAge <= 20
    ) {
      return 1500;
    }
    return 5000;
  }
  if (market && market.is_open === false) return 5000;
  return 5000;
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Request failed: ${response.status} ${message}`);
  }
  return response.json();
}

function updateLastUpdated(text) {
  const el = document.getElementById("tickerUpdated");
  if (el) el.textContent = text;
}

function announceToScreenReader(message) {
  const el = document.getElementById("srStatus");
  if (!el) return;
  el.textContent = "";
  requestAnimationFrame(() => {
    el.textContent = message;
  });
}

function setActiveTab(targetTab) {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");
  if (!targetTab) return;

  tabs.forEach((tab) => {
    const isActive = tab === targetTab;
    tab.classList.toggle("is-active", isActive);
    tab.setAttribute("aria-selected", isActive ? "true" : "false");
    tab.setAttribute("tabindex", isActive ? "0" : "-1");
  });

  panels.forEach((panel) => {
    const isActive = panel.id === targetTab.getAttribute("data-panel");
    panel.classList.toggle("is-active", isActive);
    panel.hidden = !isActive;
  });
}

function activateTabs() {
  const tabs = document.querySelectorAll(".tab");
  if (!tabs.length) return;

  const active = document.querySelector(".tab.is-active") || tabs[0];
  setActiveTab(active);

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      setActiveTab(tab);
    });

    tab.addEventListener("keydown", (event) => {
      const currentIndex = Array.from(tabs).indexOf(tab);
      let nextIndex = currentIndex;

      if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % tabs.length;
      if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") nextIndex = 0;
      if (event.key === "End") nextIndex = tabs.length - 1;
      if (nextIndex === currentIndex) return;

      event.preventDefault();
      const nextTab = tabs[nextIndex];
      setActiveTab(nextTab);
      nextTab.focus();
    });
  });
}

function renderSymbolOptions(symbols) {
  const select = document.getElementById("tickerSelect");
  if (!select) return;

  select.innerHTML = "";
  if (!Array.isArray(symbols) || symbols.length === 0) {
    const empty = document.createElement("option");
    empty.value = "";
    empty.textContent = "No matching stocks";
    empty.disabled = true;
    empty.selected = true;
    select.appendChild(empty);
    return;
  }

  symbols.forEach((symbol) => {
    const option = document.createElement("option");
    option.value = symbol;
    option.textContent = getSymbolProminentLabel(symbol);
    select.appendChild(option);
  });

  if (symbols.includes(state.symbol)) {
    select.value = state.symbol;
  } else {
    select.value = symbols[0];
  }
}

function filterSymbolsByQuery(query) {
  const term = String(query || "").trim().toLowerCase();
  if (!term) return [...state.symbolList];
  return state.symbolList.filter((symbol) => {
    const s = String(symbol || "").toUpperCase();
    const fullName = getSymbolName(s);
    return s.toLowerCase().includes(term) || fullName.toLowerCase().includes(term);
  });
}

function applySymbolSearch(query) {
  state.filteredSymbolList = filterSymbolsByQuery(query);
  renderSymbolOptions(state.filteredSymbolList);
}

function populateSymbols(symbols) {
  state.symbolList = Array.isArray(symbols) ? [...symbols] : [];
  if (state.symbolList.length && !state.symbolList.includes(state.symbol)) {
    state.symbol = state.symbolList[0];
  }

  const search = document.getElementById("tickerSearch");
  applySymbolSearch(search ? search.value : "");
  updateSelectedSymbolHeader(state.symbol);
}

async function selectSymbol(symbol) {
  if (!symbol || symbol === state.symbol) return;
  state.symbol = symbol;
  state.chart.manualView = false;
  state.chart.viewMinTs = null;
  state.chart.viewMaxTs = null;
  updateSelectedSymbolHeader(symbol);

  const select = document.getElementById("tickerSelect");
  if (select) select.value = symbol;

  updateLastUpdated(`Loading ${getSymbolDisplayName(symbol)}...`);
  await Promise.all([
    safeLoad("ticker", loadTicker, { force: true }),
    safeLoad("news", loadNews, { force: true }),
  ]);
  scheduleTickerRefresh();
}

function renderMovers(data) {
  const tbody = document.querySelector("#moversTable tbody");
  tbody.innerHTML = "";
  const movers = data.movers || [];

  if (!movers.length) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td colspan="6">No movers available yet.</td>`;
    tbody.appendChild(tr);
  }

  movers.forEach((row) => {
    const tr = document.createElement("tr");
    const pctClass = row.percent_change >= 0 ? "txt-up" : "txt-down";
    const symbol = row.symbol || "-";
    const displayName = getSymbolDisplayName(symbol) || symbol;
    tr.innerHTML = `
      <td>${displayName}</td>
      <td>${formatMoney(row.price)}</td>
      <td>${formatSigned(row.change, 2)}</td>
      <td class="${pctClass}">${formatPercent(row.percent_change, 2)}</td>
      <td>${formatMoney(row.high)}</td>
      <td>${formatMoney(row.low)}</td>
    `;
    tbody.appendChild(tr);
  });

  const updated = document.getElementById("dashboardUpdated");
  if (updated) updated.textContent = data.updated_at_est || "-";
}

function updateText(id, text, className = null) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  if (className !== null) el.className = className;
}

function renderTicker(data) {
  const quote = data.quote || {};
  const forecast = data.forecast || {};
  const market = data.market_status || {};
  const metrics = data.live_oos_metrics || {};
  const timings = data.backend_timings_ms || {};

  updateText("tickerUpdated", data.updated_at_est || "-");
  updateText("priceValue", formatMoney(quote.price));
  updateText("priceChange", formatPercent(quote.change_pct, 2), quote.change_pct >= 0 ? "txt-up" : "txt-down");

  const direction = String(forecast.direction || "FLAT").toUpperCase();
  updateText("forecastDirection", direction, directionClass(direction));
  updateText("forecastReturn", formatPercent(forecast.predicted_return_pct, 3), directionClass(direction));
  updateText("forecastConfidence", `${Number(forecast.confidence || 0).toFixed(1)}%`);
  updateText("forecastStatus", forecast.model_status || "-");

  const marketLabel = String(market.label || "UNKNOWN").toUpperCase();
  const marketClass = market.is_open === true ? "txt-up" : market.is_open === false ? "txt-down" : "txt-flat";
  const timezoneText = market.timezone ? ` | ${market.timezone}` : "";
  updateText("marketStatus", `${marketLabel}${timezoneText}`, marketClass);

  const marketPrintParts = [];
  if (data.last_market_print_est) {
    const ageSec = Number(data.last_market_print_age_seconds);
    const ageText = Number.isFinite(ageSec) ? `${ageSec}s ago` : "-";
    marketPrintParts.push(`${data.last_market_print_est} (${ageText})`);
  } else {
    marketPrintParts.push("No market print");
  }
  if (market.next_open_local) {
    marketPrintParts.push(`Next open: ${market.next_open_local}`);
  }
  if (data.quote_timestamp_est) {
    marketPrintParts.push(`Quote ts: ${data.quote_timestamp_est}`);
  }
  if (market.is_live_streaming === true) {
    marketPrintParts.push("Live bars");
  } else if (market.is_open === true) {
    marketPrintParts.push("Open (delayed feed)");
  } else if (market.is_open === false) {
    marketPrintParts.push("Closed session");
  }
  const marketPrint = marketPrintParts.join(" | ");
  updateText("lastMarketPrint", marketPrint);

  const streamQuality = String(market.stream_quality || "UNKNOWN").toUpperCase();
  const streamClass =
    streamQuality === "REALTIME"
      ? "txt-up"
      : streamQuality === "DELAYED" || streamQuality === "MARKET_CLOSED"
      ? "txt-down"
      : "txt-flat";

  const quoteAgeSeconds = Number(data.quote_age_seconds);
  const quoteAgeText = Number.isFinite(quoteAgeSeconds) ? `${quoteAgeSeconds}s` : "-";
  const barAgeSeconds = Number(data.last_market_print_age_seconds);
  const barAgeText = Number.isFinite(barAgeSeconds) ? `${barAgeSeconds}s` : "-";

  const totalMs = Number(timings.total);
  const totalText = Number.isFinite(totalMs) ? `${totalMs} ms` : "-";
  const parts = [];
  if (Number.isFinite(Number(timings.bars_fetch))) parts.push(`bars ${Number(timings.bars_fetch)}ms`);
  if (Number.isFinite(Number(timings.quote_fetch))) parts.push(`quote ${Number(timings.quote_fetch)}ms`);
  if (Number.isFinite(Number(timings.forecast_fetch))) parts.push(`forecast ${Number(timings.forecast_fetch)}ms`);
  const breakdownText = parts.length ? parts.join(" | ") : "-";

  updateText("streamQuality", streamQuality, streamClass);
  updateText("quoteAge", quoteAgeText);
  updateText("barAge", barAgeText);
  updateText("backendTotal", totalText);
  updateText("backendBreakdown", breakdownText);

  state.refresh.tickerPollMs = computeTickerPollMs(market, data.last_market_print_age_seconds, data.quote_age_seconds);

  updateText("oosTotal", String(metrics.total_scored ?? "-"));
  updateText("oosAcc3", metrics.accuracy_3class_pct != null ? `${Number(metrics.accuracy_3class_pct).toFixed(2)}%` : "-");
  updateText(
    "oosAccBin",
    metrics.accuracy_binary_excl_flat_pct != null ? `${Number(metrics.accuracy_binary_excl_flat_pct).toFixed(2)}%` : "-"
  );
  updateText("oosMae", metrics.mae_return != null ? Number(metrics.mae_return).toFixed(6) : "-");

  const displayName = getSymbolProminentLabel(state.symbol);
  updateLastUpdated(data.updated_at_est || "-");
  document.title = `${displayName} | Futursia`;
  updateSelectedSymbolHeader(state.symbol);
  const select = document.getElementById("tickerSelect");
  if (select) select.value = state.symbol;
  drawChart(data.bars || [], data.forecast_path || []);
}

function renderNews(data) {
  const list = document.getElementById("newsList");
  list.innerHTML = "";

  const headlines = data.headlines || [];
  if (headlines.length === 0) {
    const empty = document.createElement("div");
    empty.className = "news-item";
    empty.setAttribute("role", "listitem");
    empty.textContent = "No headlines available.";
    list.appendChild(empty);
  }

  headlines.forEach((item) => {
    const card = document.createElement("article");
    card.className = "news-item";
    card.setAttribute("role", "listitem");

    const title = document.createElement("h3");
    title.textContent = item.title || "Untitled";
    card.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "news-meta";
    meta.textContent = `${item.source || "Unknown"} | ${item.timestamp_est || "-"}`;
    card.appendChild(meta);

    const body = document.createElement("p");
    body.textContent = item.summary || "No summary available.";
    card.appendChild(body);

    if (item.url) {
      const link = document.createElement("a");
      link.href = item.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Open source";
      link.setAttribute("aria-label", `Open source: ${item.title || "headline"}`);
      card.appendChild(link);
    }

    list.appendChild(card);
  });

  updateText("newsUpdated", data.updated_at_est || "-");
}

function getChartLayout() {
  const svg = document.getElementById("priceChart");
  const rect = svg ? svg.getBoundingClientRect() : null;
  const width = Math.max(480, Math.round(rect?.width || 960));
  const height = Math.max(280, Math.round(rect?.height || 360));
  const padLeft = 56;
  const padRight = 12;
  const padTop = 10;
  const padBottom = 36;
  return {
    width,
    height,
    padLeft,
    padRight,
    padTop,
    padBottom,
    innerW: width - padLeft - padRight,
    innerH: height - padTop - padBottom,
  };
}

function parseNumber(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function parseTimeMs(value) {
  const ts = new Date(value).getTime();
  return Number.isFinite(ts) ? ts : null;
}

function inferMedianStepMs(rows, fallbackMs = 60 * 1000) {
  const times = (rows || [])
    .map((row) => Number(row.t))
    .filter((ts) => Number.isFinite(ts))
    .sort((a, b) => a - b);
  const deltas = [];
  for (let i = 1; i < times.length; i += 1) {
    const delta = times[i] - times[i - 1];
    if (delta > 0) deltas.push(delta);
  }
  if (!deltas.length) {
    return Number.isFinite(fallbackMs) ? fallbackMs : 60 * 1000;
  }
  deltas.sort((a, b) => a - b);
  return deltas[Math.floor(deltas.length / 2)];
}

function normalizeLiveCandles(bars) {
  return (bars || [])
    .map((bar) => {
      const t = parseTimeMs(bar.timestamp);
      const open = parseNumber(bar.open);
      const high = parseNumber(bar.high);
      const low = parseNumber(bar.low);
      const close = parseNumber(bar.close);
      if (t == null || open == null || high == null || low == null || close == null) return null;
      if (open < 0 || high < 0 || low < 0 || close < 0) return null;
      return { t, open, high, low, close };
    })
    .filter(Boolean)
    .sort((a, b) => a.t - b.t);
}

function normalizeForecastCandles(forecastPath, liveBars = []) {
  const points = (forecastPath || [])
    .map((row) => {
      const t = parseTimeMs(row.timestamp);
      const price = parseNumber(row.price);
      if (t == null || price == null) return null;
      return { t, price };
    })
    .filter(Boolean)
    .sort((a, b) => a.t - b.t);
  if (points.length < 2) return [];

  const lastLiveTs = liveBars.length ? Number(liveBars[liveBars.length - 1].t) : null;
  const liveStepMs = inferMedianStepMs(liveBars, null);
  const forecastStepMs = inferMedianStepMs(points, 60 * 1000);
  const stepMs = Number.isFinite(liveStepMs) && liveStepMs > 0 ? liveStepMs : forecastStepMs;

  const firstForecastTs = Number(points[1]?.t);
  const expectedFirstTs = Number.isFinite(lastLiveTs) ? lastLiveTs + stepMs : firstForecastTs;
  const shouldRealign =
    Number.isFinite(lastLiveTs) &&
    Number.isFinite(firstForecastTs) &&
    (firstForecastTs <= lastLiveTs || Math.abs(firstForecastTs - expectedFirstTs) > stepMs * 1.5);

  const candles = [];
  for (let i = 1; i < points.length; i += 1) {
    const curr = points[i];
    const prev = points[i - 1];
    const open = prev.price;
    const close = curr.price;
    if (!Number.isFinite(open) || !Number.isFinite(close) || open < 0 || close < 0) continue;
    candles.push({
      t: shouldRealign ? lastLiveTs + i * stepMs : curr.t,
      open,
      high: Math.max(open, close),
      low: Math.min(open, close),
      close,
    });
  }
  return candles;
}

function getChartDataStats() {
  const all = [...state.chart.liveBars, ...state.chart.forecastBars];
  if (!all.length) return null;

  const times = all.map((row) => row.t).sort((a, b) => a - b);
  const minTs = times[0];
  const maxTs = times[times.length - 1];

  const deltas = [];
  for (let i = 1; i < times.length; i += 1) {
    const delta = times[i] - times[i - 1];
    if (delta > 0) deltas.push(delta);
  }
  deltas.sort((a, b) => a - b);
  const stepMs = deltas.length ? deltas[Math.floor(deltas.length / 2)] : 60 * 1000;
  const padMs = Math.max(stepMs * 3, 2 * 60 * 1000);

  return { minTs, maxTs, stepMs, padMs };
}

function fitChartToData() {
  const stats = getChartDataStats();
  if (!stats) return;
  const hasForecast = state.chart.forecastBars.length > 0;
  if (hasForecast) {
    const maxTs = stats.maxTs;
    const threeHoursMs = 3 * 60 * 60 * 1000;
    state.chart.viewMinTs = Math.max(stats.minTs, maxTs - threeHoursMs);
    state.chart.viewMaxTs = maxTs;
  } else {
    state.chart.viewMinTs = stats.minTs - stats.padMs;
    state.chart.viewMaxTs = stats.maxTs;
  }
  state.chart.manualView = false;
}

function clampChartView() {
  const stats = getChartDataStats();
  if (!stats) return;

  const hardMin = stats.minTs - stats.padMs * 2;
  const hardMax = stats.maxTs + stats.padMs * 2;
  const hardRange = Math.max(hardMax - hardMin, stats.stepMs * 10);

  const minRange = Math.max(stats.stepMs * 6, 2 * 60 * 1000);
  const maxRange = Math.max(hardRange * 1.5, minRange * 2);

  let vMin = Number(state.chart.viewMinTs);
  let vMax = Number(state.chart.viewMaxTs);
  if (!Number.isFinite(vMin) || !Number.isFinite(vMax) || vMax <= vMin) {
    vMin = hardMin;
    vMax = hardMax;
  }

  let range = vMax - vMin;
  const center = vMin + range / 2;

  if (range < minRange) {
    range = minRange;
    vMin = center - range / 2;
    vMax = center + range / 2;
  }
  if (range > maxRange) {
    range = maxRange;
    vMin = center - range / 2;
    vMax = center + range / 2;
  }

  if (vMin < hardMin) {
    const shift = hardMin - vMin;
    vMin += shift;
    vMax += shift;
  }
  if (vMax > hardMax) {
    const shift = vMax - hardMax;
    vMin -= shift;
    vMax -= shift;
  }

  state.chart.viewMinTs = vMin;
  state.chart.viewMaxTs = vMax;
}

function formatAxisTime(tsMs) {
  const dt = new Date(tsMs);
  const mo = String(dt.getMonth() + 1).padStart(2, "0");
  const da = String(dt.getDate()).padStart(2, "0");
  const hh = String(dt.getHours()).padStart(2, "0");
  const mm = String(dt.getMinutes()).padStart(2, "0");
  return `${mo}-${da} ${hh}:${mm}`;
}

function createSvgNode(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function renderChartMessage(message) {
  const svg = document.getElementById("priceChart");
  if (!svg) return;
  svg.innerHTML = "";
  svg.appendChild(
    createSvgNode("text", {
      x: "50%",
      y: "50%",
      "text-anchor": "middle",
      fill: "#7f9199",
      "font-size": 12,
    })
  ).textContent = message;
}

function getVisibleCandles(rows, minTs, maxTs, stepMs) {
  const left = minTs - stepMs * 2;
  const right = maxTs + stepMs * 2;
  return rows.filter((row) => row.t >= left && row.t <= right);
}

function renderChartFromState() {
  const svg = document.getElementById("priceChart");
  if (!svg) return;
  svg.innerHTML = "";

  const layout = getChartLayout();
  svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  const stats = getChartDataStats();
  if (!stats) {
    renderChartMessage("Not enough chart data yet");
    return;
  }

  clampChartView();
  const minT = Number(state.chart.viewMinTs);
  const maxT = Number(state.chart.viewMaxTs);
  const rangeT = Math.max(maxT - minT, stats.stepMs * 6);

  const liveVisible = getVisibleCandles(state.chart.liveBars, minT, maxT, stats.stepMs);
  const forecastVisible = getVisibleCandles(state.chart.forecastBars, minT, maxT, stats.stepMs);
  const visible = [...liveVisible, ...forecastVisible];
  if (!visible.length) {
    renderChartMessage("No visible bars in current view");
    return;
  }

  const isValidPriceBar = (row) =>
    Number.isFinite(row.low) &&
    Number.isFinite(row.high) &&
    row.low > 0 &&
    row.high > 0;
  const validForRange = visible.filter(isValidPriceBar);
  const source = validForRange.length ? validForRange : visible;

  let minP = Math.min(...source.map((row) => row.low));
  let maxP = Math.max(...source.map((row) => row.high));
  if (!Number.isFinite(minP) || !Number.isFinite(maxP)) {
    renderChartMessage("Not enough chart data yet");
    return;
  }
  minP = Math.max(0, minP);
  if (maxP <= minP) {
    maxP = minP + Math.max(minP * 0.01, 0.1);
  }
  const padP = Math.max((maxP - minP) * 0.08, minP * 0.002);
  minP = Math.max(0, minP - padP);
  maxP += padP;

  const x = (t) => layout.padLeft + ((t - minT) / rangeT) * layout.innerW;
  const y = (p) => layout.padTop + ((maxP - p) / (maxP - minP)) * layout.innerH;

  const yTicks = 5;
  for (let i = 0; i <= yTicks; i += 1) {
    const gy = layout.padTop + (layout.innerH * i) / yTicks;
    svg.appendChild(
      createSvgNode("line", {
        x1: layout.padLeft,
        x2: layout.width - layout.padRight,
        y1: gy,
        y2: gy,
        stroke: "#253037",
        "stroke-width": 1,
      })
    );

    const price = maxP - ((maxP - minP) * i) / yTicks;
    const label = createSvgNode("text", {
      x: 8,
      y: gy + 4,
      fill: "#90a4ad",
      "font-size": 10,
    });
    label.textContent = price.toFixed(2);
    svg.appendChild(label);
  }

  const xTicks = 6;
  for (let i = 0; i <= xTicks; i += 1) {
    const tx = layout.padLeft + (layout.innerW * i) / xTicks;
    const ts = minT + (rangeT * i) / xTicks;
    svg.appendChild(
      createSvgNode("line", {
        x1: tx,
        x2: tx,
        y1: layout.padTop,
        y2: layout.height - layout.padBottom,
        stroke: "#1d272d",
        "stroke-width": 1,
      })
    );
    const label = createSvgNode("text", {
      x: tx,
      y: layout.height - 8,
      fill: "#89a0a9",
      "font-size": 10,
      "text-anchor": "middle",
    });
    label.textContent = formatAxisTime(ts);
    svg.appendChild(label);
  }

  svg.appendChild(
    createSvgNode("line", {
      x1: layout.padLeft,
      x2: layout.width - layout.padRight,
      y1: layout.height - layout.padBottom,
      y2: layout.height - layout.padBottom,
      stroke: "#45555d",
      "stroke-width": 1,
    })
  );
  svg.appendChild(
    createSvgNode("line", {
      x1: layout.padLeft,
      x2: layout.padLeft,
      y1: layout.padTop,
      y2: layout.height - layout.padBottom,
      stroke: "#45555d",
      "stroke-width": 1,
    })
  );

  const allVisibleTimes = visible.map((row) => row.t).sort((a, b) => a - b);
  const timeDiffs = [];
  for (let i = 1; i < allVisibleTimes.length; i += 1) {
    const dt = allVisibleTimes[i] - allVisibleTimes[i - 1];
    if (dt > 0) timeDiffs.push(dt);
  }
  timeDiffs.sort((a, b) => a - b);
  const stepMs = timeDiffs.length ? timeDiffs[Math.floor(timeDiffs.length / 2)] : stats.stepMs;
  const baseBarWidth = Math.max(2, Math.min(14, (stepMs / rangeT) * layout.innerW * 0.7));

  function drawCandle(row, color, width, opacity = 1) {
    const cx = x(row.t);
    const yHigh = y(row.high);
    const yLow = y(row.low);
    const yOpen = y(row.open);
    const yClose = y(row.close);
    const bodyTop = Math.min(yOpen, yClose);
    const bodyHeight = Math.max(1.1, Math.abs(yOpen - yClose));

    svg.appendChild(
      createSvgNode("line", {
        x1: cx,
        x2: cx,
        y1: yHigh,
        y2: yLow,
        stroke: color,
        "stroke-width": 1.2,
        "stroke-opacity": opacity,
      })
    );

    svg.appendChild(
      createSvgNode("rect", {
        x: cx - width / 2,
        y: bodyTop,
        width,
        height: bodyHeight,
        fill: color,
        "fill-opacity": opacity,
        stroke: color,
        "stroke-opacity": opacity,
      })
    );
  }

  liveVisible.forEach((row) => {
    if (!isValidPriceBar(row)) return;
    const color = row.close >= row.open ? "#25c26e" : "#ef5350";
    drawCandle(row, color, baseBarWidth, 0.95);
  });

  const forecastBarWidth = baseBarWidth;
  forecastVisible.forEach((row) => {
    if (!isValidPriceBar(row)) return;
    drawCandle(row, "#ff9f43", forecastBarWidth, 0.9);
  });
}

function zoomChart(factor, anchorRatio = 0.5) {
  const stats = getChartDataStats();
  if (!stats) return;
  clampChartView();

  const minT = Number(state.chart.viewMinTs);
  const maxT = Number(state.chart.viewMaxTs);
  const range = maxT - minT;
  const anchorT = minT + range * Math.min(1, Math.max(0, anchorRatio));

  const newRange = range * factor;
  let nextMin = anchorT - (anchorT - minT) * factor;
  let nextMax = nextMin + newRange;

  state.chart.viewMinTs = nextMin;
  state.chart.viewMaxTs = nextMax;
  state.chart.manualView = true;
  clampChartView();
  renderChartFromState();
}

function resetChartView() {
  fitChartToData();
  renderChartFromState();
}

function setupChartInteractions() {
  if (state.chart.interactionsReady) return;
  const svg = document.getElementById("priceChart");
  if (!svg) return;
  state.chart.interactionsReady = true;

  svg.addEventListener(
    "wheel",
    (event) => {
      const stats = getChartDataStats();
      if (!stats) return;
      event.preventDefault();
      const layout = getChartLayout();
      const rect = svg.getBoundingClientRect();
      const relX = ((event.clientX - rect.left) / rect.width) * layout.width;
      if (relX <= layout.padLeft || relX >= layout.width - layout.padRight) return;
      const anchorRatio = (relX - layout.padLeft) / layout.innerW;
      zoomChart(event.deltaY < 0 ? 0.85 : 1.15, anchorRatio);
    },
    { passive: false }
  );

  svg.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const stats = getChartDataStats();
    if (!stats) return;
    clampChartView();
    state.chart.dragging = true;
    state.chart.dragStartX = event.clientX;
    state.chart.dragMinTs = Number(state.chart.viewMinTs);
    state.chart.dragMaxTs = Number(state.chart.viewMaxTs);
    state.chart.manualView = true;
    svg.setPointerCapture(event.pointerId);
  });

  svg.addEventListener("pointermove", (event) => {
    if (!state.chart.dragging) return;
    const layout = getChartLayout();
    const dragRange = state.chart.dragMaxTs - state.chart.dragMinTs;
    const dx = event.clientX - state.chart.dragStartX;
    const dt = (-dx / layout.innerW) * dragRange;
    state.chart.viewMinTs = state.chart.dragMinTs + dt;
    state.chart.viewMaxTs = state.chart.dragMaxTs + dt;
    clampChartView();
    renderChartFromState();
  });

  const stopDrag = () => {
    state.chart.dragging = false;
  };
  svg.addEventListener("pointerup", stopDrag);
  svg.addEventListener("pointercancel", stopDrag);
  svg.addEventListener("dblclick", () => resetChartView());
  svg.addEventListener("keydown", (event) => {
    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomChart(0.85, 0.5);
    } else if (event.key === "-" || event.key === "_") {
      event.preventDefault();
      zoomChart(1.15, 0.5);
    } else if (event.key === "0") {
      event.preventDefault();
      resetChartView();
    }
  });

  const zoomInBtn = document.getElementById("chartZoomInBtn");
  if (zoomInBtn) zoomInBtn.addEventListener("click", () => zoomChart(0.85, 0.5));

  const zoomOutBtn = document.getElementById("chartZoomOutBtn");
  if (zoomOutBtn) zoomOutBtn.addEventListener("click", () => zoomChart(1.15, 0.5));

  const resetBtn = document.getElementById("chartResetBtn");
  if (resetBtn) resetBtn.addEventListener("click", () => resetChartView());
}

function drawChart(bars, forecastPath) {
  const liveBars = normalizeLiveCandles(bars);
  state.chart.liveBars = liveBars;
  state.chart.forecastBars = normalizeForecastCandles(forecastPath, liveBars);

  if (!state.chart.liveBars.length && !state.chart.forecastBars.length) {
    renderChartMessage("Not enough chart data yet");
    return;
  }

  if (!state.chart.manualView || state.chart.viewMinTs == null || state.chart.viewMaxTs == null) {
    fitChartToData();
  }
  renderChartFromState();
}

async function safeLoad(name, fn, options = {}) {
  const force = Boolean(options.force);
  if (state.pending[name] && !force) return;
  if (!force) state.pending[name] = true;
  try {
    await fn();
  } catch (error) {
    console.error(`${name} refresh failed`, error);
    updateLastUpdated(`Error: ${error.message}`);
    announceToScreenReader(`Error: ${error.message}`);
  } finally {
    if (!force) state.pending[name] = false;
  }
}

async function loadSettings() {
  const settings = await getJson("/api/settings");
  state.settings = settings;
  state.symbolNames = settings.symbol_names || {};
  const pollSec = Number(settings?.refresh?.quote_poll_interval);
  if (Number.isFinite(pollSec) && pollSec > 0) {
    state.refresh.tickerPollMs = Math.max(500, Math.round(pollSec * 1000));
  }
  populateSymbols(settings.symbols || []);
  const dump = document.getElementById("settingsDump");
  dump.textContent = JSON.stringify(settings, null, 2);
}

async function loadDashboard() {
  const data = await getJson("/api/dashboard?limit=10");
  renderMovers(data);
}

async function loadTicker() {
  const requestedSymbol = state.symbol;
  const requestId = ++state.requestIds.ticker;
  const data = await getJson(`/api/ticker/${encodeURIComponent(requestedSymbol)}`);
  if (requestId !== state.requestIds.ticker) return;
  if (requestedSymbol !== state.symbol) return;
  renderTicker(data);
}

async function loadNews() {
  const requestedSymbol = state.symbol;
  const requestId = ++state.requestIds.news;
  const data = await getJson(`/api/news/${encodeURIComponent(requestedSymbol)}?limit=10`);
  if (requestId !== state.requestIds.news) return;
  if (requestedSymbol !== state.symbol) return;
  renderNews(data);
}

async function refreshAll() {
  await Promise.all([
    safeLoad("dashboard", loadDashboard),
    safeLoad("ticker", loadTicker),
    safeLoad("news", loadNews),
  ]);
}

function scheduleTickerRefresh() {
  if (state.refresh.tickerTimerId) {
    clearTimeout(state.refresh.tickerTimerId);
  }

  const loop = async () => {
    await safeLoad("ticker", loadTicker);
    state.refresh.tickerTimerId = setTimeout(loop, state.refresh.tickerPollMs);
  };

  state.refresh.tickerTimerId = setTimeout(loop, state.refresh.tickerPollMs);
}

function wireEvents() {
  const search = document.getElementById("tickerSearch");
  if (search) {
    search.addEventListener("input", (e) => {
      applySymbolSearch(e.target.value);
    });
    search.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        const first = state.filteredSymbolList[0];
        if (!first) return;
        e.preventDefault();
        selectSymbol(first);
      } else if (e.key === "Escape") {
        e.preventDefault();
        search.value = "";
        applySymbolSearch("");
      }
    });
  }

  const select = document.getElementById("tickerSelect");
  if (select) {
    select.addEventListener("change", (e) => {
      const symbol = e.target.value;
      if (symbol) selectSymbol(symbol);
    });
  }
}

async function init() {
  activateTabs();
  wireEvents();
  setupChartInteractions();
  window.addEventListener("resize", () => {
    if (state.chart.liveBars.length || state.chart.forecastBars.length) {
      renderChartFromState();
    }
  });
  await loadSettings();
  await refreshAll();

  setInterval(() => {
    safeLoad("dashboard", loadDashboard);
  }, 5000);
  scheduleTickerRefresh();

  setInterval(() => {
    safeLoad("news", loadNews);
  }, 15000);
}

window.addEventListener("DOMContentLoaded", () => {
  init().catch((error) => {
    console.error("Initialization failed", error);
    updateLastUpdated(`Initialization failed: ${error.message}`);
    announceToScreenReader(`Initialization failed: ${error.message}`);
  });
});
