import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import '../styles/TickerDetail.css'

const POPULAR_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

export default function TickerDetail() {
  const [symbol, setSymbol] = useState('AAPL')
  const [quote, setQuote] = useState(null)
  const [forecast, setForecast] = useState(null)
  const [bars, setBars] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const fetchData = async (sym) => {
    try {
      setLoading(true)
      const [quoteRes, forecastRes, barsRes] = await Promise.all([
        axios.get(`/api/quote/${sym}`),
        axios.get(`/api/forecast/${sym}`),
        axios.get(`/api/bars/${sym}`)
      ])

      setQuote(quoteRes.data)
      setForecast(forecastRes.data)
      setBars(barsRes.data.bars || [])
      setError(null)
    } catch (err) {
      setError(`Failed to load data for ${sym}`)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData(symbol)
    const interval = setInterval(() => fetchData(symbol), 1000)
    return () => clearInterval(interval)
  }, [symbol])

  if (!quote || !forecast) {
    return <div className="loading">Loading ticker data...</div>
  }

  const price = parseFloat(quote.last_price || quote.c || 0)
  const prevClose = parseFloat(quote.pc || price)
  const change = price - prevClose
  const changePct = (change / prevClose * 100).toFixed(2)
  const direction = forecast.direction || 'FLAT'
  const confidence = forecast.confidence || 0
  const forecastReturn = forecast.predicted_return || 0

  const chartData = bars.slice(-120).map(bar => ({
    time: new Date(bar.timestamp).toLocaleTimeString(),
    price: parseFloat(bar.close),
    timestamp: bar.timestamp
  }))

  return (
    <div className="ticker-detail">
      <div className="detail-header">
        <h2>Ticker Detail and 40-Minute Forecast</h2>
        <p className="subtitle">Trading-style live view with forecast path and confidence.</p>
      </div>

      <div className="selector-card">
        <label>Symbol</label>
        <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {POPULAR_STOCKS.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="price-card">
        <div className="price-info">
          <div className="price-label">
            Price: <span className="price-value">${price.toFixed(2)}</span>
          </div>
          <div className={`change-label ${change >= 0 ? 'positive' : 'negative'}`}>
            Change: {change >= 0 ? '+' : ''}{change.toFixed(2)} ({changePct}%)
          </div>
          <div className="high-low">
            High: ${parseFloat(quote.h || quote.high || price).toFixed(2)} |
            Low: ${parseFloat(quote.l || quote.low || price).toFixed(2)}
          </div>
        </div>
      </div>

      <div className="forecast-box">
        <h3>40-Minute Forecast</h3>
        <div className="forecast-metrics">
          <div className={`direction ${direction.toLowerCase()}`}>
            Direction: <strong>{direction}</strong>
          </div>
          <div className="confidence">Confidence: <strong>{confidence.toFixed(1)}%</strong></div>
          <div className="return">Return: <strong>{forecastReturn.toFixed(3)}%</strong></div>
        </div>

        <div className="chart-container">
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={['dataMin - 1', 'dataMax + 1']} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="price" stroke="#1f77b4" dot={false} name="Price" />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div className="no-data">No chart data available</div>
          )}
        </div>
      </div>

      <div className="stats-card">
        <h3>Price and Forecast Stats</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-label">Current Price</span>
            <span className="stat-value">${price.toFixed(2)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Previous Close</span>
            <span className="stat-value">${prevClose.toFixed(2)}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Daily Change</span>
            <span className="stat-value">{change.toFixed(2)} ({changePct}%)</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Forecast Return</span>
            <span className="stat-value">{forecastReturn.toFixed(3)}%</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Confidence</span>
            <span className="stat-value">{confidence.toFixed(1)}%</span>
          </div>
        </div>
      </div>
    </div>
  )
}
