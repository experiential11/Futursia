import React, { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/News.css'

const POPULAR_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']

export default function News() {
  const [symbol, setSymbol] = useState('AAPL')
  const [headlines, setHeadlines] = useState([])
  const [loading, setLoading] = useState(true)

  const fetchNews = async (sym) => {
    try {
      setLoading(true)
      const response = await axios.get(`/api/news/${sym}`)
      setHeadlines(response.data.headlines || [])
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchNews(symbol)
    const interval = setInterval(() => fetchNews(symbol), 5000)
    return () => clearInterval(interval)
  }, [symbol])

  return (
    <div className="news">
      <div className="news-header">
        <h2>Financial News and Headlines</h2>
        <p className="subtitle">Symbol-level headline feed in EST with source and summary.</p>
      </div>

      <div className="selector-card">
        <label>Symbol</label>
        <select value={symbol} onChange={(e) => setSymbol(e.target.value)}>
          {POPULAR_STOCKS.map(s => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <div className="news-container">
        {loading ? (
          <div className="loading">Loading headlines...</div>
        ) : headlines.length > 0 ? (
          <div className="headlines-list">
            {headlines.map((headline, idx) => (
              <div key={idx} className="headline-card">
                <div className="headline-meta">
                  <span className="headline-number">#{idx + 1}</span>
                  <span className="headline-source">{headline.source || 'Unknown'}</span>
                  <span className="headline-date">
                    {new Date(headline.datetime || headline.timestamp).toLocaleString()}
                  </span>
                </div>
                <h3 className="headline-title">{headline.headline || headline.title}</h3>
                <p className="headline-summary">{headline.summary || headline.text}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-data">No headlines available</div>
        )}
      </div>
    </div>
  )
}
