import React, { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/Dashboard.css'

export default function Dashboard() {
  const [movers, setMovers] = useState([])
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [error, setError] = useState(null)

  const fetchMovers = async () => {
    try {
      const response = await axios.get('/api/movers')
      setMovers(response.data.movers || [])
      setLastUpdate(new Date().toLocaleTimeString())
      setError(null)
    } catch (err) {
      setError('Failed to load market data')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchMovers()
    const interval = setInterval(fetchMovers, 1000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>Market Dashboard</h2>
        <p className="subtitle">Top movers and intraday range with locked 1-second refresh (EST).</p>
      </div>

      <div className="controls-card">
        <div className="status-label">
          {loading ? 'Loading market data...' : `Updated: ${lastUpdate}`}
        </div>
        <button className="refresh-btn" onClick={fetchMovers} disabled={loading}>
          Refresh Now
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      <div className="table-card">
        <h3>Top 10 Movers</h3>
        <div className="table-wrapper">
          <table className="movers-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Price</th>
                <th>Change $</th>
                <th>Change %</th>
                <th>High</th>
                <th>Low</th>
              </tr>
            </thead>
            <tbody>
              {movers.slice(0, 10).map((mover, idx) => (
                <tr key={idx}>
                  <td className="symbol">{mover.symbol}</td>
                  <td>${parseFloat(mover.price).toFixed(2)}</td>
                  <td>${parseFloat(mover.change).toFixed(2)}</td>
                  <td className={parseFloat(mover.percent_change) >= 0 ? 'positive' : 'negative'}>
                    {parseFloat(mover.percent_change).toFixed(2)}%
                  </td>
                  <td>${parseFloat(mover.high).toFixed(2)}</td>
                  <td>${parseFloat(mover.low).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
