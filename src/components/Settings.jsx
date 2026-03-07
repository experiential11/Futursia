import React, { useState, useEffect } from 'react'
import axios from 'axios'
import '../styles/Settings.css'

export default function Settings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const response = await axios.get('/api/config')
        setConfig(response.data)
      } catch (err) {
        console.error(err)
      } finally {
        setLoading(false)
      }
    }

    fetchConfig()
  }, [])

  if (loading || !config) {
    return <div className="loading">Loading diagnostics...</div>
  }

  return (
    <div className="settings">
      <div className="settings-header">
        <h2>Settings and Diagnostics</h2>
        <p className="subtitle">Provider configuration, forecast settings, and runtime diagnostics.</p>
      </div>

      <div className="info-card">
        <h3>System Information</h3>
        <div className="info-content">
          <pre>{config.info || 'No configuration data available'}</pre>
        </div>
      </div>

      <div className="config-grid">
        <div className="config-section">
          <h4>Market Provider</h4>
          <p>{config.provider || 'N/A'}</p>
        </div>
        <div className="config-section">
          <h4>Forecast Horizon</h4>
          <p>{config.horizon_minutes || 'N/A'} minutes</p>
        </div>
        <div className="config-section">
          <h4>Model Type</h4>
          <p>{config.primary_model || 'N/A'}</p>
        </div>
        <div className="config-section">
          <h4>Auto-Refresh</h4>
          <p>1 second (locked)</p>
        </div>
      </div>

      <div className="features-card">
        <h3>Features</h3>
        <ul className="features-list">
          <li>Real-time stock quotes</li>
          <li>40-minute forecasts</li>
          <li>Forecast path chart</li>
          <li>Financial news headlines</li>
          <li>1-second auto-refresh</li>
          <li>Non-blocking UI updates</li>
        </ul>
      </div>
    </div>
  )
}
