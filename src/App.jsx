import React, { useState } from 'react'
import Dashboard from './components/Dashboard'
import TickerDetail from './components/TickerDetail'
import News from './components/News'
import Settings from './components/Settings'
import './styles/App.css'

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <h1>Futursia</h1>
          <p>Real-time Stock Forecasting & Market Dashboard</p>
        </div>
      </header>

      <nav className="tabs">
        <button
          className={`tab ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          Market
        </button>
        <button
          className={`tab ${activeTab === 'ticker' ? 'active' : ''}`}
          onClick={() => setActiveTab('ticker')}
        >
          Ticker + Forecast
        </button>
        <button
          className={`tab ${activeTab === 'news' ? 'active' : ''}`}
          onClick={() => setActiveTab('news')}
        >
          News
        </button>
        <button
          className={`tab ${activeTab === 'settings' ? 'active' : ''}`}
          onClick={() => setActiveTab('settings')}
        >
          Diagnostics
        </button>
      </nav>

      <main className="main-content">
        {activeTab === 'dashboard' && <Dashboard />}
        {activeTab === 'ticker' && <TickerDetail />}
        {activeTab === 'news' && <News />}
        {activeTab === 'settings' && <Settings />}
      </main>

      <footer className="app-footer">
        <p>Futursia Forecasting &copy; 2024 | Real-time 40-minute stock forecasts</p>
      </footer>
    </div>
  )
}
