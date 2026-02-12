"""Database operations for market data storage."""

import sqlite3
import os
import json
from datetime import datetime
import pandas as pd


class MarketDB:
    """SQLite database for market data and features."""
    
    def __init__(self, db_path='storage/market.db'):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._init_schema()
    
    def _get_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_schema(self):
        """Initialize database schema if it doesn't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Intraday bars table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intraday_bars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                vwap REAL,
                created_at TEXT,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Quotes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bid REAL,
                ask REAL,
                bid_size INTEGER,
                ask_size INTEGER,
                last_price REAL,
                last_size INTEGER,
                created_at TEXT,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Forecasts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                forecast_time INTEGER NOT NULL,
                prediction_return REAL,
                confidence REAL,
                direction TEXT,
                interval_50_lower REAL,
                interval_50_upper REAL,
                interval_90_lower REAL,
                interval_90_upper REAL,
                model_status TEXT,
                created_at TEXT,
                UNIQUE(symbol, forecast_time)
            )
        ''')
        
        # API calls log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                endpoint TEXT NOT NULL,
                symbol TEXT,
                status_code INTEGER,
                timestamp INTEGER,
                error_message TEXT,
                created_at TEXT
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_bars_symbol_time ON intraday_bars(symbol, timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quotes_symbol_time ON quotes(symbol, timestamp DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_time ON forecasts(symbol, forecast_time DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_api_calls_time ON api_calls(timestamp DESC)')
        
        conn.commit()
        conn.close()
    
    def save_bars(self, symbol, bars):
        """Save intraday bars to database.
        
        Args:
            symbol: Stock symbol
            bars: DataFrame with OHLCV data
        """
        if bars.empty:
            return
        
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        for _, row in bars.iterrows():
            try:
                timestamp = int(pd.Timestamp(row['timestamp']).timestamp())
                cursor.execute('''
                    INSERT OR REPLACE INTO intraday_bars
                    (symbol, timestamp, open, high, low, close, volume, vwap, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    timestamp,
                    float(row.get('open')),
                    float(row.get('high')),
                    float(row.get('low')),
                    float(row.get('close')),
                    int(row.get('volume', 0)),
                    float(row.get('vwap')) if 'vwap' in row else None,
                    now
                ))
            except (ValueError, TypeError):
                continue
        
        conn.commit()
        conn.close()
    
    def get_bars(self, symbol, limit=1000):
        """Get intraday bars from database.
        
        Args:
            symbol: Stock symbol
            limit: Maximum rows to return
        
        Returns:
            DataFrame: Historical bars
        """
        conn = self._get_connection()
        query = '''
            SELECT timestamp, open, high, low, close, volume, vwap
            FROM intraday_bars
            WHERE symbol = ?
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(symbol, limit))
        conn.close()
        
        if df.empty:
            return df
        
        # Sort in ascending order for analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
        return df.sort_values('timestamp').reset_index(drop=True)
    
    def save_quote(self, symbol, quote):
        """Save quote to database.
        
        Args:
            symbol: Stock symbol
            quote: Quote dict with bid, ask, etc.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        timestamp = int(datetime.utcnow().timestamp())
        
        cursor.execute('''
            INSERT OR REPLACE INTO quotes
            (symbol, timestamp, bid, ask, bid_size, ask_size, last_price, last_size, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            timestamp,
            quote.get('bid'),
            quote.get('ask'),
            quote.get('bid_size'),
            quote.get('ask_size'),
            quote.get('last_price'),
            quote.get('last_size'),
            now
        ))
        
        conn.commit()
        conn.close()
    
    def save_forecast(self, symbol, forecast):
        """Save forecast to database.
        
        Args:
            symbol: Stock symbol
            forecast: Forecast dict
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        forecast_time = int(datetime.utcnow().timestamp())
        
        cursor.execute('''
            INSERT OR REPLACE INTO forecasts
            (symbol, forecast_time, prediction_return, confidence, direction,
             interval_50_lower, interval_50_upper, interval_90_lower, interval_90_upper,
             model_status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            forecast_time,
            forecast.get('prediction_return'),
            forecast.get('confidence'),
            forecast.get('direction'),
            forecast.get('interval_50_lower'),
            forecast.get('interval_50_upper'),
            forecast.get('interval_90_lower'),
            forecast.get('interval_90_upper'),
            forecast.get('model_status'),
            now
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_forecast(self, symbol):
        """Get latest forecast for symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            dict: Latest forecast or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM forecasts
            WHERE symbol = ?
            ORDER BY forecast_time DESC
            LIMIT 1
        ''', (symbol,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def log_api_call(self, endpoint, symbol=None, status_code=None, error_message=None):
        """Log API call for diagnostics.
        
        Args:
            endpoint: API endpoint name
            symbol: Stock symbol (optional)
            status_code: HTTP status code
            error_message: Error message if any
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        timestamp = int(datetime.utcnow().timestamp())
        
        cursor.execute('''
            INSERT INTO api_calls
            (endpoint, symbol, status_code, timestamp, error_message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (endpoint, symbol, status_code, timestamp, error_message, now))
        
        conn.commit()
        conn.close()
    
    def get_api_call_stats(self, minutes=5):
        """Get API call statistics from last N minutes.
        
        Args:
            minutes: Number of minutes to look back
        
        Returns:
            dict: API call statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff_time = int(datetime.utcnow().timestamp()) - (minutes * 60)
        
        cursor.execute('''
            SELECT COUNT(*) as total, status_code,
                   COUNT(CASE WHEN error_message IS NOT NULL THEN 1 END) as errors
            FROM api_calls
            WHERE timestamp > ?
            GROUP BY status_code
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        stats = {'total': 0, 'errors': 0, 'by_status': {}}
        for row in rows:
            stats['total'] += row['total']
            stats['errors'] += row['errors']
            if row['status_code']:
                stats['by_status'][row['status_code']] = row['total']
        
        return stats
    
    def cleanup_old_data(self, days=30):
        """Remove data older than specified days.
        
        Args:
            days: Days of data to keep
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cutoff_time = int(datetime.utcnow().timestamp()) - (days * 86400)
        
        cursor.execute('DELETE FROM intraday_bars WHERE timestamp < ?', (cutoff_time,))
        cursor.execute('DELETE FROM quotes WHERE timestamp < ?', (cutoff_time,))
        cursor.execute('DELETE FROM api_calls WHERE timestamp < ?', (cutoff_time,))
        
        conn.commit()
        conn.close()
