"""State management for watchlist and user preferences."""

import json
import os
from datetime import datetime


class StateManager:
    """Manage user state (watchlist, settings, etc.)."""
    
    DEFAULT_STATE = {
        'watchlist': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
        'pinned_tickers': [],
        'model_selection': 'ridge',
        'refresh_interval': 5,
        'forecast_interval': 60,
        'theme': 'light',
        'last_active_ticker': 'AAPL',
        'created_at': None,
        'updated_at': None
    }
    
    def __init__(self, state_file='storage/state.json'):
        """Initialize state manager.
        
        Args:
            state_file: Path to state JSON file
        """
        self.state_file = state_file
        os.makedirs(os.path.dirname(state_file) or '.', exist_ok=True)
        self._state = self._load()
    
    def _load(self):
        """Load state from file or create default."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    # Merge with defaults for any missing keys
                    return {**self.DEFAULT_STATE, **state}
            except Exception:
                pass
        
        state = self.DEFAULT_STATE.copy()
        state['created_at'] = datetime.utcnow().isoformat()
        state['updated_at'] = state['created_at']
        return state
    
    def _save(self):
        """Save state to file."""
        try:
            self._state['updated_at'] = datetime.utcnow().isoformat()
            os.makedirs(os.path.dirname(self.state_file) or '.', exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self._state, f, indent=2)
            return True
        except Exception:
            return False
    
    def get_watchlist(self):
        """Get user's watchlist."""
        return self._state.get('watchlist', self.DEFAULT_STATE['watchlist'])
    
    def set_watchlist(self, tickers):
        """Set user's watchlist.
        
        Args:
            tickers: List of stock symbols
        """
        # Limit to max tickers
        max_tickers = 20
        self._state['watchlist'] = tickers[:max_tickers]
        self._save()
    
    def add_to_watchlist(self, ticker):
        """Add ticker to watchlist.
        
        Args:
            ticker: Stock symbol
        """
        watchlist = self.get_watchlist()
        ticker = ticker.upper()
        if ticker not in watchlist and len(watchlist) < 20:
            watchlist.append(ticker)
            self.set_watchlist(watchlist)
    
    def remove_from_watchlist(self, ticker):
        """Remove ticker from watchlist.
        
        Args:
            ticker: Stock symbol
        """
        watchlist = self.get_watchlist()
        ticker = ticker.upper()
        if ticker in watchlist:
            watchlist.remove(ticker)
            self.set_watchlist(watchlist)
    
    def get_pinned_tickers(self):
        """Get pinned tickers."""
        return self._state.get('pinned_tickers', [])
    
    def set_pinned_tickers(self, tickers):
        """Set pinned tickers.
        
        Args:
            tickers: List of stock symbols
        """
        self._state['pinned_tickers'] = [t.upper() for t in tickers]
        self._save()
    
    def get_model_selection(self):
        """Get selected forecast model."""
        return self._state.get('model_selection', 'ridge')
    
    def set_model_selection(self, model):
        """Set forecast model.
        
        Args:
            model: Model name (ridge, xgboost, etc.)
        """
        self._state['model_selection'] = model
        self._save()
    
    def get_setting(self, key, default=None):
        """Get arbitrary setting.
        
        Args:
            key: Setting key
            default: Default value if not found
        
        Returns:
            Setting value
        """
        return self._state.get(key, default)
    
    def set_setting(self, key, value):
        """Set arbitrary setting.
        
        Args:
            key: Setting key
            value: Setting value
        """
        self._state[key] = value
        self._save()
    
    def get_all(self):
        """Get entire state."""
        return self._state.copy()
    
    def reset_to_defaults(self):
        """Reset to default state."""
        self._state = self.DEFAULT_STATE.copy()
        self._state['created_at'] = datetime.utcnow().isoformat()
        self._save()
