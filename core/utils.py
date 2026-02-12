"""Utility functions for the forecaster."""

import os
import json
from datetime import datetime, timedelta
import pytz
import pandas as pd


def get_api_key(config=None):
    """Get market API key with provider-aware fallback order.

    Returns:
        str: API key or None if not set
    """
    env_databento = os.getenv('DATABENTO_API_KEY')
    if env_databento:
        return env_databento

    env_fmp = os.getenv('FMP_API_KEY')
    if env_fmp:
        return env_fmp

    env_finnhub = os.getenv('FINNHUB_API_KEY')
    if env_finnhub:
        return env_finnhub

    env_massive = os.getenv('MASSIVE_API_KEY')
    if env_massive:
        return env_massive

    if not isinstance(config, dict):
        return None

    provider = (config.get('market_api_provider') or '').strip().lower()
    if provider == 'databento':
        return config.get('databento_api', {}).get('api_key')
    if provider == 'fmp':
        return config.get('fmp_api', {}).get('api_key')
    if provider == 'finnhub':
        return config.get('finnhub_api', {}).get('api_key')
    if provider == 'massive':
        return config.get('massive_api', {}).get('api_key')

    return (
        config.get('databento_api', {}).get('api_key')
        or config.get('fmp_api', {}).get('api_key')
        or config.get('finnhub_api', {}).get('api_key')
        or config.get('massive_api', {}).get('api_key')
    )


def is_market_open(timezone='US/Eastern'):
    """Check if US stock market is currently open.
    
    Args:
        timezone: Timezone for market check (default US/Eastern)
    
    Returns:
        bool: True if market is open, False otherwise
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    # Market open: 9:30 AM to 4:00 PM ET, Monday-Friday
    weekday = now.weekday()
    is_trading_day = weekday < 5  # Monday = 0, Friday = 4
    
    if not is_trading_day:
        return False
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close


def get_market_status_info(timezone='US/Eastern'):
    """Get human-readable market status information.
    
    Returns:
        dict: Status info with 'is_open', 'next_open', 'next_close'
    """
    tz = pytz.timezone(timezone)
    now = datetime.now(tz)
    
    is_open = is_market_open(timezone)
    
    weekday = now.weekday()
    is_trading_day = weekday < 5
    
    if is_open:
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        next_open = None
    else:
        if is_trading_day:
            # Market will open tomorrow
            next_open = (now + timedelta(days=1)).replace(hour=9, minute=30, second=0, microsecond=0)
            if weekday == 4:  # Friday
                next_open = (now + timedelta(days=3)).replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            # Weekend - market opens Monday
            days_until_monday = (7 - weekday) % 7 or 1
            next_open = (now + timedelta(days=days_until_monday)).replace(hour=9, minute=30, second=0, microsecond=0)
        
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return {
        'is_open': is_open,
        'now': now,
        'next_open': next_open,
        'market_close': market_close if is_open else None,
        'timezone': timezone
    }


def format_datetime(dt):
    """Format datetime for display.
    
    Args:
        dt: datetime object
    
    Returns:
        str: Formatted datetime string
    """
    if dt is None:
        return 'N/A'
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def safe_round(value, decimals=2):
    """Safely round a value, handling None and infinity.
    
    Args:
        value: Value to round
        decimals: Number of decimal places
    
    Returns:
        float or None: Rounded value
    """
    if value is None or pd.isna(value):
        return None
    try:
        if abs(value) == float('inf'):
            return None
        return round(float(value), decimals)
    except (ValueError, TypeError):
        return None


def percent_format(value):
    """Format value as percentage string.
    
    Args:
        value: Numeric value
    
    Returns:
        str: Formatted percentage
    """
    if value is None or pd.isna(value):
        return 'N/A'
    try:
        return f"{float(value) * 100:.2f}%"
    except (ValueError, TypeError):
        return 'N/A'


def load_json(filepath, default=None):
    """Load JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        default: Default value if file doesn't exist
    
    Returns:
        dict: Loaded JSON or default
    """
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception:
        pass
    return default or {}


def save_json(filepath, data):
    """Save data to JSON file with error handling.
    
    Args:
        filepath: Path to JSON file
        data: Data to save
    
    Returns:
        bool: True if successful
    """
    try:
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception:
        return False


def truncate_string(s, max_length=50):
    """Truncate string with ellipsis.
    
    Args:
        s: String to truncate
        max_length: Maximum length
    
    Returns:
        str: Truncated string
    """
    if s is None:
        return 'N/A'
    s = str(s)
    if len(s) <= max_length:
        return s
    return s[:max_length-3] + '...'


def mask_api_key(key):
    """Mask API key for display (security).
    
    Args:
        key: API key string
    
    Returns:
        str: Masked key
    """
    if not key:
        return '[NOT SET]'
    if len(key) <= 8:
        return '****'
    return key[:4] + '*' * (len(key) - 8) + key[-4:]
