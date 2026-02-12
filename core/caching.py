"""Caching layer for API responses."""

import time
from datetime import datetime, timedelta
from functools import wraps


class Cache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self):
        """Initialize cache."""
        self._cache = {}
    
    def get(self, key, default=None):
        """Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if expired or not found
        
        Returns:
            Cached value or default
        """
        if key in self._cache:
            value, expiry_time = self._cache[key]
            if time.time() < expiry_time:
                return value
            else:
                del self._cache[key]
        return default
    
    def set(self, key, value, ttl_seconds=300):
        """Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
        """
        expiry_time = time.time() + ttl_seconds
        self._cache[key] = (value, expiry_time)
    
    def clear(self):
        """Clear all cached items."""
        self._cache.clear()
    
    def cleanup_expired(self):
        """Remove expired items."""
        current_time = time.time()
        expired_keys = [k for k, (_, exp_time) in self._cache.items() if current_time >= exp_time]
        for key in expired_keys:
            del self._cache[key]


class APIThrottler:
    """Rate limiting and throttling for API calls."""
    
    def __init__(self, calls_per_second=10, backoff_factor=1.5):
        """Initialize throttler.
        
        Args:
            calls_per_second: Rate limit
            backoff_factor: Exponential backoff factor
        """
        self.calls_per_second = calls_per_second
        self.backoff_factor = backoff_factor
        self._last_call_time = {}
        self._backoff_until = {}
        self.min_interval = 1.0 / calls_per_second
    
    def should_throttle(self, endpoint):
        """Check if endpoint should be throttled.
        
        Args:
            endpoint: Endpoint identifier
        
        Returns:
            bool: True if should throttle
        """
        current_time = time.time()
        
        # Check if in backoff period
        if endpoint in self._backoff_until:
            if current_time < self._backoff_until[endpoint]:
                return True
            else:
                del self._backoff_until[endpoint]
        
        # Check if minimum interval elapsed
        if endpoint in self._last_call_time:
            elapsed = current_time - self._last_call_time[endpoint]
            if elapsed < self.min_interval:
                return True
        
        return False
    
    def record_call(self, endpoint):
        """Record successful API call.
        
        Args:
            endpoint: Endpoint identifier
        """
        self._last_call_time[endpoint] = time.time()
        # Clear backoff on success
        if endpoint in self._backoff_until:
            del self._backoff_until[endpoint]
    
    def record_rate_limit(self, endpoint, backoff_seconds=None):
        """Record rate limit hit - trigger backoff.
        
        Args:
            endpoint: Endpoint identifier
            backoff_seconds: Initial backoff time (None = auto-calculate)
        """
        if backoff_seconds is None:
            # Get current backoff or start with 1 second
            if endpoint not in self._backoff_until:
                backoff_seconds = 1.0
            else:
                backoff_seconds = (self._backoff_until[endpoint] - time.time()) * self.backoff_factor
        
        self._backoff_until[endpoint] = time.time() + backoff_seconds
    
    def get_wait_time(self, endpoint):
        """Get seconds to wait before next call.
        
        Args:
            endpoint: Endpoint identifier
        
        Returns:
            float: Seconds to wait (0 if can call now)
        """
        current_time = time.time()
        
        # Check backoff
        if endpoint in self._backoff_until:
            wait = self._backoff_until[endpoint] - current_time
            if wait > 0:
                return wait
        
        # Check minimum interval
        if endpoint in self._last_call_time:
            elapsed = current_time - self._last_call_time[endpoint]
            wait = self.min_interval - elapsed
            if wait > 0:
                return wait
        
        return 0
