"""Top 10 movers calculation and management."""

import pandas as pd
from datetime import timedelta
from core.logging_setup import get_logger


logger = get_logger()


class Top10Manager:
    """Manage top 10 movers calculation."""
    
    def __init__(self, config, api_client):
        """Initialize top 10 manager.
        
        Args:
            config: Configuration dictionary
            api_client: MassiveClient instance
        """
        self.config = config.get('top10', {})
        self.api = api_client
        self.metric = self.config.get('metric', 'percent_change_1h')
        self.lookback_minutes = self.config.get('lookback_minutes', 60)
    
    def get_top_movers(self):
        """Get top 10 movers based on configured metric.
        
        Returns:
            DataFrame: Top movers with metrics
        """
        if self.metric == 'percent_change_1h':
            return self._calculate_percent_change_movers()
        elif self.metric == 'volume_spike':
            return self._calculate_volume_spike_movers()
        else:
            return self._calculate_percent_change_movers()
    
    def _calculate_percent_change_movers(self):
        """Calculate top movers by 1-hour percent change.
        
        Returns:
            DataFrame: Top 10 movers
        """
        try:
            # Try API first
            api_movers = self.api.get_top_movers(limit=20)
            
            if api_movers:
                df = pd.DataFrame(api_movers)
                
                # Normalize columns
                if 'change_percent' in df.columns:
                    df = df.sort_values('change_percent', ascending=False)
                elif 'change' in df.columns:
                    df = df.sort_values('change', ascending=False)
                
                return df.head(10)
        
        except Exception as e:
            logger.error(f"Error calculating percent change movers: {e}")
        
        return pd.DataFrame()
    
    def _calculate_volume_spike_movers(self):
        """Calculate top movers by volume spike.
        
        Returns:
            DataFrame: Top 10 by volume spike
        """
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META', 'AMZN']
            movers = []
            
            for symbol in symbols:
                bars = self.api.get_bars(symbol, limit=int(self.lookback_minutes))
                if not bars.empty:
                    recent_volume = bars.tail(5)['volume'].sum()
                    overall_volume = bars['volume'].sum()
                    volume_ratio = recent_volume / (overall_volume + 1e-9)
                    
                    # Also get current price
                    quote = self.api.get_quote(symbol)
                    current_price = quote.get('last_price', 0) if quote else 0
                    
                    movers.append({
                        'symbol': symbol,
                        'volume_spike_ratio': volume_ratio,
                        'recent_volume': recent_volume,
                        'price': current_price,
                        'volume': recent_volume
                    })
            
            if movers:
                df = pd.DataFrame(movers)
                df = df.sort_values('volume_spike_ratio', ascending=False)
                return df.head(10)
        
        except Exception as e:
            logger.error(f"Error calculating volume spike movers: {e}")
        
        return pd.DataFrame()
