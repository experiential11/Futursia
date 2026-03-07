"""Factory to return the appropriate market data client based on config."""
from typing import Dict
import logging
from .finnhub_client import FinnhubClient
from .fmp_client import FMPClient
from .yfinance_client import YFinanceClient

logger = logging.getLogger(__name__)


def get_market_client(config: Dict):
    """Return a market data client instance depending on configuration.

    Preference order:
      - Explicit `market_api_provider` value if provided
      - FMP if fmp config exists
      - YFinance if yfinance config exists
      - Else Finnhub
    """
    provider = (config.get('market_api_provider') or '').strip().lower()

    if provider == 'fmp':
        logger.info("Using FMPClient as market data provider (explicit)")
        return FMPClient(config)
    if provider == 'finnhub':
        logger.info("Using FinnhubClient as market data provider (explicit)")
        return FinnhubClient(config)
    if provider == 'yfinance':
        logger.info("Using YFinanceClient as market data provider (explicit)")
        return YFinanceClient(config)
    if provider == 'databento':
        logger.warning("Databento provider disabled; falling back to YFinanceClient")
        return YFinanceClient(config)

    if config.get('fmp_api'):
        logger.info("Using FMPClient as market data provider (fallback)")
        return FMPClient(config)
    if config.get('yfinance_api'):
        logger.info("Using YFinanceClient as market data provider (fallback)")
        return YFinanceClient(config)

    logger.info("Using FinnhubClient as market data provider (fallback)")
    return FinnhubClient(config)
