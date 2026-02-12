"""Logging setup for the forecaster application."""

import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(config):
    """Configure logging with rotating file handler and console output.
    
    Args:
        config: Configuration dictionary from config.yaml
    """
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    log_file = log_config.get('log_file', 'storage/app.log')
    max_size = log_config.get('max_log_size', 10485760)
    backup_count = log_config.get('backup_count', 5)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('forecaster')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_size,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    
    # Console handler - only show WARNING and above to reduce clutter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name='forecaster'):
    """Get the configured logger instance."""
    return logging.getLogger(name)
