"""
Structured logging configuration module for CarPricePredictor.
"""
from __future__ import annotations

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Initializes structured logging configuration.

    Args:
        log_level: String logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').

    Returns:
        Configured root application logger.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger = logging.getLogger("car_price_predictor")
    logger.setLevel(level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
