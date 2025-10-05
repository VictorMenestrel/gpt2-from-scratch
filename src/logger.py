"""Logger configuration module."""

import logging
from pathlib import Path


def configure_logger(
    name: str,
    log_file: str = "app.log",
    level=logging.DEBUG,
) -> logging.Logger:
    """Configure and return a logger instance."""
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler with a specific format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    )
    logger.addHandler(console_handler)

    # Create file handler to log messages to a file
    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )
        logger.addHandler(file_handler)
    except OSError as e:
        logger.exception("Failed to create log file handler", exc_info=e)

    return logger


# Configure the logger for this module
log_file_path = Path(__file__).parent / "logs" / "app.log"
log_file_path.parent.mkdir(parents=True, exist_ok=True)
logger = configure_logger(__name__, log_file=log_file_path)
