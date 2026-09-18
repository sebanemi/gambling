"""Logging estructurado.

Uso interno de la aplicación: stderr con formato estructurado y nivel
configurable. La CLI se encarga de presentar salida amigable por separado.
"""

import logging
import sys

_APP_LOGGER_NAME = "football_predictor"
_CONFIGURED = False


class StructuredFormatter(logging.Formatter):
    """Formato estructurado: ``timestamp  LEVEL  logger  mensaje``."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(7)
        name = record.name.ljust(26)
        message = record.getMessage()
        return f"{ts}  {level}  {name}  {message}"


def configure_logging(level: str = "INFO") -> None:
    """Configura el logger raíz de la aplicación (idempotente)."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(StructuredFormatter())

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.addHandler(handler)
    logger.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retorna un logger dentro del namespace de la aplicación."""
    return logging.getLogger(f"{_APP_LOGGER_NAME}.{name}")