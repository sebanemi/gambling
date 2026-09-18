"""Configuración de ventanas temporales para features."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    """Ventanas de forma, goles y estadísticas. ``goals_window=None`` = todo el historial."""

    form_windows: tuple[int, ...] = (3, 5, 10)
    home_away_window: int = 5
    goals_window: int | None = None
    stats_window: int | None = None