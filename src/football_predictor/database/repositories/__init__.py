"""Repositories: única capa que accede a BD con lógica de datos (no estadística)."""

from .match_statistics import MatchStatisticsRepository

__all__ = ["MatchStatisticsRepository"]