"""Protocolos de dominio: contratos intercambiables.

Los modelos, features y la CLI dependen de estos protocolos, nunca de
implementaciones concretas. Una fuente externa es reemplazable sin tocar
el resto del sistema.
"""

from datetime import date
from typing import Protocol

from football_predictor.domain.entities import (
    CompetitionInfo,
    HistoricalMatch,
    MatchRecord,
    TeamInfo,
)


class FootballDataProvider(Protocol):
    """Fuente de datos de fútbol (CSV, API HTTP, ...)."""

    def get_matches(self) -> list[MatchRecord]:
        ...

    def get_teams(self) -> list[TeamInfo]:
        ...

    def get_competitions(self) -> list[CompetitionInfo]:
        ...


class HistoryProvider(Protocol):
    """ÚNICA puerta de acceso al histórico para features y modelos.

    La condición ``date < cutoff`` (estricta) es la garantía central de
    ausencia de data leakage: ningún feature puede ver un partido con
    fecha igual o posterior al momento de la predicción.
    """

    def get_matches_before(self, cutoff: date) -> list[HistoricalMatch]:
        """Partidos finalizados con ``date < cutoff``, en orden cronológico."""

    def get_all_matches(self) -> list[HistoricalMatch]:
        """Todo el histórico disponible (finalizados, con goles), cronológico.

        Se usa para entrenar modelos y armar el dataset de features; las
        FEATURES de cada partido siguen usando la puerta por fecha.
        """