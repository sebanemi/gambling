"""Contratos y tipos compartidos de la capa de modelos."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

from football_predictor.domain.features import FeatureVector


class Outcome(IntEnum):
    """Etiqueta de resultado: orden estable para ML/probabilidades (1X2)."""

    HOME = 0
    DRAW = 1
    AWAY = 2


def outcome_from_goals(home_goals: int, away_goals: int) -> Outcome:
    if home_goals > away_goals:
        return Outcome.HOME
    if home_goals < away_goals:
        return Outcome.AWAY
    return Outcome.DRAW


@dataclass(frozen=True)
class PredictionResult:
    """Probabilidades P(1), P(X), P(2) y goles esperados (opcionales).

    Las probabilidades SIEMPRE suman 1.0 (se normalizan en el
    constructor para no depender de la precisión del modelo).
    """

    match_id: int
    home_win: float
    draw: float
    away_win: float
    home_goals: float | None = None
    away_goals: float | None = None
    features_snapshot: dict[str, float] | None = None

    def __post_init__(self) -> None:
        total = self.home_win + self.draw + self.away_win
        if total <= 0.0:
            raise ValueError("las probabilidades deben sumar un valor positivo")
        if abs(total - 1.0) > 1e-9:
            factor = 1.0 / total
            object.__setattr__(self, "home_win", self.home_win * factor)
            object.__setattr__(self, "draw", self.draw * factor)
            object.__setattr__(self, "away_win", self.away_win * factor)


@dataclass(frozen=True)
class StatisticsPredictionResult:
    """Esperanzas por equipo de las 10 métricas recolectadas."""

    match_id: int
    home_yellow_cards: float = 0.0
    away_yellow_cards: float = 0.0
    home_red_cards: float = 0.0
    away_red_cards: float = 0.0
    home_corners: float = 0.0
    away_corners: float = 0.0
    home_shots: float = 0.0
    away_shots: float = 0.0
    home_shots_on_target: float = 0.0
    away_shots_on_target: float = 0.0
    home_fouls: float = 0.0
    away_fouls: float = 0.0
    home_throw_ins: float = 0.0
    away_throw_ins: float = 0.0
    home_penalties: float = 0.0
    away_penalties: float = 0.0
    home_xg: float = 0.0
    away_xg: float = 0.0
    home_possession: float = 0.0
    away_possession: float = 0.0


class Predictor(Protocol):
    """Modelo listo para predecir un partido (entrenado cuando aplique)."""

    def predict(self, features: FeatureVector) -> PredictionResult:
        ...