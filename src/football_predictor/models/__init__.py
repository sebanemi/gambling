"""Modelos de predicción de resultados. Sin acceso a BD."""

from football_predictor.models.base import Outcome, PredictionResult, StatisticsPredictionResult, outcome_from_goals
from football_predictor.models.elo_model import EloModel
from football_predictor.models.ensemble import EnsembleModel
from football_predictor.models.ml import MLModel
from football_predictor.models.poisson import PoissonModel
from football_predictor.models.statistics_model import StatisticsModel

__all__ = [
    "EloModel",
    "EnsembleModel",
    "MLModel",
    "Outcome",
    "PoissonModel",
    "PredictionResult",
    "StatisticsModel",
    "StatisticsPredictionResult",
    "outcome_from_goals",
]