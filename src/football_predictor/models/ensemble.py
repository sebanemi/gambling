"""Ensemble: media ponderada de las probabilidades de varios modelos (§16)."""

from collections.abc import Sequence

from football_predictor.domain.features import FeatureVector
from football_predictor.models.base import PredictionResult, Predictor


class EnsembleModel:
    """Combina modelos ya entrenados. Los pesos no necesitan sumar 1:
    se normalizan aquí para evitar errores de configuración."""

    def __init__(self, models: Sequence[Predictor], weights: Sequence[float]) -> None:
        if len(models) != len(weights):
            raise ValueError("models y weights deben tener el mismo largo")
        if not models:
            raise ValueError("EnsembleModel requiere al menos un modelo")

        total = sum(weights)
        if total <= 0:
            raise ValueError("la suma de pesos debe ser positiva")

        self._models = list(models)
        self._weights = [float(w) / total for w in weights]

    def predict(self, features: FeatureVector) -> PredictionResult:
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        goals_home: list[float] = []
        goals_away: list[float] = []

        for model, weight in zip(self._models, self._weights):
            result = model.predict(features)
            home_win += weight * result.home_win
            draw += weight * result.draw
            away_win += weight * result.away_win
            if result.home_goals is not None:
                goals_home.append(result.home_goals)
                goals_away.append(result.away_goals)

        mean_goals = (sum(goals_home) / len(goals_home), sum(goals_away) / len(goals_away)) if goals_home else (None, None)

        return PredictionResult(
            match_id=features.match_id,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            home_goals=mean_goals[0],
            away_goals=mean_goals[1],
            features_snapshot=features.flatten(),
        )