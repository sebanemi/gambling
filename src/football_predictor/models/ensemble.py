"""Ensemble: media ponderada de las probabilidades de varios modelos (§16)."""

import inspect
from collections.abc import Callable, Sequence

from football_predictor.domain.features import FeatureVector
from football_predictor.models.base import PredictionResult, Predictor


def _member_predict_fn(model: Predictor) -> Callable[[FeatureVector], PredictionResult]:
    """Bind de ``model.predict`` sin snapshot si el miembro lo soporta.

    Los miembros solo aportan probs/goles al ensemble (sus snapshots se
    descartaban igual); llamar sin snapshot evita un ``flatten()`` por
    miembro. Acepta miembros con el protocolo viejo ``predict(features)``.
    """
    try:
        accepts_flag = "include_snapshot" in inspect.signature(model.predict).parameters
    except (TypeError, ValueError):
        accepts_flag = False
    if accepts_flag:
        return lambda features: model.predict(features, include_snapshot=False)
    return lambda features: model.predict(features)


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
        self._predict_fns = [_member_predict_fn(model) for model in self._models]

    def predict(self, features: FeatureVector, *, include_snapshot: bool = True) -> PredictionResult:
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        goals_home: list[float] = []
        goals_away: list[float] = []

        for predict_fn, weight in zip(self._predict_fns, self._weights):
            result = predict_fn(features)
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
            features_snapshot=features.flatten() if include_snapshot else None,
        )