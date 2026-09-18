"""Modelo Elo como predictor de resultado (§13 de la especificación).

ESTÁ SEPARADO de la feature Elo: la feature da el rating ANTES del
partido (calculado en el FeatureBuilder); este modelo sólo toma esa
información y la convierte en probabilidades 1X2.

P(local gana) se obtiene del expected_score logístico clásico usando el
rating local + ventaja de local. La masa de empate es una función que
crece cuando los ratings están parejos y decae con la diferencia de
ratings (gaussiana centrada en 0), repartiendo el resto entre los dos
resultados:

    P_draw = draw_max * exp(-diff² / (2σ²));  P_1/P_2 = p·(1-P_draw) / (1-p)·...
"""

import math

from football_predictor.domain.features import FeatureVector
from football_predictor.features.elo import EloCalculator
from football_predictor.models.base import PredictionResult


class EloModel:
    def __init__(self, draw_max: float = 0.30, draw_sigma: float = 300.0) -> None:
        self._draw_max = float(draw_max)
        self._draw_sigma = float(draw_sigma)

    def predict(self, features: FeatureVector) -> PredictionResult:
        home_rating = features.home_elo + features.home_advantage
        p_home = EloCalculator.expected_score(home_rating, features.away_elo)
        rating_diff = home_rating - features.away_elo

        draw_prob = self._draw_max * math.exp(-(rating_diff ** 2) / (2.0 * self._draw_sigma ** 2))
        residual = 1.0 - draw_prob

        return PredictionResult(
            match_id=features.match_id,
            home_win=p_home * residual,
            draw=draw_prob,
            away_win=(1.0 - p_home) * residual,
            features_snapshot=features.flatten(),
        )