"""Entrenamiento de todos los modelos a partir del histórico (sin BD).

Determinista: misma configuración y mismo histórico → mismos modelos.
Los nombres de modelo son las claves de ``train_all()``:
``poisson``, ``elo``, ``ml`` y ``ensemble``.
"""

from football_predictor.config.settings import Settings, get_settings
from football_predictor.domain.protocols import HistoryProvider
from football_predictor.features.builder import FeatureBuilder
from football_predictor.features.config import FeatureConfig
from football_predictor.features.elo import EloCalculator
from football_predictor.features.stream import FeatureStream
from football_predictor.models.base import Predictor, outcome_from_goals
from football_predictor.models.elo_model import EloModel
from football_predictor.models.ensemble import EnsembleModel
from football_predictor.models.ml import MLModel
from football_predictor.models.poisson import PoissonModel

MODEL_POISSON = "poisson"
MODEL_ELO = "elo"
MODEL_ML = "ml"
MODEL_ENSEMBLE = "ensemble"
ALL_MODELS = (MODEL_POISSON, MODEL_ELO, MODEL_ML, MODEL_ENSEMBLE)


def parse_ensemble_weights(raw: str) -> list[float]:
    parts = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("models_ensemble_weights debe tener 3 valores: poisson,elo,ml")
    if sum(parts) <= 0:
        raise ValueError("la suma de los pesos del ensemble debe ser positiva")
    return parts


class ModelTrainer:
    def __init__(self, history_provider: HistoryProvider, settings: Settings | None = None) -> None:
        self._provider = history_provider
        self._settings = settings or get_settings()

    @property
    def _feature_config(self) -> FeatureConfig:
        s = self._settings
        return FeatureConfig(
            goals_window=s.features_goals_window,
            home_away_window=s.features_home_away_window,
            stats_window=s.features_stats_window,
        )

    @property
    def _elo_calculator(self) -> EloCalculator:
        s = self._settings
        return EloCalculator(
            initial_rating=s.elo_initial_rating,
            k_factor=s.elo_k_factor,
            home_advantage=s.elo_home_advantage,
        )

    def feature_builder(self) -> FeatureBuilder:
        return FeatureBuilder(self._provider, self._feature_config, self._elo_calculator)

    def train_all(self) -> dict[str, Predictor]:
        matches = sorted(self._provider.get_all_matches(), key=lambda m: (m.date, m.match_id))
        if not matches:
            raise ValueError("no hay partidos históricos para entrenar: importa datos primero")

        stream = FeatureStream(self._feature_config, self._elo_calculator)
        vectors = stream.build_all(matches)
        outcomes = [outcome_from_goals(match.home_goals, match.away_goals) for match in matches]

        poisson = PoissonModel(
            regularization=self._settings.models_poisson_regularization,
            rho=self._settings.models_poisson_rho,
        )
        poisson.fit(matches)

        elo = EloModel(
            draw_max=self._settings.models_elo_draw_max,
            draw_sigma=self._settings.models_elo_draw_sigma,
        )

        ml = MLModel(random_state=self._settings.models_ml_random_state)
        ml.fit(vectors, outcomes)

        weights = parse_ensemble_weights(self._settings.models_ensemble_weights)
        ensemble = EnsembleModel([poisson, elo, ml], weights)

        return {
            MODEL_POISSON: poisson,
            MODEL_ELO: elo,
            MODEL_ML: ml,
            MODEL_ENSEMBLE: ensemble,
        }