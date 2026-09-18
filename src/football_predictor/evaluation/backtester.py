"""Backtesting walk-forward (§18 de la especificación).

Para cada partido del histórico, en orden cronológico:

1. Entrena cada modelo SOLO con partidos estrictamente anteriores.
2. Construye las features del target (puerta por fecha, anti-leakage).
3. Predice y compara contra el resultado real.

Optimización de rendimiento: las features de TODOS los partidos se
construyen una sola vez en orden cronológico con ``FeatureStream``
(O(n) en vez de O(n²/n³)), la matriz del ML se precomputa y se reutiliza
por slices, y el Poisson se recalienta entre pasos con su último óptimo
(el objetivo y el óptimo son los mismos; solo cambia el punto inicial).

Las predicciones de un partido N-ésimo SOLO pueden depender de los
partidos 1..N-1; esto se verifica en los tests con un escenario causal.
El reporte agrega ranking_loss y top1 por modelo sobre toda la caminata.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from football_predictor.config.settings import Settings, get_settings
from football_predictor.domain.protocols import HistoryProvider
from football_predictor.evaluation.ensemble_optimizer import optimize_weights
from football_predictor.evaluation.metrics import EvaluationReport, evaluate_model
from football_predictor.features.config import FeatureConfig
from football_predictor.features.elo import EloCalculator
from football_predictor.features.stream import FeatureStream
from football_predictor.models.base import Predictor, outcome_from_goals
from football_predictor.models.elo_model import EloModel
from football_predictor.models.ensemble import EnsembleModel
from football_predictor.models.ml import MLModel
from football_predictor.models.poisson import PoissonModel
from football_predictor.services.model_trainer import (
    MODEL_ELO,
    MODEL_ENSEMBLE,
    MODEL_ML,
    MODEL_POISSON,
    parse_ensemble_weights,
)

_DefaultModelNames = ("poisson", "elo", "ml", "ensemble")

PredictionSurrogate = tuple[float, float, float]


@dataclass(frozen=True)
class BacktestReport:
    """Métricas agregadas por modelo + muestras por paso (auditables)."""

    evaluated: int
    min_prior_matches: int
    reports: dict[str, EvaluationReport]
    _samples: dict[str, tuple[tuple[int, PredictionSurrogate, int, int], ...]] = field(
        repr=False, compare=False
    )
    recommended_weights: dict[str, float] | None = None
    recommended_log_loss: float | None = None

    def samples(self, model_name: str) -> list[tuple[int, PredictionSurrogate, int, int]]:
        return list(self._samples.get(model_name, []))


class Backtester:
    def __init__(
        self,
        history_provider: HistoryProvider,
        settings: Settings | None = None,
        min_prior_matches: int = 5,
    ) -> None:
        self._provider = history_provider
        self._settings = settings or get_settings()
        self._min_prior_matches = max(1, min_prior_matches)

    def run(self, model_names: Sequence[str] = _DefaultModelNames) -> BacktestReport:
        matches = sorted(self._provider.get_all_matches(), key=lambda m: (m.date, m.match_id))
        if not matches:
            raise ValueError("no hay partidos históricos para backtestear")

        names = tuple(name for name in model_names if name in _DefaultModelNames)
        if not names:
            raise ValueError("debes pedir al menos un modelo válido (poisson, elo, ml, ensemble)")

        weights = parse_ensemble_weights(self._settings.models_ensemble_weights)
        s = self._settings

        stream = FeatureStream(
            FeatureConfig(
                goals_window=s.features_goals_window,
                home_away_window=s.features_home_away_window,
                stats_window=s.features_stats_window,
            ),
            EloCalculator(
                initial_rating=s.elo_initial_rating,
                k_factor=s.elo_k_factor,
                home_advantage=s.elo_home_advantage,
            ),
        )
        vectors = stream.build_all(matches)
        outcomes = [outcome_from_goals(m.home_goals, m.away_goals) for m in matches]

        need_ml = MODEL_ML in names or MODEL_ENSEMBLE in names
        need_poisson = MODEL_POISSON in names or MODEL_ENSEMBLE in names
        matrix = MLModel.make_matrix(vectors) if need_ml else None
        outcome_matrix = np.array([int(o) for o in outcomes], dtype=int) if need_ml else None

        samples: dict[str, list[tuple[int, PredictionSurrogate, int, int]]] = {name: [] for name in names}
        evaluated = 0
        prev_poisson: np.ndarray | None = None

        for index, target_match in enumerate(matches):
            if index < self._min_prior_matches:
                continue

            trained: dict[str, Predictor] = {}
            if need_poisson:
                poisson = PoissonModel(
                    regularization=s.models_poisson_regularization,
                    rho=s.models_poisson_rho,
                )
                try:
                    poisson.fit(matches[:index], warm_start=prev_poisson)
                    trained[MODEL_POISSON] = poisson
                    prev_poisson = poisson.last_point
                except (ValueError, RuntimeError):
                    prev_poisson = None

            trained[MODEL_ELO] = EloModel(draw_max=s.models_elo_draw_max, draw_sigma=s.models_elo_draw_sigma)

            if need_ml:
                ml = MLModel(random_state=s.models_ml_random_state)
                try:
                    ml.fit_matrix(matrix[:index], outcome_matrix[:index])
                    trained[MODEL_ML] = ml
                except (ValueError, RuntimeError):
                    pass

            members = [trained[name] for name in (MODEL_POISSON, MODEL_ELO, MODEL_ML) if name in trained]
            if len(members) >= 2:
                trained[MODEL_ENSEMBLE] = EnsembleModel(members, weights[: len(members)])
            elif len(members) == 1:
                trained[MODEL_ENSEMBLE] = members[0]

            features = vectors[index]
            for name in names:
                model = trained.get(name)
                if model is None:
                    continue
                result = model.predict(features)
                samples[name].append(
                    (
                        target_match.match_id,
                        (result.home_win, result.draw, result.away_win),
                        target_match.home_goals,
                        target_match.away_goals,
                    )
                )
            evaluated += 1

        reports = {
            name: evaluate_model(name, [(prob, hg, ag) for _, prob, hg, ag in samples[name]])
            for name in names
        }

        recommended_weights: dict[str, float] | None = None
        recommended_log_loss: float | None = None
        if MODEL_ENSEMBLE in names:
            member_samples = {
                name: samples[name]
                for name in (MODEL_POISSON, MODEL_ELO, MODEL_ML)
                if name in samples
            }
            optimized = optimize_weights(member_samples)
            if optimized is not None:
                recommended_weights, recommended_log_loss = optimized

        return BacktestReport(
            evaluated=evaluated,
            min_prior_matches=self._min_prior_matches,
            reports=reports,
            _samples={name: tuple(samples[name]) for name in names},
            recommended_weights=recommended_weights,
            recommended_log_loss=recommended_log_loss,
        )