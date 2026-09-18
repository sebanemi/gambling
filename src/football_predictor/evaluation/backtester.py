"""Backtesting walk-forward (§18 de la especificación).

Para cada partido del histórico, en orden cronológico:

1. Entrena cada modelo SOLO con partidos estrictamente anteriores.
2. Construye las features del target (puerta por fecha, anti-leakage).
3. Predice y compara contra el resultado real.

Las predicciones de un partido N-ésimo SOLO pueden depender de los
partidos 1..N-1; esto se verifica en los tests con un escenario causal.
El reporte agrega ranking_loss y top1 por modelo sobre toda la caminata.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

from football_predictor.config.settings import Settings, get_settings
from football_predictor.domain.entities import TargetMatch
from football_predictor.domain.protocols import HistoryProvider
from football_predictor.evaluation.metrics import EvaluationReport, evaluate_model
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
    ModelTrainer,
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
        self._trainer = ModelTrainer(history_provider, self._settings)

    def run(self, model_names: Sequence[str] = _DefaultModelNames) -> BacktestReport:
        matches = sorted(self._provider.get_all_matches(), key=lambda m: (m.date, m.match_id))
        if not matches:
            raise ValueError("no hay partidos históricos para backtestear")

        names = tuple(name for name in model_names if name in _DefaultModelNames)
        if not names:
            raise ValueError("debes pedir al menos un modelo válido (poisson, elo, ml, ensemble)")

        weights = parse_ensemble_weights(self._settings.models_ensemble_weights)
        builder = self._trainer.feature_builder()
        samples: dict[str, list[tuple[int, PredictionSurrogate, int, int]]] = {name: [] for name in names}
        evaluated = 0

        for index, target_match in enumerate(matches):
            prior = matches[:index]
            if len(prior) < self._min_prior_matches:
                continue

            trained = self._train_step(prior, builder, weights)
            features = builder.build_for_match(
                TargetMatch(
                    match_id=target_match.match_id,
                    date=target_match.date,
                    home_team_id=target_match.home_team_id,
                    away_team_id=target_match.away_team_id,
                )
            )

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
        return BacktestReport(
            evaluated=evaluated,
            min_prior_matches=self._min_prior_matches,
            reports=reports,
            _samples={name: tuple(samples[name]) for name in names},
        )

    def _train_step(self, prior, builder, weights) -> dict[str, Predictor]:
        """Entrena cada modelo con ``prior``; los que no convergen se omiten."""
        s = self._settings
        trained: dict[str, Predictor] = {}

        poisson = PoissonModel(
            regularization=s.models_poisson_regularization,
            draw_correction=s.models_poisson_draw_correction,
        )
        try:
            poisson.fit(prior)
            trained[MODEL_POISSON] = poisson
        except (ValueError, RuntimeError):
            pass

        trained[MODEL_ELO] = EloModel(draw_max=s.models_elo_draw_max, draw_sigma=s.models_elo_draw_sigma)

        features = []
        outcomes = []
        for match in prior:
            features.append(
                builder.build_for_match(
                    TargetMatch(
                        match_id=match.match_id,
                        date=match.date,
                        home_team_id=match.home_team_id,
                        away_team_id=match.away_team_id,
                    )
                )
            )
            outcomes.append(outcome_from_goals(match.home_goals, match.away_goals))
        ml = MLModel(random_state=s.models_ml_random_state)
        try:
            ml.fit(features, outcomes)
            trained[MODEL_ML] = ml
        except (ValueError, RuntimeError):
            pass

        members = [trained[name] for name in (MODEL_POISSON, MODEL_ELO, MODEL_ML) if name in trained]
        if len(members) >= 2:
            trained[MODEL_ENSEMBLE] = EnsembleModel(members, weights[: len(members)])
        elif len(members) == 1:
            trained[MODEL_ENSEMBLE] = members[0]

        return trained