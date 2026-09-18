"""Predicción de un partido y evaluación de predicciones guardadas.

Servicio sobre una Session: llama al trainer (puro), construye las
features del partido target con la puerta de histórico y persiste cada
predicción con su snapshot en ``predictions``.
"""

from football_predictor.config.settings import get_settings
from football_predictor.database.models.match import Match
from football_predictor.database.repositories.match_history import MatchHistoryRepository
from football_predictor.database.repositories.prediction import PredictionRepository
from football_predictor.domain.entities import TargetMatch
from football_predictor.evaluation.metrics import EvaluationReport, evaluate_model
from football_predictor.models.base import PredictionResult, StatisticsPredictionResult
from football_predictor.models.statistics_model import StatisticsModel
from football_predictor.services.model_trainer import ALL_MODELS, ModelTrainer


class PredictionService:
    def __init__(self, session) -> None:
        self._session = session
        self._settings = get_settings()
        self._history = MatchHistoryRepository(session)
        self._trainer = ModelTrainer(self._history, self._settings)
        self._stats_model = StatisticsModel()

    def predict_match(
        self, match_id: int, model_name: str = "ensemble"
    ) -> tuple[dict[str, PredictionResult], StatisticsPredictionResult]:
        if model_name != "all" and model_name not in ALL_MODELS:
            raise ValueError(f"modelo desconocido: {model_name}. Válidos: 'all', poisson, elo, ml, ensemble")

        match = self._session.get(Match, match_id)
        if match is None:
            raise ValueError(f"no existe partido con id {match_id}")

        models = self._trainer.train_all()
        features = self._trainer.feature_builder().build_for_match(
            TargetMatch(match_id=match.id, date=match.date, home_team_id=match.home_team_id, away_team_id=match.away_team_id)
        )

        # Entrenar modelo de estadísticas con todo el histórico
        all_matches = self._history.get_all_matches()
        self._stats_model.fit(all_matches)
        stats_pred = self._stats_model.expected_counts(match.home_team_id, match.away_team_id)
        stats_result = StatisticsPredictionResult(
            match_id=match.id,
            home_yellow_cards=stats_pred["yellow_cards"][0],
            away_yellow_cards=stats_pred["yellow_cards"][1],
            home_red_cards=stats_pred["red_cards"][0],
            away_red_cards=stats_pred["red_cards"][1],
            home_corners=stats_pred["corners"][0],
            away_corners=stats_pred["corners"][1],
        )

        names = ALL_MODELS if model_name == "all" else (model_name,)
        repository = PredictionRepository(self._session)

        results: dict[str, PredictionResult] = {}
        for name in names:
            result = models[name].predict(features)
            repository.upsert(match_id=match.id, model_name=name, result=result)
            results[name] = result

        self._session.commit()
        return results, stats_result

    def evaluate(self, model_name: str):
        from football_predictor.database.repositories.prediction import PredictionRepository
        from football_predictor.evaluation.metrics import EvaluationReport, evaluate_model
        samples = PredictionRepository(self._session).for_evaluation(model_name)
        return evaluate_model(model_name, samples)