from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.match import Match
from football_predictor.database.models.prediction import Prediction
from football_predictor.models.base import PredictionResult


class PredictionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, *, match_id: int, model_name: str, result: PredictionResult) -> None:
        """Graba o actualiza la predicción (match_id, model_name) única."""
        prediction = self._session.scalar(
            select(Prediction).where(
                Prediction.match_id == match_id,
                Prediction.model_name == model_name,
            )
        )
        if prediction is None:
            prediction = Prediction(match_id=match_id, model_name=model_name)
            self._session.add(prediction)

        prediction.home_win = result.home_win
        prediction.draw = result.draw
        prediction.away_win = result.away_win
        prediction.home_goals = result.home_goals
        prediction.away_goals = result.away_goals
        prediction.features_snapshot = result.features_snapshot

    def count(self, *, match_id: int | None = None, model_name: str | None = None) -> int:
        stmt = select(Prediction.id)
        if match_id is not None:
            stmt = stmt.where(Prediction.match_id == match_id)
        if model_name is not None:
            stmt = stmt.where(Prediction.model_name == model_name)
        return len(self._session.execute(stmt).all())

    def for_evaluation(self, model_name: str) -> list[tuple[tuple[float, float, float], int, int]]:
        """(probas 1X2, goles local, goles visitante) para partidos ya jugados."""
        rows = self._session.execute(
            select(
                Prediction.home_win,
                Prediction.draw,
                Prediction.away_win,
                Match.home_goals,
                Match.away_goals,
            )
            .join(Match, Prediction.match_id == Match.id)
            .where(
                Prediction.model_name == model_name,
                Match.status == "finished",
                Match.home_goals.is_not(None),
                Match.away_goals.is_not(None),
            )
        ).all()
        return [((home_win, draw, away_win), hg, ag) for home_win, draw, away_win, hg, ag in rows]