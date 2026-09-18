from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.match import Match
from football_predictor.database.models.match_statistics import MatchStatistics
from football_predictor.domain.entities import HistoricalMatch


class MatchHistoryRepository:
    """Implementación de ``HistoryProvider`` contra PostgreSQL.

    ÚNICO punto de acceso al histórico para features y modelos. La
    condición estricta ``date < cutoff`` es la garantía central anti-leakage.
    Retorna una lista en orden cronológico (tiebreak por match id).
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_matches_before(self, cutoff: date) -> list[HistoricalMatch]:
        return self._query(Match.date < cutoff)

    def get_all_matches(self) -> list[HistoricalMatch]:
        return self._query()

    def _query(self, *conditions) -> list[HistoricalMatch]:
        stmt = select(
            Match.id,
            Match.date,
            Match.home_team_id,
            Match.away_team_id,
            Match.home_goals,
            Match.away_goals,
            MatchStatistics.home_yellow_cards,
            MatchStatistics.away_yellow_cards,
            MatchStatistics.home_red_cards,
            MatchStatistics.away_red_cards,
            MatchStatistics.home_corners,
            MatchStatistics.away_corners,
            MatchStatistics.home_shots,
            MatchStatistics.away_shots,
            MatchStatistics.home_shots_on_target,
            MatchStatistics.away_shots_on_target,
            MatchStatistics.home_fouls,
            MatchStatistics.away_fouls,
            MatchStatistics.home_throw_ins,
            MatchStatistics.away_throw_ins,
            MatchStatistics.home_penalties,
            MatchStatistics.away_penalties,
            MatchStatistics.home_xg,
            MatchStatistics.away_xg,
            MatchStatistics.home_possession,
            MatchStatistics.away_possession,
        ).join(
            MatchStatistics, MatchStatistics.match_id == Match.id, isouter=True
        ).where(
            Match.status == "finished",
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
            *conditions,
        ).order_by(Match.date, Match.id)

        rows = self._session.execute(stmt).all()

        return [
            HistoricalMatch(
                match_id=match_id,
                date=match_date,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_goals=home_goals,
                away_goals=away_goals,
                home_yellow_cards=home_yellow,
                away_yellow_cards=away_yellow,
                home_red_cards=home_red,
                away_red_cards=away_red,
                home_corners=home_corners,
                away_corners=away_corners,
                home_shots=home_shots,
                away_shots=away_shots,
                home_shots_on_target=home_shots_on_target,
                away_shots_on_target=away_shots_on_target,
                home_fouls=home_fouls,
                away_fouls=away_fouls,
                home_throw_ins=home_throw_ins,
                away_throw_ins=away_throw_ins,
                home_penalties=home_penalties,
                away_penalties=away_penalties,
                home_xg=home_xg,
                away_xg=away_xg,
                home_possession=home_possession,
                away_possession=away_possession,
            )
            for (
                match_id,
                match_date,
                home_team_id,
                away_team_id,
                home_goals,
                away_goals,
                home_yellow,
                away_yellow,
                home_red,
                away_red,
                home_corners,
                away_corners,
                home_shots,
                away_shots,
                home_shots_on_target,
                away_shots_on_target,
                home_fouls,
                away_fouls,
                home_throw_ins,
                away_throw_ins,
                home_penalties,
                away_penalties,
                home_xg,
                away_xg,
                home_possession,
                away_possession,
            ) in rows
        ]