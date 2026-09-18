from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.match import Match
from football_predictor.database.models.match_statistics import MatchStatistics


class MatchStatisticsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(
        self,
        match_id: int,
        *,
        home_yellow_cards: int | None = None,
        away_yellow_cards: int | None = None,
        home_red_cards: int | None = None,
        away_red_cards: int | None = None,
        home_corners: int | None = None,
        away_corners: int | None = None,
        home_shots: int | None = None,
        home_shots_on_target: int | None = None,
        home_fouls: int | None = None,
        home_throw_ins: int | None = None,
        home_penalties: int | None = None,
        away_shots: int | None = None,
        away_shots_on_target: int | None = None,
        away_fouls: int | None = None,
        away_throw_ins: int | None = None,
        away_penalties: int | None = None,
    ) -> MatchStatistics:
        stats = self._session.scalar(select(MatchStatistics).where(MatchStatistics.match_id == match_id))
        if stats is None:
            stats = MatchStatistics(match_id=match_id)
            self._session.add(stats)

        stats.home_yellow_cards = home_yellow_cards
        stats.away_yellow_cards = away_yellow_cards
        stats.home_red_cards = home_red_cards
        stats.away_red_cards = away_red_cards
        stats.home_corners = home_corners
        stats.away_corners = away_corners
        stats.home_shots = home_shots
        stats.home_shots_on_target = home_shots_on_target
        stats.home_fouls = home_fouls
        stats.home_throw_ins = home_throw_ins
        stats.home_penalties = home_penalties
        stats.away_shots = away_shots
        stats.away_shots_on_target = away_shots_on_target
        stats.away_fouls = away_fouls
        stats.away_throw_ins = away_throw_ins
        stats.away_penalties = away_penalties

        return stats

    def get_for_match(self, match_id: int) -> MatchStatistics | None:
        return self._session.scalar(select(MatchStatistics).where(MatchStatistics.match_id == match_id))

    def get_stats_for_matches_before(self, cutoff: date) -> dict[int, MatchStatistics]:
        stmt = select(MatchStatistics).join(Match).where(
            Match.status == "finished",
            Match.home_goals.is_not(None),
            Match.away_goals.is_not(None),
            Match.date < cutoff,
        )
        rows = self._session.execute(stmt).scalars().all()
        return {s.match_id: s for s in rows}