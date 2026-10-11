"""Repository for match odds operations."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.match_odds import MatchOdds


class MatchOddsRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def upsert(self, match_id: int, bookmaker: str, home_odds: float, draw_odds: float, away_odds: float) -> MatchOdds:
        """Insert or update odds for a match/bookmaker combination."""
        existing = self._session.execute(
            select(MatchOdds).where(MatchOdds.match_id == match_id, MatchOdds.bookmaker == bookmaker)
        ).scalar_one_or_none()

        if existing:
            existing.home_win_odds = home_odds
            existing.draw_odds = draw_odds
            existing.away_win_odds = away_odds
            return existing

        odds = MatchOdds(
            match_id=match_id,
            bookmaker=bookmaker,
            home_win_odds=home_odds,
            draw_odds=draw_odds,
            away_win_odds=away_odds,
        )
        self._session.add(odds)
        self._session.flush()
        return odds

    def get_for_match(self, match_id: int, bookmaker: str | None = None) -> list[MatchOdds]:
        """Get odds for a match, optionally filtered by bookmaker."""
        query = select(MatchOdds).where(MatchOdds.match_id == match_id)
        if bookmaker:
            query = query.where(MatchOdds.bookmaker == bookmaker)
        return list(self._session.execute(query).scalars().all())

    def get_best_odds(self, match_id: int) -> dict[str, float]:
        """Get best odds across all bookmakers for a match.

        Returns:
            Dict with keys 'home', 'draw', 'away' and best decimal odds
        """
        odds_list = self.get_for_match(match_id)
        if not odds_list:
            return {"home": 0.0, "draw": 0.0, "away": 0.0}

        best = {
            "home": max(float(o.home_win_odds) for o in odds_list),
            "draw": max(float(o.draw_odds) for o in odds_list),
            "away": max(float(o.away_win_odds) for o in odds_list),
        }
        return best
