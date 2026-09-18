from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.season import Season


class SeasonRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def all_ids(self) -> dict[tuple[int, str], int]:
        """Mapa (competition_id, name) -> id."""
        rows = self._session.execute(
            select(Season.id, Season.competition_id, Season.name)
        ).all()
        return {(competition_id, name): season_id for season_id, competition_id, name in rows}

    def create(self, *, competition_id: int, name: str) -> int:
        season = Season(competition_id=competition_id, name=name)
        self._session.add(season)
        self._session.flush()
        return season.id

    def update_dates(self, *, season_id: int, start_date: date, end_date: date) -> None:
        season = self._session.get(Season, season_id)
        if season is not None:
            season.start_date = start_date
            season.end_date = end_date