from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.database.models.competition import Competition


class CompetitionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def all_ids(self) -> dict[tuple[str, str], int]:
        """Mapa (name, country) -> id."""
        rows = self._session.execute(
            select(Competition.id, Competition.name, Competition.country)
        ).all()
        return {(name, country): comp_id for comp_id, name, country in rows}

    def create(self, *, name: str, country: str = "", type: str = "league") -> int:
        competition = Competition(name=name, country=country, type=type)
        self._session.add(competition)
        self._session.flush()
        return competition.id