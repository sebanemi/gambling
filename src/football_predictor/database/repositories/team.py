from sqlalchemy import select
from sqlalchemy.orm import Session

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.database.models.team import Team

_normalizer = NameNormalizer()


class TeamRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def all_ids(self) -> dict[tuple[str, str], int]:
        """Mapa (name, country) -> id para upsert en memoria."""
        rows = self._session.execute(select(Team.id, Team.name, Team.country)).all()
        return {(name, country): team_id for team_id, name, country in rows}

    def find_id(self, name: str, country: str = "") -> int | None:
        """Resuelve el id por nombre normalizado (para la CLI)."""
        candidate = _normalizer.normalize(name)
        return self._session.scalar(select(Team.id).where(Team.name == candidate))

    def create(self, *, name: str, country: str = "", source: str = "") -> int:
        team = Team(name=name, country=country, source=source)
        self._session.add(team)
        self._session.flush()
        return team.id