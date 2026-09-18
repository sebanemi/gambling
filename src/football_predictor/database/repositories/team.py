from sqlalchemy import func, select
from sqlalchemy.orm import Session

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.database.models.match import Match
from football_predictor.database.models.team import Team

_normalizer = NameNormalizer()

# Términos decorativos del nombre de un club que no distinguen equipos
# (ej. "Brentford" vs "Brentford FC", "Chelsea" vs "Chelsea FC").
_TEAM_JUNK_TOKENS = frozenset(
    {"fc", "afc", "cf", "sc", "ac", "cc", "f.c.", "f c", "utd", "sad"}
)


def _fuzzy_key(name: str) -> str:
    """Clave canónica para resolver nombres con variantes tipográficas.

    Baja a minúsculas, normaliza diacríticos/espacios y descarta sufijos
    y prefijos decorativos (``FC``, ``AFC``, ``CF``, ...).
    """
    key = (
        _normalizer.normalize(name)
        .lower()
        .replace("&", " ")
        .replace(".", " ")
        .replace("-", " ")
    )
    return " ".join(token for token in key.split() if token not in _TEAM_JUNK_TOKENS)


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

    def resolve(self, name: str) -> int | None:
        """Resuelve tolerando variantes (``Brentford`` → ``Brentford FC``).

        Estrategia: compara claves canónicas (sin diacríticos, espacios
        colapsados ni sufijos tipo FC/AFC). Si hay varios candidatos
        (duplicados entre fuentes), elige el que tiene más partidos, que
        suele ser el equipo activo.
        """
        target = _fuzzy_key(name)
        if not target:
            return None

        rows = self._session.execute(
            select(
                Team.id,
                Team.name,
                func.count(Match.id).label("n_matches"),
            )
            .outerjoin(
                Match,
                (Match.home_team_id == Team.id) | (Match.away_team_id == Team.id),
            )
            .group_by(Team.id)
        ).all()

        best: tuple[int, int, int] | None = None  # (rank, -n_matches, id)
        for team_id, stored_name, n_matches in rows:
            stored_key = _fuzzy_key(stored_name)
            if stored_key == target:
                rank = 0
            elif len(target) >= 4 and (target in stored_key or stored_key in target):
                rank = 1
            else:
                continue
            if best is None or (rank, -n_matches) < best:
                best = (rank, -int(n_matches or 0), team_id)
        return best[2] if best else None

    def get_name(self, team_id: int) -> str | None:
        team = self._session.get(Team, team_id)
        return team.name if team is not None else None

    def search(self, text: str, limit: int = 6) -> list[str]:
        """Nombres almacenados que contienen `text` (para sugerencias)."""
        token = _normalizer.normalize(text)
        rows = self._session.execute(
            select(Team.name).where(Team.name.ilike(f"%{token}%")).limit(limit)
        ).all()
        return [row[0] for row in rows]

    def create(self, *, name: str, country: str = "", source: str = "") -> int:
        team = Team(name=name, country=country, source=source)
        self._session.add(team)
        self._session.flush()
        return team.id