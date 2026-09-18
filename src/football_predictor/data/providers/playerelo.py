"""Provider playerelo.football.

Ratings Elo de equipos (Team Elo), ligas y predicciones. NO expone
resultados historicos (``get_matches`` = lista vacía); aporta equipos con
su Elo externo, util para comparar/sembrar el modelo Elo interno.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient, ProviderError
from football_predictor.data.providers.helpers import derive_competitions, derive_teams
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo

_PAGE_SIZE = 100
_DEFAULT_MAX_PAGES = 200


@dataclass
class ClubElo:
    team_id: str
    name: str
    team_elo: float
    league_slug: str = ""
    league: str = ""
    country: str = ""


@dataclass
class PlayerEloProvider:
    """Clubs con Team Elo y ligas desde playerelo.football.

    Si se fija ``league`` (slug o nombre normalizado), ``get_teams`` solo
    devuelve los clubs de esa liga.
    """

    client: ApiClient
    league: str | None = None
    max_pages: int = _DEFAULT_MAX_PAGES
    name_normalizer: NameNormalizer | None = None

    source: str = "playerelo"
    name: str = "playerelo.football"

    _clubs: list[ClubElo] | None = field(default=None, init=False)

    def get_matches(self) -> list[MatchRecord]:
        """Esta fuente no dispone de partidos historicos."""
        return []

    def get_teams(self) -> list[TeamInfo]:
        clubs = self._all_clubs()
        teams: dict[str, TeamInfo] = {}
        for club in clubs:
            if self.league and not self._matches_league(club):
                continue
            name = self._normalizer(club.name)
            teams.setdefault(
                name,
                TeamInfo(name=name, country=self._normalizer(club.country), source=self.source),
            )
        return list(teams.values())

    def get_competitions(self) -> list[CompetitionInfo]:
        leagues = self._leagues()
        if self.league:
            return [l for l in leagues if self._matches_name(l.name, self.league)]
        return leagues

    def available_leagues(self) -> list[CompetitionInfo]:
        return self._leagues()

    def team_elo(self) -> dict[str, float]:
        """Mapa equipo normalizado -> Team Elo (última página barrida)."""
        result: dict[str, float] = {}
        for club in self._all_clubs():
            result[self._normalizer(club.name)] = club.team_elo
        return result

    def top_clubs(self, limit: int = 10) -> list[ClubElo]:
        return self._all_clubs()[:limit]

    def _all_clubs(self) -> list[ClubElo]:
        if self._clubs is not None:
            return self._clubs
        clubs: list[ClubElo] = []
        for page in range(1, self.max_pages + 1):
            payload = self.client.get(
                "/v1/clubs", params={"limit": _PAGE_SIZE, "offset": (page - 1) * _PAGE_SIZE}
            )
            batch = payload if isinstance(payload, list) else payload.get("data", [])
            if not isinstance(batch, list):
                raise ProviderError(
                    f"playerelo: /v1/clubs con formato desconocido ({type(payload).__name__})"
                )
            for raw in batch:
                clubs.append(self._to_club(raw))
            if len(batch) < _PAGE_SIZE:
                break
        self._clubs = clubs
        return clubs

    def _leagues(self) -> list[CompetitionInfo]:
        payload = self.client.get("/v1/leagues")
        items = payload if isinstance(payload, list) else payload.get("leagues", payload.get("data", []))
        if not isinstance(items, list):
            raise ProviderError(
                f"playerelo: /v1/leagues con formato desconocido ({type(payload).__name__})"
            )
        leagues: list[CompetitionInfo] = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            leagues.append(
                CompetitionInfo(
                    name=self._normalizer(str(raw.get("name", "") or raw.get("league_name", ""))),
                    country=self._normalizer(str(raw.get("country", ""))),
                )
            )
        return leagues

    def _to_club(self, raw: dict) -> ClubElo:
        return ClubElo(
            team_id=str(raw.get("team_id", "")),
            name=str(raw.get("name", "")),
            team_elo=float(raw.get("team_elo", 0.0) or 0.0),
            league_slug=str(raw.get("league_slug", "")),
            league=str(raw.get("league_name", "")),
            country=str(raw.get("country", "")),
        )

    def _matches_league(self, club: ClubElo) -> bool:
        return self._matches_name(club.league, self.league) or self._matches_name(
            club.league_slug, self.league
        )

    def _matches_name(self, candidate: str | None, expected: str) -> bool:
        if not candidate:
            return False
        return self._normalizer(candidate) == self._normalizer(expected)