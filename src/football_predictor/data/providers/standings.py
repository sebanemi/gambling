"""Provider azharimm/football-standings-api (datos ESPN, sin clave).

Expone competiciones y equipos (tablas por temporada); NO tiene endpoint
de partidos, asi que ``get_matches`` devuelve lista vacía. Se usa para
poblar el catalogo de competiciones/equipos de la BD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient, ProviderError
from football_predictor.data.providers.helpers import derive_competitions, derive_teams
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo


@dataclass
class StandingLeague:
    id: str
    name: str
    abbr: str = ""
    slug: str = ""


@dataclass
class FootballStandingsProvider:
    """Competiciones/equipos desde football-standings-api (ESPN)."""

    client: ApiClient
    league_id: str | None = None
    season: str | None = None
    name_normalizer: NameNormalizer | None = None

    source: str = "standings"
    name: str = "football-standings-api"

    _leagues: list[StandingLeague] | None = field(default=None, init=False)

    def get_matches(self) -> list[MatchRecord]:
        """Esta fuente no dispone de partidos historicos."""
        return []

    def get_teams(self) -> list[TeamInfo]:
        if self.league_id is None:
            return []
        params = {"sort": "asc"}
        if self.season:
            params["season"] = self.season
        payload = self.client.get(
            f"/leagues/{self.league_id}/standings", params=params
        )
        standings = payload.get("data", {}).get("standings", [])
        teams: dict[str, TeamInfo] = {}
        for row in standings:
            team = row.get("team", {})
            name = str(team.get("displayName", "") or team.get("name", ""))
            if name:
                teams.setdefault(
                    self._normalizer(name), TeamInfo(name=self._normalizer(name))
                )
        return list(teams.values())

    def get_competitions(self) -> list[CompetitionInfo]:
        if self.league_id is not None:
            league = self._find_league(self.league_id)
            if league is not None:
                return [CompetitionInfo(name=self._normalizer(league.name))]
        return [
            CompetitionInfo(name=self._normalizer(league.name))
            for league in self._catalog()
        ]

    def available_leagues(self) -> list[StandingLeague]:
        return self._catalog()

    def _catalog(self) -> list[StandingLeague]:
        if self._leagues is not None:
            return self._leagues
        payload = self.client.get("/leagues")
        data = payload.get("data", [])
        leagues: list[StandingLeague] = []
        for raw in data:
            leagues.append(
                StandingLeague(
                    id=str(raw.get("id", "")),
                    name=str(raw.get("name", "")),
                    abbr=str(raw.get("abbr", "")),
                    slug=str(raw.get("slug", "")),
                )
            )
        self._leagues = leagues
        return leagues

    def _find_league(self, league_id: str) -> StandingLeague | None:
        for league in self._catalog():
            if league.id == league_id:
                return league
        raise ProviderError(
            f"football-standings-api: liga {league_id!r} no encontrada "
            "(usar --list para ver los ids disponibles)"
        )