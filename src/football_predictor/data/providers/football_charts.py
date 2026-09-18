"""Provider football-charts.com.

Historico por liga + temporada: ``GET /leagues/{league}/results/?season=``
(lista de partidos finalizados con marcador) y ``GET /leagues/`` para el
catalogo. Token Bearer opcional (gratis, 5k req/dia); la URL de base se
autodescribe en ``GET /api/v1/``.
"""

from __future__ import annotations

from dataclasses import dataclass

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient
from football_predictor.data.providers.helpers import (
    derive_competitions,
    derive_teams,
    parse_score,
    pick,
)
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo


@dataclass
class FootballChartsLeague:
    league: str
    name: str
    country: str = ""
    seasons: list[str] | None = None


@dataclass
class FootballChartsProvider:
    """Partidos finalizados de una liga/temporada desde football-charts.com."""

    client: ApiClient
    league_key: str = "premier"
    season: str | None = None
    name_normalizer: NameNormalizer | None = None

    source: str = "football-charts"
    name: str = "football-charts.com"

    def __post_init__(self) -> None:
        self._normalizer = self.name_normalizer or NameNormalizer()

    def get_matches(self) -> list[MatchRecord]:
        params: dict[str, str] = {}
        if self.season:
            params["season"] = self.season
        payload = self.client.get(f"/leagues/{self.league_key}/results/", params=params)
        raw_matches = payload.get("matches", [])
        if not isinstance(raw_matches, list):
            raise ValueError(
                "football-charts: respuesta de results sin lista 'matches': "
                f"{list(payload)[:5]}"
            )

        records: list[MatchRecord] = []
        for raw in raw_matches:
            date_value = str(raw.get("date", ""))
            if not date_value:
                continue
            home_goals, away_goals = parse_score(raw.get("score", "0:0"))
            records.append(
                MatchRecord(
                    competition_name=self._normalizer(self._league_name()),
                    season_name=self._normalizer(self._resolve_season(date_value)),
                    date=parse_iso_date(date_value, field="date"),
                    home_team=self._normalizer(str(raw.get("homeTeam", ""))),
                    away_team=self._normalizer(str(raw.get("awayTeam", ""))),
                    home_goals=home_goals,
                    away_goals=away_goals,
                    status="finished",
                )
            )
        return records

    def get_teams(self) -> list[TeamInfo]:
        return derive_teams(self.get_matches())

    def get_competitions(self) -> list[CompetitionInfo]:
        all_leagues = self._catalog()
        matched = [l for l in all_leagues if l.league == self.league_key]
        if matched:
            league = matched[0]
            return [
                CompetitionInfo(
                    name=self._normalizer(league.name), country=league.country
                )
            ]
        return derive_competitions(self.get_matches())

    def available_leagues(self) -> list[FootballChartsLeague]:
        """Catalogo completo con temporadas disponibles."""
        return self._catalog()

    def _catalog(self) -> list[FootballChartsLeague]:
        payload = self.client.get("/leagues/")
        leagues: list[FootballChartsLeague] = []
        for raw in payload.get("leagues", []):
            leagues.append(
                FootballChartsLeague(
                    league=str(raw.get("league", "")),
                    name=str(raw.get("name", "")),
                    country=str(raw.get("country", "")),
                    seasons=[str(s) for s in raw.get("seasons", [])],
                )
            )
        return leagues

    def _league_name(self) -> str:
        for league in self._catalog():
            if league.league == self.league_key:
                return league.name
        return self.league_key

    def _resolve_season(self, date_value: str) -> str:
        if self.season:
            return self.season
        return date_value[:4]