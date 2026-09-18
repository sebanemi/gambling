"""Provider football-data.org (v4).

Historico por competicion + temporada: ``GET /competitions/{code}/matches?season=``
con header ``X-Auth-Token``. Sobre el plan gratuito cada competicion tiene
limites de temporadas disponibles; los partidos sin marcador llegan con
``score.fullTime = null`` y se importan como ``scheduled``/``postponed``.
"""

from __future__ import annotations

from dataclasses import dataclass

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient
from football_predictor.data.providers.helpers import (
    derive_competitions,
    derive_teams,
    parse_iso_date,
)
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo

_STATUS_MAP = {
    "FINISHED": "finished",
    "SCHEDULED": "scheduled",
    "TIMED": "scheduled",
    "POSTPONED": "postponed",
    "CANCELLED": "cancelled",
    "SUSPENDED": "postponed",
    "AWARDED": "finished",
}

_TYPE_MAP = {"LEAGUE": "league", "CUP": "cup", "PLAYOFFS": "cup"}


class FootballDataOrgError(ValueError):
    """Competicion o temporada inválida para football-data.org."""


@dataclass
class FootballDataCompetition:
    code: str
    name: str
    country: str = ""
    type: str = "league"


@dataclass
class FootballDataOrgProvider:
    """Partidos de una competicion/temporada desde football-data.org."""

    client: ApiClient
    competition: str = "PL"
    season: str | None = None
    competition_name: str = ""
    competition_country: str = ""
    name_normalizer: NameNormalizer | None = None

    source: str = "football-data"
    name: str = "football-data.org"

    def __post_init__(self) -> None:
        self._normalizer = self.name_normalizer or NameNormalizer()

    def _matches_payload(self) -> dict:
        params: dict[str, str] = {}
        if self.season:
            params["season"] = self.season
        return self.client.get(f"/competitions/{self.competition}/matches", params=params)

    def get_matches(self) -> list[MatchRecord]:
        payload = self._matches_payload()
        competition = payload.get("competition", {})
        comp_name = self._competition_name(competition)
        country = self._country(competition)
        season_name = self._season_name(payload)

        records: list[MatchRecord] = []
        for raw in payload.get("matches", []):
            status = _STATUS_MAP.get(str(raw.get("status", "")), "scheduled")
            goals = raw.get("score", {}).get("fullTime") or {}
            home_goals = goals.get("home") if isinstance(goals, dict) else None
            away_goals = goals.get("away") if isinstance(goals, dict) else None
            utc_date = str(raw.get("utcDate", ""))
            if not utc_date:
                continue
            records.append(
                MatchRecord(
                    competition_name=self._normalizer(comp_name),
                    season_name=self._normalizer(season_name),
                    date=parse_iso_date(utc_date, field="utcDate"),
                    home_team=self._normalizer(raw["homeTeam"]["name"]),
                    away_team=self._normalizer(raw["awayTeam"]["name"]),
                    home_goals=home_goals,
                    away_goals=away_goals,
                    status=status,
                )
            )
        return records

    def get_teams(self) -> list[TeamInfo]:
        return derive_teams(self.get_matches())

    def get_competitions(self) -> list[CompetitionInfo]:
        payload = self._matches_payload()
        competition = payload.get("competition", {})
        if not competition:
            return derive_competitions(self.get_matches())
        return [
            CompetitionInfo(
                name=self._normalizer(self._competition_name(competition)),
                country=self._country(competition),
                type=_TYPE_MAP.get(str(competition.get("type", "")), "league"),
            )
        ]

    def available_competitions(self) -> list[FootballDataCompetition]:
        """Catalogo completo ``/competitions`` (depende del plan)."""
        payload = self.client.get("/competitions")
        result: list[FootballDataCompetition] = []
        for raw in payload.get("competitions", []):
            result.append(
                FootballDataCompetition(
                    code=str(raw.get("code", "")),
                    name=str(raw.get("name", "")),
                    country=str(raw.get("area", {}).get("name", "")),
                    type=_TYPE_MAP.get(str(raw.get("type", "")), "league"),
                )
            )
        return result

    def _competition_name(self, competition: dict) -> str:
        if self.competition_name:
            return self.competition_name
        return str(competition.get("name", self.competition))

    def _country(self, competition: dict) -> str:
        if self.competition_country:
            return self.competition_country
        return str(competition.get("area", {}).get("name", ""))

    def _season_name(self, payload: dict) -> str:
        if self.season:
            return self.season
        season = payload.get("season", {})
        start = str(season.get("startDate", ""))[:4]
        return start or str(season.get("id", ""))