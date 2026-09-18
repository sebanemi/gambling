"""Provider 5dollarfootballapi.com.

Historico/live por liga: ``/v1/leagues/{id}/fixtures`` (paginado). La
ventana historica depende del plan (3 meses Free, 12 meses Pro, hasta
2014 en Ultra). Resultados en ``goals.home/away``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient, ProviderError
from football_predictor.data.providers.helpers import derive_teams, parse_iso_date
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo

_DEFAULT_MAX_PAGES = 500
_PER_PAGE = 100


@dataclass
class FiveDollarProvider:
    """Partidos (finalizados) de una liga desde 5dollarfootballapi."""

    client: ApiClient
    league_id: str | None = None
    season: str | None = None
    max_pages: int = _DEFAULT_MAX_PAGES
    name_normalizer: NameNormalizer | None = None

    source: str = "5dollar"
    name: str = "5dollarfootballapi"

    _matches: list[MatchRecord] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._normalizer = self.name_normalizer or NameNormalizer()

    def get_matches(self) -> list[MatchRecord]:
        if self._matches is not None:
            return list(self._matches)
        if self.league_id is None:
            raise ProviderError(
                "5dollarfootballapi: se requiere --competition (id de liga). "
                "Liste ligas con --list."
            )
        records: list[MatchRecord] = []
        fixtures = self._fetch_page(self.league_id, page=1)
        for fixture in fixtures:
            record = self._to_record(fixture)
            if record is not None:
                records.append(record)
        self._matches = records
        return records

    def get_teams(self) -> list[TeamInfo]:
        return derive_teams(self.get_matches())

    def get_competitions(self) -> list[CompetitionInfo]:
        if self.league_id is None:
            return self._all_leagues()
        for league in self._all_leagues():
            if str(league.id) == str(self.league_id):
                return [
                    CompetitionInfo(name=league.name, country=league.country)
                ]
        raise ProviderError(f"5dollarfootballapi: liga {self.league_id} no encontrada")

    def available_leagues(self) -> list[CompetitionInfo]:
        return self._all_leagues()

    def _all_leagues(self) -> list[CompetitionInfo]:
        leagues: list[CompetitionInfo] = []
        page = 1
        while True:
            payload = self.client.get("/leagues", params={"per_page": 100, "page": page})
            data = payload.get("data", [])
            for raw in data:
                country = (raw.get("country") or {}).get("name") or ""
                leagues.append(
                    CompetitionInfo(
                        name=self._normalizer(str(raw.get("name", ""))),
                        country=self._normalizer(country),
                    )
                )
            pagination = payload.get("pagination", {})
            if not data or not pagination.get("has_more"):
                break
            page += 1
        return leagues

    def _fetch_page(self, league_id: str, page: int) -> list[dict]:
        params: dict[str, object] = {
            "status": "finished",
            "per_page": _PER_PAGE,
            "page": page,
        }
        if self.season:
            params["season"] = self.season
        payload = self.client.get(f"/leagues/{league_id}/fixtures", params=params)
        return [f for f in payload.get("data", []) if isinstance(f, dict)]

    def _to_record(self, fixture: dict) -> MatchRecord | None:
        kickoff = str(fixture.get("kickoff_utc", ""))
        if not kickoff:
            return None
        match_date = parse_iso_date(kickoff, field="kickoff_utc")
        if self.season and not self._in_season(match_date.year, self.season):
            return None

        teams = fixture.get("teams", {}) or {}
        home = (teams.get("home") or {}).get("name", "")
        away = (teams.get("away") or {}).get("name", "")
        if not home or not away:
            return None

        goals = fixture.get("goals") or {}
        home_goals = goals.get("home") if isinstance(goals, dict) else None
        away_goals = goals.get("away") if isinstance(goals, dict) else None

        cards = fixture.get("cards") or {}
        home_yellow = cards.get("home", {}).get("yellow") if isinstance(cards, dict) else None
        away_yellow = cards.get("away", {}).get("yellow") if isinstance(cards, dict) else None
        home_red = cards.get("home", {}).get("red") if isinstance(cards, dict) else None
        away_red = cards.get("away", {}).get("red") if isinstance(cards, dict) else None

        corners = fixture.get("corners") or {}
        home_corners = corners.get("home") if isinstance(corners, dict) else None
        away_corners = corners.get("away") if isinstance(corners, dict) else None

        status = "finished" if home_goals is not None else "scheduled"
        league_name = (fixture.get("league") or {}).get("name", "")
        return MatchRecord(
            competition_name=self._normalizer(league_name),
            season_name=self._normalizer(self.season or str(match_date.year)),
            date=match_date,
            home_team=self._normalizer(str(home)),
            away_team=self._normalizer(str(away)),
            home_goals=home_goals,
            away_goals=away_goals,
            status=status,
            home_yellow_cards=home_yellow,
            away_yellow_cards=away_yellow,
            home_red_cards=home_red,
            away_red_cards=away_red,
            home_corners=home_corners,
            away_corners=away_corners,
        )

    @staticmethod
    def _in_season(year: int, season: str) -> bool:
        try:
            start = int(season[:4])
        except ValueError:
            return False
        return start <= year <= start + 1