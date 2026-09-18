"""Provider SofaScore (API pública no oficial).

Endpoints:
- Torneos: ``/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events``
- Partido: ``/api/v1/event/{event_id}`` (stats, lineups, incidents incluidos)
- Stats: ``/api/v1/event/{event_id}/statistics``
- Alineaciones: ``/api/v1/event/{event_id}/lineups``
- Incidentes: ``/api/v1/event/{event_id}/incidents``

Cubre: Top 5 Europa, UCL, Libertadores, Liga Argentina, etc.
Rate limit: ~60 req/min. Sin auth requerida.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient, ProviderError
from football_predictor.data.providers.helpers import parse_iso_date
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo

# IDs de torneos principales en SofaScore
TOURNAMENT_IDS = {
    "premier-league": 17,
    "la-liga": 8,
    "bundesliga": 35,
    "serie-a": 23,
    "ligue-1": 34,
    "champions-league": 7,
    "copa-libertadores": 13,
    "liga-argentina": 325,
}


@dataclass
class SofaScoreProvider:
    """Partidos y estadísticas desde SofaScore."""

    client: ApiClient
    tournament: str = "la-liga"
    season: str | None = None
    name_normalizer: NameNormalizer | None = None

    source: str = "sofascore"
    name: str = "SofaScore"

    _matches: list[MatchRecord] | None = field(default=None, init=False)
    _season_id_cache: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._normalizer = self.name_normalizer or NameNormalizer()
        self._tournament_id = TOURNAMENT_IDS.get(self.tournament)
        if self._tournament_id is None:
            raise ProviderError(f"Torneo desconocido: {self.tournament}. Disponibles: {list(TOURNAMENT_IDS.keys())}")

    def get_matches(self) -> list[MatchRecord]:
        if self._matches is not None:
            return list(self._matches)
        if self._tournament_id is None:
            raise ProviderError("Torneo no configurado")

        season_id = self._resolve_season_id()
        events = self._fetch_events(season_id)
        records: list[MatchRecord] = []
        for ev in events:
            rec = self._to_record(ev)
            if rec is not None:
                records.append(rec)
        self._matches = records
        return records

    def get_teams(self) -> list[TeamInfo]:
        from football_predictor.data.providers.helpers import derive_teams
        return derive_teams(self.get_matches())

    def get_competitions(self) -> list[CompetitionInfo]:
        return [
            CompetitionInfo(
                name=self._normalizer(self.tournament.replace("-", " ").title()),
                country="",
                type="league",
            )
        ]

    def available_competitions(self) -> list[str]:
        return list(TOURNAMENT_IDS.keys())

    def _resolve_season_id(self) -> int:
        if self._season_id_cache is not None:
            return self._season_id_cache

        if self.season is None:
            # Sin season: usar la más reciente disponible
            seasons = self._fetch_seasons()
            if not seasons:
                raise ProviderError(f"No se encontraron temporadas para {self.tournament}")
            self._season_id_cache = seasons[0]["id"]
            return self._season_id_cache

        # Buscar temporada por año
        seasons = self._fetch_seasons()
        for s in seasons:
            year = str(s.get("year", ""))
            if year == self.season or year.startswith(self.season):
                self._season_id_cache = s["id"]
                return s["id"]

        # Fallback: más reciente
        self._season_id_cache = seasons[0]["id"] if seasons else 0
        return self._season_id_cache

    def _fetch_seasons(self) -> list[dict]:
        url = f"/api/v1/unique-tournament/{self._tournament_id}/seasons"
        try:
            payload = self.client.get(url)
            return payload.get("seasons", [])
        except ProviderError:
            return []

    def _fetch_events(self, season_id: int) -> list[dict]:
        url = f"/api/v1/unique-tournament/{self._tournament_id}/season/{season_id}/events"
        payload = self.client.get(url)
        return payload.get("events", [])

    def _to_record(self, event: dict) -> MatchRecord | None:
        if event.get("status", {}).get("type") != "finished":
            return None

        start_ts = event.get("startTimestamp")
        if not start_ts:
            return None
        from datetime import datetime, timezone
        match_date = datetime.fromtimestamp(start_ts, tz=timezone.utc).date()

        home = event.get("homeTeam", {}).get("name", "")
        away = event.get("awayTeam", {}).get("name", "")
        if not home or not away:
            return None

        home_score = event.get("homeScore", {}).get("current")
        away_score = event.get("awayScore", {}).get("current")

        # Obtener stats detalladas del evento
        stats = self._fetch_event_statistics(event.get("id"))
        incidents = self._fetch_event_incidents(event.get("id"))

        return MatchRecord(
            competition_name=self._normalizer(self.tournament.replace("-", " ").title()),
            season_name=self._normalizer(self.season or str(match_date.year)),
            date=match_date,
            home_team=self._normalizer(home),
            away_team=self._normalizer(away),
            home_goals=home_score,
            away_goals=away_score,
            status="finished",
            home_yellow_cards=stats.get("home_yellow_cards"),
            away_yellow_cards=stats.get("away_yellow_cards"),
            home_red_cards=stats.get("home_red_cards"),
            away_red_cards=stats.get("away_red_cards"),
            home_corners=stats.get("home_corners"),
            away_corners=stats.get("away_corners"),
            home_shots=stats.get("home_shots"),
            away_shots=stats.get("away_shots"),
            home_shots_on_target=stats.get("home_shots_on_target"),
            away_shots_on_target=stats.get("away_shots_on_target"),
            home_fouls=stats.get("home_fouls"),
            away_fouls=stats.get("away_fouls"),
            home_throw_ins=stats.get("home_throw_ins"),
            away_throw_ins=stats.get("away_throw_ins"),
            home_penalties=stats.get("home_penalties"),
            away_penalties=stats.get("away_penalties"),
        )

    def _fetch_event_statistics(self, event_id: int) -> dict:
        if not event_id:
            return {}
        try:
            payload = self.client.get(f"/api/v1/event/{event_id}/statistics")
            return self._parse_statistics(payload)
        except ProviderError:
            return {}

    def _parse_statistics(self, payload: dict) -> dict:
        """Extrae stats por equipo de la respuesta de statistics."""
        result = {}
        for group in payload.get("statistics", []):
            for item in group.get("groups", []):
                for stat in item.get("statisticsItems", []):
                    name = stat.get("name", "").lower()
                    home = stat.get("home")
                    away = stat.get("away")
                    if home is None or away is None:
                        continue
                    if "yellow card" in name:
                        result["home_yellow_cards"] = int(home)
                        result["away_yellow_cards"] = int(away)
                    elif "red card" in name:
                        result["home_red_cards"] = int(home)
                        result["away_red_cards"] = int(away)
                    elif "corner" in name:
                        result["home_corners"] = int(home)
                        result["away_corners"] = int(away)
                    elif "shot" in name and "on target" not in name and "blocked" not in name:
                        result["home_shots"] = int(home)
                        result["away_shots"] = int(away)
                    elif "shots on target" in name or "on target" in name:
                        result["home_shots_on_target"] = int(home)
                        result["away_shots_on_target"] = int(away)
                    elif "foul" in name:
                        result["home_fouls"] = int(home)
                        result["away_fouls"] = int(away)
                    elif "throw" in name or "throw-in" in name:
                        result["home_throw_ins"] = int(home)
                        result["away_throw_ins"] = int(away)
                    elif "penalty" in name and "saved" not in name and "won" not in name:
                        result["home_penalties"] = int(home)
                        result["away_penalties"] = int(away)
        return result

    def _fetch_event_incidents(self, event_id: int) -> list[dict]:
        if not event_id:
            return []
        try:
            payload = self.client.get(f"/api/v1/event/{event_id}/incidents")
            return payload.get("incidents", [])
        except ProviderError:
            return []