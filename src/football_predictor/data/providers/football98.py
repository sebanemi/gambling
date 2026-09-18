"""Provider football98 (RapidAPI, GiulianoCrescimbeni).

``/{championship}/results/`` devuelve los partidos de una competicion. La
respuesta no está formalmente documentada y puede variar, asi que el
parser es tolerante (varias variantes de campos) y falla con un mensaje
claro ante un esquema no reconocido. Requiere ``X-RapidAPI-Key``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from football_predictor.data.normalizers.names import NameNormalizer
from football_predictor.data.providers.base import ApiClient, ProviderError
from football_predictor.data.providers.helpers import (
    derive_competitions,
    derive_teams,
    parse_iso_date,
    parse_score,
    pick,
)
from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo

_DATE_KEYS = ("date", "date_utc", "utcDate", "match_date", "kick_off", "kickoff")
_HOME_KEYS = ("home", "homeTeam", "home_team", "home-team", "team_home", "home_team_name")
_AWAY_KEYS = ("away", "awayTeam", "away_team", "away-team", "team_away", "away_team_name")
_SCORE_KEYS = ("score", "ftscore", "ft_score", "result", "full_time", "goals")


@dataclass
class Football98Provider:
    """Partidos de un championship desde football98 (RapidAPI)."""

    client: ApiClient
    championship: str = "premierleague"
    season: str | None = None
    name_normalizer: NameNormalizer | None = None

    source: str = "football98"
    name: str = "football98"

    _matches: list[MatchRecord] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._normalizer = self.name_normalizer or NameNormalizer()

    def get_matches(self) -> list[MatchRecord]:
        if self._matches is not None:
            return list(self._matches)
        payload = self.client.get(f"/{self.championship}/results/")
        raw_matches = self._extract_matches(payload)
        records: list[MatchRecord] = []
        for index, raw in enumerate(raw_matches):
            record = self._to_record(raw, index)
            if record is not None:
                records.append(record)
        self._matches = records
        return records

    def get_teams(self) -> list[TeamInfo]:
        return derive_teams(self.get_matches())

    def get_competitions(self) -> list[CompetitionInfo]:
        return derive_competitions(self.get_matches())

    def available_competitions(self) -> list[str]:
        """Identificadores del endpoint ``/competitions/``."""
        payload = self.client.get("/competitions/")
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, dict):
            values = payload.get("competitions") or payload.get("data") or payload.get("championships")
            if isinstance(values, list):
                return [str(item) for item in values]
        raise ProviderError(
            f"football98: /competitions/ con formato desconocido ({type(payload).__name__})"
        )

    def _extract_matches(self, payload: Any) -> list[dict]:
        if isinstance(payload, list):
            return [m for m in payload if isinstance(m, dict)]
        if isinstance(payload, dict):
            for key in ("matches", "results", "data", self.championship):
                if isinstance(payload.get(key), list):
                    return [m for m in payload[key] if isinstance(m, dict)]
            if not payload:
                return []
        raise ProviderError(
            "football98: respuesta de /results/ con esquema no reconocido "
            f"({type(payload).__name__}). Revisar el formato actual de la API."
        )

    def _to_record(self, raw: dict, index: int) -> MatchRecord | None:
        date_value = pick(raw, _DATE_KEYS)
        if date_value is None:
            return None
        home = pick(raw, _HOME_KEYS)
        away = pick(raw, _AWAY_KEYS)
        if not home or not away:
            return None

        home_goals: int | None = None
        away_goals: int | None = None
        score_value = pick(raw, _SCORE_KEYS)
        if score_value:
            try:
                if isinstance(score_value, dict):
                    hg = pick(score_value, ("home", "home_goals", "homeTeam"))
                    ag = pick(score_value, ("away", "away_goals", "awayTeam"))
                    if hg is not None and ag is not None:
                        home_goals, away_goals = int(hg), int(ag)
                else:
                    home_goals, away_goals = parse_score(score_value)
            except ValueError:
                home_goals = away_goals = None

        season_name = self.season or str(date_value)[:4]
        return MatchRecord(
            competition_name=self._normalizer(self.championship),
            season_name=self._normalizer(season_name),
            date=parse_iso_date(str(date_value), field="date"),
            home_team=self._normalizer(str(home)),
            away_team=self._normalizer(str(away)),
            home_goals=home_goals,
            away_goals=away_goals,
            status="finished" if home_goals is not None else "scheduled",
        )