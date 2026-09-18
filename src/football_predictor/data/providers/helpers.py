"""Helpers compartidos para normalizar respuestas de APIs a entidades.

Los providers traducen JSON diverso a :class:`MatchRecord`; estos helpers
mantienen la tolerancia en nombres de campos sin silenciar errores.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from football_predictor.domain.entities import CompetitionInfo, MatchRecord, TeamInfo


def parse_iso_date(value: str, *, field: str = "fecha") -> date:
    """Fecha ISO-8601 (puede traer hora/offset), p.ej. ``2024-08-16`` o
    ``2024-08-16T19:00:00Z``."""
    try:
        raw = str(value).strip()
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"{field} inválida: {value!r}") from exc


def parse_score(value: Any, *, field: str = "score") -> tuple[int, int]:
    """``"2:1"`` -> ``(2, 1)``. Acepta también ``2-1``."""
    if isinstance(value, dict):
        home = value.get("home")
        away = value.get("away")
        if home is not None and away is not None:
            return int(home), int(away)
        raise ValueError(f"{field}: objeto de score sin home/away: {value!r}")
    parts = str(value).replace("−", "-").split(":")
    if len(parts) != 2:
        parts = str(value).replace("−", "-").split("-")
    if len(parts) != 2:
        raise ValueError(f"{field}: formato de score no reconocido: {value!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"{field}: score no numérico: {value!r}") from exc


def pick(obj: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Primera clave presente en ``obj`` (tolerante a variantes de campo)."""
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return default


def derive_teams(records: list[MatchRecord]) -> list[TeamInfo]:
    """Equipos únicos a partir de los partidos (misma lógica que el CSV)."""
    teams: dict[tuple[str, str], TeamInfo] = {}
    for record in records:
        for name in (record.home_team, record.away_team):
            teams.setdefault((name, ""), TeamInfo(name=name))
    return list(teams.values())


def derive_competitions(records: list[MatchRecord]) -> list[CompetitionInfo]:
    """Competiciones únicas a partir de los partidos."""
    competitions: dict[str, CompetitionInfo] = {}
    for record in records:
        competitions.setdefault(
            record.competition_name, CompetitionInfo(name=record.competition_name)
        )
    return list(competitions.values())