"""Entidades de dominio puras: sin SQLAlchemy, sin I/O.

Representan información intercambiada entre proveedores, importadores,
features y modelos. Nombres canónicos (normalizados) para equipos y
competiciones.
"""

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CompetitionInfo:
    name: str
    country: str = ""
    type: str = "league"


@dataclass(frozen=True)
class TeamInfo:
    name: str
    country: str = ""
    source: str = ""


@dataclass(frozen=True)
class MatchRecord:
    """Un partido con ids resueltos por nombre (dominio), no por PK de BD."""

    competition_name: str
    season_name: str
    date: date
    home_team: str
    away_team: str
    home_goals: int | None = None
    away_goals: int | None = None
    status: str = "finished"
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_fouls: int | None = None
    away_fouls: int | None = None
    home_throw_ins: int | None = None
    away_throw_ins: int | None = None
    home_penalties: int | None = None
    away_penalties: int | None = None
    home_xg: float | None = None
    away_xg: float | None = None
    home_possession: float | None = None
    away_possession: float | None = None


@dataclass(frozen=True)
class HistoricalMatch:
    """Un partido ya disputado, con ids numéricos para features.

    Proviene únicamente de :class:`HistoryProvider` (nunca del target).
    ``home_goals``/``away_goals`` no son None: solo se usan partidos
    finalizados.
    """

    match_id: int
    date: date
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None
    home_shots: int | None = None
    away_shots: int | None = None
    home_shots_on_target: int | None = None
    away_shots_on_target: int | None = None
    home_fouls: int | None = None
    away_fouls: int | None = None
    home_throw_ins: int | None = None
    away_throw_ins: int | None = None
    home_penalties: int | None = None
    away_penalties: int | None = None
    home_xg: float | None = None
    away_xg: float | None = None
    home_possession: float | None = None
    away_possession: float | None = None
    home_yellow_cards: int | None = None
    away_yellow_cards: int | None = None
    home_red_cards: int | None = None
    away_red_cards: int | None = None
    home_corners: int | None = None
    away_corners: int | None = None


@dataclass(frozen=True)
class TargetMatch:
    """El partido a predecir. Todas sus features usan sólo partidos
    con ``date < target.date`` (regla anti-leakage)."""

    match_id: int
    date: date
    home_team_id: int
    away_team_id: int