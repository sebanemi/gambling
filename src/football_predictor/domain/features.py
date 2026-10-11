"""FeatureVector: representación tipada de las features de un partido.

Es puro (Pydantic, sin BD). ``flatten()`` produce el diccionario de
features numéricas usado por el modelo de ML; también es lo que se
persiste como snapshots auditables en ``predictions.features_snapshot``.
"""

from datetime import date

from pydantic import BaseModel

STATISTIC_METRICS: frozenset[str] = frozenset({
    "yellow_cards",
    "red_cards",
    "corners",
    "shots",
    "shots_on_target",
    "fouls",
    "throw_ins",
    "penalties",
    "xg",
    "possession",
})

# Orden canónico y determinista de las métricas (el frozenset no garantiza
# orden entre procesos; ordenarlo fija el orden de columnas en todos lados).
SORTED_METRICS: tuple[str, ...] = tuple(sorted(STATISTIC_METRICS))

_FORM_FIELDS: tuple[str, ...] = (
    "played",
    "wins",
    "draws",
    "losses",
    "points_per_game",
    "goals_for",
    "goals_against",
    "goal_difference",
)

_COLUMNS_CACHE: tuple[str, ...] | None = None


def feature_columns() -> tuple[str, ...]:
    """Nombres de columna en el orden canónico de ``flatten()``/``to_array()``.

    Se calcula una sola vez por proceso (una única llamada a ``flatten()``
    sobre un template en vez de una por celda de matriz).
    """
    global _COLUMNS_CACHE
    if _COLUMNS_CACHE is None:
        _COLUMNS_CACHE = tuple(
            FeatureVector(
                match_id=0,
                home_team_id=0,
                away_team_id=0,
                date=date(2000, 1, 1),
                home_elo=0.0,
                away_elo=0.0,
                elo_difference=0.0,
                home_form=TeamForm(),
                away_form=TeamForm(),
                home_goals_for_avg=0.0,
                home_goals_against_avg=0.0,
                away_goals_for_avg=0.0,
                away_goals_against_avg=0.0,
                home_advantage=0.0,
            ).flatten()
        )
    return _COLUMNS_CACHE


def matrix_from_vectors(vectors: object) -> object:
    """Matriz (N x 128) sin dicts intermedios: un ``to_array()`` por fila.

    Reemplaza el patrón ``[[fv.flatten()[c] for c in cols] for fv in vecs]``
    que llamaba a ``flatten()`` N×128 veces (cada llamada construye un dict
    de 128 claves). Lazy-importa numpy para no contaminar el dominio.
    """
    import numpy as np

    rows = [fv.to_array() for fv in vectors]  # type: ignore[union-attr]
    return np.array(rows, dtype=float)


class FormWindow(BaseModel):
    """Estadísticas de forma en una ventana de ``played`` partidos."""

    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    points_per_game: float = 0.0
    goals_for: int = 0
    goals_against: int = 0
    goal_difference: int = 0


class TeamForm(BaseModel):
    """Forma de un equipo: general (3/5/10) y por contexto (local/visitante)."""

    overall_3: FormWindow = FormWindow()
    overall_5: FormWindow = FormWindow()
    overall_10: FormWindow = FormWindow()
    home_5: FormWindow = FormWindow()
    away_5: FormWindow = FormWindow()


class FeatureVector(BaseModel):
    """Features calculadas SÓLO con partidos anteriores al target."""

    match_id: int
    home_team_id: int
    away_team_id: int
    date: date

    home_elo: float
    away_elo: float
    elo_difference: float

    home_form: TeamForm
    away_form: TeamForm

    home_goals_for_avg: float
    home_goals_against_avg: float
    away_goals_for_avg: float
    away_goals_against_avg: float

    home_yellow_cards_for_avg: float = 0.0
    home_yellow_cards_against_avg: float = 0.0
    away_yellow_cards_for_avg: float = 0.0
    away_yellow_cards_against_avg: float = 0.0
    home_red_cards_for_avg: float = 0.0
    home_red_cards_against_avg: float = 0.0
    away_red_cards_for_avg: float = 0.0
    away_red_cards_against_avg: float = 0.0
    home_corners_for_avg: float = 0.0
    home_corners_against_avg: float = 0.0
    away_corners_for_avg: float = 0.0
    away_corners_against_avg: float = 0.0
    home_shots_for_avg: float = 0.0
    home_shots_against_avg: float = 0.0
    away_shots_for_avg: float = 0.0
    away_shots_against_avg: float = 0.0
    home_shots_on_target_for_avg: float = 0.0
    home_shots_on_target_against_avg: float = 0.0
    away_shots_on_target_for_avg: float = 0.0
    away_shots_on_target_against_avg: float = 0.0
    home_fouls_for_avg: float = 0.0
    home_fouls_against_avg: float = 0.0
    away_fouls_for_avg: float = 0.0
    away_fouls_against_avg: float = 0.0
    home_throw_ins_for_avg: float = 0.0
    home_throw_ins_against_avg: float = 0.0
    away_throw_ins_for_avg: float = 0.0
    away_throw_ins_against_avg: float = 0.0
    home_penalties_for_avg: float = 0.0
    home_penalties_against_avg: float = 0.0
    away_penalties_for_avg: float = 0.0
    away_penalties_against_avg: float = 0.0
    home_xg_for_avg: float = 0.0
    home_xg_against_avg: float = 0.0
    away_xg_for_avg: float = 0.0
    away_xg_against_avg: float = 0.0
    home_possession_for_avg: float = 0.0
    home_possession_against_avg: float = 0.0
    away_possession_for_avg: float = 0.0
    away_possession_against_avg: float = 0.0

    home_advantage: float

    def flatten(self) -> dict[str, float]:
        """Features numéricas planas (estables y serializables)."""
        flat: dict[str, float] = {
            "home_elo": self.home_elo,
            "away_elo": self.away_elo,
            "elo_difference": self.elo_difference,
            "home_goals_for_avg": self.home_goals_for_avg,
            "home_goals_against_avg": self.home_goals_against_avg,
            "away_goals_for_avg": self.away_goals_for_avg,
            "away_goals_against_avg": self.away_goals_against_avg,
            "home_advantage": self.home_advantage,
        }
        for metric in SORTED_METRICS:
            flat[f"home_{metric}_for_avg"] = getattr(self, f"home_{metric}_for_avg")
            flat[f"home_{metric}_against_avg"] = getattr(self, f"home_{metric}_against_avg")
            flat[f"away_{metric}_for_avg"] = getattr(self, f"away_{metric}_for_avg")
            flat[f"away_{metric}_against_avg"] = getattr(self, f"away_{metric}_against_avg")
        for side, form in (("home", self.home_form), ("away", self.away_form)):
            for window in (3, 5, 10):
                window_form = getattr(form, f"overall_{window}")
                for field in _FORM_FIELDS:
                    flat[f"{side}_form_{field}_{window}"] = float(getattr(window_form, field))
            for context in ("home", "away"):
                context_form = getattr(form, f"{context}_5")
                for field in _FORM_FIELDS:
                    flat[f"{side}_form_{context}_{field}_5"] = float(getattr(context_form, field))
        return flat

    def to_array(self) -> list[float]:
        """Fila numérica en el orden de :func:`feature_columns` (vía rápida).

        Mismo contenido que ``flatten().values()`` pero sin construir el
        dict intermedio: para matrices N×128 evita N×128 llamadas a
        ``flatten()``.
        """
        row: list[float] = [
            self.home_elo,
            self.away_elo,
            self.elo_difference,
            self.home_goals_for_avg,
            self.home_goals_against_avg,
            self.away_goals_for_avg,
            self.away_goals_against_avg,
            self.home_advantage,
        ]
        for metric in SORTED_METRICS:
            row.append(getattr(self, f"home_{metric}_for_avg"))
            row.append(getattr(self, f"home_{metric}_against_avg"))
            row.append(getattr(self, f"away_{metric}_for_avg"))
            row.append(getattr(self, f"away_{metric}_against_avg"))
        for form in (self.home_form, self.away_form):
            for window in (3, 5, 10):
                window_form = getattr(form, f"overall_{window}")
                for field in _FORM_FIELDS:
                    row.append(float(getattr(window_form, field)))
            for context in ("home", "away"):
                context_form = getattr(form, f"{context}_5")
                for field in _FORM_FIELDS:
                    row.append(float(getattr(context_form, field)))
        return row