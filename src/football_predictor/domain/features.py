"""FeatureVector: representación tipada de las features de un partido.

Es puro (Pydantic, sin BD). ``flatten()`` produce el diccionario de
features numéricas usado por el modelo de ML; también es lo que se
persiste como snapshots auditables en ``predictions.features_snapshot``.
"""

from datetime import date

from pydantic import BaseModel


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
            "home_yellow_cards_for_avg": self.home_yellow_cards_for_avg,
            "home_yellow_cards_against_avg": self.home_yellow_cards_against_avg,
            "away_yellow_cards_for_avg": self.away_yellow_cards_for_avg,
            "away_yellow_cards_against_avg": self.away_yellow_cards_against_avg,
            "home_red_cards_for_avg": self.home_red_cards_for_avg,
            "home_red_cards_against_avg": self.home_red_cards_against_avg,
            "away_red_cards_for_avg": self.away_red_cards_for_avg,
            "away_red_cards_against_avg": self.away_red_cards_against_avg,
            "home_corners_for_avg": self.home_corners_for_avg,
            "home_corners_against_avg": self.home_corners_against_avg,
            "away_corners_for_avg": self.away_corners_for_avg,
            "away_corners_against_avg": self.away_corners_against_avg,
            "home_advantage": self.home_advantage,
        }
        for side, form in (("home", self.home_form), ("away", self.away_form)):
            for window in (3, 5, 10):
                window_form = getattr(form, f"overall_{window}")
                for field in (
                    "played",
                    "wins",
                    "draws",
                    "losses",
                    "points_per_game",
                    "goals_for",
                    "goals_against",
                    "goal_difference",
                ):
                    flat[f"{side}_form_{field}_{window}"] = float(getattr(window_form, field))
            for context in ("home", "away"):
                context_form = getattr(form, f"{context}_5")
                for field in (
                    "played",
                    "wins",
                    "draws",
                    "losses",
                    "points_per_game",
                    "goals_for",
                    "goals_against",
                    "goal_difference",
                ):
                    flat[f"{side}_form_{context}_{field}_5"] = float(getattr(context_form, field))
        return flat