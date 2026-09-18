"""Sistema Elo para equipos.

Es la implementación única usada por DOS consumidores distintos
(§13 de la especificación):

- Como **feature** (``elo_difference`` para ML), vía ``ratings_after``.
- Como **modelo independiente** (probará `features/elo` + `models/elo`),
  usando ``expected_score`` con el rating anterior al partido.

Regla de leakage §11: el rating usado para predecir un partido es el
resultado de procesar ÚNICAMENTE partidos anteriores (el que llama
garantiza que ``matches`` ya cumple ``date < cutoff``).
"""

from collections.abc import Sequence

from football_predictor.domain.entities import HistoricalMatch


def result_score(home_goals: int, away_goals: int) -> float:
    """Puntos resultantes desde la perspectiva local: 1.0 / 0.5 / 0.0."""
    if home_goals > away_goals:
        return 1.0
    if home_goals < away_goals:
        return 0.0
    return 0.5


class EloCalculator:
    """Elo cronológico con k-factor y ventaja de local configurables."""

    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 20.0, home_advantage: float = 60.0) -> None:
        self._initial = float(initial_rating)
        self._k = float(k_factor)
        self.home_advantage = float(home_advantage)

    @property
    def initial_rating(self) -> float:
        return self._initial

    @staticmethod
    def expected_score(own_rating: float, opponent_rating: float) -> float:
        """Probabilidad esperada (modelo logístico estándar, escala 400)."""
        return 1.0 / (1.0 + 10 ** ((opponent_rating - own_rating) / 400.0))

    def ratings_after(self, matches: Sequence[HistoricalMatch]) -> dict[int, float]:
        """Actualiza ratings EN ORDEN cronológico y devuelve el estado final.

        Si ``matches`` contiene todos los partidos con ``date < D``, el
        resultado es el rating disponible ANTES de cualquier partido en ``D``.
        """
        ratings: dict[int, float] = {}
        ordered = sorted(matches, key=lambda m: (m.date, m.match_id))

        for match in ordered:
            home_rating = ratings.get(match.home_team_id, self._initial)
            away_rating = ratings.get(match.away_team_id, self._initial)

            expected_home = self.expected_score(home_rating + self.home_advantage, away_rating)
            actual_home = result_score(match.home_goals, match.away_goals)

            ratings[match.home_team_id] = home_rating + self._k * (actual_home - expected_home)
            ratings[match.away_team_id] = away_rating + self._k * ((1.0 - actual_home) - (1.0 - expected_home))

        return ratings