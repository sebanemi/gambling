"""Modelo Poisson de goles de cada equipo (§6-§8 de la especificación).

Cada equipo tiene un rating de ataque (``attack[team]``) y otro de
defensa (``defense[team]``); la ventaja de local es un parámetro global.
Los goles esperados salen de regresiones de Poisson independientes:

    λ_local  = exp(γ_local  + attack[local]  - defense[visitante] + advance)
    λ_visit  = exp(γ_visit  + attack[visitante] - defense[local])

``defense`` es la DEBILIDAD defensiva del rival: un valor alto aumenta
los goles del oponente. Los parámetros se estiman por máxima verosimilitud
(todos los partidos del histórico previo), con regularización ridge sobre
attack/defense para evitar valores extremos con pocos datos.

P(1), P(X), P(2) se derivan de la distribución conjunta de dos Poisson
independientes con corrección de empate opcional (factor de Dixon-Coles
diagonal, ``draw_correction=1.0`` lo desactiva).
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from football_predictor.domain.entities import HistoricalMatch
from football_predictor.domain.features import FeatureVector
from football_predictor.models.base import PredictionResult

_MAX_GOALS = 15


class PoissonModel:
    def __init__(
        self,
        regularization: float = 0.1,
        draw_correction: float = 1.0,
    ) -> None:
        self._regularization = float(regularization)
        self.draw_correction = float(draw_correction)
        self._gamma_home = 0.0
        self._gamma_away = 0.0
        self._home_advantage = 0.0
        self._attack: dict[int, float] = {}
        self._defense: dict[int, float] = {}

    def fit(self, history: Sequence[HistoricalMatch]) -> None:
        """Estima parámetros por MLE sobre el histórico previo."""
        if not history:
            raise ValueError("PoissonModel.fit requiere al menos un partido")

        teams = sorted({m.home_team_id for m in history} | {m.away_team_id for m in history})
        index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        h_idx = np.array([index[m.home_team_id] for m in history], dtype=int)
        a_idx = np.array([index[m.away_team_id] for m in history], dtype=int)
        hg = np.array([m.home_goals for m in history], dtype=float)
        ag = np.array([m.away_goals for m in history], dtype=float)

        def to_params(z: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
            att = z[:n_teams]
            defense = z[n_teams : 2 * n_teams]
            return (z[2 * n_teams], z[2 * n_teams + 1], z[2 * n_teams + 2], att, defense)

        def nll(z: np.ndarray) -> float:
            gamma_h, gamma_a, adv, att, defense = to_params(z)
            lam_home = np.exp(gamma_h + att[h_idx] + defense[a_idx] + adv)
            lam_away = np.exp(gamma_a + att[a_idx] + defense[h_idx])
            lik = np.sum(lam_home - hg * np.log(lam_home)) + np.sum(lam_away - ag * np.log(lam_away))
            ridge = self._regularization * (np.sum(att ** 2) + np.sum(defense ** 2))
            return float(lik + ridge)

        result = minimize(
            nll,
            x0=np.zeros(2 * n_teams + 3, dtype=float),
            method="L-BFGS-B",
            options={"maxiter": 2000, "maxfun": 200000},
        )
        if not result.success:
            raise RuntimeError(f"PoissonModel.fit no convergió: {result.message}")

        gamma_h, gamma_a, adv, att, defense = to_params(result.x)
        self._gamma_home = float(gamma_h)
        self._gamma_away = float(gamma_a)
        self._home_advantage = float(adv)
        self._attack = {team: float(att[i]) for team, i in index.items()}
        self._defense = {team: float(defense[i]) for team, i in index.items()}

    def expected_goals(self, home_team_id: int, away_team_id: int) -> tuple[float, float]:
        lam_home = math.exp(
            self._gamma_home
            + self._attack.get(home_team_id, 0.0)
            + self._defense.get(away_team_id, 0.0)
            + self._home_advantage
        )
        lam_away = math.exp(
            self._gamma_away
            + self._attack.get(away_team_id, 0.0)
            + self._defense.get(home_team_id, 0.0)
        )
        return lam_home, lam_away

    @staticmethod
    def _poisson_pmf(goles: int, lam: float) -> float:
        return math.exp(-lam) * lam ** goles / math.factorial(goles)

    def _joint_distribution(self, lam_home: float, lam_away: float) -> np.ndarray:
        pmf_home = [self._poisson_pmf(x, lam_home) for x in range(_MAX_GOALS)]
        pmf_away = [self._poisson_pmf(y, lam_away) for y in range(_MAX_GOALS)]
        outer = np.outer(pmf_home, pmf_away)

        if self.draw_correction != 1.0:
            tau = np.ones_like(outer)
            for i in range(min(_MAX_GOALS, 5)):
                tau[i, i] = self.draw_correction
            outer = outer * tau

        total = outer.sum()
        if total <= 0.0:
            raise ValueError("probabilidad conjunta nula en el modelo Poisson")
        return outer / total

    def predict(self, features: FeatureVector) -> PredictionResult:
        lam_home, lam_away = self.expected_goals(features.home_team_id, features.away_team_id)
        joint = self._joint_distribution(lam_home, lam_away)

        home_win = float(np.tril(joint, -1).sum())
        draw = float(np.trace(joint))
        away_win = float(np.triu(joint, 1).sum())

        return PredictionResult(
            match_id=features.match_id,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            home_goals=lam_home,
            away_goals=lam_away,
            features_snapshot=features.flatten(),
        )