"""Modelo Poisson de goles de cada equipo (§6-§8 de la especificación).

Cada equipo tiene un rating de ataque (``attack[team]``) y otro de
defensa (``defense[team]``); la ventaja de local es un parámetro global.
Los goles esperados salen de regresiones de Poisson independientes:

    λ_local  = exp(γ_local  + attack[local]  - defense[visitante] + advance)
    λ_visit  = exp(γ_visit  + attack[visitante] - defense[local])

``defense`` es la DEBILIDAD defensiva del rival: un valor alto aumenta
los goles del oponente.

Corrección de Dixon-Coles (1997): las independientes sobreestiman los
empates bajos y subestiman 1-0/0-1/1-1. Se introduce un factor
`tau(x,y,rho)` sobre esas celdas:

    tau(0,0) = 1;  tau(0,1) = tau(1,0) = 1 − rho;  tau(1,1) = 1 + rho

El parámetro ``rho`` (negativo típico en fútbol) se estima junto con el
resto por MLE, o se puede fijar (``rho=None`` ⇒ se estima; si se pasa un
valor, queda fijo). Parámetros previos: MLE con regularización ridge.
"""

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import minimize

from football_predictor.domain.entities import HistoricalMatch
from football_predictor.domain.features import FeatureVector
from football_predictor.models.base import PredictionResult

_MAX_GOALS = 15
_RHO_BOUND = 0.5


class PoissonModel:
    def __init__(
        self,
        regularization: float = 0.1,
        rho: float | None = None,
    ) -> None:
        self._regularization = float(regularization)
        self._rho_fixed = None if rho is None else float(rho)
        if self._rho_fixed is not None and not -_RHO_BOUND <= self._rho_fixed <= _RHO_BOUND:
            raise ValueError(f"rho debe estar en [{-_RHO_BOUND}, {_RHO_BOUND}]")
        self._gamma_home = 0.0
        self._gamma_away = 0.0
        self._home_advantage = 0.0
        self._attack: dict[int, float] = {}
        self._defense: dict[int, float] = {}
        self._rho_fitted = 0.0
        self._last_x: np.ndarray | None = None

    @property
    def rho(self) -> float:
        """Factor Dixon-Coles efectivo (fijo o estimado por MLE)."""
        return self._rho_fixed if self._rho_fixed is not None else self._rho_fitted

    @property
    def last_point(self) -> np.ndarray | None:
        """Último vector de parámetros (para warm-start en walk-forward)."""
        return self._last_x

    def fit(
        self, history: Sequence[HistoricalMatch], *, warm_start: np.ndarray | None = None
    ) -> None:
        """Estima parámetros por MLE sobre el histórico previo.

        ``warm_start`` (opcional) es el vector de parámetros de un fit
        anterior sobre un histórico casi igual: acelera la convergencia del
        backtest walk-forward (mismo óptimo, menos iteraciones).
        """
        if not history:
            raise ValueError("PoissonModel.fit requiere al menos un partido")

        teams = sorted({m.home_team_id for m in history} | {m.away_team_id for m in history})
        index = {team: i for i, team in enumerate(teams)}
        n_teams = len(teams)

        h_idx = np.array([index[m.home_team_id] for m in history], dtype=int)
        a_idx = np.array([index[m.away_team_id] for m in history], dtype=int)
        hg = np.array([m.home_goals for m in history], dtype=int)
        ag = np.array([m.away_goals for m in history], dtype=int)

        gamma_h_idx = 2 * n_teams
        gamma_a_idx = 2 * n_teams + 1
        adv_idx = 2 * n_teams + 2
        rho_idx = 2 * n_teams + 3

        # Conteos de las celdas corregidas por Dixon-Coles.
        n_01_10 = int(((hg == 1) & (ag == 0)).sum() + ((hg == 0) & (ag == 1)).sum())
        n_11 = int(((hg == 1) & (ag == 1)).sum())

        def to_params(z: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
            att = z[:n_teams]
            defense = z[n_teams : 2 * n_teams]
            return (z[gamma_h_idx], z[gamma_a_idx], z[adv_idx], att, defense)

        def objective_and_gradient(z: np.ndarray) -> tuple[float, np.ndarray]:
            gamma_h, gamma_a, adv, att, defense = to_params(z)
            rho_z = z[rho_idx]
            lam_home = np.exp(gamma_h + att[h_idx] + defense[a_idx] + adv)
            lam_away = np.exp(gamma_a + att[a_idx] + defense[h_idx])

            residuals_home = lam_home - hg
            residuals_away = lam_away - ag

            nll = (
                np.sum(lam_home - hg * np.log(lam_home))
                + np.sum(lam_away - ag * np.log(lam_away))
                + self._regularization * (np.sum(att ** 2) + np.sum(defense ** 2))
            )
            # Término Dixon-Coles: −Σ log tau(x,y,rho).
            if n_01_10:
                nll -= n_01_10 * math.log(1.0 - rho_z)
            if n_11:
                nll -= n_11 * math.log(1.0 + rho_z)

            jac = np.zeros_like(z)
            jac[gamma_h_idx] = residuals_home.sum()
            jac[gamma_a_idx] = residuals_away.sum()
            jac[adv_idx] = residuals_home.sum()
            np.add.at(jac, h_idx, residuals_home)
            np.add.at(jac, a_idx, residuals_away)
            np.add.at(jac, n_teams + a_idx, residuals_home)
            np.add.at(jac, n_teams + h_idx, residuals_away)
            jac[:n_teams] += 2 * self._regularization * att
            jac[n_teams : 2 * n_teams] += 2 * self._regularization * defense
            jac[rho_idx] = (n_01_10 / (1.0 - rho_z) - n_11 / (1.0 + rho_z)) if n_01_10 or n_11 else 0.0
            return float(nll), jac

        def nll(z: np.ndarray) -> float:
            return objective_and_gradient(z)[0]

        def grad(z: np.ndarray) -> np.ndarray:
            return objective_and_gradient(z)[1]

        size = 2 * n_teams + 4
        if warm_start is not None and warm_start.size == size:
            x0 = np.asarray(warm_start, dtype=float).copy()
        elif warm_start is not None and warm_start.size == size - 1:
            # Compatibilidad con vectores pre-Dixon-Coles (sin rho).
            x0 = np.append(np.asarray(warm_start, dtype=float), self.rho)
        else:
            x0 = np.zeros(size, dtype=float)
            x0[rho_idx] = self.rho

        rho_hat = self.rho
        bounds = [(None, None)] * size
        if self._rho_fixed is None:
            bounds[rho_idx] = (-_RHO_BOUND, _RHO_BOUND)
        else:
            bounds[rho_idx] = (rho_hat, rho_hat)

        result = minimize(
            nll,
            x0=x0,
            jac=grad,
            bounds=bounds,
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
        self._rho_fitted = float(result.x[rho_idx])
        self._last_x = result.x

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

        rho_z = self.rho
        if rho_z != 0.0:
            tau = np.ones_like(outer)
            tau[1, 0] = 1.0 - rho_z
            tau[0, 1] = 1.0 - rho_z
            tau[1, 1] = 1.0 + rho_z
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