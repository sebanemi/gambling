"""Optimización de pesos del ensemble por máxima verosimilitud logística.

El problema es estrictamente convexo (log-sum-exp de una combinación
lineal con pesos en el simplex), así que SLSQP encuentra el óptimo
global: la mezcla convexa de las probabilidades de cada modelo que
minimiza el log-loss del resultado real.

Las muestras provienen del backtest walk-forward: cada probabilidad se
generó entrenando SOLO con el pasado del partido, por lo que los pesos
aprendidos son out-of-sample (sin leakage). Es la base de un stacking
de nivel 1 directo.
"""

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from football_predictor.models.base import Outcome, outcome_from_goals

# Orden canónico de los miembros del ensemble.
MEMBER_ORDER = ("poisson", "elo", "ml")

_MIN_SAMPLES = 50  # menos que esto ⇒ la estimación no es fiable

Sample = tuple[int, tuple[float, float, float], int, int]


def _aligned(
    samples: Mapping[str, Sequence[Sample]], canon: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Matrices (N, K, 3) de probabilidades + outcomes, alineadas por match."""
    by_match: dict[int, dict[str, tuple[tuple[float, float, float], int, int]]] = {}
    for model in canon:
        for match_id, prob, home_goals, away_goals in samples[model]:
            by_match.setdefault(match_id, {})[model] = (prob, home_goals, away_goals)

    rows: list[tuple[list[tuple[float, float, float]], Outcome]] = []
    for entry in by_match.values():
        if not all(model in entry for model in canon):
            continue
        prob, home_goals, away_goals = entry[canon[0]]
        rows.append(([entry[m][0] for m in canon], outcome_from_goals(home_goals, away_goals)))

    if not rows:
        return None
    prob_matrix = np.array([row[0] for row in rows], dtype=float)
    actual = np.array([int(row[1]) for row in rows], dtype=int)
    return prob_matrix, actual, len(rows)


def optimize_weights(
    samples: Mapping[str, Sequence[Sample]],
) -> tuple[dict[str, float], float] | None:
    """Devuelve (pesos óptimos por miembro, log-loss medio) o None si no alcanza."""
    canon = [name for name in MEMBER_ORDER if samples.get(name)]
    if len(canon) < 2:
        return None
    aligned = _aligned(samples, canon)
    if aligned is None:
        return None
    prob_matrix, actual, n = aligned
    if n < _MIN_SAMPLES:
        return None

    k = len(canon)

    def neg_log_likelihood(w: np.ndarray) -> float:
        mixed = np.einsum("k,nkc->nc", w, prob_matrix)  # (N, 3)
        picked = mixed[np.arange(n), actual]
        return -float(np.sum(np.log(np.maximum(picked, 1e-15))))

    constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    result = minimize(
        neg_log_likelihood,
        x0=np.full(k, 1.0 / k),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[constraint],
        options={"maxiter": 2000, "ftol": 1e-14},
    )
    if not result.success:
        return None

    weights = {canon[i]: float(w) for i, w in enumerate(result.x)}
    return weights, float(neg_log_likelihood(result.x) / n)


def mix_log_loss(
    samples: Mapping[str, Sequence[Sample]], weights: Mapping[str, float]
) -> float | None:
    """Log-loss medio de mezclar los modelos con ``weights`` (mismo alineado)."""
    canon = [name for name in weights if name in samples]
    if len(canon) < 2:
        return None
    aligned = _aligned(samples, canon)
    if aligned is None:
        return None
    prob_matrix, actual, n = aligned
    w = np.array([weights[name] for name in canon], dtype=float)
    w = w / w.sum()
    mixed = np.einsum("k,nkc->nc", w, prob_matrix)
    picked = mixed[np.arange(n), actual]
    return float(-np.sum(np.log(np.maximum(picked, 1e-15))) / n)