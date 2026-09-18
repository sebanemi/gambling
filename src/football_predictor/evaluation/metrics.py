"""Métricas de evaluación de predicciones (§17 de la especificación).

Objetivo: medir la CALIDAD de las probabilidades 1X2 (no solo el ranking).
Métricas:

- ``ranking_loss``: 1 · P(resultado real). Ideal 0; mide ranking.
- ``log_loss``: −ln P(resultado real). Métrica estrictamente propia de
  calibración (penaliza sobreconfianza), en nats.
- ``brier``: error cuadrático medio multiclase entre probabilidades y el
  one-hot real (0 a 2; ideal 0). Mide calibración y resolución.
- ``top1_accuracy`` (pos1): fracción donde el argmax acierta.
"""

import math
from dataclasses import dataclass

from football_predictor.models.base import Outcome, outcome_from_goals

_EPS = 1e-9


@dataclass(frozen=True)
class EvaluationReport:
    model_name: str
    matches: int
    ranking_loss: float
    top1_accuracy: float
    correct_top1: int
    log_loss: float = 0.0
    brier: float = 0.0


def outcome_proba(probabilities: tuple[float, float, float], outcome: Outcome) -> float:
    return {
        Outcome.HOME: probabilities[0],
        Outcome.DRAW: probabilities[1],
        Outcome.AWAY: probabilities[2],
    }[outcome]


def ranking_loss(probabilities: tuple[float, float, float], home_goals: int, away_goals: int) -> float:
    proba = outcome_proba(probabilities, outcome_from_goals(home_goals, away_goals))
    return 1.0 - proba


def log_loss(
    probabilities: tuple[float, float, float], home_goals: int, away_goals: int
) -> float:
    """Log-loss del resultado real (−ln P(actual)); 0 = perfecto, +inf = peor."""
    proba = outcome_proba(probabilities, outcome_from_goals(home_goals, away_goals))
    return -math.log(max(proba, _EPS))


def brier(
    probabilities: tuple[float, float, float], home_goals: int, away_goals: int
) -> float:
    """Brier multiclase (media cuadrática sobre las 3 clases). 0 = perfecto."""
    actual = outcome_from_goals(home_goals, away_goals)
    return sum((probabilities[k] - (1.0 if Outcome(k) == actual else 0.0)) ** 2 for k in range(3))


def top1_correct(probabilities: tuple[float, float, float], home_goals: int, away_goals: int) -> bool:
    predicted = Outcome(probabilities.index(max(probabilities)))
    return predicted == outcome_from_goals(home_goals, away_goals)


def evaluate_model(
    model_name: str,
    samples: list[tuple[tuple[float, float, float], int, int]],
) -> EvaluationReport:
    """Agrega ranking_loss, log_loss, brier y top1 sobre las muestras."""
    if not samples:
        return EvaluationReport(
            model_name=model_name,
            matches=0,
            ranking_loss=0.0,
            top1_accuracy=0.0,
            correct_top1=0,
            log_loss=0.0,
            brier=0.0,
        )

    losses = [ranking_loss(p, hg, ag) for p, hg, ag in samples]
    lls = [log_loss(p, hg, ag) for p, hg, ag in samples]
    briers = [brier(p, hg, ag) for p, hg, ag in samples]
    correct = sum(top1_correct(p, hg, ag) for p, hg, ag in samples)

    return EvaluationReport(
        model_name=model_name,
        matches=len(samples),
        ranking_loss=sum(losses) / len(losses),
        top1_accuracy=correct / len(samples),
        correct_top1=correct,
        log_loss=sum(lls) / len(lls),
        brier=sum(briers) / len(briers),
    )