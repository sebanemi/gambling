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
    """Optimizado: acceso directo por índice en lugar de dict lookup."""
    return probabilities[outcome.value]


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
    # Optimizado: precomputar one-hot vector
    one_hot = [1.0 if Outcome(k) == actual else 0.0 for k in range(3)]
    return sum((probabilities[k] - one_hot[k]) ** 2 for k in range(3))


def top1_correct(probabilities: tuple[float, float, float], home_goals: int, away_goals: int) -> bool:
    predicted = Outcome(probabilities.index(max(probabilities)))
    return predicted == outcome_from_goals(home_goals, away_goals)


def evaluate_model(
    model_name: str,
    samples: list[tuple[tuple[float, float, float], int, int]],
) -> EvaluationReport:
    """Agrega ranking_loss, log_loss, brier y top1 sobre las muestras.
    
    Optimizado: single pass sobre las muestras para calcular todas las métricas.
    """
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

    n = len(samples)
    total_ranking_loss = 0.0
    total_log_loss = 0.0
    total_brier = 0.0
    correct = 0

    # Single pass optimizado
    for probabilities, home_goals, away_goals in samples:
        outcome = outcome_from_goals(home_goals, away_goals)
        proba = outcome_proba(probabilities, outcome)
        
        total_ranking_loss += 1.0 - proba
        total_log_loss += -math.log(max(proba, _EPS))
        
        # Brier calculation
        one_hot = [1.0 if Outcome(k) == outcome else 0.0 for k in range(3)]
        total_brier += sum((probabilities[k] - one_hot[k]) ** 2 for k in range(3))
        
        # Top-1 accuracy
        if Outcome(probabilities.index(max(probabilities))) == outcome:
            correct += 1

    return EvaluationReport(
        model_name=model_name,
        matches=n,
        ranking_loss=total_ranking_loss / n,
        top1_accuracy=correct / n,
        correct_top1=correct,
        log_loss=total_log_loss / n,
        brier=total_brier / n,
    )