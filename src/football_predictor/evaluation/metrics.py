"""Métricas de evaluación de predicciones (§17 de la especificación).

Objetivo: medir la CALIDAD del ranking de resultados (1X2), no la
magnitud. Dos métricas complementarias:

- ``ranking_loss``: 1 menos la probabilidad asignada al resultado real.
  Es estrictamente decreciente en la confianza correcta (ideal = 0) e
  insensible a la escala de las demás clases.
- ``top1_accuracy`` (pos1): fracción de partidos donde el resultado con
  mayor probabilidad coincide con el real. Complementa con el caso duro
  de empates poco probables.
"""

from dataclasses import dataclass

from football_predictor.models.base import Outcome, outcome_from_goals


@dataclass(frozen=True)
class EvaluationReport:
    model_name: str
    matches: int
    ranking_loss: float
    top1_accuracy: float
    correct_top1: int


def outcome_proba(probabilities: tuple[float, float, float], outcome: Outcome) -> float:
    return {
        Outcome.HOME: probabilities[0],
        Outcome.DRAW: probabilities[1],
        Outcome.AWAY: probabilities[2],
    }[outcome]


def ranking_loss(probabilities: tuple[float, float, float], home_goals: int, away_goals: int) -> float:
    proba = outcome_proba(probabilities, outcome_from_goals(home_goals, away_goals))
    return 1.0 - proba


def top1_correct(probabilities: tuple[float, float, float], home_goals: int, away_goals: int) -> bool:
    predicted = Outcome(probabilities.index(max(probabilities)))
    return predicted == outcome_from_goals(home_goals, away_goals)


def evaluate_model(
    model_name: str,
    samples: list[tuple[tuple[float, float, float], int, int]],
) -> EvaluationReport:
    """Agrega ranking_loss y top1 sobre las muestras (1X2, goles reales)."""
    if not samples:
        return EvaluationReport(model_name=model_name, matches=0, ranking_loss=0.0, top1_accuracy=0.0, correct_top1=0)

    losses = [ranking_loss(probabilities, home_goals, away_goals) for probabilities, home_goals, away_goals in samples]
    correct = sum(top1_correct(probabilities, home_goals, away_goals) for probabilities, home_goals, away_goals in samples)

    return EvaluationReport(
        model_name=model_name,
        matches=len(samples),
        ranking_loss=sum(losses) / len(losses),
        top1_accuracy=correct / len(samples),
        correct_top1=correct,
    )