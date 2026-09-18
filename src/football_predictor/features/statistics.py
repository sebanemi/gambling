"""Promedios de estadísticas por equipo usando SÓLO partidos anteriores.

El catálogo ``STATISTIC_METRICS`` centraliza QUÉ métricas de recuento
calcular: las usan el calculador, el ``FeatureVector`` (flatten), el
``FeatureStream`` incremental y el ``StatisticsModel`` de predicción.
"""

from collections.abc import Sequence

from football_predictor.domain.entities import HistoricalMatch
from football_predictor.domain.features import STATISTIC_METRICS


class StatisticsAverageCalculator:
    """Calcula promedios de estadísticas de un equipo sobre su histórico previo."""

    @staticmethod
    def averages_for(
        team_matches: Sequence[HistoricalMatch],
        team_id: int,
        window: int | None = None,
    ) -> dict[str, float]:
        """Retorna dict con promedios por estadística.

        ``window=None`` usa todo el historial; si ``window>0`` usa los
        últimos ``window`` partidos. Sin partidos → 0.0 para todo.
        Las métricas ausentes (``None``) cuentan como 0 en la división
        por el total de partidos del equipo.
        """
        if window and window > 0:
            team_matches = team_matches[-window:]

        count = len(team_matches)
        result: dict[str, float] = {}
        for metric in STATISTIC_METRICS:
            result[f"{metric}_for"] = 0.0
            result[f"{metric}_against"] = 0.0
        if count == 0:
            return result

        sums = {f"{metric}_{side}": 0 for metric in STATISTIC_METRICS for side in ("for", "against")}

        for match in team_matches:
            is_home = match.home_team_id == team_id
            for metric in STATISTIC_METRICS:
                home_val = getattr(match, f"home_{metric}", None)
                away_val = getattr(match, f"away_{metric}", None)
                if is_home:
                    if home_val is not None:
                        sums[f"{metric}_for"] += home_val
                    if away_val is not None:
                        sums[f"{metric}_against"] += away_val
                else:
                    if away_val is not None:
                        sums[f"{metric}_for"] += away_val
                    if home_val is not None:
                        sums[f"{metric}_against"] += home_val

        return {key: value / count for key, value in sums.items()}