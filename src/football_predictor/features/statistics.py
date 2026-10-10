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
        
        Optimizado: reduce llamadas a getattr usando acceso directo a atributos
        y cache de team_id para evitar comparaciones repetidas.
        """
        if window and window > 0:
            team_matches = team_matches[-window:]

        count = len(team_matches)
        if count == 0:
            return {f"{metric}_{side}": 0.0 for metric in STATISTIC_METRICS for side in ("for", "against")}

        # Inicializar sums con 0
        sums = {f"{metric}_{side}": 0 for metric in STATISTIC_METRICS for side in ("for", "against")}

        for match in team_matches:
            is_home = match.home_team_id == team_id
            # Acceso directo a atributos específicos para evitar getattr overhead
            if is_home:
                # Home team
                if match.home_yellow_cards is not None:
                    sums["yellow_cards_for"] += match.home_yellow_cards
                if match.away_yellow_cards is not None:
                    sums["yellow_cards_against"] += match.away_yellow_cards
                if match.home_red_cards is not None:
                    sums["red_cards_for"] += match.home_red_cards
                if match.away_red_cards is not None:
                    sums["red_cards_against"] += match.away_red_cards
                if match.home_corners is not None:
                    sums["corners_for"] += match.home_corners
                if match.away_corners is not None:
                    sums["corners_against"] += match.away_corners
                if match.home_shots is not None:
                    sums["shots_for"] += match.home_shots
                if match.away_shots is not None:
                    sums["shots_against"] += match.away_shots
                if match.home_shots_on_target is not None:
                    sums["shots_on_target_for"] += match.home_shots_on_target
                if match.away_shots_on_target is not None:
                    sums["shots_on_target_against"] += match.away_shots_on_target
                if match.home_fouls is not None:
                    sums["fouls_for"] += match.home_fouls
                if match.away_fouls is not None:
                    sums["fouls_against"] += match.away_fouls
                if match.home_throw_ins is not None:
                    sums["throw_ins_for"] += match.home_throw_ins
                if match.away_throw_ins is not None:
                    sums["throw_ins_against"] += match.away_throw_ins
                if match.home_penalties is not None:
                    sums["penalties_for"] += match.home_penalties
                if match.away_penalties is not None:
                    sums["penalties_against"] += match.away_penalties
                if match.home_xg is not None:
                    sums["xg_for"] += match.home_xg
                if match.away_xg is not None:
                    sums["xg_against"] += match.away_xg
                if match.home_possession is not None:
                    sums["possession_for"] += match.home_possession
                if match.away_possession is not None:
                    sums["possession_against"] += match.away_possession
            else:
                # Away team
                if match.away_yellow_cards is not None:
                    sums["yellow_cards_for"] += match.away_yellow_cards
                if match.home_yellow_cards is not None:
                    sums["yellow_cards_against"] += match.home_yellow_cards
                if match.away_red_cards is not None:
                    sums["red_cards_for"] += match.away_red_cards
                if match.home_red_cards is not None:
                    sums["red_cards_against"] += match.home_red_cards
                if match.away_corners is not None:
                    sums["corners_for"] += match.away_corners
                if match.home_corners is not None:
                    sums["corners_against"] += match.home_corners
                if match.away_shots is not None:
                    sums["shots_for"] += match.away_shots
                if match.home_shots is not None:
                    sums["shots_against"] += match.home_shots
                if match.away_shots_on_target is not None:
                    sums["shots_on_target_for"] += match.away_shots_on_target
                if match.home_shots_on_target is not None:
                    sums["shots_on_target_against"] += match.home_shots_on_target
                if match.away_fouls is not None:
                    sums["fouls_for"] += match.away_fouls
                if match.home_fouls is not None:
                    sums["fouls_against"] += match.home_fouls
                if match.away_throw_ins is not None:
                    sums["throw_ins_for"] += match.away_throw_ins
                if match.home_throw_ins is not None:
                    sums["throw_ins_against"] += match.home_throw_ins
                if match.away_penalties is not None:
                    sums["penalties_for"] += match.away_penalties
                if match.home_penalties is not None:
                    sums["penalties_against"] += match.home_penalties
                if match.away_xg is not None:
                    sums["xg_for"] += match.away_xg
                if match.home_xg is not None:
                    sums["xg_against"] += match.home_xg
                if match.away_possession is not None:
                    sums["possession_for"] += match.away_possession
                if match.home_possession is not None:
                    sums["possession_against"] += match.home_possession

        return {key: value / count for key, value in sums.items()}