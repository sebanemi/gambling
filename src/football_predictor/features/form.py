"""Forma reciente por equipo: general (3/5/10) y contextual (local/visitante)."""

from collections.abc import Sequence

from football_predictor.domain.entities import HistoricalMatch
from football_predictor.domain.features import FormWindow, TeamForm


class FormCalculator:
    """Calcula ventanas de forma a partir de partidos ya ordenados.

    ``team_matches`` debe contener sólo partidos ANTERIORES al momento de
    la predicción (lo garantiza el HistoryProvider).
    
    Optimizado: usa list comprehension para filtrar partidos y reduce overhead.
    """

    @classmethod
    def team_form(
        cls,
        team_matches: Sequence[HistoricalMatch],
        team_id: int,
        form_windows: tuple[int, ...],
        home_away_window: int,
    ) -> TeamForm:
        # Optimizado: usar list comprehension en lugar de loop for
        home_matches = [m for m in team_matches if m.home_team_id == team_id]
        away_matches = [m for m in team_matches if m.away_team_id == team_id]

        form = TeamForm()
        for window in form_windows:
            setattr(form, f"overall_{window}", cls.window_stats(team_matches, team_id, window))
        setattr(form, f"home_{home_away_window}", cls.window_stats(home_matches, team_id, home_away_window))
        setattr(form, f"away_{home_away_window}", cls.window_stats(away_matches, team_id, home_away_window))
        return form

    @staticmethod
    def window_stats(
        chronological: Sequence[HistoricalMatch],
        team_id: int,
        window: int,
    ) -> FormWindow:
        """Últimos ``window`` partidos (cola de una lista cronológica).
        
        Optimizado: single pass para calcular todas las estadísticas.
        """
        recent = chronological[-window:] if window > 0 else chronological
        stats = FormWindow(played=len(recent))

        # Single pass optimizado
        for match in recent:
            is_home = match.home_team_id == team_id
            goals_for = match.home_goals if is_home else match.away_goals
            goals_against = match.away_goals if is_home else match.home_goals
            
            if goals_for > goals_against:
                stats.wins += 1
            elif goals_for < goals_against:
                stats.losses += 1
            else:
                stats.draws += 1

            stats.goals_for += goals_for
            stats.goals_against += goals_against

        stats.goal_difference = stats.goals_for - stats.goals_against
        if stats.played > 0:
            stats.points_per_game = (stats.wins * 3 + stats.draws) / stats.played
        return stats