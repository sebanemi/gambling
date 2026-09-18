"""Promedios de goles a favor / en contra usando SÓLO partidos anteriores."""

from collections.abc import Sequence

from football_predictor.domain.entities import HistoricalMatch


class GoalsAverageCalculator:
    """Calcula promedios de goles de un equipo sobre su histórico previo."""

    @staticmethod
    def averages_for(
        team_matches: Sequence[HistoricalMatch],
        team_id: int,
        window: int | None = None,
    ) -> tuple[float, float]:
        """Retorna ``(goals_for_avg, goals_against_avg)``.

        ``window=None`` usa todo el historial; si ``window>0`` usa los
        últimos ``window`` partidos. Sin partidos → (0.0, 0.0) (sin div/0).
        """
        if window and window > 0:
            team_matches = team_matches[-window:]

        count = len(team_matches)
        if count == 0:
            return 0.0, 0.0

        goals_for = 0
        goals_against = 0
        for match in team_matches:
            if match.home_team_id == team_id:
                goals_for += match.home_goals
                goals_against += match.away_goals
            else:
                goals_for += match.away_goals
                goals_against += match.home_goals

        return goals_for / count, goals_against / count