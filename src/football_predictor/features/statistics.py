"""Promedios de estadísticas por equipo usando SÓLO partidos anteriores."""

from collections.abc import Sequence

from football_predictor.domain.entities import HistoricalMatch


class StatisticsAverageCalculator:
    """Calcula promedios de tarjetas y córners de un equipo sobre su histórico previo."""

    @staticmethod
    def averages_for(
        team_matches: Sequence[HistoricalMatch],
        team_id: int,
        window: int | None = None,
    ) -> dict[str, float]:
        """Retorna dict con promedios por estadística.

        ``window=None`` usa todo el historial; si ``window>0`` usa los
        últimos ``window`` partidos. Sin partidos → 0.0 para todo.
        """
        if window and window > 0:
            team_matches = team_matches[-window:]

        count = len(team_matches)
        if count == 0:
            return {
                "yellow_cards_for": 0.0,
                "yellow_cards_against": 0.0,
                "red_cards_for": 0.0,
                "red_cards_against": 0.0,
                "corners_for": 0.0,
                "corners_against": 0.0,
            }

        yellow_for = yellow_against = 0
        red_for = red_against = 0
        corners_for = corners_against = 0

        for match in team_matches:
            is_home = match.home_team_id == team_id
            if is_home:
                if match.home_yellow_cards is not None:
                    yellow_for += match.home_yellow_cards
                if match.away_yellow_cards is not None:
                    yellow_against += match.away_yellow_cards
                if match.home_red_cards is not None:
                    red_for += match.home_red_cards
                if match.away_red_cards is not None:
                    red_against += match.away_red_cards
                if match.home_corners is not None:
                    corners_for += match.home_corners
                if match.away_corners is not None:
                    corners_against += match.away_corners
            else:
                if match.away_yellow_cards is not None:
                    yellow_for += match.away_yellow_cards
                if match.home_yellow_cards is not None:
                    yellow_against += match.home_yellow_cards
                if match.away_red_cards is not None:
                    red_for += match.away_red_cards
                if match.home_red_cards is not None:
                    red_against += match.home_red_cards
                if match.away_corners is not None:
                    corners_for += match.away_corners
                if match.home_corners is not None:
                    corners_against += match.home_corners

        return {
            "yellow_cards_for": yellow_for / count,
            "yellow_cards_against": yellow_against / count,
            "red_cards_for": red_for / count,
            "red_cards_against": red_against / count,
            "corners_for": corners_for / count,
            "corners_against": corners_against / count,
        }