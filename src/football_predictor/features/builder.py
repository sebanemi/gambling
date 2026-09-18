"""FeatureBuilder: ensambla un :class:`FeatureVector` para un partido target.

Toda la información sale del :class:`HistoryProvider` (partidos con
``date < cutoff``). El builder agrega una segunda defensa: filtra de nuevo
``date < target.date`` por las dudas, de modo que un provedor defectuoso
no pueda inducir leakage.
"""

from collections.abc import Sequence
from datetime import date

from football_predictor.domain.entities import HistoricalMatch, TargetMatch
from football_predictor.domain.features import FeatureVector
from football_predictor.domain.protocols import HistoryProvider
from football_predictor.features.config import FeatureConfig
from football_predictor.features.elo import EloCalculator
from football_predictor.features.form import FormCalculator
from football_predictor.features.goals import GoalsAverageCalculator
from football_predictor.features.statistics import StatisticsAverageCalculator


class FeatureBuilder:
    """Construye features con ventanas temporales correctas vía snapshot."""

    def __init__(
        self,
        history_provider: HistoryProvider,
        feature_config: FeatureConfig | None = None,
        elo_calculator: EloCalculator | None = None,
    ) -> None:
        self._provider = history_provider
        self._config = feature_config or FeatureConfig()
        self._elo = elo_calculator or EloCalculator()
        self._snapshots: dict[date, tuple[HistoricalMatch, ...]] = {}

    def build_for_match(self, target: TargetMatch) -> FeatureVector:
        history = self._snapshot(target.date)

        home_team_matches = self._team_matches(history, target.home_team_id)
        away_team_matches = self._team_matches(history, target.away_team_id)

        elo_ratings = self._elo.ratings_after(history)
        home_elo = elo_ratings.get(target.home_team_id, self._elo.initial_rating)
        away_elo = elo_ratings.get(target.away_team_id, self._elo.initial_rating)

        home_form = FormCalculator.team_form(
            home_team_matches, target.home_team_id, self._config.form_windows, self._config.home_away_window
        )
        away_form = FormCalculator.team_form(
            away_team_matches, target.away_team_id, self._config.form_windows, self._config.home_away_window
        )

        home_for, home_against = GoalsAverageCalculator.averages_for(
            home_team_matches, target.home_team_id, self._config.goals_window
        )
        away_for, away_against = GoalsAverageCalculator.averages_for(
            away_team_matches, target.away_team_id, self._config.goals_window
        )

        home_stats = StatisticsAverageCalculator.averages_for(
            home_team_matches, target.home_team_id, self._config.stats_window
        )
        away_stats = StatisticsAverageCalculator.averages_for(
            away_team_matches, target.away_team_id, self._config.stats_window
        )

        return FeatureVector(
            match_id=target.match_id,
            home_team_id=target.home_team_id,
            away_team_id=target.away_team_id,
            date=target.date,
            home_elo=home_elo,
            away_elo=away_elo,
            elo_difference=home_elo - away_elo,
            home_form=home_form,
            away_form=away_form,
            home_goals_for_avg=home_for,
            home_goals_against_avg=home_against,
            away_goals_for_avg=away_for,
            away_goals_against_avg=away_against,
            home_yellow_cards_for_avg=home_stats["yellow_cards_for"],
            home_yellow_cards_against_avg=home_stats["yellow_cards_against"],
            away_yellow_cards_for_avg=away_stats["yellow_cards_for"],
            away_yellow_cards_against_avg=away_stats["yellow_cards_against"],
            home_red_cards_for_avg=home_stats["red_cards_for"],
            home_red_cards_against_avg=home_stats["red_cards_against"],
            away_red_cards_for_avg=away_stats["red_cards_for"],
            away_red_cards_against_avg=away_stats["red_cards_against"],
            home_corners_for_avg=home_stats["corners_for"],
            home_corners_against_avg=home_stats["corners_against"],
            away_corners_for_avg=away_stats["corners_for"],
            away_corners_against_avg=away_stats["corners_against"],
            home_advantage=self._elo.home_advantage,
        )

    def _snapshot(self, cutoff: date) -> tuple[HistoricalMatch, ...]:
        """Snapshot del histórico estrictamente anterior a ``cutoff`` (cache)."""
        cached = self._snapshots.get(cutoff)
        if cached is not None:
            return cached

        matches = list(self._provider.get_matches_before(cutoff))
        matches = [m for m in matches if m.date < cutoff]
        matches.sort(key=lambda m: (m.date, m.match_id))
        snapshot = tuple(matches)
        self._snapshots[cutoff] = snapshot
        return snapshot

    @staticmethod
    def _team_matches(
        history: Sequence[HistoricalMatch], team_id: int
    ) -> list[HistoricalMatch]:
        return [
            m for m in history if m.home_team_id == team_id or m.away_team_id == team_id
        ]