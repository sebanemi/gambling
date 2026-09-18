"""FeatureStream: construye TODOS los FeatureVector en una sola pasada O(n).

Reemplaza el bucle ``build_for_match`` por partido (que era O(n²)-O(n³)):
procesa los partidos ordenados cronológicamente manteniendo estado
incremental (Elo, promedios, forma) y emite el ``FeatureVector`` de cada
partido ANTES de actualizar el estado con él.

Garantía anti-leakage idéntica a ``FeatureBuilder``: el stream agrupa por
fecha; todos los partidos de una misma fecha se construyen con el estado
de las fechas ESTRICTAMENTE anteriores, y solo después se actualiza el
estado con todo el grupo. Así ningún partido ve otro de su misma fecha.
"""

from collections import deque
from collections.abc import Sequence

from football_predictor.domain.entities import HistoricalMatch, TargetMatch
from football_predictor.domain.features import STATISTIC_METRICS, FeatureVector, TeamForm
from football_predictor.features.config import FeatureConfig
from football_predictor.features.elo import EloCalculator, result_score
from football_predictor.features.form import FormCalculator


class _WindowStats:
    """Suma corrida con ventana deslizante opcional (equivalente a ``[-window:]``)."""

    __slots__ = ("_count", "_dq", "_total", "_window")

    def __init__(self, window: int | None) -> None:
        self._window = int(window) if window and window > 0 else 0
        self._dq: deque[float] = deque()
        self._total = 0.0
        self._count = 0

    def add(self, value: float | None) -> None:
        current = float(value or 0.0)
        if self._window > 0:
            self._dq.append(current)
            self._total += current
            if len(self._dq) > self._window:
                self._total -= self._dq.popleft()
            self._count = len(self._dq)
        else:
            self._total += current
            self._count += 1

    def avg(self) -> float:
        return self._total / self._count if self._count else 0.0


class FeatureStream:
    """Estado incremental cronológico para construir features en O(n)."""

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        elo_calculator: EloCalculator | None = None,
    ) -> None:
        self._config = feature_config or FeatureConfig()
        self._elo = elo_calculator or EloCalculator()

        self._ratings: dict[int, float] = {}
        self._goals: dict[int, tuple[_WindowStats, _WindowStats]] = {}
        self._stats: dict[int, dict[str, dict[str, _WindowStats]]] = {}
        self._recent: dict[int, deque[HistoricalMatch]] = {}
        self._home_recent: dict[int, deque[HistoricalMatch]] = {}
        self._away_recent: dict[int, deque[HistoricalMatch]] = {}

    def build_all(self, matches: Sequence[HistoricalMatch]) -> list[FeatureVector]:
        """Construye un FeatureVector por partido, en el orden dado.

        ``matches`` debe venir ordenado por ``(date, match_id)`` (lo
        garantiza ``HistoryProvider``). Los partidos de una misma fecha se
        procesan con el estado de las fechas anteriores, no entre sí.
        """
        vectors: list[FeatureVector] = []
        ordered = sorted(matches, key=lambda m: (m.date, m.match_id))
        index = 0
        total = len(ordered)
        while index < total:
            group_end = index
            while group_end < total and ordered[group_end].date == ordered[index].date:
                group_end += 1
            for match in ordered[index:group_end]:
                vectors.append(self._build_vector(match))
            for match in ordered[index:group_end]:
                self._consume(match)
            index = group_end
        return vectors

    def build_all_keys(self, matches: Sequence[HistoricalMatch]) -> list[FeatureVector]:
        """Alias de :meth:`build_all` (equivalente a la firma del builder)."""
        return self.build_all(matches)

    def build_target(self, target: TargetMatch) -> FeatureVector:
        """Features de un target con el estado actual (partidos previos ya consumidos)."""
        return self._build_vector(target)

    def _team_recent(self, team_id: int) -> deque[HistoricalMatch]:
        if team_id not in self._recent:
            self._recent[team_id] = deque(maxlen=max(self._config.form_windows))
        return self._recent[team_id]

    def _team_home_recent(self, team_id: int) -> deque[HistoricalMatch]:
        if team_id not in self._home_recent:
            self._home_recent[team_id] = deque(maxlen=self._config.home_away_window)
        return self._home_recent[team_id]

    def _team_away_recent(self, team_id: int) -> deque[HistoricalMatch]:
        if team_id not in self._away_recent:
            self._away_recent[team_id] = deque(maxlen=self._config.home_away_window)
        return self._away_recent[team_id]

    def _team_goals(self, team_id: int) -> tuple[_WindowStats, _WindowStats]:
        stats = self._goals.get(team_id)
        if stats is None:
            stats = (_WindowStats(self._config.goals_window), _WindowStats(self._config.goals_window))
            self._goals[team_id] = stats
        return stats

    def _team_metric_stats(self, team_id: int) -> dict[str, dict[str, _WindowStats]]:
        stats = self._stats.get(team_id)
        if stats is None:
            stats = {
                metric: {
                    "for": _WindowStats(self._config.stats_window),
                    "against": _WindowStats(self._config.stats_window),
                }
                for metric in STATISTIC_METRICS
            }
            self._stats[team_id] = stats
        return stats

    def _build_vector(self, match: TargetMatch | HistoricalMatch) -> FeatureVector:
        home_id = match.home_team_id
        away_id = match.away_team_id
        initial = self._elo.initial_rating

        home_elo = self._ratings.get(home_id, initial)
        away_elo = self._ratings.get(away_id, initial)

        home_goals = self._team_goals(home_id)
        away_goals = self._team_goals(away_id)

        home_stats = self._team_metric_stats(home_id)
        away_stats = self._team_metric_stats(away_id)

        stat_fields: dict[str, float] = {}
        for metric in STATISTIC_METRICS:
            stat_fields[f"home_{metric}_for_avg"] = home_stats[metric]["for"].avg()
            stat_fields[f"home_{metric}_against_avg"] = home_stats[metric]["against"].avg()
            stat_fields[f"away_{metric}_for_avg"] = away_stats[metric]["for"].avg()
            stat_fields[f"away_{metric}_against_avg"] = away_stats[metric]["against"].avg()

        return FeatureVector(
            match_id=match.match_id,
            home_team_id=home_id,
            away_team_id=away_id,
            date=match.date,
            home_elo=home_elo,
            away_elo=away_elo,
            elo_difference=home_elo - away_elo,
            home_form=self._form_for(home_id),
            away_form=self._form_for(away_id),
            home_goals_for_avg=home_goals[0].avg(),
            home_goals_against_avg=home_goals[1].avg(),
            away_goals_for_avg=away_goals[0].avg(),
            away_goals_against_avg=away_goals[1].avg(),
            home_advantage=self._elo.home_advantage,
            **stat_fields,
        )

    def _form_for(self, team_id: int) -> TeamForm:
        recent = list(self._recent.get(team_id, ()))
        form = TeamForm()
        for window in self._config.form_windows:
            setattr(form, f"overall_{window}", FormCalculator.window_stats(recent[-window:], team_id, window))
        context = self._config.home_away_window
        setattr(
            form,
            f"home_{context}",
            FormCalculator.window_stats(list(self._home_recent.get(team_id, ()))[-context:], team_id, context),
        )
        setattr(
            form,
            f"away_{context}",
            FormCalculator.window_stats(list(self._away_recent.get(team_id, ()))[-context:], team_id, context),
        )
        return form

    def _consume(self, match: HistoricalMatch) -> None:
        home_id = match.home_team_id
        away_id = match.away_team_id

        home_goal_stats = self._team_goals(home_id)
        home_goal_stats[0].add(match.home_goals)
        home_goal_stats[1].add(match.away_goals)
        away_goal_stats = self._team_goals(away_id)
        away_goal_stats[0].add(match.away_goals)
        away_goal_stats[1].add(match.home_goals)

        home_stats = self._team_metric_stats(home_id)
        away_stats = self._team_metric_stats(away_id)
        for metric in STATISTIC_METRICS:
            home_metric = getattr(match, f"home_{metric}", None)
            away_metric = getattr(match, f"away_{metric}", None)
            home_stats[metric]["for"].add(home_metric)
            home_stats[metric]["against"].add(away_metric)
            away_stats[metric]["for"].add(away_metric)
            away_stats[metric]["against"].add(home_metric)

        self._team_recent(home_id).append(match)
        self._team_recent(away_id).append(match)
        self._team_home_recent(home_id).append(match)
        self._team_away_recent(away_id).append(match)

        home_rating = self._ratings.get(home_id, self._elo.initial_rating)
        away_rating = self._ratings.get(away_id, self._elo.initial_rating)
        expected_home = self._elo.expected_score(home_rating + self._elo.home_advantage, away_rating)
        actual_home = result_score(match.home_goals, match.away_goals)
        self._ratings[home_id] = home_rating + self._elo.k_factor * (actual_home - expected_home)
        self._ratings[away_id] = away_rating + self._elo.k_factor * ((1.0 - actual_home) - (1.0 - expected_home))