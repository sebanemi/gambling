"""Helpers compartidos para tests de features y modelos."""

from datetime import date

from football_predictor.domain.entities import HistoricalMatch, TargetMatch
from football_predictor.domain.features import FeatureVector, TeamForm


def day(day_number: int) -> date:
    return date(2026, 1, day_number)


def hm(match_id: int, day_number: int, home_id: int, away_id: int, hg: int, ag: int) -> HistoricalMatch:
    return HistoricalMatch(
        match_id=match_id,
        date=day(day_number),
        home_team_id=home_id,
        away_team_id=away_id,
        home_goals=hg,
        away_goals=ag,
    )


def target(match_id: int, day_number: int, home_id: int, away_id: int) -> TargetMatch:
    return TargetMatch(match_id=match_id, date=day(day_number), home_team_id=home_id, away_team_id=away_id)


def fv(
    match_id: int,
    home_id: int,
    away_id: int,
    *,
    home_elo: float = 1500.0,
    away_elo: float = 1500.0,
    home_goals_for_avg: float = 0.0,
    home_goals_against_avg: float = 0.0,
    away_goals_for_avg: float = 0.0,
    away_goals_against_avg: float = 0.0,
    home_advantage: float = 60.0,
) -> FeatureVector:
    return FeatureVector(
        match_id=match_id,
        home_team_id=home_id,
        away_team_id=away_id,
        date=day(1),
        home_elo=home_elo,
        away_elo=away_elo,
        elo_difference=home_elo - away_elo,
        home_form=TeamForm(),
        away_form=TeamForm(),
        home_goals_for_avg=home_goals_for_avg,
        home_goals_against_avg=home_goals_against_avg,
        away_goals_for_avg=away_goals_for_avg,
        away_goals_against_avg=away_goals_against_avg,
        home_advantage=home_advantage,
    )


class FakeHistoryProvider:
    """HistoryProvider controlado. ``leaky=True`` devuelve TODO el dataset
    (simula un provedor defectuoso) para verificar la defensa del builder."""

    def __init__(self, matches: list[HistoricalMatch], *, leaky: bool = False) -> None:
        self._matches = matches
        self._leaky = leaky

    def get_matches_before(self, cutoff: date) -> list[HistoricalMatch]:
        if self._leaky:
            return sorted(self._matches, key=lambda m: (m.date, m.match_id))
        return sorted((m for m in self._matches if m.date < cutoff), key=lambda m: (m.date, m.match_id))

    def get_all_matches(self) -> list[HistoricalMatch]:
        return sorted(self._matches, key=lambda m: (m.date, m.match_id))