"""Equivalencia entre FeatureBuilder (por partido, O(n²)) y FeatureStream (incremental O(n))."""

import random
from datetime import timedelta

import pytest

from football_predictor.domain.entities import HistoricalMatch, TargetMatch
from football_predictor.features.builder import FeatureBuilder
from football_predictor.features.config import FeatureConfig
from football_predictor.features.elo import EloCalculator
from football_predictor.features.stream import FeatureStream
from tests.helpers import day, hm


def _approx_equal(a, b) -> bool:
    if isinstance(a, (int, float)):
        return a == b or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))
    if isinstance(a, dict):
        return set(a) == set(b) and all(_approx_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_approx_equal(x, y) for x, y in zip(a, b))
    return a == b


def _build_dataset(n: int = 40, seed: int = 7) -> list[HistoricalMatch]:
    rng = random.Random(seed)
    matches = []
    base = day(1)
    for i in range(n):
        date = base + timedelta(days=rng.randint(0, 6))
        home = rng.randint(1, 6)
        away = rng.randint(1, 6)
        while away == home:
            away = rng.randint(1, 6)
        stats = {}
        if rng.random() < 0.7:
            stats = {
                "home_yellow_cards": rng.randint(0, 6),
                "away_yellow_cards": rng.randint(0, 6),
                "home_red_cards": rng.randint(0, 1),
                "away_red_cards": rng.randint(0, 1),
                "home_corners": rng.randint(0, 15),
                "away_corners": rng.randint(0, 15),
                "home_shots": rng.randint(0, 25),
                "away_shots": rng.randint(0, 25),
                "home_shots_on_target": rng.randint(0, 10),
                "away_shots_on_target": rng.randint(0, 10),
                "home_fouls": rng.randint(0, 20),
                "away_fouls": rng.randint(0, 20),
                "home_throw_ins": rng.randint(0, 30),
                "away_throw_ins": rng.randint(0, 30),
                "home_penalties": rng.randint(0, 2),
                "away_penalties": rng.randint(0, 2),
                "home_xg": round(rng.uniform(0, 4), 2),
                "away_xg": round(rng.uniform(0, 4), 2),
                "home_possession": round(rng.uniform(30, 70), 1),
                "away_possession": round(rng.uniform(30, 70), 1),
            }
        matches.append(
            HistoricalMatch(
                match_id=i + 1,
                date=date,
                home_team_id=home,
                away_team_id=away,
                home_goals=rng.randint(0, 4),
                away_goals=rng.randint(0, 4),
                **stats,
            )
        )
    return sorted(matches, key=lambda m: (m.date, m.match_id))


class _Provider:
    def __init__(self, matches):
        self._matches = matches

    def get_all_matches(self):
        return self._matches

    def get_matches_before(self, cutoff):
        return [m for m in self._matches if m.date < cutoff]


@pytest.mark.parametrize("config", [FeatureConfig(), FeatureConfig(goals_window=5, stats_window=7)])
def test_stream_matches_builder_for_all_matches(config):
    matches = _build_dataset()
    elo = EloCalculator(initial_rating=1500.0, k_factor=20.0, home_advantage=60.0)
    builder = FeatureBuilder(_Provider(matches), config, elo)
    stream = FeatureStream(config, elo)

    expected = [builder.build_for_match(TargetMatch(m.match_id, m.date, m.home_team_id, m.away_team_id)) for m in matches]
    got = stream.build_all(matches)

    assert len(got) == len(expected)
    for original, fast in zip(expected, got):
        assert _approx_equal(original.model_dump(), fast.model_dump()), f"mismatch en match {original.match_id}"


def test_stream_matches_builder_for_arbitrary_target():
    matches = _build_dataset()
    elo = EloCalculator()
    builder = FeatureBuilder(_Provider(matches), FeatureConfig(), elo)
    stream = FeatureStream(FeatureConfig(), elo)

    last_date = max(m.date for m in matches)
    target = TargetMatch(match_id=999, date=last_date + timedelta(days=1), home_team_id=1, away_team_id=2)
    expected = builder.build_for_match(target)

    stream.build_all(matches)
    got = stream.build_target(target)
    assert _approx_equal(expected.model_dump(), got.model_dump())


def test_stream_same_day_matches_are_isolated():
    matches = _build_dataset()
    polluted = matches + [hm(500, 20, 1, 2, 9, 9)]
    polluted = sorted(polluted, key=lambda m: (m.date, m.match_id))
    elo = EloCalculator()
    config = FeatureConfig()

    builder = FeatureBuilder(_Provider(polluted), config, elo)
    stream = FeatureStream(config, elo)
    vectors = stream.build_all(polluted)

    index = next(i for i, m in enumerate(polluted) if m.match_id == 500)
    got = vectors[index]
    expected = builder.build_for_match(TargetMatch(500, day(20), 1, 2))
    assert _approx_equal(expected.model_dump(), got.model_dump())