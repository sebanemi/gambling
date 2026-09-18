"""Cobertura de stats ricas: parser SofaScore y StatisticsModel extendido."""

from football_predictor.data.providers.sofascore import SofaScoreProvider
from football_predictor.domain.features import STATISTIC_METRICS
from football_predictor.models.statistics_model import StatisticsModel
from tests.helpers import hm


def _stats():
    provider = SofaScoreProvider.__new__(SofaScoreProvider)
    payload = {
        "statistics": [
            {
                "groups": [
                    {
                        "statisticsItems": [
                            {"name": "Yellow Cards", "home": 4, "away": 2},
                            {"name": "Red Cards", "home": 0, "away": 1},
                            {"name": "Corners", "home": 8, "away": 3},
                            {"name": "Ball Possession", "home": "58%", "away": "42%"},
                            {"name": "Total Shots", "home": 14, "away": 6},
                            {"name": "Shots on Target", "home": 5, "away": 2},
                            {"name": "Fouls", "home": 10, "away": 15},
                            {"name": "Throw-ins", "home": 22, "away": 18},
                            {"name": "Penalties", "home": 1, "away": 0},
                            {"name": "Expected Goals", "home": "1.8", "away": "0.6"},
                        ]
                    }
                ]
            }
        ]
    }
    parsed = provider._parse_statistics(payload)
    assert parsed["home_yellow_cards"] == 4
    assert parsed["away_red_cards"] == 1
    assert parsed["home_corners"] == 8
    assert parsed["home_shots"] == 14
    assert parsed["away_shots_on_target"] == 2
    assert parsed["home_fouls"] == 10
    assert parsed["home_throw_ins"] == 22
    assert parsed["away_penalties"] == 0
    assert parsed["home_xg"] == 1.8
    assert parsed["away_xg"] == 0.6
    assert parsed["home_possession"] == 58.0
    assert parsed["away_possession"] == 42.0


def test_statistics_model_covers_all_metrics():
    assert tuple(StatisticsModel._STATS) == tuple(STATISTIC_METRICS)


def test_statistics_model_predicts_rich_stats():
    history = [
        hm(1, 1, 1, 2, 2, 1),
        hm(2, 2, 2, 3, 1, 0),
        hm(3, 3, 3, 1, 0, 0),
        hm(4, 4, 1, 4, 3, 2),
    ]
    for m in history:
        m = m.__class__(
            match_id=m.match_id,
            date=m.date,
            home_team_id=m.home_team_id,
            away_team_id=m.away_team_id,
            home_goals=m.home_goals,
            away_goals=m.away_goals,
            home_shots=m.match_id * 10 % 20,
            away_shots=m.match_id * 5 % 18,
            home_xg=m.match_id * 0.3,
            away_xg=m.match_id * 0.2,
        )
    model = StatisticsModel()
    model.fit(history)
    result = model.expected_counts(1, 2)
    assert set(result) == set(STATISTIC_METRICS)
    for stat in ("yellow_cards", "red_cards", "corners", "shots", "xg", "possession"):
        home, away = result[stat]
        assert isinstance(home, float) and isinstance(away, float)