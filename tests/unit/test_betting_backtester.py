import pytest

from football_predictor.betting.backtester import BettingBacktester
from football_predictor.config.settings import Settings
from tests.helpers import FakeHistoryProvider, hm


def _matches(n=12):
    # Alternating results between 4 teams so Elo/ML can train.
    matches = []
    for i in range(n):
        home_goals, away_goals = (2, 0) if i % 2 == 0 else (0, 1)
        matches.append(hm(i + 1, i + 1, (i % 4) + 1, ((i + 1) % 4) + 1, home_goals, away_goals))
    return matches


def _generous_odds(match_id: int) -> dict[str, float]:
    # High odds on every outcome -> almost every prediction is a value bet.
    return {"home": 5.0, "draw": 5.0, "away": 5.0}


def _settings(**overrides):
    base = Settings()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_no_odds_means_no_bets():
    provider = FakeHistoryProvider(_matches())
    backtester = BettingBacktester(provider, _settings(), min_prior_matches=3, odds_lookup=None)
    result = backtester.run(model_name="elo", strategy_name="fixed")
    assert result.total_bets == 0
    assert result.skipped_no_odds == 9  # 12 - 3 warmup
    assert result.final_bankroll == pytest.approx(1000.0)


def test_generous_odds_places_bets_and_updates_bankroll():
    provider = FakeHistoryProvider(_matches())
    backtester = BettingBacktester(
        provider, _settings(), min_prior_matches=3, odds_lookup=_generous_odds
    )
    result = backtester.run(model_name="elo", strategy_name="fixed", initial_bankroll=1000.0)
    assert result.total_bets == 9
    assert result.winning_bets + result.losing_bets == 9
    assert 0.0 <= result.win_rate <= 1.0
    assert result.total_staked == pytest.approx(9 * 10.0)
    assert result.final_bankroll != pytest.approx(1000.0)
    assert result.profit == pytest.approx(result.final_bankroll - 1000.0)
    assert len(result.bankroll_history) == 9
    assert len(result.bets) == 9
    # ROI consistency: (return - staked) / staked
    assert result.roi == pytest.approx((result.total_return - result.total_staked) / result.total_staked)


def test_strict_odds_filter_skips_everything():
    provider = FakeHistoryProvider(_matches())
    backtester = BettingBacktester(
        provider, _settings(), min_prior_matches=3, odds_lookup=_generous_odds
    )
    result = backtester.run(model_name="elo", min_odds=6.0, max_odds=10.0)
    assert result.total_bets == 0
    assert result.skipped_no_value > 0


def test_empty_history_raises():
    with pytest.raises(ValueError, match="no hay partidos"):
        BettingBacktester(FakeHistoryProvider([]), _settings(), odds_lookup=_generous_odds).run()


def test_unknown_model_raises():
    provider = FakeHistoryProvider(_matches())
    backtester = BettingBacktester(provider, _settings(), odds_lookup=_generous_odds)
    with pytest.raises(ValueError, match="modelo desconocido"):
        backtester.run(model_name="martingale")


def test_zero_threshold_bets_more_than_high_threshold():
    provider = FakeHistoryProvider(_matches())
    low = BettingBacktester(provider, _settings(), min_prior_matches=3, odds_lookup=_generous_odds).run(
        model_name="elo", value_threshold=0.0
    )
    high = BettingBacktester(provider, _settings(), min_prior_matches=3, odds_lookup=_generous_odds).run(
        model_name="elo", value_threshold=5.0
    )
    assert low.total_bets >= high.total_bets
