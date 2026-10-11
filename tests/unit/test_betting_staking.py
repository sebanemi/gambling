import pytest

from football_predictor.betting.staking import (
    ConfidenceBasedStakeStrategy,
    FixedStakeStrategy,
    KellyCriterionStrategy,
    ProportionalStakeStrategy,
    StakingContext,
    get_strategy,
)


def _ctx(bankroll=1000.0, prob=0.5, odds=2.2, prediction="HOME"):
    return StakingContext(bankroll=bankroll, model_probability=prob, odds=odds, prediction=prediction)


def test_fixed_stake():
    assert FixedStakeStrategy(stake=10.0).calculate_stake(_ctx()) == pytest.approx(10.0)


def test_kelly_positive_edge():
    stake = KellyCriterionStrategy(kelly_fraction=1.0).calculate_stake(_ctx(prob=0.6, odds=2.0))
    # full kelly = ((1.0 * 0.6) - 0.4) / 1.0 = 0.2 -> 200 on 1000 bankroll
    assert stake == pytest.approx(200.0)


def test_kelly_no_edge_returns_zero():
    assert KellyCriterionStrategy().calculate_stake(_ctx(prob=0.4, odds=2.0)) == 0.0
    assert KellyCriterionStrategy().calculate_stake(_ctx(prob=0.5, odds=1.0)) == 0.0


def test_kelly_fractional_scales():
    full = KellyCriterionStrategy(kelly_fraction=1.0).calculate_stake(_ctx(prob=0.6, odds=2.0))
    quarter = KellyCriterionStrategy(kelly_fraction=0.25).calculate_stake(_ctx(prob=0.6, odds=2.0))
    assert quarter == pytest.approx(full * 0.25)


def test_proportional():
    assert ProportionalStakeStrategy(percentage=0.02).calculate_stake(_ctx()) == pytest.approx(20.0)


def test_confidence_scales_with_ev():
    low = ConfidenceBasedStakeStrategy().calculate_stake(_ctx(prob=0.5, odds=2.0))  # EV=0
    assert low == 0.0
    mid = ConfidenceBasedStakeStrategy().calculate_stake(_ctx(prob=0.55, odds=2.0))  # EV=0.1
    assert mid == pytest.approx(1000.0 * 0.05)


def test_get_strategy_factory():
    assert isinstance(get_strategy("fixed"), FixedStakeStrategy)
    assert isinstance(get_strategy("kelly"), KellyCriterionStrategy)
    assert isinstance(get_strategy("proportional"), ProportionalStakeStrategy)
    assert isinstance(get_strategy("confidence"), ConfidenceBasedStakeStrategy)
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy("martingale")
