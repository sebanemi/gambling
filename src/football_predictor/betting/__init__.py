"""Betting strategy simulation module."""

from football_predictor.betting.staking import (
    ConfidenceBasedStakeStrategy,
    FixedStakeStrategy,
    KellyCriterionStrategy,
    ProportionalStakeStrategy,
    StakingContext,
    StakingStrategy,
    get_strategy,
)
from football_predictor.betting.value import edge, expected_value, implied_probability, is_value_bet

__all__ = [
    "ConfidenceBasedStakeStrategy",
    "FixedStakeStrategy",
    "KellyCriterionStrategy",
    "ProportionalStakeStrategy",
    "StakingContext",
    "StakingStrategy",
    "edge",
    "expected_value",
    "get_strategy",
    "implied_probability",
    "is_value_bet",
]
