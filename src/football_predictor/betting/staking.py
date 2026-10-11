"""Bankroll management strategies for betting."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StakingContext:
    """Context for staking calculation."""

    bankroll: float
    model_probability: float
    odds: float
    prediction: str  # HOME, DRAW, AWAY


class StakingStrategy(Protocol):
    """Protocol for staking strategies."""

    def calculate_stake(self, context: StakingContext) -> float:
        """Calculate stake amount given the context."""
        ...


class FixedStakeStrategy:
    """Always bet a fixed amount."""

    def __init__(self, stake: float = 10.0) -> None:
        self._stake = float(stake)

    def calculate_stake(self, context: StakingContext) -> float:
        return self._stake


class KellyCriterionStrategy:
    """Kelly criterion with fractional Kelly for safety.

    Kelly = (bp - q) / b
    Where:
    - b = odds - 1 (net odds)
    - p = probability of winning
    - q = probability of losing (1 - p)
    """

    def __init__(self, kelly_fraction: float = 0.25) -> None:
        """Initialize Kelly strategy.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0.25 = quarter Kelly)
        """
        self._kelly_fraction = float(kelly_fraction)

    def calculate_stake(self, context: StakingContext) -> float:
        if context.odds <= 1.0:
            return 0.0

        # Calculate edge
        ev = context.model_probability * context.odds - 1.0
        if ev <= 0:
            return 0.0

        # Full Kelly formula
        b = context.odds - 1.0  # net odds
        p = context.model_probability
        q = 1.0 - p

        full_kelly = (b * p - q) / b
        fractional_kelly = full_kelly * self._kelly_fraction

        # Kelly can suggest >100% of bankroll, cap at 100%
        stake_fraction = max(0.0, min(fractional_kelly, 1.0))

        return context.bankroll * stake_fraction


class ProportionalStakeStrategy:
    """Bet a fixed percentage of bankroll."""

    def __init__(self, percentage: float = 0.02) -> None:
        """Initialize proportional strategy.

        Args:
            percentage: Percentage of bankroll to stake (0.02 = 2%)
        """
        self._percentage = float(percentage)

    def calculate_stake(self, context: StakingContext) -> float:
        return context.bankroll * self._percentage


class ConfidenceBasedStakeStrategy:
    """Stake proportional to model confidence (expected value)."""

    def __init__(self, max_stake_percentage: float = 0.05) -> None:
        """Initialize confidence-based strategy.

        Args:
            max_stake_percentage: Maximum stake as percentage of bankroll
        """
        self._max_stake_percentage = float(max_stake_percentage)

    def calculate_stake(self, context: StakingContext) -> float:
        from football_predictor.betting.value import expected_value

        ev = expected_value(context.model_probability, context.odds)
        if ev <= 0:
            return 0.0

        # Normalize EV: cap at 10% EV for scaling
        normalized_ev = min(ev / 0.10, 1.0)

        stake_percentage = self._max_stake_percentage * normalized_ev
        return context.bankroll * stake_percentage


def get_strategy(name: str, **kwargs) -> StakingStrategy:
    """Factory function to get a staking strategy.

    Args:
        name: Strategy name (fixed, kelly, proportional, confidence)
        **kwargs: Strategy-specific parameters

    Returns:
        StakingStrategy instance
    """
    strategies = {
        "fixed": FixedStakeStrategy,
        "kelly": KellyCriterionStrategy,
        "proportional": ProportionalStakeStrategy,
        "confidence": ConfidenceBasedStakeStrategy,
    }

    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name](**kwargs)
