"""Value bet detection and expected value calculations."""


def expected_value(model_probability: float, odds: float) -> float:
    """Calculate expected value of a bet.

    Args:
        model_probability: Our model's probability of the outcome (0-1)
        odds: Decimal odds offered by bookmaker

    Returns:
        Expected value per unit staked. Positive = profitable edge.
    """
    if model_probability <= 0 or model_probability > 1:
        raise ValueError("model_probability must be in (0, 1]")
    if odds <= 0:
        raise ValueError("odds must be positive")

    # EV = (prob * (odds - 1)) - ((1 - prob) * 1)
    # Equivalent to: prob * odds - 1
    return model_probability * odds - 1.0


def is_value_bet(model_probability: float, odds: float, threshold: float = 0.05) -> bool:
    """Check if a bet has positive expected value above threshold.

    Args:
        model_probability: Our model's probability of the outcome (0-1)
        odds: Decimal odds offered by bookmaker
        threshold: Minimum EV to consider a value bet (default 5%)

    Returns:
        True if EV > threshold
    """
    ev = expected_value(model_probability, odds)
    return ev > threshold


def implied_probability(odds: float) -> float:
    """Calculate implied probability from decimal odds.

    Args:
        odds: Decimal odds

    Returns:
        Implied probability (0-1)
    """
    if odds <= 0:
        raise ValueError("odds must be positive")
    return 1.0 / odds


def edge(model_probability: float, odds: float) -> float:
    """Calculate the edge (advantage) over the bookmaker.

    Args:
        model_probability: Our model's probability
        odds: Decimal odds

    Returns:
        Edge as percentage (model_prob - implied_prob)
    """
    return model_probability - implied_probability(odds)
