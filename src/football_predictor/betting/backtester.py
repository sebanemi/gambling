"""Betting simulation backtester.

Walk-forward simulation of betting strategies with historical odds.
Respects anti-leakage: predictions only use past matches, and a bet is
only placed when real historical odds exist for the match.

Performance: mirrors ``evaluation.Backtester`` — a single ``FeatureStream``
pass O(n), a precomputed ML matrix reused by slices, and Poisson warm-start
between steps.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from football_predictor.betting.staking import StakingStrategy, get_strategy
from football_predictor.betting.value import expected_value, is_value_bet
from football_predictor.config.settings import Settings, get_settings
from football_predictor.domain.protocols import HistoryProvider
from football_predictor.features.config import FeatureConfig
from football_predictor.features.elo import EloCalculator
from football_predictor.features.stream import FeatureStream
from football_predictor.models.base import Outcome, Predictor, outcome_from_goals
from football_predictor.models.elo_model import EloModel
from football_predictor.models.ensemble import EnsembleModel
from football_predictor.models.gbm import GBMModel
from football_predictor.models.ml import MLModel
from football_predictor.models.poisson import PoissonModel
from football_predictor.services.model_trainer import (
    MODEL_ELO,
    MODEL_ENSEMBLE,
    MODEL_GBM,
    MODEL_ML,
    MODEL_POISSON,
    parse_ensemble_weights,
)
from football_predictor.utils.logging import get_logger

log = get_logger("betting.backtester")

# match_id -> {"home": float, "draw": float, "away": float} (decimal odds).
# None / zeros = no odds available for that match.
OddsLookup = Callable[[int], dict[str, float] | None]

_OUTCOME_TO_ODDS_KEY = {Outcome.HOME: "home", Outcome.DRAW: "draw", Outcome.AWAY: "away"}
_STR_TO_OUTCOME = {"HOME": Outcome.HOME, "DRAW": Outcome.DRAW, "AWAY": Outcome.AWAY}


@dataclass(frozen=True)
class BettingResult:
    """Result of a single bet."""

    match_id: int
    match_date: date
    prediction: str  # HOME, DRAW, AWAY
    model_probability: float
    bookmaker: str
    odds: float
    stake: float
    profit_loss: float
    bankroll_after: float
    actual_outcome: str  # HOME, DRAW, AWAY


@dataclass(frozen=True)
class BettingSimulationResult:
    """Summary of a betting simulation."""

    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: float
    total_staked: float
    total_return: float
    roi: float
    max_drawdown: float
    final_bankroll: float
    bankroll_history: list[tuple[date, float]]
    bets: Sequence[BettingResult] = field(default_factory=list)
    initial_bankroll: float = 0.0
    skipped_no_odds: int = 0
    skipped_no_value: int = 0

    @property
    def profit(self) -> float:
        return self.final_bankroll - self.initial_bankroll


class BettingBacktester:
    """Walk-forward betting simulation with real historical odds."""

    def __init__(
        self,
        history_provider: HistoryProvider,
        settings: Settings | None = None,
        min_prior_matches: int = 10,
        odds_lookup: OddsLookup | None = None,
        bookmaker: str = "best",
    ) -> None:
        self._provider = history_provider
        self._settings = settings or get_settings()
        self._min_prior_matches = max(1, min_prior_matches)
        self._odds_lookup = odds_lookup
        self._bookmaker = bookmaker

    def run(
        self,
        model_name: str = "ensemble",
        strategy_name: str = "kelly",
        initial_bankroll: float = 1000.0,
        value_threshold: float = 0.05,
        min_odds: float = 1.1,
        max_odds: float = 10.0,
        bookmaker: str | None = None,
    ) -> BettingSimulationResult:
        """Run betting simulation.

        Only matches with real odds (via ``odds_lookup``) are bettable;
        matches without odds are counted in ``skipped_no_odds``.
        """
        matches = sorted(self._provider.get_all_matches(), key=lambda m: (m.date, m.match_id))
        if not matches:
            raise ValueError("no hay partidos históricos para simular apuestas")
        if model_name not in (MODEL_POISSON, MODEL_ELO, MODEL_ML, MODEL_GBM, MODEL_ENSEMBLE):
            raise ValueError(
                f"modelo desconocido: {model_name}. Válidos: poisson, elo, ml, gbm, ensemble"
            )

        active_bookmaker = bookmaker or self._bookmaker
        strategy = self._build_strategy(strategy_name)

        s = self._settings
        stream = FeatureStream(
            FeatureConfig(
                goals_window=s.features_goals_window,
                home_away_window=s.features_home_away_window,
                stats_window=s.features_stats_window,
            ),
            EloCalculator(
                initial_rating=s.elo_initial_rating,
                k_factor=s.elo_k_factor,
                home_advantage=s.elo_home_advantage,
            ),
        )
        vectors = stream.build_all(matches)
        outcomes = [outcome_from_goals(m.home_goals, m.away_goals) for m in matches]

        need_members = self._required_members(model_name)
        matrix: np.ndarray | None = None
        outcome_matrix: np.ndarray | None = None
        if MODEL_ML in need_members or MODEL_GBM in need_members:
            matrix = MLModel.make_matrix(vectors)
            outcome_matrix = np.array([int(o) for o in outcomes], dtype=int)

        weights = parse_ensemble_weights(s.models_ensemble_weights)

        bankroll = float(initial_bankroll)
        bankroll_history: list[tuple[date, float]] = []
        bets: list[BettingResult] = []
        total_staked = 0.0
        total_return = 0.0
        winning_bets = 0
        losing_bets = 0
        skipped_no_odds = 0
        skipped_no_value = 0
        max_bankroll = bankroll
        max_drawdown = 0.0
        prev_poisson: np.ndarray | None = None

        for index, match in enumerate(matches):
            if index < self._min_prior_matches:
                continue

            odds = self._lookup_odds(match.match_id)
            if odds is None:
                skipped_no_odds += 1
                continue

            trained, prev_poisson = self._train_step(
                matches, matrix, outcome_matrix, need_members, weights, index, prev_poisson
            )
            model = trained.get(model_name)
            if model is None:
                skipped_no_value += 1
                continue

            result = model.predict(vectors[index], include_snapshot=False)
            probs = {"HOME": result.home_win, "DRAW": result.draw, "AWAY": result.away_win}
            predicted_str = max(probs, key=lambda k: probs[k])
            model_prob = probs[predicted_str]
            outcome = _STR_TO_OUTCOME[predicted_str]
            odd = float(odds.get(_OUTCOME_TO_ODDS_KEY[outcome], 0.0) or 0.0)

            if not (min_odds <= odd <= max_odds):
                skipped_no_value += 1
                continue
            if not is_value_bet(model_prob, odd, value_threshold):
                skipped_no_value += 1
                continue

            from football_predictor.betting.staking import StakingContext

            stake = strategy.calculate_stake(
                StakingContext(
                    bankroll=bankroll,
                    model_probability=model_prob,
                    odds=odd,
                    prediction=predicted_str,
                )
            )
            stake = self._cap_stake(stake, bankroll)
            if stake <= 0:
                skipped_no_value += 1
                continue

            actual = outcome_from_goals(match.home_goals, match.away_goals)
            actual_str = {Outcome.HOME: "HOME", Outcome.DRAW: "DRAW", Outcome.AWAY: "AWAY"}[actual]
            won = predicted_str == actual_str
            profit_loss = stake * (odd - 1.0) if won else -stake

            bankroll += profit_loss
            total_staked += stake
            if won:
                total_return += stake + profit_loss
                winning_bets += 1
            else:
                losing_bets += 1

            max_bankroll = max(max_bankroll, bankroll)
            drawdown = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            bankroll_history.append((match.date, bankroll))
            bets.append(
                BettingResult(
                    match_id=match.match_id,
                    match_date=match.date,
                    prediction=predicted_str,
                    model_probability=model_prob,
                    bookmaker=active_bookmaker,
                    odds=odd,
                    stake=stake,
                    profit_loss=profit_loss,
                    bankroll_after=bankroll,
                    actual_outcome=actual_str,
                )
            )

        total_bets = winning_bets + losing_bets
        win_rate = winning_bets / total_bets if total_bets else 0.0
        roi = (total_return - total_staked) / total_staked if total_staked > 0 else 0.0

        return BettingSimulationResult(
            total_bets=total_bets,
            winning_bets=winning_bets,
            losing_bets=losing_bets,
            win_rate=win_rate,
            total_staked=total_staked,
            total_return=total_return,
            roi=roi,
            max_drawdown=max_drawdown,
            final_bankroll=bankroll,
            bankroll_history=bankroll_history,
            bets=bets,
            initial_bankroll=float(initial_bankroll),
            skipped_no_odds=skipped_no_odds,
            skipped_no_value=skipped_no_value,
        )

    # --- internals ---

    def _required_members(self, model_name: str) -> tuple[str, ...]:
        if model_name == MODEL_ENSEMBLE:
            return (MODEL_POISSON, MODEL_ELO, MODEL_ML, MODEL_GBM)
        return (model_name,)

    def _build_strategy(self, strategy_name: str) -> StakingStrategy:
        s = self._settings
        kwargs: dict[str, float] = {}
        if strategy_name == "fixed":
            kwargs = {"stake": s.betting_fixed_stake}
        elif strategy_name == "kelly":
            kwargs = {"kelly_fraction": s.betting_kelly_fraction}
        elif strategy_name == "proportional":
            kwargs = {"percentage": s.betting_proportional_percentage}
        elif strategy_name == "confidence":
            kwargs = {"max_stake_percentage": s.betting_max_stake_percentage}
        return get_strategy(strategy_name, **kwargs)

    def _cap_stake(self, stake: float, bankroll: float) -> float:
        if stake <= 0 or bankroll <= 0:
            return 0.0
        max_stake = bankroll * self._settings.betting_max_stake_percentage
        # Fixed strategy with a tiny bankroll: allow the fixed stake if it fits.
        if stake <= bankroll and stake <= max(1.0, max_stake):
            return min(stake, bankroll)
        return min(stake, bankroll, max_stake)

    def _lookup_odds(self, match_id: int) -> dict[str, float] | None:
        if self._odds_lookup is None:
            return None
        try:
            odds = self._odds_lookup(match_id)
        except Exception as exc:  # noqa: BLE001 — una fuente de odds rota no debe tumbar la simu
            log.warning("odds lookup failed for match %s: %s", match_id, exc)
            return None
        if not odds:
            return None
        try:
            home = float(odds.get("home", 0.0) or 0.0)
            draw = float(odds.get("draw", 0.0) or 0.0)
            away = float(odds.get("away", 0.0) or 0.0)
        except (TypeError, ValueError):
            return None
        if home <= 1.0 or draw <= 1.0 or away <= 1.0:
            return None
        return {"home": home, "draw": draw, "away": away}

    def _train_step(
        self,
        matches: Sequence,
        matrix: np.ndarray | None,
        outcome_matrix: np.ndarray | None,
        need_members: tuple[str, ...],
        weights: list[float],
        index: int,
        prev_poisson: np.ndarray | None,
    ) -> tuple[dict[str, Predictor], np.ndarray | None]:
        s = self._settings
        trained: dict[str, Predictor] = {}

        if MODEL_POISSON in need_members:
            poisson = PoissonModel(
                regularization=s.models_poisson_regularization,
                rho=s.models_poisson_rho,
            )
            try:
                poisson.fit(matches[:index], warm_start=prev_poisson)
                trained[MODEL_POISSON] = poisson
                prev_poisson = poisson.last_point
            except (ValueError, RuntimeError) as exc:
                log.warning("Poisson fit failed at index %d: %s", index, exc)
                prev_poisson = None

        if MODEL_ELO in need_members:
            trained[MODEL_ELO] = EloModel(
                draw_max=s.models_elo_draw_max, draw_sigma=s.models_elo_draw_sigma
            )

        if MODEL_ML in need_members and matrix is not None and outcome_matrix is not None:
            ml = MLModel(random_state=s.models_ml_random_state)
            try:
                ml.fit_matrix(matrix[:index], outcome_matrix[:index])
                trained[MODEL_ML] = ml
            except (ValueError, RuntimeError) as exc:
                log.warning("ML fit failed at index %d: %s", index, exc)

        if MODEL_GBM in need_members and matrix is not None and outcome_matrix is not None:
            gbm = GBMModel(
                backend=s.models_gbm_backend,
                n_estimators=s.models_gbm_n_estimators,
                max_depth=s.models_gbm_max_depth,
                learning_rate=s.models_gbm_learning_rate,
                random_state=s.models_gbm_random_state,
            )
            try:
                gbm.fit_matrix(matrix[:index], outcome_matrix[:index])
                trained[MODEL_GBM] = gbm
            except (ValueError, RuntimeError, ImportError) as exc:
                log.warning("GBM fit failed at index %d: %s", index, exc)

        if len(need_members) > 1:
            members = [
                trained[name]
                for name in (MODEL_POISSON, MODEL_ELO, MODEL_ML, MODEL_GBM)
                if name in trained
            ]
            if len(members) >= 2:
                trained[MODEL_ENSEMBLE] = EnsembleModel(members, weights[: len(members)])
            elif len(members) == 1:
                trained[MODEL_ENSEMBLE] = members[0]

        return trained, prev_poisson


__all__ = [
    "BettingBacktester",
    "BettingResult",
    "BettingSimulationResult",
    "OddsLookup",
    "expected_value",
]
