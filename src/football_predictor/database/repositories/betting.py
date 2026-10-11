"""Persistencia de simulaciones de apuestas en BD."""

import json

from sqlalchemy.orm import Session

from football_predictor.betting.backtester import BettingSimulationResult
from football_predictor.database.models.betting_simulation import BettingHistory, BettingSimulation


class BettingRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save_simulation(
        self,
        *,
        model_name: str,
        strategy: str,
        result: BettingSimulationResult,
        parameters: dict | None = None,
    ) -> BettingSimulation:
        simulation = BettingSimulation(
            model_name=model_name,
            strategy=strategy,
            initial_bankroll=result.initial_bankroll,
            final_bankroll=result.final_bankroll,
            total_bets=result.total_bets,
            winning_bets=result.winning_bets,
            roi=result.roi,
            max_drawdown=result.max_drawdown,
            parameters=json.dumps(parameters or {}, ensure_ascii=False),
        )
        self._session.add(simulation)
        self._session.flush()
        for bet in result.bets:
            self._session.add(
                BettingHistory(
                    simulation_id=simulation.id,
                    match_id=bet.match_id,
                    prediction=bet.prediction,
                    model_probability=bet.model_probability,
                    bookmaker=bet.bookmaker,
                    odds=bet.odds,
                    stake=bet.stake,
                    profit_loss=bet.profit_loss,
                    bankroll_after=bet.bankroll_after,
                )
            )
        self._session.commit()
        return simulation


__all__ = ["BettingRepository"]
