from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from football_predictor.database.models.base import Base


class BettingSimulation(Base):
    """Summary of a betting simulation run."""

    __tablename__ = "betting_simulations"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    strategy: Mapped[str] = mapped_column(String(50), nullable=False)
    initial_bankroll: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    final_bankroll: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    total_bets: Mapped[int] = mapped_column(Integer, nullable=False)
    winning_bets: Mapped[int] = mapped_column(Integer, nullable=False)
    roi: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    max_drawdown: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    parameters: Mapped[str] = mapped_column(String(1000), nullable=True)  # JSON string
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    history: Mapped[list["BettingHistory"]] = relationship(
        back_populates="simulation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("initial_bankroll > 0", name="ck_betting_sim_initial_positive"),
        CheckConstraint("total_bets >= 0", name="ck_betting_sim_total_nonneg"),
        CheckConstraint("winning_bets >= 0", name="ck_betting_sim_winning_nonneg"),
        CheckConstraint("winning_bets <= total_bets", name="ck_betting_sim_winning_le_total"),
        Index("ix_betting_simulations_model", "model_name"),
        Index("ix_betting_simulations_strategy", "strategy"),
    )

    def __repr__(self) -> str:
        return (
            f"<BettingSimulation id={self.id} model={self.model_name} "
            f"strategy={self.strategy} roi={self.roi}% bets={self.total_bets}>"
        )


class BettingHistory(Base):
    """Individual bet in a simulation."""

    __tablename__ = "betting_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    simulation_id: Mapped[int] = mapped_column(
        ForeignKey("betting_simulations.id", ondelete="CASCADE"), nullable=False
    )
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    prediction: Mapped[str] = mapped_column(String(10), nullable=False)  # HOME, DRAW, AWAY
    model_probability: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    stake: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    profit_loss: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    bankroll_after: Mapped[float] = mapped_column(Numeric(15, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    simulation: Mapped["BettingSimulation"] = relationship(back_populates="history")
    match: Mapped["Match"] = relationship(  # noqa: F821
        backref="betting_history",
    )

    __table_args__ = (
        CheckConstraint("stake >= 0", name="ck_betting_history_stake_nonneg"),
        CheckConstraint("odds > 0", name="ck_betting_history_odds_positive"),
        CheckConstraint("prediction IN ('HOME', 'DRAW', 'AWAY')", name="ck_betting_history_prediction"),
        Index("ix_betting_history_simulation_id", "simulation_id"),
        Index("ix_betting_history_match_id", "match_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<BettingHistory sim_id={self.simulation_id} match_id={self.match_id} "
            f"pred={self.prediction} odds={self.odds} pnl={self.profit_loss}>"
        )
