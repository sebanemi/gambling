from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from football_predictor.database.models.base import Base


class MatchOdds(Base):
    """Historical odds for a match from a bookmaker."""

    __tablename__ = "match_odds"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"), nullable=False)
    bookmaker: Mapped[str] = mapped_column(String(50), nullable=False)
    home_win_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    draw_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    away_win_odds: Mapped[float] = mapped_column(Numeric(10, 4), nullable=False)
    odds_type: Mapped[str] = mapped_column(String(20), nullable=False, default="decimal", server_default="decimal")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    match: Mapped["Match"] = relationship(  # noqa: F821
        backref="odds",
    )

    __table_args__ = (
        UniqueConstraint("match_id", "bookmaker", name="uq_match_odds_match_bookmaker"),
        CheckConstraint("home_win_odds > 0", name="ck_match_odds_home_positive"),
        CheckConstraint("draw_odds > 0", name="ck_match_odds_draw_positive"),
        CheckConstraint("away_win_odds > 0", name="ck_match_odds_away_positive"),
        CheckConstraint(
            "odds_type IN ('decimal', 'fractional', 'american')",
            name="ck_match_odds_type",
        ),
        Index("ix_match_odds_match_id", "match_id"),
        Index("ix_match_odds_bookmaker", "bookmaker"),
    )

    def __repr__(self) -> str:
        return (
            f"<MatchOdds match_id={self.match_id} bookmaker={self.bookmaker} "
            f"home={self.home_win_odds} draw={self.draw_odds} away={self.away_win_odds}>"
        )
