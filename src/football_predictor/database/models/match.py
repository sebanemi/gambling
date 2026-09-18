from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from football_predictor.database.models.base import Base


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[int] = mapped_column(primary_key=True)
    competition_id: Mapped[int] = mapped_column(
        ForeignKey("competitions.id"), nullable=False
    )
    season_id: Mapped[int] = mapped_column(ForeignKey("seasons.id"), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    home_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    away_team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    home_goals: Mapped[int | None] = mapped_column(nullable=True)
    away_goals: Mapped[int | None] = mapped_column(nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="finished", server_default="finished"
    )
    source: Mapped[str] = mapped_column(
        String(50), nullable=False, default="csv", server_default="csv"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )

    competition: Mapped["Competition"] = relationship()  # noqa: F821
    season: Mapped["Season"] = relationship(back_populates="matches")
    home_team: Mapped["Team"] = relationship(  # noqa: F821
        back_populates="home_matches", foreign_keys=[home_team_id]
    )
    away_team: Mapped["Team"] = relationship(
        back_populates="away_matches", foreign_keys=[away_team_id]
    )
    statistics: Mapped["MatchStatistics"] = relationship(  # noqa: F821
        back_populates="match", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint(
            "competition_id",
            "season_id",
            "home_team_id",
            "away_team_id",
            "date",
            name="uq_matches_competition_season_teams_date",
        ),
        CheckConstraint("home_team_id <> away_team_id", name="ck_matches_distinct_teams"),
        CheckConstraint("home_goals IS NULL OR home_goals >= 0", name="ck_matches_home_goals_nonneg"),
        CheckConstraint("away_goals IS NULL OR away_goals >= 0", name="ck_matches_away_goals_nonneg"),
        CheckConstraint(
            "status IN ('scheduled', 'finished', 'postponed', 'cancelled')",
            name="ck_matches_status",
        ),
        Index("ix_matches_date", "date"),
        Index("ix_matches_home_team_id", "home_team_id"),
        Index("ix_matches_away_team_id", "away_team_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<Match {self.date} {self.home_team.name} {self.home_goals}-{self.away_goals} "
            f"{self.away_team.name}>"
        )