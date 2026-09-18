from datetime import datetime

from sqlalchemy import DateTime, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from football_predictor.database.models.base import Base


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    country: Mapped[str] = mapped_column(String(200), nullable=False, default="", server_default="")
    type: Mapped[str] = mapped_column(String(50), nullable=False, default="league", server_default="league")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    seasons: Mapped[list["Season"]] = relationship(back_populates="competition")

    __table_args__ = (
        UniqueConstraint("name", "country", name="uq_competitions_name_country"),
    )