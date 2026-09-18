from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from football_predictor.database.models.base import Base


class Prediction(Base):
    """Predicción persistida por modelo para un partido (snapshot auditable).

    ``features_snapshot`` guarda el ``FeatureVector.flatten()`` exacto que
    produjo la predicción: permite explicar y auditar cualquier resultado.
    El unique (match_id, model_name) hace que re-predecir actualice en vez
    de acumular filas huérfanas.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(50), nullable=False)
    home_win: Mapped[float] = mapped_column(Float, nullable=False)
    draw: Mapped[float] = mapped_column(Float, nullable=False)
    away_win: Mapped[float] = mapped_column(Float, nullable=False)
    home_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    away_goals: Mapped[float | None] = mapped_column(Float, nullable=True)
    features_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()
    )

    match: Mapped["Match"] = relationship()  # noqa: F821

    __table_args__ = (
        UniqueConstraint("match_id", "model_name", name="uq_predictions_match_model"),
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction match={self.match_id} model={self.model_name} "
            f"{self.home_win:.3f}/{self.draw:.3f}/{self.away_win:.3f}>"
        )