"""predictions

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-17 00:00:00
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id"), nullable=False),
        sa.Column("model_name", sa.String(length=50), nullable=False),
        sa.Column("home_win", sa.Float(), nullable=False),
        sa.Column("draw", sa.Float(), nullable=False),
        sa.Column("away_win", sa.Float(), nullable=False),
        sa.Column("home_goals", sa.Float(), nullable=True),
        sa.Column("away_goals", sa.Float(), nullable=True),
        sa.Column("features_snapshot", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("match_id", "model_name", name="uq_predictions_match_model"),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])


def downgrade() -> None:
    op.drop_index("ix_predictions_match_id", table_name="predictions")
    op.drop_table("predictions")