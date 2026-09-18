"""match_statistics

Revision ID: ba87c7d82c65
Revises: 0003
Create Date: 2026-09-17 17:36:57.245121
"""

import sqlalchemy as sa
from alembic import op


revision = 'ba87c7d82c65'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "match_statistics",
        sa.Column("match_id", sa.Integer(), sa.ForeignKey("matches.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("home_yellow_cards", sa.Integer(), nullable=True),
        sa.Column("home_red_cards", sa.Integer(), nullable=True),
        sa.Column("home_corners", sa.Integer(), nullable=True),
        sa.Column("away_yellow_cards", sa.Integer(), nullable=True),
        sa.Column("away_red_cards", sa.Integer(), nullable=True),
        sa.Column("away_corners", sa.Integer(), nullable=True),
        sa.Column("home_shots", sa.Integer(), nullable=True),
        sa.Column("home_shots_on_target", sa.Integer(), nullable=True),
        sa.Column("home_fouls", sa.Integer(), nullable=True),
        sa.Column("home_throw_ins", sa.Integer(), nullable=True),
        sa.Column("home_penalties", sa.Integer(), nullable=True),
        sa.Column("away_shots", sa.Integer(), nullable=True),
        sa.Column("away_shots_on_target", sa.Integer(), nullable=True),
        sa.Column("away_fouls", sa.Integer(), nullable=True),
        sa.Column("away_throw_ins", sa.Integer(), nullable=True),
        sa.Column("away_penalties", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.CheckConstraint("home_yellow_cards IS NULL OR home_yellow_cards >= 0", name="ck_match_stats_home_yellow_nonneg"),
        sa.CheckConstraint("home_red_cards IS NULL OR home_red_cards >= 0", name="ck_match_stats_home_red_nonneg"),
        sa.CheckConstraint("home_corners IS NULL OR home_corners >= 0", name="ck_match_stats_home_corners_nonneg"),
        sa.CheckConstraint("away_yellow_cards IS NULL OR away_yellow_cards >= 0", name="ck_match_stats_away_yellow_nonneg"),
        sa.CheckConstraint("away_red_cards IS NULL OR away_red_cards >= 0", name="ck_match_stats_away_red_nonneg"),
        sa.CheckConstraint("away_corners IS NULL OR away_corners >= 0", name="ck_match_stats_away_corners_nonneg"),
    )


def downgrade() -> None:
    op.drop_table("match_statistics")