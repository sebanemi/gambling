"""rich_match_statistics

Persiste las stats ricas (xG y posesión) que antes se descartaban en el
import (FeatureBuilder/StatisticsModel ya soportan las 10 métricas).

Revision ID: 0004_rich_stats
Revises: ba87c7d82c65
Create Date: 2026-09-18
"""

import sqlalchemy as sa
from alembic import op

revision = '0004_rich_stats'
down_revision = 'ba87c7d82c65'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("match_statistics", sa.Column("home_xg", sa.Float(), nullable=True))
    op.add_column("match_statistics", sa.Column("away_xg", sa.Float(), nullable=True))
    op.add_column("match_statistics", sa.Column("home_possession", sa.Float(), nullable=True))
    op.add_column("match_statistics", sa.Column("away_possession", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("match_statistics", "away_possession")
    op.drop_column("match_statistics", "home_possession")
    op.drop_column("match_statistics", "away_xg")
    op.drop_column("match_statistics", "home_xg")