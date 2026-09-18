"""core tables

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-17 00:00:00
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "competitions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("country", sa.String(length=200), nullable=False, server_default=sa.text("''")),
        sa.Column("type", sa.String(length=50), nullable=False, server_default=sa.text("'league'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", "country", name="uq_competitions_name_country"),
    )

    op.create_table(
        "teams",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("country", sa.String(length=200), nullable=False, server_default=sa.text("''")),
        sa.Column("external_id", sa.String(length=100), nullable=True),
        sa.Column("source", sa.String(length=50), nullable=False, server_default=sa.text("''")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("name", "country", name="uq_teams_name_country"),
    )

    op.create_table(
        "seasons",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("competition_id", "name", name="uq_seasons_competition_name"),
    )
    op.create_index("ix_seasons_competition_id", "seasons", ["competition_id"])

    op.create_table(
        "matches",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("competition_id", sa.Integer(), sa.ForeignKey("competitions.id"), nullable=False),
        sa.Column("season_id", sa.Integer(), sa.ForeignKey("seasons.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("home_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("away_team_id", sa.Integer(), sa.ForeignKey("teams.id"), nullable=False),
        sa.Column("home_goals", sa.Integer(), nullable=True),
        sa.Column("away_goals", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default=sa.text("'finished'")),
        sa.Column("source", sa.String(length=50), nullable=False, server_default=sa.text("'csv'")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "competition_id",
            "season_id",
            "home_team_id",
            "away_team_id",
            "date",
            name="uq_matches_competition_season_teams_date",
        ),
        sa.CheckConstraint("home_team_id <> away_team_id", name="ck_matches_distinct_teams"),
        sa.CheckConstraint("home_goals IS NULL OR home_goals >= 0", name="ck_matches_home_goals_nonneg"),
        sa.CheckConstraint("away_goals IS NULL OR away_goals >= 0", name="ck_matches_away_goals_nonneg"),
        sa.CheckConstraint(
            "status IN ('scheduled', 'finished', 'postponed', 'cancelled')",
            name="ck_matches_status",
        ),
    )
    op.create_index("ix_matches_date", "matches", ["date"])
    op.create_index("ix_matches_home_team_id", "matches", ["home_team_id"])
    op.create_index("ix_matches_away_team_id", "matches", ["away_team_id"])


def downgrade() -> None:
    op.drop_table("matches")
    op.drop_table("seasons")
    op.drop_table("teams")
    op.drop_table("competitions")