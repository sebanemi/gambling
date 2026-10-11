"""betting_tables

Agrega tablas para simulación de apuestas: odds históricos, simulaciones
e historial de apuestas individuales.

Revision ID: 0005_betting_tables
Revises: 0004_rich_stats
Create Date: 2026-10-10
"""

import sqlalchemy as sa
from alembic import op

revision = '0005_betting_tables'
down_revision = '0004_rich_stats'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tabla match_odds
    op.create_table(
        'match_odds',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('bookmaker', sa.String(length=50), nullable=False),
        sa.Column('home_win_odds', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('draw_odds', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('away_win_odds', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('odds_type', sa.String(length=20), nullable=False, server_default='decimal'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('match_id', 'bookmaker', name='uq_match_odds_match_bookmaker'),
        sa.CheckConstraint('home_win_odds > 0', name='ck_match_odds_home_positive'),
        sa.CheckConstraint('draw_odds > 0', name='ck_match_odds_draw_positive'),
        sa.CheckConstraint('away_win_odds > 0', name='ck_match_odds_away_positive'),
        sa.CheckConstraint("odds_type IN ('decimal', 'fractional', 'american')", name='ck_match_odds_type'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_match_odds_match_id'), 'match_odds', ['match_id'], unique=False)
    op.create_index(op.f('ix_match_odds_bookmaker'), 'match_odds', ['bookmaker'], unique=False)

    # Tabla betting_simulations
    op.create_table(
        'betting_simulations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('strategy', sa.String(length=50), nullable=False),
        sa.Column('initial_bankroll', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('final_bankroll', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('total_bets', sa.Integer(), nullable=False),
        sa.Column('winning_bets', sa.Integer(), nullable=False),
        sa.Column('roi', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('max_drawdown', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('parameters', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.CheckConstraint('initial_bankroll > 0', name='ck_betting_sim_initial_positive'),
        sa.CheckConstraint('total_bets >= 0', name='ck_betting_sim_total_nonneg'),
        sa.CheckConstraint('winning_bets >= 0', name='ck_betting_sim_winning_nonneg'),
        sa.CheckConstraint('winning_bets <= total_bets', name='ck_betting_sim_winning_le_total'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_betting_simulations_model'), 'betting_simulations', ['model_name'], unique=False)
    op.create_index(op.f('ix_betting_simulations_strategy'), 'betting_simulations', ['strategy'], unique=False)

    # Tabla betting_history
    op.create_table(
        'betting_history',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('simulation_id', sa.Integer(), nullable=False),
        sa.Column('match_id', sa.Integer(), nullable=False),
        sa.Column('prediction', sa.String(length=10), nullable=False),
        sa.Column('model_probability', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('bookmaker', sa.String(length=50), nullable=False),
        sa.Column('odds', sa.Numeric(precision=10, scale=4), nullable=False),
        sa.Column('stake', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('profit_loss', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('bankroll_after', sa.Numeric(precision=15, scale=2), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False),
        sa.ForeignKeyConstraint(['simulation_id'], ['betting_simulations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['match_id'], ['matches.id']),
        sa.CheckConstraint('stake >= 0', name='ck_betting_history_stake_nonneg'),
        sa.CheckConstraint('odds > 0', name='ck_betting_history_odds_positive'),
        sa.CheckConstraint("prediction IN ('HOME', 'DRAW', 'AWAY')", name='ck_betting_history_prediction'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_betting_history_simulation_id'), 'betting_history', ['simulation_id'], unique=False)
    op.create_index(op.f('ix_betting_history_match_id'), 'betting_history', ['match_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_betting_history_match_id'), table_name='betting_history')
    op.drop_index(op.f('ix_betting_history_simulation_id'), table_name='betting_history')
    op.drop_table('betting_history')

    op.drop_index(op.f('ix_betting_simulations_strategy'), table_name='betting_simulations')
    op.drop_index(op.f('ix_betting_simulations_model'), table_name='betting_simulations')
    op.drop_table('betting_simulations')

    op.drop_index(op.f('ix_match_odds_bookmaker'), table_name='match_odds')
    op.drop_index(op.f('ix_match_odds_match_id'), table_name='match_odds')
    op.drop_table('match_odds')
