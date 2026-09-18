"""baseline

Revision ID: 0001
Revises:
Create Date: 2026-09-17 00:00:00
"""

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Baseline vacío. Las tablas se crean en la migración 0002 (Fase 2)."""
    pass


def downgrade() -> None:
    pass