"""add partial index for pending timeout sweep

Revision ID: 20260215184500
Revises: 20260214143000
Create Date: 2026-02-15 18:45:00
"""

from alembic import op


revision = "20260215184500"
down_revision = "20260214143000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_gap_analyses_pending_updated_at
        ON gap_analyses (updated_at)
        WHERE status = 'PENDING'
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_gap_analyses_pending_updated_at")

