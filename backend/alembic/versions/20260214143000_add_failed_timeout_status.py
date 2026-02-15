"""add failed timeout status

Revision ID: 20260214143000
Revises: 20260213173000
Create Date: 2026-02-14 14:30:00
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260214143000"
down_revision = "20260213173000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE gap_analysis_status ADD VALUE IF NOT EXISTS 'FAILED_TIMEOUT'")


def downgrade() -> None:
    # PostgreSQL enum value removal is non-trivial and intentionally skipped.
    pass
