"""add error_message to jd_clean_runs

Revision ID: 20260213173000
Revises: 20260213170000
Create Date: 2026-02-13 17:30:00
"""

from alembic import op
import sqlalchemy as sa


revision = "20260213173000"
down_revision = "20260213170000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jd_clean_runs", sa.Column("error_message", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("jd_clean_runs", "error_message")
