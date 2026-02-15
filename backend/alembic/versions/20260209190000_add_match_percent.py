"""add match_percent to gap_results

Revision ID: 20260209190000
Revises: 20260209130000
Create Date: 2026-02-09 19:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "20260209190000"
down_revision = "20260209130000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gap_results", sa.Column("match_percent", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("gap_results", "match_percent")
