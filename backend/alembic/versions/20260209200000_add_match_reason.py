"""add match_reason to gap_results

Revision ID: 20260209200000
Revises: 20260209190000
Create Date: 2026-02-09 20:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260209200000"
down_revision = "20260209190000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("gap_results", sa.Column("match_reason", sa.Text(), nullable=True))
    op.add_column("gap_results", sa.Column("top_priority_skills", postgresql.JSONB(), nullable=True))
 

def downgrade() -> None:
    op.drop_column("gap_results", "top_priority_skills")
    op.drop_column("gap_results", "match_reason")
