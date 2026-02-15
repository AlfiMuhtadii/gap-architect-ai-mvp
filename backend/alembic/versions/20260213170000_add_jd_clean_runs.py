"""add jd_clean_runs metadata table

Revision ID: 20260213170000
Revises: 20260209200000
Create Date: 2026-02-13 17:00:00
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260213170000"
down_revision = "20260209200000"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Idempotent enum creation (handles partial failed previous migration runs).
    op.execute(
        """
        DO $$
        BEGIN
            CREATE TYPE jd_clean_status AS ENUM ('SUCCESS', 'FAILED');
        EXCEPTION
            WHEN duplicate_object THEN NULL;
        END $$;
        """
    )
    jd_clean_status = postgresql.ENUM("SUCCESS", "FAILED", name="jd_clean_status", create_type=False)

    op.create_table(
        "jd_clean_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("input_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("clean_strategy", sa.String(length=64), nullable=False),
        sa.Column("status", jd_clean_status, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("input_hash"),
    )
    op.create_index("ix_jd_clean_runs_input_hash", "jd_clean_runs", ["input_hash"], unique=True)
    op.create_index("ix_jd_clean_runs_status", "jd_clean_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_jd_clean_runs_status", table_name="jd_clean_runs")
    op.drop_index("ix_jd_clean_runs_input_hash", table_name="jd_clean_runs")
    op.drop_table("jd_clean_runs")
    jd_clean_status = postgresql.ENUM("SUCCESS", "FAILED", name="jd_clean_status", create_type=False)
    jd_clean_status.drop(op.get_bind(), checkfirst=True)
