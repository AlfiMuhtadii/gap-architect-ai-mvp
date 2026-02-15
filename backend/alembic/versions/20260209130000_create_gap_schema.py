"""create gap schema

Revision ID: 20260209130000
Revises: 
Create Date: 2026-02-09 13:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260209130000"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    gap_analysis_status = sa.Enum(
        "PENDING",
        "DONE",
        "FAILED_VALIDATION",
        "FAILED_LLM",
        name="gap_analysis_status",
        create_type=False,
    )
    llm_run_status = sa.Enum("SUCCESS", "FAILED", name="llm_run_status", create_type=False)

    op.create_table(
        "gap_analyses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("fingerprint", sa.CHAR(length=64), nullable=False),
        sa.Column("resume_text", sa.Text(), nullable=False),
        sa.Column("jd_text", sa.Text(), nullable=False),
        sa.Column("status", gap_analysis_status, nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("prompt_version", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("fingerprint", name="uq_gap_analyses_fingerprint"),
    )
    op.create_index("ix_gap_analyses_status", "gap_analyses", ["status"], unique=False)
    op.create_index("ix_gap_analyses_created_at", "gap_analyses", ["created_at"], unique=False)

    op.create_table(
        "gap_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("gap_analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("missing_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("action_steps", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("interview_questions", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("roadmap_markdown", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["gap_analysis_id"], ["gap_analyses.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("gap_analysis_id", name="uq_gap_results_gap_analysis_id"),
    )

    op.create_table(
        "llm_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("gap_analysis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=255), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("response_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", llm_run_status, nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["gap_analysis_id"], ["gap_analyses.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_llm_runs_gap_analysis_id", "llm_runs", ["gap_analysis_id"], unique=False)
    op.create_index("ix_llm_runs_request_hash", "llm_runs", ["request_hash"], unique=False)
    op.create_index("ix_llm_runs_created_at", "llm_runs", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_llm_runs_created_at", table_name="llm_runs")
    op.drop_index("ix_llm_runs_request_hash", table_name="llm_runs")
    op.drop_index("ix_llm_runs_gap_analysis_id", table_name="llm_runs")
    op.drop_table("llm_runs")

    op.drop_table("gap_results")

    op.drop_index("ix_gap_analyses_created_at", table_name="gap_analyses")
    op.drop_index("ix_gap_analyses_status", table_name="gap_analyses")
    op.drop_table("gap_analyses")

    llm_run_status = sa.Enum("SUCCESS", "FAILED", name="llm_run_status", create_type=False)
    gap_analysis_status = sa.Enum(
        "PENDING",
        "DONE",
        "FAILED_VALIDATION",
        "FAILED_LLM",
        name="gap_analysis_status",
        create_type=False,
    )

    llm_run_status.drop(op.get_bind(), checkfirst=True)
    gap_analysis_status.drop(op.get_bind(), checkfirst=True)
