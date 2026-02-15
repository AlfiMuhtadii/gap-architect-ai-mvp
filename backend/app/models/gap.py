import uuid
from datetime import datetime
from sqlalchemy import String, Text, DateTime, Enum, ForeignKey, CHAR, Index, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func
from app.db.base import Base
import enum

class GapAnalysisStatus(str, enum.Enum):
    PENDING = "PENDING"
    DONE = "DONE"
    FAILED_VALIDATION = "FAILED_VALIDATION"
    FAILED_LLM = "FAILED_LLM"
    FAILED_TIMEOUT = "FAILED_TIMEOUT"

class LlmRunStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class JdCleanStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

class GapAnalysis(Base):
    __tablename__ = "gap_analyses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fingerprint: Mapped[str] = mapped_column(CHAR(64), unique=True, nullable=False)
    resume_text: Mapped[str] = mapped_column(Text, nullable=False)
    jd_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[GapAnalysisStatus] = mapped_column(
        Enum(GapAnalysisStatus, name="gap_analysis_status"), nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    gap_result = relationship("GapResult", back_populates="gap_analysis", uselist=False, cascade="all, delete-orphan")
    llm_runs = relationship("LlmRun", back_populates="gap_analysis", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_gap_analyses_created_at", "created_at"),
    )

class GapResult(Base):
    __tablename__ = "gap_results"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gap_analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("gap_analyses.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    missing_skills: Mapped[dict] = mapped_column(JSONB, nullable=False)
    top_priority_skills: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    action_steps: Mapped[dict] = mapped_column(JSONB, nullable=False)
    interview_questions: Mapped[dict] = mapped_column(JSONB, nullable=False)
    roadmap_markdown: Mapped[str] = mapped_column(Text, nullable=False)
    match_percent: Mapped[float | None] = mapped_column(Float, nullable=True)
    match_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    gap_analysis = relationship("GapAnalysis", back_populates="gap_result")

class LlmRun(Base):
    __tablename__ = "llm_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    gap_analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("gap_analyses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    response_json: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[LlmRunStatus] = mapped_column(Enum(LlmRunStatus, name="llm_run_status"), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    gap_analysis = relationship("GapAnalysis", back_populates="llm_runs")

    __table_args__ = (
        Index("ix_llm_runs_created_at", "created_at"),
    )


class JdCleanRun(Base):
    __tablename__ = "jd_clean_runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    input_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False, unique=True, index=True)
    clean_strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[JdCleanStatus] = mapped_column(
        Enum(JdCleanStatus, name="jd_clean_status"), nullable=False, index=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
