import uuid
from dataclasses import dataclass

from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult


@dataclass
class GapAnalysisPayloadFactory:
    resume_text: str = (
        "Senior engineer with eight years building backend services using Python, FastAPI, "
        "PostgreSQL, Docker, Kubernetes, AWS, CI/CD, monitoring, and testing. "
        "Led migration projects, designed APIs, improved reliability, mentored teammates, "
        "collaborated with product and design, and delivered production systems with measurable impact. "
        "Comfortable with incident management, on-call rotation, documentation, code reviews, "
        "and iterative delivery in agile teams."
    )
    jd_text: str = (
        "We are looking for a backend engineer to build API services with Python, SQL, AWS, "
        "Docker, Kubernetes, observability, security, testing, and scalable architecture. "
        "Candidates should collaborate cross-functionally, write clean code, review pull requests, "
        "own incident response, and deliver reliable production features. "
        "The role includes sprint planning, technical documentation, mentoring, and continuous "
        "improvement of system reliability and developer experience."
    )
    model: str = "gpt-test"
    prompt_version: str = "v1"

    def build(self) -> dict:
        return {
            "resume_text": self.resume_text,
            "jd_text": self.jd_text,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }


def make_gap_analysis(**kwargs) -> GapAnalysis:
    defaults = dict(
        id=uuid.uuid4(),
        fingerprint="f" * 64,
        resume_text="r",
        jd_text="j",
        status=GapAnalysisStatus.PENDING,
        model="m",
        prompt_version="v1",
    )
    defaults.update(kwargs)
    return GapAnalysis(**defaults)


def make_gap_result(gap_analysis_id) -> GapResult:
    return GapResult(
        gap_analysis_id=gap_analysis_id,
        missing_skills=["a", "b"],
        top_priority_skills=["a"],
        action_steps=[{"title": "t", "why": "w", "deliverable": "d"}] * 3,
        interview_questions=[
            {"question": "q", "focus_gap": "g", "what_good_looks_like": "w"}
        ]
        * 3,
        roadmap_markdown="rm",
        match_percent=80.0,
        match_reason="Matched 8 of 10 skills",
    )
