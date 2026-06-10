"""Discover saved book projects for the maintenance dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ai_book_batch_writer.models import BookProject
from ai_book_batch_writer.project_store import load_project


@dataclass(frozen=True)
class ProjectSummary:
    """Small, display-friendly representation of one saved project."""

    path: Path
    title: str
    status: str
    completed_chapters: int
    total_chapters: int
    failed_chapters: int
    total_tokens: int
    updated_at: datetime


def discover_projects(directory: str | Path) -> list[ProjectSummary]:
    """Return valid project files sorted by most recent update."""
    root = Path(directory)
    if not root.exists():
        return []
    candidates = set(root.rglob("*.json"))
    summaries: list[ProjectSummary] = []
    for path in candidates:
        try:
            project = load_project(path)
        except Exception:
            continue
        summaries.append(_summary(project, path))
    return sorted(
        summaries,
        key=lambda summary: summary.updated_at,
        reverse=True,
    )


def _summary(project: BookProject, path: Path) -> ProjectSummary:
    completed = sum(chapter.is_complete for chapter in project.chapters)
    failed = sum(chapter.status == "failed" for chapter in project.chapters)
    total = len(project.chapters)
    status = "completed" if total > 0 and completed == total else "incomplete"
    return ProjectSummary(
        path=path,
        title=project.settings.title or "Untitled Book",
        status=status,
        completed_chapters=completed,
        total_chapters=total,
        failed_chapters=failed,
        total_tokens=project.token_usage.total_tokens,
        updated_at=project.updated_at,
    )
