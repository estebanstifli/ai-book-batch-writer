from __future__ import annotations

from ai_book_batch_writer.models import BookChapter, BookProject, BookSettings
from ai_book_batch_writer.project_index import discover_projects
from ai_book_batch_writer.project_store import load_project, save_project


def test_discover_projects_lists_complete_and_incomplete(tmp_path) -> None:
    complete = BookProject(
        settings=BookSettings(title="Complete", idea="Done"),
        chapters=[
            BookChapter(
                number=1,
                title="One",
                summary="Done",
                content="Text",
                status="completed",
            )
        ],
    )
    incomplete = BookProject(
        settings=BookSettings(title="Incomplete", idea="Continue"),
        chapters=[
            BookChapter(
                number=1,
                title="One",
                summary="Pending",
                status="failed",
                error="Temporary provider error",
            )
        ],
    )
    save_project(complete, tmp_path / "complete" / "project.json")
    save_project(incomplete, tmp_path / "incomplete" / "draft-state.json")

    summaries = discover_projects(tmp_path)

    assert {summary.status for summary in summaries} == {
        "completed",
        "incomplete",
    }
    assert {summary.title for summary in summaries} == {
        "Complete",
        "Incomplete",
    }


def test_loading_interrupted_project_resets_generating_status(tmp_path) -> None:
    project = BookProject(
        settings=BookSettings(title="Interrupted", idea="Resume"),
        chapters=[
            BookChapter(
                number=1,
                title="One",
                summary="Interrupted",
                status="generating",
            )
        ],
    )
    path = tmp_path / "project.json"
    save_project(project, path)

    loaded = load_project(path)

    assert loaded.chapters[0].status == "pending"
