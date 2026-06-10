from __future__ import annotations

from datetime import datetime, timezone

from ai_book_batch_writer.utils import (
    create_project_directory,
    safe_filename,
)


def test_safe_filename_handles_unicode_and_punctuation() -> None:
    assert safe_filename("Guía: IA & Python") == "guia-ia-python"


def test_create_project_directory_avoids_collisions(tmp_path) -> None:
    created = datetime(2026, 6, 10, 12, 30, tzinfo=timezone.utc)

    first = create_project_directory(tmp_path, "My Book", created)
    second = create_project_directory(tmp_path, "My Book", created)

    assert first.name.endswith("-my-book")
    assert second.name.endswith("-my-book-2")
