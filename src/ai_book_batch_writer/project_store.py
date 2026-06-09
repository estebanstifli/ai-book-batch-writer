"""Safe JSON persistence for book projects."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from ai_book_batch_writer.models import BookProject
from ai_book_batch_writer.utils import ensure_parent_dir


def save_project(project: BookProject, path: str | Path) -> None:
    """Save a project atomically without persisting API keys."""
    target = ensure_parent_dir(path)
    project.touch()
    payload = project.model_dump(mode="json", exclude={"llm_settings": {"api_key"}})

    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.stem}-",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, target)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def load_project(path: str | Path) -> BookProject:
    """Load and validate a project JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BookProject.model_validate(payload)

