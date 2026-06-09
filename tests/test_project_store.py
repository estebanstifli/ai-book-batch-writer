from __future__ import annotations

import json

from ai_book_batch_writer.project_store import load_project, save_project


def test_project_round_trip_omits_api_key(tmp_path, sample_project) -> None:
    sample_project.token_usage.add_outline(125)
    sample_project.token_usage.add_book(875)
    path = tmp_path / "project.json"
    save_project(sample_project, path)

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert "api_key" not in raw["llm_settings"]

    loaded = load_project(path)
    assert loaded.settings.title == sample_project.settings.title
    assert loaded.chapters[0].sections[0].content.startswith("Document")
    assert loaded.llm_settings is not None
    assert loaded.llm_settings.api_key is None
    assert loaded.token_usage.total_tokens == 1000
