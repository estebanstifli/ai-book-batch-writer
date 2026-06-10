from __future__ import annotations

import json

from ai_book_batch_writer.project_store import load_project, save_project
from ai_book_batch_writer.settings_store import SettingsStore


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


def test_settings_store_removes_nested_secrets(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save(
        {
            "language": "es",
            "global_llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "api_key": "must-not-be-saved",
                "max_tokens": 4000,
            },
        }
    )

    stored = store.load()
    assert stored["global_llm"]["model"] == "gpt-4o-mini"
    assert stored["global_llm"]["max_tokens"] == 4000
    assert "api_key" not in stored["global_llm"]


def test_settings_store_merges_updates(tmp_path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.save({"language": "en", "appearance": "Dark"})

    stored = store.save_merged({"language": "es"})

    assert stored["language"] == "es"
    assert stored["appearance"] == "Dark"


def test_settings_store_migrates_old_3000_token_default(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "global_llm": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "max_tokens": 3000,
                }
            }
        ),
        encoding="utf-8",
    )

    stored = SettingsStore(path).load()

    assert stored["global_llm"]["max_tokens"] == 8096
    assert stored["settings_schema_version"] == 2
