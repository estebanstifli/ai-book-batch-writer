from __future__ import annotations

import json

from ai_book_batch_writer.model_catalog import ModelCatalogStore


def test_model_catalog_round_trip(tmp_path) -> None:
    store = ModelCatalogStore(tmp_path / "model_cache.json")

    stored = store.update(
        "openai",
        ["gpt-4o-mini", "gpt-4.1", "gpt-4o-mini", "  "],
    )

    assert stored == ["gpt-4.1", "gpt-4o-mini"]
    assert store.models_for("openai") == stored
    payload = json.loads(store.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "updated_at" in payload["providers"]["openai"]


def test_invalid_model_catalog_returns_empty_data(tmp_path) -> None:
    path = tmp_path / "model_cache.json"
    path.write_text("{not json", encoding="utf-8")

    assert ModelCatalogStore(path).models_for("openai") == []
