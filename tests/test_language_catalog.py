from __future__ import annotations

import json

from ai_book_batch_writer.language_catalog import load_output_languages


def test_bundled_language_catalog_is_large() -> None:
    languages = load_output_languages()

    assert len(languages) >= 600
    assert "English" in languages
    assert "Spanish" in languages
    assert languages == sorted(set(languages), key=str.casefold)


def test_language_catalog_falls_back_when_invalid(tmp_path) -> None:
    path = tmp_path / "languages.json"
    path.write_text(json.dumps({"invalid": []}), encoding="utf-8")

    assert load_output_languages(path) == ["English", "Spanish"]
