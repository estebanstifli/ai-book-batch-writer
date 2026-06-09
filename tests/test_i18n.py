from __future__ import annotations

import json
import re
from pathlib import Path

from ai_book_batch_writer.i18n import Translator


def test_translation_loading_and_nested_keys() -> None:
    translator = Translator()

    assert translator.t("app.title") == "AI Book Batch Writer"
    assert translator.t("log.chapter_started", number=2, title="Testing") == (
        "Starting chapter 2: Testing"
    )
    translator.set_language("es")
    assert translator.t("button.generate_book") == "Generar libro"


def test_missing_translation_falls_back_to_english(tmp_path) -> None:
    (tmp_path / "en.json").write_text(
        json.dumps({"button": {"save": "Save"}}),
        encoding="utf-8",
    )
    (tmp_path / "es.json").write_text(
        json.dumps({"button": {}}),
        encoding="utf-8",
    )
    translator = Translator(language="es", locales_dir=tmp_path)

    assert translator.t("button.save") == "Save"
    assert translator.t("missing.key") == "missing.key"


def test_static_ui_translation_keys_exist() -> None:
    translator = Translator()
    app_source = (
        Path(__file__).parents[1]
        / "src"
        / "ai_book_batch_writer"
        / "app.py"
    ).read_text(encoding="utf-8")
    keys = set(
        re.findall(
            r'(?<![\w.])(?:self\.)?t\("([^"]+)"',
            app_source,
        )
    )

    missing = [key for key in sorted(keys) if translator.t(key) == key]
    assert missing == []


def test_english_and_spanish_have_matching_keys() -> None:
    locales_dir = Path(__file__).parents[1] / "locales"
    english = json.loads((locales_dir / "en.json").read_text(encoding="utf-8"))
    spanish = json.loads((locales_dir / "es.json").read_text(encoding="utf-8"))

    def flatten(data: dict, prefix: str = "") -> set[str]:
        keys: set[str] = set()
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                keys.update(flatten(value, full_key))
            else:
                keys.add(full_key)
        return keys

    assert flatten(english) == flatten(spanish)
