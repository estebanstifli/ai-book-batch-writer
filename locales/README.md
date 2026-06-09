# Locales

The desktop UI reads all visible text from JSON locale files.

To add a language:

English (`en.json`) and Spanish (`es.json`) are included.

To add another language:

1. Copy `en.json` to a new ISO-language file such as `fr.json`.
2. Translate values without changing keys or format placeholders.
3. Add the language name under `language` in every locale.
4. Keep placeholders such as `{number}`, `{path}` and `{error}` unchanged.
5. Run `pytest tests/test_i18n.py`.

Missing keys fall back to English. If a key is absent from English too, the key
itself is displayed so untranslated UI text is easy to spot during development.
