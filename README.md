# AI Book Batch Writer

AI Book Batch Writer is a Windows-friendly Python desktop application for
creating structured long-form drafts with large language models. It generates
an editable JSON outline first, then writes the project chapter by chapter or
section by section with progress reporting, local project files and practical
export formats.

> This tool creates drafts, not publish-ready books. Review, fact-check and edit
> every generated result before publication or professional use.

## Screenshots

Application screenshots will be added under [`screenshots/`](screenshots/) as
the interface evolves.

| Home dashboard | Guided creation | Book maintenance |
| --- | --- | --- |
| Large task cards | Locked 1 → 2 → 3 workflow | Load, review and continue projects |

## Features

- Outline-first writing workflow with a human review step
- Dashboard with Create Books, Book Maintenance, Global Settings and Utilities
- Guided 1 → 2 → 3 creation flow that locks the outline after writing starts
- Reusable global model defaults with per-book overrides
- Provider connection tests for cloud credentials and local Ollama
- Editable model dropdowns populated from each provider's live catalog
- Local JSON model cache for fast startup and offline reuse
- Friendly multi-field validation without exposing internal exceptions
- Editable output-language selector backed by 693 Unicode CLDR entries
- Modal generation progress with cancellation and live chapter status
- Automatic project save plus Markdown, TXT, DOCX and PDF exports
- OpenAI, OpenRouter, Anthropic Claude, Google Gemini and local Ollama
  providers through dedicated LangChain integrations
- Real LangChain prompt templates, chat model abstraction and invocation chains
- Pydantic validation for settings, outlines and saved projects
- JSON extraction plus model-assisted repair for malformed outlines
- Sequential chapter and section generation with continuity context
- Persistent outline, book and total token usage counters
- Responsive CustomTkinter UI using a background worker and event queue
- Cancellation between model requests
- Save and load JSON projects without persisting API keys
- Markdown, TXT and DOCX export
- English and Spanish UI backed by nested JSON translations and English fallback
- Retry with exponential backoff and file-based diagnostic logging
- PyInstaller build script for a standalone Windows executable

## Why This Project Exists

The project began as a small batch-generation script. The original concept was
useful: create a structure, generate each part separately, preserve recent
context and show progress. This repository rebuilds that workflow as a
maintainable portfolio application with clear boundaries between UI, model
providers, generation, persistence, validation and export.

The legacy script is intentionally not included. It mixed credentials, model
calls, global state and UI updates in one file.

## Tech Stack

- Python 3.11+
- CustomTkinter
- LangChain with OpenAI, OpenRouter, Anthropic, Google Gemini and Ollama integrations
- Pydantic 2
- python-dotenv
- python-docx
- ReportLab
- tenacity
- pytest
- PyInstaller

## Architecture

```text
src/ai_book_batch_writer/
├─ app.py                 # Dashboard, guided workflow and worker queue
├─ models.py              # Pydantic domain models
├─ llm_providers.py       # Multi-provider LangChain factory
├─ provider_discovery.py  # Connection tests and live model catalogs
├─ model_catalog.py       # Credential-free JSON model cache
├─ generation_service.py  # Outline and long-form orchestration
├─ prompts.py             # ChatPromptTemplate definitions
├─ json_utils.py          # Extraction, validation and repair
├─ project_store.py       # API-key-safe JSON persistence
├─ exporters.py           # Markdown, TXT and DOCX
└─ i18n.py                # JSON translation loader
```

Provider-specific code is isolated in `llm_providers.py`. Each integration
returns LangChain's common `BaseChatModel` interface, so the prompt and
generation services remain provider-independent.

Model discovery uses each provider's official REST catalog. Refreshed model
identifiers are stored under the user's application data directory in
`model_cache.json`; the cache contains model names and timestamps only.

The output-language selector is backed by the Unicode CLDR language catalog
and remains editable for custom names or language variants. The bundled source
and attribution are documented in [`data/README.md`](data/README.md).

## Install From Source

```powershell
git clone https://github.com/estebanstifli/ai-book-batch-writer.git
cd ai-book-batch-writer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ai_book_batch_writer.main
```

The console script is also available after installation:

```powershell
ai-book-batch-writer
```

## Using OpenAI

Copy `.env.example` to `.env` and add a key:

```env
OPENAI_API_KEY=
```

Alternatively, enter a key in the desktop UI for the current session. Keys
entered in the UI are kept in memory and are not written to project files or
desktop preference files.

Example settings:

```text
Provider: OpenAI
Model: gpt-4o-mini
```

Model availability changes over time. Enter any compatible model name enabled
for your account.

## Other Cloud Providers

The application also supports dedicated LangChain integrations for:

| Provider | Environment variable | Example model |
| --- | --- | --- |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |
| Anthropic Claude | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| Google Gemini | `GOOGLE_API_KEY` | `gemini-2.5-flash` |

Cloud API endpoints are managed by their integrations and are not exposed in
the UI. The API Base URL field is shown only for Ollama. Use **Test** to
validate the current credential, or **Refresh Models** to update the editable
model dropdown from the provider.

## Using Ollama Locally

Install [Ollama](https://ollama.com/), then download and run a model:

```bash
ollama pull llama3.1
ollama serve
```

Use these application settings:

```text
Provider: Ollama
Model: llama3.1
API Base URL: http://localhost:11434
```

The Windows executable does not include Ollama or model weights. Users must
install Ollama and download their chosen model separately.

## Workflow

1. Save reusable provider and model defaults under **Global Settings**.
2. Choose **Create Books** from the dashboard.
3. Adjust the preloaded model settings for this book when needed.
4. Describe the book, guide, manual, tutorial or structured article.
5. Generate, review and edit the JSON outline.
6. Start book generation. The setup and outline steps are then locked to keep
   generated content consistent with its structure.
7. Monitor each chapter or section in the modal progress window and log.
8. When generation finishes, the project and every export format are saved
   automatically in a dedicated project folder.

Saved projects can be reopened from **Book Maintenance**. Export commands and
working-folder shortcuts are also available under **Other Utilities**.

## Export Formats

- **JSON:** complete editable project state, excluding API keys
- **Markdown:** metadata, linked table of contents, chapters and sections
- **TXT:** portable plain text
- **DOCX:** Word document with title, chapter and section heading levels
- **PDF:** clean A4 reading/printing document without project metadata

Automatic completion creates:

```text
projects/<timestamp>-<book-title>/
├─ project.json
├─ book.md
├─ book.txt
├─ book.docx
└─ book.pdf
```

When running the Windows executable, these folders are created under
`Documents\AI Book Batch Writer\Projects` so they remain available after the
one-file application exits.

## Build a Windows Executable

From a Windows command prompt:

```bat
build_windows.bat
```

The executable is created under `dist\AI Book Batch Writer.exe`. The script
bundles the locale catalog and CustomTkinter assets and declares all supported
provider integration entry modules for PyInstaller. It builds inside an isolated
`.build-venv` so unrelated packages from a global Python installation are not
accidentally bundled.

PyInstaller output should be tested on a clean Windows machine before release.
Code signing and an installer are recommended for public distribution.

## Tests

```powershell
pytest
```

The test suite covers translations, JSON extraction and repair behavior,
prompt construction, project persistence, exporters and the generation
workflow with a fake LangChain chat model. It also constructs all five provider
adapters without making API calls.

## Localization

All UI text is loaded from JSON locale files through nested keys such as:

```python
t("button.generate_outline")
t("label.model")
t("error.missing_api_key")
```

English and Spanish are included. The interface can switch languages at runtime
without losing the form, outline, preview or log contents. See
[`locales/README.md`](locales/README.md) for adding another language. Missing
keys fall back to English.

## Security Notes

- Never commit `.env` or real credentials.
- Project JSON files and preference files omit API keys.
- UI-entered credentials, including keys entered in Global Settings, are
  session-only.
- The model catalog cache never stores credentials or request headers.
- Automatic project JSON files exclude API keys.
- Revoke a key immediately if it is ever exposed in source code, logs,
  screenshots or commit history.
- Review exported content before sharing it; generated text can contain
  inaccurate or unsafe claims.

## Ethical and Content Note

Use this application for drafting and assisted research, not unattended mass
publication. Respect copyright, provider terms, privacy, disclosure
requirements and applicable law. Human authors remain responsible for
originality, factual accuracy, citations and the impact of published content.

## Roadmap

- Structured outline editor with inline chapter and section controls
- Resume/regenerate controls for individual failed sections
- Streaming output and richer usage/cost metrics
- Optional LiteLLM and Azure OpenAI adapters
- Optional encrypted credential storage
- Additional UI translations
- Release automation, code signing and installer packaging

## License

Released under the [MIT License](LICENSE).
