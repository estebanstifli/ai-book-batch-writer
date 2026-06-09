"""CustomTkinter desktop application."""

from __future__ import annotations

import json
import queue
import threading
from datetime import datetime
from tkinter import filedialog, messagebox
from typing import Any, Callable

import customtkinter as ctk
from pydantic import ValidationError

from ai_book_batch_writer.config import PROJECT_ROOT, load_environment
from ai_book_batch_writer.exporters import (
    export_docx,
    export_markdown,
    export_txt,
    render_markdown,
)
from ai_book_batch_writer.generation_service import BookGenerationService
from ai_book_batch_writer.i18n import Translator
from ai_book_batch_writer.json_utils import parse_outline_document
from ai_book_batch_writer.llm_providers import (
    PROVIDER_SPECS,
    get_provider_spec,
    provider_api_key,
    provider_key_env_display,
    provider_requires_api_key,
    provider_supports_api_base,
)
from ai_book_batch_writer.models import (
    BookOutline,
    BookProject,
    BookSettings,
    GenerationEvent,
    LLMSettings,
    TokenUsage,
)
from ai_book_batch_writer.project_store import load_project, save_project
from ai_book_batch_writer.settings_store import SettingsStore
from ai_book_batch_writer.utils import CancelToken, GenerationCancelled

WorkerFunction = Callable[[], Any]


class AIBookBatchWriterApp(ctk.CTk):
    """Main desktop window and UI-thread event coordinator."""

    def __init__(self) -> None:
        load_environment()
        self.settings_store = SettingsStore()
        self.preferences = self.settings_store.load()
        self.translator = Translator(self.preferences.get("language", "en"))
        self._refresh_locale_maps()

        appearance = self.preferences.get("appearance", "System")
        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.project: BookProject | None = None
        self.cancel_token = CancelToken()
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.is_busy = False

        self.title(self.t("app.title"))
        self.geometry("1420x900")
        self.minsize(1120, 720)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._create_variables()
        self._build_ui()
        self._apply_provider_defaults(force=False)
        self.after(100, self._poll_worker_queue)

    def t(self, key: str, **kwargs: Any) -> str:
        return self.translator.t(key, **kwargs)

    def _refresh_locale_maps(self) -> None:
        self.provider_labels = {
            self.t(f"provider.{code}"): code for code in PROVIDER_SPECS
        }
        self.appearance_labels = {
            self.t("appearance.system"): "System",
            self.t("appearance.dark"): "Dark",
            self.t("appearance.light"): "Light",
        }
        self.language_labels = {
            self.t(f"language.{code}"): code
            for code in self.translator.available_languages()
        }

    def _create_variables(self) -> None:
        provider_code = str(self.preferences.get("provider", "openai")).lower()
        if provider_code not in PROVIDER_SPECS:
            provider_code = "openai"
        provider = next(
            (
                label
                for label, code in self.provider_labels.items()
                if code == provider_code
            ),
            self.t("provider.openai"),
        )
        self.provider_var = ctk.StringVar(value=provider)
        self.model_var = ctk.StringVar(
            value=self.preferences.get(
                "model",
                get_provider_spec(provider_code).default_model,
            )
        )
        self.api_key_var = ctk.StringVar()
        self.api_base_var = ctk.StringVar(
            value=self.preferences.get("api_base", "")
        )
        self.temperature_var = ctk.StringVar(value=self.t("default.temperature"))
        self.max_tokens_var = ctk.StringVar(value=self.t("default.max_tokens"))
        self.title_var = ctk.StringVar()
        self.output_language_var = ctk.StringVar(
            value=self.t("default.output_language")
        )
        self.tone_var = ctk.StringVar(value=self.t("default.tone"))
        self.audience_var = ctk.StringVar(value=self.t("default.target_audience"))
        self.chapter_count_var = ctk.StringVar(
            value=self.t("default.chapter_count")
        )
        self.words_per_chapter_var = ctk.StringVar(
            value=self.t("default.words_per_chapter")
        )
        self.status_var = ctk.StringVar(value=self.t("status.ready"))
        self.task_var = ctk.StringVar(value=self.t("status.ready"))
        self.token_var = ctk.StringVar(value="0")
        self.language_var = ctk.StringVar(
            value=next(
                (
                    label
                    for label, code in self.language_labels.items()
                    if code == self.translator.language
                ),
                self.t("language.en"),
            )
        )
        self.appearance_var = ctk.StringVar(
            value=next(
                (
                    label
                    for label, code in self.appearance_labels.items()
                    if code == self.preferences.get("appearance", "System")
                ),
                self.t("appearance.system"),
            )
        )

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_box = ctk.CTkFrame(header, fg_color="transparent")
        title_box.grid(row=0, column=0, padx=20, pady=14, sticky="w")
        ctk.CTkLabel(
            title_box,
            text=self.t("app.title"),
            font=ctk.CTkFont(size=24, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_box,
            text=self.t("app.tagline"),
            text_color=("gray35", "gray70"),
        ).pack(anchor="w")

        header_controls = ctk.CTkFrame(header, fg_color="transparent")
        header_controls.grid(row=0, column=1, padx=20, pady=14, sticky="e")
        ctk.CTkLabel(
            header_controls,
            text=self.t("label.interface_language"),
        ).grid(row=0, column=0, padx=(0, 6))
        self.language_menu = ctk.CTkOptionMenu(
            header_controls,
            values=list(self.language_labels),
            variable=self.language_var,
            command=self._change_language,
            width=110,
        )
        self.language_menu.grid(row=0, column=1, padx=(0, 14))
        ctk.CTkLabel(
            header_controls,
            text=self.t("label.appearance"),
        ).grid(row=0, column=2, padx=(0, 6))
        ctk.CTkOptionMenu(
            header_controls,
            values=list(self.appearance_labels),
            variable=self.appearance_var,
            command=self._change_appearance,
            width=100,
        ).grid(row=0, column=3)

        actions = ctk.CTkFrame(self, corner_radius=0)
        actions.grid(row=1, column=0, padx=0, pady=(1, 0), sticky="ew")
        button_specs = [
            ("button.generate_outline", self._generate_outline),
            ("button.generate_book", self._generate_book),
            ("button.cancel", self._cancel_generation),
            ("button.save_project", self._save_project),
            ("button.load_project", self._load_project),
            ("button.export_markdown", lambda: self._export("md")),
            ("button.export_txt", lambda: self._export("txt")),
            ("button.export_docx", lambda: self._export("docx")),
        ]
        self.action_buttons: dict[str, ctk.CTkButton] = {}
        for column, (key, command) in enumerate(button_specs):
            button = ctk.CTkButton(
                actions,
                text=self.t(key),
                command=command,
                height=34,
            )
            button.grid(row=0, column=column, padx=(12 if column == 0 else 4, 4), pady=10)
            self.action_buttons[key] = button
        self.action_buttons["button.cancel"].configure(state="disabled")

        self.tabs = ctk.CTkTabview(self)
        self.tabs.grid(row=2, column=0, padx=12, pady=12, sticky="nsew")
        self.setup_tab = self.tabs.add(self.t("tab.setup"))
        self.outline_tab = self.tabs.add(self.t("tab.outline"))
        self.book_tab = self.tabs.add(self.t("tab.book"))
        self._build_setup_tab()
        self._build_outline_tab()
        self._build_book_tab()

        footer = ctk.CTkFrame(self, corner_radius=0)
        footer.grid(row=3, column=0, sticky="ew")
        footer.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            footer,
            text=self.t("label.current_task"),
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=0, padx=(16, 8), pady=10)
        ctk.CTkLabel(footer, textvariable=self.task_var).grid(
            row=0,
            column=1,
            sticky="w",
        )
        self.progress_bar = ctk.CTkProgressBar(footer, width=280)
        self.progress_bar.grid(row=0, column=2, padx=12, pady=10)
        self.progress_bar.set(0)
        ctk.CTkLabel(
            footer,
            text=self.t("label.total_tokens"),
            font=ctk.CTkFont(weight="bold"),
        ).grid(row=0, column=3, padx=(0, 6), pady=10)
        ctk.CTkLabel(
            footer,
            textvariable=self.token_var,
            width=90,
            anchor="w",
        ).grid(row=0, column=4, padx=(0, 12), pady=10)
        ctk.CTkLabel(
            footer,
            textvariable=self.status_var,
            width=170,
            anchor="e",
        ).grid(row=0, column=5, padx=(0, 16))

    def _build_setup_tab(self) -> None:
        self.setup_tab.grid_columnconfigure((0, 1), weight=1)
        self.setup_tab.grid_rowconfigure(0, weight=1)

        model_panel = ctk.CTkScrollableFrame(
            self.setup_tab,
            label_text=self.t("panel.model_settings"),
        )
        model_panel.grid(row=0, column=0, padx=(8, 5), pady=8, sticky="nsew")
        model_panel.grid_columnconfigure(1, weight=1)

        self._label(model_panel, 0, "label.provider")
        self.provider_menu = ctk.CTkOptionMenu(
            model_panel,
            values=list(self.provider_labels),
            variable=self.provider_var,
            command=lambda _: self._apply_provider_defaults(force=True),
        )
        self.provider_menu.grid(row=0, column=1, padx=10, pady=8, sticky="ew")
        self._entry(model_panel, 1, "label.model", self.model_var)
        self.api_key_label = self._label(
            model_panel,
            2,
            "label.api_key",
        )
        self.api_key_entry = ctk.CTkEntry(
            model_panel,
            textvariable=self.api_key_var,
            show="*",
            placeholder_text=self.t("placeholder.api_key"),
        )
        self.api_key_entry.grid(row=2, column=1, padx=10, pady=8, sticky="ew")
        self.api_base_label = self._label(
            model_panel,
            3,
            "label.api_base",
        )
        self.api_base_entry = ctk.CTkEntry(
            model_panel,
            textvariable=self.api_base_var,
        )
        self.api_base_entry.grid(row=3, column=1, padx=10, pady=8, sticky="ew")
        self._entry(model_panel, 4, "label.temperature", self.temperature_var)
        self._entry(model_panel, 5, "label.max_tokens", self.max_tokens_var)

        book_panel = ctk.CTkScrollableFrame(
            self.setup_tab,
            label_text=self.t("panel.book_settings"),
        )
        book_panel.grid(row=0, column=1, padx=(5, 8), pady=8, sticky="nsew")
        book_panel.grid_columnconfigure(1, weight=1)

        self._entry(book_panel, 0, "label.book_title", self.title_var)
        self._label(book_panel, 1, "label.idea")
        self.idea_text = ctk.CTkTextbox(book_panel, height=120)
        self.idea_text.grid(row=1, column=1, padx=10, pady=8, sticky="ew")
        self._set_placeholder(self.idea_text, "placeholder.main_idea")
        self._entry(
            book_panel,
            2,
            "label.output_language",
            self.output_language_var,
        )
        self._entry(book_panel, 3, "label.tone", self.tone_var)
        self._entry(book_panel, 4, "label.target_audience", self.audience_var)
        self._entry(
            book_panel,
            5,
            "label.chapter_count",
            self.chapter_count_var,
        )
        self._entry(
            book_panel,
            6,
            "label.words_per_chapter",
            self.words_per_chapter_var,
        )
        self._label(book_panel, 7, "label.additional_instructions")
        self.instructions_text = ctk.CTkTextbox(book_panel, height=110)
        self.instructions_text.grid(row=7, column=1, padx=10, pady=8, sticky="ew")
        self._set_placeholder(
            self.instructions_text,
            "placeholder.additional_instructions",
        )

    def _build_outline_tab(self) -> None:
        self.outline_tab.grid_columnconfigure(0, weight=1)
        self.outline_tab.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            self.outline_tab,
            text=self.t("panel.outline_editor"),
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        self.outline_text = ctk.CTkTextbox(
            self.outline_tab,
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=13),
        )
        self.outline_text.grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")
        self._set_placeholder(self.outline_text, "placeholder.outline")

    def _build_book_tab(self) -> None:
        self.book_tab.grid_columnconfigure(0, weight=3)
        self.book_tab.grid_columnconfigure(1, weight=2)
        self.book_tab.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self.book_tab,
            text=self.t("panel.result_preview"),
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")

        log_header = ctk.CTkFrame(self.book_tab, fg_color="transparent")
        log_header.grid(row=0, column=1, padx=12, pady=(8, 0), sticky="ew")
        log_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            log_header,
            text=self.t("panel.progress_log"),
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            log_header,
            text=self.t("button.clear_log"),
            width=100,
            command=self._clear_log,
        ).grid(row=0, column=1, sticky="e")

        self.preview_text = ctk.CTkTextbox(
            self.book_tab,
            font=ctk.CTkFont(size=13),
        )
        self.preview_text.grid(
            row=1,
            column=0,
            padx=(12, 6),
            pady=(4, 12),
            sticky="nsew",
        )
        self._set_placeholder(self.preview_text, "placeholder.preview")
        self.log_text = ctk.CTkTextbox(
            self.book_tab,
            font=ctk.CTkFont(family="Consolas", size=12),
        )
        self.log_text.grid(
            row=1,
            column=1,
            padx=(6, 12),
            pady=(4, 12),
            sticky="nsew",
        )
        self._set_placeholder(self.log_text, "placeholder.log")

    def _label(
        self,
        parent: ctk.CTkBaseClass,
        row: int,
        key: str,
    ) -> ctk.CTkLabel:
        label = ctk.CTkLabel(
            parent,
            text=self.t(key),
            anchor="w",
        )
        label.grid(row=row, column=0, padx=10, pady=8, sticky="nw")
        return label

    def _entry(
        self,
        parent: ctk.CTkBaseClass,
        row: int,
        label_key: str,
        variable: ctk.StringVar,
        show: str | None = None,
        placeholder_key: str | None = None,
    ) -> ctk.CTkEntry:
        self._label(parent, row, label_key)
        entry = ctk.CTkEntry(
            parent,
            textvariable=variable,
            show=show or "",
            placeholder_text=self.t(placeholder_key) if placeholder_key else "",
        )
        entry.grid(row=row, column=1, padx=10, pady=8, sticky="ew")
        return entry

    def _set_placeholder(self, textbox: ctk.CTkTextbox, key: str) -> None:
        textbox.insert("1.0", self.t(key))
        textbox.configure(text_color=("gray50", "gray60"))
        textbox.bind(
            "<FocusIn>",
            lambda event, box=textbox, translation_key=key: self._clear_placeholder(
                box,
                translation_key,
            ),
        )

    def _clear_placeholder(self, textbox: ctk.CTkTextbox, key: str) -> None:
        if textbox.get("1.0", "end").strip() == self.t(key):
            textbox.delete("1.0", "end")
            textbox.configure(text_color=("gray10", "gray90"))

    def _textbox_value(self, textbox: ctk.CTkTextbox, placeholder_key: str) -> str:
        value = textbox.get("1.0", "end").strip()
        return "" if value == self.t(placeholder_key) else value

    def _replace_text(self, textbox: ctk.CTkTextbox, value: str) -> None:
        textbox.delete("1.0", "end")
        textbox.insert("1.0", value)
        textbox.configure(text_color=("gray10", "gray90"))

    def _change_appearance(self, value: str) -> None:
        ctk.set_appearance_mode(self.appearance_labels[value])

    def _change_language(self, value: str) -> None:
        if self.is_busy:
            return

        language = self.language_labels[value]
        if language == self.translator.language:
            return

        provider = self._provider_code()
        appearance = self.appearance_labels[self.appearance_var.get()]
        current_tab = self.tabs.get()
        tab_key = next(
            (
                key
                for key in ("tab.setup", "tab.outline", "tab.book")
                if self.t(key) == current_tab
            ),
            "tab.setup",
        )
        text_state = self._capture_text_state()
        variable_state = {
            "model": self.model_var.get(),
            "api_key": self.api_key_var.get(),
            "api_base": self.api_base_var.get(),
            "temperature": self.temperature_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "title": self.title_var.get(),
            "output_language": self.output_language_var.get(),
            "tone": self.tone_var.get(),
            "audience": self.audience_var.get(),
            "chapter_count": self.chapter_count_var.get(),
            "words_per_chapter": self.words_per_chapter_var.get(),
            "tokens": self.token_var.get(),
        }

        self.translator.set_language(language)
        self._refresh_locale_maps()
        self.title(self.t("app.title"))

        for child in self.winfo_children():
            child.destroy()
        self._recreate_variables(
            variable_state,
            provider=provider,
            appearance=appearance,
            language=language,
        )
        self._build_ui()
        self._restore_text_state(text_state)
        self._apply_provider_defaults(force=False)
        self._update_token_display()
        self.tabs.set(self.t(tab_key))

    def _recreate_variables(
        self,
        state: dict[str, str],
        provider: str,
        appearance: str,
        language: str,
    ) -> None:
        self.provider_var = ctk.StringVar(
            value=next(
                label
                for label, code in self.provider_labels.items()
                if code == provider
            )
        )
        self.model_var = ctk.StringVar(value=state["model"])
        self.api_key_var = ctk.StringVar(value=state["api_key"])
        self.api_base_var = ctk.StringVar(value=state["api_base"])
        self.temperature_var = ctk.StringVar(value=state["temperature"])
        self.max_tokens_var = ctk.StringVar(value=state["max_tokens"])
        self.title_var = ctk.StringVar(value=state["title"])
        self.output_language_var = ctk.StringVar(value=state["output_language"])
        self.tone_var = ctk.StringVar(value=state["tone"])
        self.audience_var = ctk.StringVar(value=state["audience"])
        self.chapter_count_var = ctk.StringVar(value=state["chapter_count"])
        self.words_per_chapter_var = ctk.StringVar(
            value=state["words_per_chapter"]
        )
        self.status_var = ctk.StringVar(value=self.t("status.ready"))
        self.task_var = ctk.StringVar(value=self.t("status.ready"))
        self.token_var = ctk.StringVar(value=state["tokens"])
        self.language_var = ctk.StringVar(
            value=next(
                label
                for label, code in self.language_labels.items()
                if code == language
            )
        )
        self.appearance_var = ctk.StringVar(
            value=next(
                label
                for label, code in self.appearance_labels.items()
                if code == appearance
            )
        )

    def _capture_text_state(self) -> dict[str, str]:
        return {
            "idea": self._textbox_value(self.idea_text, "placeholder.main_idea"),
            "instructions": self._textbox_value(
                self.instructions_text,
                "placeholder.additional_instructions",
            ),
            "outline": self._textbox_value(
                self.outline_text,
                "placeholder.outline",
            ),
            "preview": self._textbox_value(
                self.preview_text,
                "placeholder.preview",
            ),
            "log": self._textbox_value(self.log_text, "placeholder.log"),
        }

    def _restore_text_state(self, state: dict[str, str]) -> None:
        widgets = [
            (self.idea_text, "idea", "placeholder.main_idea"),
            (
                self.instructions_text,
                "instructions",
                "placeholder.additional_instructions",
            ),
            (self.outline_text, "outline", "placeholder.outline"),
            (self.preview_text, "preview", "placeholder.preview"),
            (self.log_text, "log", "placeholder.log"),
        ]
        for widget, state_key, placeholder_key in widgets:
            value = state[state_key]
            if value:
                self._replace_text(widget, value)
            else:
                widget.delete("1.0", "end")
                self._set_placeholder(widget, placeholder_key)

    def _provider_code(self) -> str:
        return self.provider_labels[self.provider_var.get()]

    def _apply_provider_defaults(self, force: bool) -> None:
        provider = self._provider_code()
        spec = get_provider_spec(provider)
        if force or not self.model_var.get():
            self.model_var.set(spec.default_model)
        if force:
            self.api_key_var.set("")
        if spec.supports_api_base:
            if force or not self.api_base_var.get():
                self.api_base_var.set(spec.default_api_base or "")
        else:
            self.api_base_var.set("")
        self._update_provider_fields()

    def _update_provider_fields(self) -> None:
        provider = self._provider_code()
        if provider_requires_api_key(provider):
            self.api_key_label.grid()
            self.api_key_entry.grid()
            self.api_key_entry.configure(
                placeholder_text=self.t(
                    "placeholder.api_key_env",
                    env_var=provider_key_env_display(provider),
                )
            )
        else:
            self.api_key_label.grid_remove()
            self.api_key_entry.grid_remove()

        if provider_supports_api_base(provider):
            self.api_base_label.grid()
            self.api_base_entry.grid()
        else:
            self.api_base_label.grid_remove()
            self.api_base_entry.grid_remove()

    def _llm_settings_from_form(
        self,
        require_api_key: bool = True,
    ) -> LLMSettings:
        provider = self._provider_code()
        settings = LLMSettings(
            provider=provider,
            model=self.model_var.get(),
            api_key=self.api_key_var.get().strip() or None,
            api_base=(
                self.api_base_var.get()
                if provider_supports_api_base(provider)
                else None
            ),
            temperature=float(self.temperature_var.get()),
            max_tokens=int(self.max_tokens_var.get()),
        )
        if (
            require_api_key
            and provider_requires_api_key(provider)
            and not provider_api_key(settings)
        ):
            raise ValueError(
                self.t(
                    "error.missing_api_key",
                    provider=self.provider_var.get(),
                    env_var=provider_key_env_display(provider),
                )
            )
        return settings

    def _book_settings_from_form(self) -> BookSettings:
        return BookSettings(
            title=self.title_var.get(),
            idea=self._textbox_value(self.idea_text, "placeholder.main_idea"),
            output_language=self.output_language_var.get(),
            tone=self.tone_var.get(),
            target_audience=self.audience_var.get(),
            chapter_count=int(self.chapter_count_var.get()),
            words_per_chapter=int(self.words_per_chapter_var.get()),
            additional_instructions=self._textbox_value(
                self.instructions_text,
                "placeholder.additional_instructions",
            ),
        )

    @staticmethod
    def _editable_outline_json(outline: BookOutline) -> str:
        payload = {
            "title": outline.title,
            "subtitle": outline.subtitle,
            "chapters": [
                {
                    "number": chapter.number,
                    "title": chapter.title,
                    "summary": chapter.summary,
                    "sections": [
                        {
                            "title": section.title,
                            "summary": section.summary,
                        }
                        for section in chapter.sections
                    ],
                }
                for chapter in outline.chapters
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _outline_from_editor(self) -> BookOutline:
        text = self._textbox_value(self.outline_text, "placeholder.outline")
        return parse_outline_document(text)

    def _sync_project_from_ui(self, require_api_key: bool = True) -> BookProject:
        outline = self._outline_from_editor()
        settings = self._book_settings_from_form()
        if not settings.title:
            settings.title = outline.title
            self.title_var.set(outline.title)
        llm_settings = self._llm_settings_from_form(
            require_api_key=require_api_key
        )

        project = self.project or BookProject(settings=settings)
        project.settings = settings
        project.llm_settings = llm_settings
        project.subtitle = outline.subtitle

        existing = {chapter.number: chapter for chapter in project.chapters}
        for chapter in outline.chapters:
            previous = existing.get(chapter.number)
            if (
                previous
                and previous.title == chapter.title
                and previous.summary == chapter.summary
            ):
                chapter.content = previous.content
                chapter.status = previous.status
                chapter.error = previous.error
                previous_sections = {
                    section.title: section for section in previous.sections
                }
                for section in chapter.sections:
                    old_section = previous_sections.get(section.title)
                    if old_section:
                        section.content = old_section.content
                        section.status = old_section.status
                        section.error = old_section.error
        project.chapters = outline.chapters
        project.touch()
        self.project = project
        return project

    def _generate_outline(self) -> None:
        if self.is_busy:
            return
        try:
            llm_settings = self._llm_settings_from_form()
            book_settings = self._book_settings_from_form()
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_settings", exc)
            return

        self.cancel_token = CancelToken()
        self.status_var.set(self.t("status.generating_outline"))
        self.task_var.set(self.t("status.generating_outline"))
        self._append_log(
            self.t(
                "log.outline_started",
                provider=self.provider_var.get(),
                model=llm_settings.model,
            )
        )

        def work() -> tuple[BookOutline, TokenUsage]:
            self.cancel_token.raise_if_cancelled()
            service = BookGenerationService(llm_settings)
            result = service.generate_outline_document(book_settings)
            self.cancel_token.raise_if_cancelled()
            return result, service.usage.model_copy(deep=True)

        self._start_worker("outline", work)

    def _generate_book(self) -> None:
        if self.is_busy:
            return
        try:
            project = self._sync_project_from_ui()
            if not project.chapters:
                raise ValueError(self.t("error.no_chapters"))
            assert project.llm_settings is not None
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_outline", exc)
            return

        self.cancel_token = CancelToken()
        self.progress_bar.set(0)
        self.status_var.set(self.t("status.generating_book"))
        self.task_var.set(self.t("status.generating_book"))

        def progress(event: GenerationEvent) -> None:
            self.worker_queue.put(("progress", event))

        def work() -> BookProject:
            service = BookGenerationService(project.llm_settings)
            return service.generate_book(
                project,
                progress_callback=progress,
                cancel_token=self.cancel_token,
            )

        self._start_worker("book", work)

    def _start_worker(self, operation: str, function: WorkerFunction) -> None:
        self._set_busy(True)

        def runner() -> None:
            try:
                result = function()
                self.worker_queue.put(("success", (operation, result)))
            except GenerationCancelled:
                self.worker_queue.put(("cancelled", operation))
            except Exception as exc:
                self.worker_queue.put(("error", (operation, exc)))

        self.worker = threading.Thread(target=runner, daemon=True)
        self.worker.start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "progress":
                    self._handle_progress(payload)
                elif kind == "success":
                    self._handle_success(*payload)
                elif kind == "cancelled":
                    self._handle_cancelled()
                elif kind == "error":
                    self._handle_worker_error(*payload)
        except queue.Empty:
            pass
        self.after(100, self._poll_worker_queue)

    def _handle_progress(self, event: GenerationEvent) -> None:
        message = self.t(event.message_key, **event.params)
        self.task_var.set(message)
        self.progress_bar.set(event.fraction)
        total_tokens = event.params.get("total_tokens")
        if isinstance(total_tokens, int):
            self.token_var.set(f"{total_tokens:,}")
        self._append_log(message)

    def _handle_success(self, operation: str, result: Any) -> None:
        self._set_busy(False)
        if operation == "outline":
            outline, usage = result
            if not self.title_var.get().strip():
                self.title_var.set(outline.title)
            settings = self._book_settings_from_form()
            settings.title = self.title_var.get().strip() or outline.title
            self.project = BookProject(
                settings=settings,
                llm_settings=self._llm_settings_from_form(),
                subtitle=outline.subtitle,
                chapters=outline.chapters,
                token_usage=usage,
            )
            self._replace_text(
                self.outline_text,
                self._editable_outline_json(outline),
            )
            self.tabs.set(self.t("tab.outline"))
            self.status_var.set(self.t("status.outline_ready"))
            self.task_var.set(self.t("status.outline_ready"))
            self._append_log(
                self.t(
                    "log.outline_completed",
                    chapters=len(outline.chapters),
                    tokens=usage.outline_tokens,
                    total_tokens=usage.total_tokens,
                )
            )
            self._update_token_display()
        else:
            self.project = result
            self._replace_text(self.preview_text, render_markdown(self.project))
            self.tabs.set(self.t("tab.book"))
            self.progress_bar.set(1)
            has_failures = any(
                chapter.status == "failed" for chapter in self.project.chapters
            )
            status_key = (
                "status.completed_with_errors"
                if has_failures
                else "status.completed"
            )
            self.status_var.set(self.t(status_key))
            self.task_var.set(self.t(status_key))
            self._update_token_display()

    def _handle_cancelled(self) -> None:
        self._set_busy(False)
        self.status_var.set(self.t("status.cancelled"))
        self.task_var.set(self.t("status.cancelled"))
        self._append_log(self.t("status.cancelled"))

    def _handle_worker_error(self, operation: str, error: Exception) -> None:
        self._set_busy(False)
        self.status_var.set(self.t("status.failed"))
        self.task_var.set(self.t("status.failed"))
        self._append_log(self.t("log.error", error=str(error)))
        key = (
            "error.generation_failed"
            if operation in {"outline", "book"}
            else "error.invalid_settings"
        )
        self._show_error(key, error)

    def _cancel_generation(self) -> None:
        if not self.is_busy:
            return
        self.cancel_token.cancel()
        self.status_var.set(self.t("status.cancelling"))
        self.task_var.set(self.t("status.cancelling"))
        self._append_log(self.t("log.cancel_requested"))

    def _set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        active_state = "disabled" if busy else "normal"
        for key, button in self.action_buttons.items():
            if key != "button.cancel":
                button.configure(state=active_state)
        self.action_buttons["button.cancel"].configure(
            state="normal" if busy else "disabled"
        )
        self.language_menu.configure(state="disabled" if busy else "normal")

    def _update_token_display(self) -> None:
        total = self.project.token_usage.total_tokens if self.project else 0
        suffix = self.t("label.estimated_suffix") if (
            self.project and self.project.token_usage.estimated
        ) else ""
        self.token_var.set(f"{total:,}{suffix}")

    def _save_project(self) -> None:
        try:
            project = self._sync_project_from_ui(require_api_key=False)
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_outline", exc)
            return
        path = filedialog.asksaveasfilename(
            title=self.t("dialog.save_project"),
            initialdir=PROJECT_ROOT / "projects",
            defaultextension=".json",
            filetypes=[
                (self.t("dialog.json_files"), "*.json"),
                (self.t("dialog.all_files"), "*.*"),
            ],
        )
        if not path:
            return
        try:
            save_project(project, path)
            self.status_var.set(self.t("status.project_saved"))
            self._append_log(self.t("log.project_saved", path=path))
            messagebox.showinfo(
                self.t("dialog.information_title"),
                self.t("message.project_saved"),
            )
        except Exception as exc:
            self._show_error("error.save_failed", exc)

    def _load_project(self) -> None:
        path = filedialog.askopenfilename(
            title=self.t("dialog.load_project"),
            initialdir=PROJECT_ROOT / "projects",
            filetypes=[
                (self.t("dialog.json_files"), "*.json"),
                (self.t("dialog.all_files"), "*.*"),
            ],
        )
        if not path:
            return
        try:
            project = load_project(path)
            self.project = project
            self._populate_project(project)
            self.status_var.set(self.t("status.project_loaded"))
            self.task_var.set(self.t("status.project_loaded"))
            self._append_log(self.t("log.project_loaded", path=path))
            messagebox.showinfo(
                self.t("dialog.information_title"),
                self.t("message.project_loaded"),
            )
        except Exception as exc:
            self._show_error("error.load_failed", exc)

    def _populate_project(self, project: BookProject) -> None:
        settings = project.settings
        self.title_var.set(settings.title or "")
        self._replace_text(self.idea_text, settings.idea)
        self.output_language_var.set(settings.output_language)
        self.tone_var.set(settings.tone)
        self.audience_var.set(settings.target_audience)
        self.chapter_count_var.set(str(settings.chapter_count))
        self.words_per_chapter_var.set(str(settings.words_per_chapter))
        self._replace_text(
            self.instructions_text,
            settings.additional_instructions or "",
        )

        if project.llm_settings:
            label = next(
                (
                    display
                    for display, code in self.provider_labels.items()
                    if code == project.llm_settings.provider
                ),
                self.t("provider.openai"),
            )
            self.provider_var.set(label)
            self.model_var.set(project.llm_settings.model)
            self.api_base_var.set(project.llm_settings.api_base or "")
            self.temperature_var.set(str(project.llm_settings.temperature))
            self.max_tokens_var.set(str(project.llm_settings.max_tokens))
        self.api_key_var.set("")
        self._apply_provider_defaults(force=False)

        outline = BookOutline(
            title=settings.title or self.t("default.untitled_book"),
            subtitle=project.subtitle,
            chapters=project.chapters,
        )
        self._replace_text(self.outline_text, self._editable_outline_json(outline))
        self._replace_text(self.preview_text, render_markdown(project))
        self._update_token_display()
        self.tabs.set(self.t("tab.outline"))

    def _export(self, format_name: str) -> None:
        try:
            project = self._sync_project_from_ui(require_api_key=False)
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_outline", exc)
            return

        options = {
            "md": (
                "dialog.export_markdown",
                ".md",
                "dialog.markdown_files",
                export_markdown,
                "Markdown",
            ),
            "txt": (
                "dialog.export_txt",
                ".txt",
                "dialog.text_files",
                export_txt,
                "TXT",
            ),
            "docx": (
                "dialog.export_docx",
                ".docx",
                "dialog.docx_files",
                export_docx,
                "DOCX",
            ),
        }
        title_key, extension, filetype_key, exporter, display_format = options[
            format_name
        ]
        filename = filedialog.asksaveasfilename(
            title=self.t(title_key),
            initialdir=PROJECT_ROOT / "exports",
            defaultextension=extension,
            filetypes=[
                (self.t(filetype_key), f"*{extension}"),
                (self.t("dialog.all_files"), "*.*"),
            ],
        )
        if not filename:
            return
        try:
            exporter(project, filename)
            self.status_var.set(self.t("status.exported"))
            self._append_log(
                self.t(
                    "log.exported",
                    format=display_format,
                    path=filename,
                )
            )
            messagebox.showinfo(
                self.t("dialog.information_title"),
                self.t("message.export_completed"),
            )
        except Exception as exc:
            self._show_error("error.export_failed", exc)

    def _append_log(self, message: str) -> None:
        placeholder = self.t("placeholder.log")
        if self.log_text.get("1.0", "end").strip() == placeholder:
            self.log_text.delete("1.0", "end")
            self.log_text.configure(text_color=("gray10", "gray90"))
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{timestamp}] {message}\n")
        self.log_text.see("end")

    def _clear_log(self) -> None:
        self.log_text.delete("1.0", "end")

    def _show_error(self, key: str, error: Exception) -> None:
        messagebox.showerror(
            self.t("dialog.error_title"),
            self.t(key, error=str(error)),
        )

    def _on_close(self) -> None:
        if self.is_busy:
            should_close = messagebox.askyesno(
                self.t("dialog.confirm_close_title"),
                self.t("dialog.confirm_close"),
            )
            if not should_close:
                return
            self.cancel_token.cancel()
        self.settings_store.save(
            {
                "language": self.translator.language,
                "appearance": self.appearance_labels[self.appearance_var.get()],
                "provider": self._provider_code(),
                "model": self.model_var.get(),
                "api_base": self.api_base_var.get(),
            }
        )
        self.destroy()


def run() -> None:
    """Launch the desktop application."""
    app = AIBookBatchWriterApp()
    app.mainloop()
