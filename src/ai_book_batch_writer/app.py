"""CustomTkinter desktop application."""

from __future__ import annotations

import json
import os
import queue
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any, Callable

import customtkinter as ctk
from pydantic import ValidationError

from ai_book_batch_writer.config import (
    exports_directory,
    load_environment,
    projects_directory,
)
from ai_book_batch_writer.exporters import (
    export_docx,
    export_markdown,
    export_pdf,
    export_project_bundle,
    export_txt,
    render_markdown,
)
from ai_book_batch_writer.generation_service import BookGenerationService
from ai_book_batch_writer.i18n import Translator
from ai_book_batch_writer.input_validation import (
    InputValidationError,
    validate_book_input,
    validate_llm_input,
)
from ai_book_batch_writer.json_utils import parse_outline_document
from ai_book_batch_writer.language_catalog import load_output_languages
from ai_book_batch_writer.llm_providers import (
    PROVIDER_SPECS,
    get_provider_spec,
    provider_api_key,
    provider_key_env_display,
    provider_requires_api_key,
    provider_supports_api_base,
)
from ai_book_batch_writer.model_catalog import ModelCatalogStore
from ai_book_batch_writer.models import (
    BookOutline,
    BookProject,
    BookSettings,
    GenerationEvent,
    LLMSettings,
    TokenUsage,
)
from ai_book_batch_writer.project_store import load_project, save_project
from ai_book_batch_writer.provider_discovery import (
    list_provider_models,
    test_provider_connection,
)
from ai_book_batch_writer.settings_store import SettingsStore
from ai_book_batch_writer.utils import (
    CancelToken,
    GenerationCancelled,
    create_project_directory,
)

WorkerFunction = Callable[[], Any]


class AIBookBatchWriterApp(ctk.CTk):
    """Main desktop window and UI-thread event coordinator."""

    def __init__(self) -> None:
        load_environment()
        self.settings_store = SettingsStore()
        self.model_catalog_store = ModelCatalogStore()
        self.output_languages = load_output_languages()
        self.preferences = self.settings_store.load()
        self.translator = Translator(self.preferences.get("language", "en"))
        self._refresh_locale_maps()

        appearance = self.preferences.get("appearance", "System")
        ctk.set_appearance_mode(appearance)
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.project: BookProject | None = None
        self.current_page = "home"
        self.current_step = 1
        self.book_generation_started = False
        self.book_generation_finished = False
        self.auto_export_completed = False
        self.project_file_path: Path | None = None
        self.project_output_dir: Path | None = None
        self.cancel_token = CancelToken()
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.is_busy = False
        self.progress_dialog: ctk.CTkToplevel | None = None
        self.progress_dialog_bar: ctk.CTkProgressBar | None = None
        self.progress_dialog_message = ctk.StringVar(value="")

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
        global_config = self.preferences.get("global_llm")
        if not isinstance(global_config, dict):
            global_config = {}
        self.global_configured = bool(
            global_config.get("provider") and global_config.get("model")
        )
        provider_code = str(
            global_config.get(
                "provider",
                self.preferences.get("provider", "openai"),
            )
        ).lower()
        if provider_code not in PROVIDER_SPECS:
            provider_code = "openai"
        provider = self._provider_label(provider_code)
        self.provider_var = ctk.StringVar(value=provider)
        self.model_var = ctk.StringVar(
            value=global_config.get(
                "model",
                self.preferences.get(
                    "model",
                    get_provider_spec(provider_code).default_model,
                ),
            )
        )
        self.api_key_var = ctk.StringVar()
        self.api_base_var = ctk.StringVar(
            value=global_config.get(
                "api_base",
                self.preferences.get("api_base", ""),
            )
        )
        self.temperature_var = ctk.StringVar(
            value=str(
                global_config.get(
                    "temperature",
                    self.t("default.temperature"),
                )
            )
        )
        self.max_tokens_var = ctk.StringVar(
            value=str(
                global_config.get(
                    "max_tokens",
                    self.t("default.max_tokens"),
                )
            )
        )
        self.global_provider_var = ctk.StringVar(value=provider)
        self.global_model_var = ctk.StringVar(
            value=global_config.get(
                "model",
                get_provider_spec(provider_code).default_model,
            )
        )
        self.global_api_key_var = ctk.StringVar()
        self.global_api_base_var = ctk.StringVar(
            value=global_config.get("api_base", "")
        )
        self.global_temperature_var = ctk.StringVar(
            value=str(
                global_config.get(
                    "temperature",
                    self.t("default.temperature"),
                )
            )
        )
        self.global_max_tokens_var = ctk.StringVar(
            value=str(
                global_config.get(
                    "max_tokens",
                    self.t("default.max_tokens"),
                )
            )
        )
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
        self.maintenance_title_var = ctk.StringVar(
            value=self.t("maintenance.no_project")
        )
        self.maintenance_detail_var = ctk.StringVar(
            value=self.t("maintenance.no_project_detail")
        )
        self.global_status_var = ctk.StringVar(
            value=(
                self.t("settings.configured")
                if self.global_configured
                else self.t("settings.not_configured")
            )
        )
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

    def _provider_label(self, provider_code: str) -> str:
        return next(
            (
                label
                for label, code in self.provider_labels.items()
                if code == provider_code
            ),
            self.t("provider.openai"),
        )

    def _model_values(self, provider: str, current: str = "") -> list[str]:
        values = self.model_catalog_store.models_for(provider)
        default = get_provider_spec(provider).default_model
        return sorted(
            {
                value.strip()
                for value in [current, default, *values]
                if value and value.strip()
            },
            key=str.casefold,
        )

    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.busy_sensitive_buttons: list[ctk.CTkButton] = []
        self.cancel_buttons: list[ctk.CTkButton] = []
        self.step_buttons: dict[int, ctk.CTkButton] = {}

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
        ctk.CTkButton(
            header_controls,
            text=self.t("button.home"),
            command=lambda: self._show_page("home"),
            width=84,
            fg_color="transparent",
            border_width=1,
            text_color=("gray15", "gray90"),
        ).grid(row=0, column=0, padx=(0, 14))
        ctk.CTkLabel(
            header_controls,
            text=self.t("label.interface_language"),
        ).grid(row=0, column=1, padx=(0, 6))
        self.language_menu = ctk.CTkOptionMenu(
            header_controls,
            values=list(self.language_labels),
            variable=self.language_var,
            command=self._change_language,
            width=110,
        )
        self.language_menu.grid(row=0, column=2, padx=(0, 14))
        ctk.CTkLabel(
            header_controls,
            text=self.t("label.appearance"),
        ).grid(row=0, column=3, padx=(0, 6))
        ctk.CTkOptionMenu(
            header_controls,
            values=list(self.appearance_labels),
            variable=self.appearance_var,
            command=self._change_appearance,
            width=100,
        ).grid(row=0, column=4)

        self.secondary_header = ctk.CTkFrame(self, corner_radius=0, height=58)
        self.secondary_header.grid(row=1, column=0, sticky="ew", pady=(1, 0))
        self.secondary_header.grid_propagate(False)

        self.page_container = ctk.CTkFrame(self, fg_color="transparent")
        self.page_container.grid(row=2, column=0, padx=16, pady=14, sticky="nsew")
        self.page_container.grid_columnconfigure(0, weight=1)
        self.page_container.grid_rowconfigure(0, weight=1)

        self.pages: dict[str, ctk.CTkFrame] = {
            name: ctk.CTkFrame(self.page_container, fg_color="transparent")
            for name in ("home", "create", "maintenance", "settings", "utilities")
        }
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")

        self._build_home_page()
        self._build_create_page()
        self._build_maintenance_page()
        self._build_global_settings_page()
        self._build_utilities_page()

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
        self._show_page(self.current_page, show_warning=False)

    def _build_home_page(self) -> None:
        page = self.pages["home"]
        page.grid_columnconfigure((0, 1), weight=1, uniform="cards")
        page.grid_rowconfigure((1, 2), weight=1, uniform="cards")
        heading = ctk.CTkFrame(page, fg_color="transparent")
        heading.grid(row=0, column=0, columnspan=2, pady=(8, 14))
        ctk.CTkLabel(
            heading,
            text=self.t("home.title"),
            font=ctk.CTkFont(size=27, weight="bold"),
        ).pack()
        ctk.CTkLabel(
            heading,
            text=self.t("home.subtitle"),
            text_color=("gray35", "gray70"),
        ).pack(pady=(4, 0))

        cards = [
            (
                "create",
                "✍",
                "home.create_title",
                "home.create_description",
                self._start_new_book,
            ),
            (
                "maintenance",
                "▤",
                "home.maintenance_title",
                "home.maintenance_description",
                lambda: self._show_page("maintenance"),
            ),
            (
                "settings",
                "⚙",
                "home.settings_title",
                "home.settings_description",
                lambda: self._show_page("settings"),
            ),
            (
                "utilities",
                "◆",
                "home.utilities_title",
                "home.utilities_description",
                lambda: self._show_page("utilities"),
            ),
        ]
        for index, (_, icon, title_key, description_key, command) in enumerate(cards):
            row = 1 + index // 2
            column = index % 2
            card = ctk.CTkFrame(page, corner_radius=16, border_width=1)
            card.grid(
                row=row,
                column=column,
                padx=(8, 5) if column == 0 else (5, 8),
                pady=7,
                sticky="nsew",
            )
            card.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                card,
                text=icon,
                font=ctk.CTkFont(size=46, weight="bold"),
                width=84,
            ).grid(row=0, column=0, rowspan=3, padx=(22, 10), pady=22)
            ctk.CTkLabel(
                card,
                text=self.t(title_key),
                font=ctk.CTkFont(size=21, weight="bold"),
                anchor="w",
            ).grid(row=0, column=1, padx=(0, 22), pady=(24, 4), sticky="ew")
            ctk.CTkLabel(
                card,
                text=self.t(description_key),
                justify="left",
                wraplength=430,
                anchor="nw",
                text_color=("gray35", "gray70"),
            ).grid(row=1, column=1, padx=(0, 22), pady=4, sticky="nsew")
            ctk.CTkButton(
                card,
                text=self.t("button.open"),
                command=command,
                width=130,
            ).grid(row=2, column=1, padx=(0, 22), pady=(8, 22), sticky="e")

    def _build_create_page(self) -> None:
        page = self.pages["create"]
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(0, weight=1)
        self.create_steps: dict[int, ctk.CTkFrame] = {
            step: ctk.CTkFrame(page, fg_color="transparent")
            for step in (1, 2, 3)
        }
        for frame in self.create_steps.values():
            frame.grid(row=0, column=0, sticky="nsew")
        self._build_setup_step()
        self._build_outline_step()
        self._build_book_step()

    def _build_setup_step(self) -> None:
        parent = self.create_steps[1]
        parent.grid_columnconfigure((0, 1), weight=1)
        parent.grid_rowconfigure(0, weight=1)

        model_panel = ctk.CTkScrollableFrame(
            parent,
            label_text=self.t("panel.project_model_settings"),
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
        self.api_key_label = self._label(
            model_panel,
            1,
            "label.api_key",
        )
        self.api_key_frame = ctk.CTkFrame(model_panel, fg_color="transparent")
        self.api_key_frame.grid(row=1, column=1, padx=10, pady=8, sticky="ew")
        self.api_key_frame.grid_columnconfigure(0, weight=1)
        self.api_key_entry = ctk.CTkEntry(
            self.api_key_frame,
            textvariable=self.api_key_var,
            show="*",
            placeholder_text=self.t("placeholder.api_key"),
        )
        self.api_key_entry.grid(row=0, column=0, sticky="ew")
        self.project_api_test_button = ctk.CTkButton(
            self.api_key_frame,
            text=self.t("button.test_connection"),
            command=lambda: self._test_connection("project"),
            width=92,
        )
        self.project_api_test_button.grid(row=0, column=1, padx=(8, 0))
        self.busy_sensitive_buttons.append(self.project_api_test_button)

        self._label(model_panel, 2, "label.model")
        self.model_frame = ctk.CTkFrame(model_panel, fg_color="transparent")
        self.model_frame.grid(row=2, column=1, padx=10, pady=8, sticky="ew")
        self.model_frame.grid_columnconfigure(0, weight=1)
        self.model_combo = ctk.CTkComboBox(
            self.model_frame,
            values=self._model_values(self._provider_code(), self.model_var.get()),
            variable=self.model_var,
            state="normal",
        )
        self.model_combo.grid(row=0, column=0, sticky="ew")
        self.project_models_button = ctk.CTkButton(
            self.model_frame,
            text=self.t("button.refresh_models"),
            command=lambda: self._refresh_models("project"),
            width=118,
        )
        self.project_models_button.grid(row=0, column=1, padx=(8, 0))
        self.busy_sensitive_buttons.append(self.project_models_button)

        self.api_base_label = self._label(
            model_panel,
            3,
            "label.api_base",
        )
        self.api_base_frame = ctk.CTkFrame(model_panel, fg_color="transparent")
        self.api_base_frame.grid(row=3, column=1, padx=10, pady=8, sticky="ew")
        self.api_base_frame.grid_columnconfigure(0, weight=1)
        self.api_base_entry = ctk.CTkEntry(
            self.api_base_frame,
            textvariable=self.api_base_var,
        )
        self.api_base_entry.grid(row=0, column=0, sticky="ew")
        self.project_local_test_button = ctk.CTkButton(
            self.api_base_frame,
            text=self.t("button.test_connection"),
            command=lambda: self._test_connection("project"),
            width=92,
        )
        self.project_local_test_button.grid(row=0, column=1, padx=(8, 0))
        self.busy_sensitive_buttons.append(self.project_local_test_button)
        self._entry(model_panel, 4, "label.temperature", self.temperature_var)
        self._entry(model_panel, 5, "label.max_tokens", self.max_tokens_var)

        book_panel = ctk.CTkScrollableFrame(
            parent,
            label_text=self.t("panel.book_settings"),
        )
        book_panel.grid(row=0, column=1, padx=(5, 8), pady=8, sticky="nsew")
        book_panel.grid_columnconfigure(1, weight=1)

        self._entry(book_panel, 0, "label.book_title", self.title_var)
        self._label(book_panel, 1, "label.idea")
        self.idea_text = ctk.CTkTextbox(book_panel, height=120)
        self.idea_text.grid(row=1, column=1, padx=10, pady=8, sticky="ew")
        self._set_placeholder(self.idea_text, "placeholder.main_idea")
        self._label(book_panel, 2, "label.output_language")
        self.output_language_combo = ctk.CTkComboBox(
            book_panel,
            values=self.output_languages,
            variable=self.output_language_var,
            state="normal",
        )
        self.output_language_combo.grid(
            row=2,
            column=1,
            padx=10,
            pady=8,
            sticky="ew",
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
        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=1, column=0, columnspan=2, padx=8, pady=(5, 0), sticky="ew")
        actions.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            actions,
            text=self.t("wizard.override_hint"),
            text_color=("gray35", "gray70"),
        ).grid(row=0, column=0, sticky="w")
        generate = ctk.CTkButton(
            actions,
            text=self.t("button.generate_outline"),
            command=self._generate_outline,
            height=38,
            width=180,
        )
        generate.grid(row=0, column=1, padx=(8, 4))
        self.busy_sensitive_buttons.append(generate)
        cancel = ctk.CTkButton(
            actions,
            text=self.t("button.cancel"),
            command=self._cancel_generation,
            state="disabled",
            width=110,
            fg_color=("gray65", "gray30"),
        )
        cancel.grid(row=0, column=2, padx=(4, 0))
        self.cancel_buttons.append(cancel)

    def _build_outline_step(self) -> None:
        parent = self.create_steps[2]
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            parent,
            text=self.t("panel.outline_editor"),
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")
        self.outline_text = ctk.CTkTextbox(
            parent,
            wrap="none",
            font=ctk.CTkFont(family="Consolas", size=13),
        )
        self.outline_text.grid(row=1, column=0, padx=12, pady=(4, 8), sticky="nsew")
        self._set_placeholder(self.outline_text, "placeholder.outline")
        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=2, column=0, padx=12, pady=(0, 4), sticky="ew")
        self.outline_back_button = ctk.CTkButton(
            actions,
            text=self.t("button.back_to_setup"),
            command=lambda: self._show_create_step(1),
            width=150,
            fg_color="transparent",
            border_width=1,
            text_color=("gray15", "gray90"),
        )
        self.outline_back_button.pack(side="left")
        save = ctk.CTkButton(
            actions,
            text=self.t("button.save_project"),
            command=self._save_project,
            width=140,
        )
        save.pack(side="right", padx=(6, 0))
        generate = ctk.CTkButton(
            actions,
            text=self.t("button.generate_book"),
            command=self._generate_book,
            width=170,
            height=38,
        )
        generate.pack(side="right", padx=(6, 0))
        self.busy_sensitive_buttons.extend([save, generate, self.outline_back_button])
        cancel = ctk.CTkButton(
            actions,
            text=self.t("button.cancel"),
            command=self._cancel_generation,
            state="disabled",
            width=110,
            fg_color=("gray65", "gray30"),
        )
        cancel.pack(side="right", padx=(6, 0))
        self.cancel_buttons.append(cancel)

    def _build_book_step(self) -> None:
        parent = self.create_steps[3]
        parent.grid_columnconfigure(0, weight=3)
        parent.grid_columnconfigure(1, weight=2)
        parent.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            parent,
            text=self.t("panel.result_preview"),
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, padx=12, pady=(12, 4), sticky="w")

        log_header = ctk.CTkFrame(parent, fg_color="transparent")
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
            parent,
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
            parent,
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
        actions = ctk.CTkFrame(parent, fg_color="transparent")
        actions.grid(row=2, column=0, columnspan=2, padx=12, pady=(0, 4), sticky="ew")
        ctk.CTkLabel(
            actions,
            text=self.t("wizard.book_locked_hint"),
            text_color=("gray35", "gray70"),
        ).pack(side="left")
        for key, command in (
            ("button.export_pdf", lambda: self._export("pdf")),
            ("button.export_docx", lambda: self._export("docx")),
            ("button.export_txt", lambda: self._export("txt")),
            ("button.export_markdown", lambda: self._export("md")),
            ("button.save_project", self._save_project),
        ):
            button = ctk.CTkButton(
                actions,
                text=self.t(key),
                command=command,
                width=125,
            )
            button.pack(side="right", padx=(6, 0))
            self.busy_sensitive_buttons.append(button)
        cancel = ctk.CTkButton(
            actions,
            text=self.t("button.cancel"),
            command=self._cancel_generation,
            state="disabled",
            width=105,
            fg_color=("gray65", "gray30"),
        )
        cancel.pack(side="right", padx=(6, 0))
        self.cancel_buttons.append(cancel)

    def _build_maintenance_page(self) -> None:
        page = self.pages["maintenance"]
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        self._page_title(page, "maintenance.title", "maintenance.description")
        ctk.CTkButton(
            page,
            text=self.t("button.load_project"),
            command=self._load_project,
            height=42,
            width=190,
        ).grid(row=1, column=0, pady=(10, 16))
        card = ctk.CTkFrame(page, corner_radius=16, border_width=1)
        card.grid(row=2, column=0, padx=80, pady=(0, 30), sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=self.t("maintenance.current_project"),
            font=ctk.CTkFont(size=17, weight="bold"),
        ).grid(row=0, column=0, padx=24, pady=(24, 6), sticky="w")
        ctk.CTkLabel(
            card,
            textvariable=self.maintenance_title_var,
            font=ctk.CTkFont(size=24, weight="bold"),
        ).grid(row=1, column=0, padx=24, pady=6, sticky="w")
        ctk.CTkLabel(
            card,
            textvariable=self.maintenance_detail_var,
            text_color=("gray35", "gray70"),
        ).grid(row=2, column=0, padx=24, pady=(0, 18), sticky="w")
        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=3, column=0, padx=24, pady=(0, 24), sticky="w")
        self.maintenance_continue_button = ctk.CTkButton(
            actions,
            text=self.t("button.continue_project"),
            command=self._continue_project,
            state="disabled",
        )
        self.maintenance_continue_button.pack(side="left")
        self.maintenance_save_button = ctk.CTkButton(
            actions,
            text=self.t("button.save_project"),
            command=self._save_project,
            state="disabled",
            fg_color="transparent",
            border_width=1,
            text_color=("gray15", "gray90"),
        )
        self.maintenance_save_button.pack(side="left", padx=8)

    def _build_global_settings_page(self) -> None:
        page = self.pages["settings"]
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(1, weight=1)
        self._page_title(page, "settings.title", "settings.description")
        panel = ctk.CTkScrollableFrame(
            page,
            label_text=self.t("panel.global_model_settings"),
            width=760,
        )
        panel.grid(row=1, column=0, padx=120, pady=16, sticky="nsew")
        panel.grid_columnconfigure(1, weight=1)
        self._label(panel, 0, "label.provider")
        self.global_provider_menu = ctk.CTkOptionMenu(
            panel,
            values=list(self.provider_labels),
            variable=self.global_provider_var,
            command=lambda _: self._apply_global_provider_defaults(force=True),
        )
        self.global_provider_menu.grid(row=0, column=1, padx=10, pady=8, sticky="ew")
        self.global_api_key_label = self._label(panel, 1, "label.api_key")
        self.global_api_key_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.global_api_key_frame.grid(
            row=1,
            column=1,
            padx=10,
            pady=8,
            sticky="ew",
        )
        self.global_api_key_frame.grid_columnconfigure(0, weight=1)
        self.global_api_key_entry = ctk.CTkEntry(
            self.global_api_key_frame,
            textvariable=self.global_api_key_var,
            show="*",
        )
        self.global_api_key_entry.grid(row=0, column=0, sticky="ew")
        self.global_api_test_button = ctk.CTkButton(
            self.global_api_key_frame,
            text=self.t("button.test_connection"),
            command=lambda: self._test_connection("global"),
            width=92,
        )
        self.global_api_test_button.grid(row=0, column=1, padx=(8, 0))
        self.busy_sensitive_buttons.append(self.global_api_test_button)

        self._label(panel, 2, "label.model")
        self.global_model_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.global_model_frame.grid(
            row=2,
            column=1,
            padx=10,
            pady=8,
            sticky="ew",
        )
        self.global_model_frame.grid_columnconfigure(0, weight=1)
        self.global_model_combo = ctk.CTkComboBox(
            self.global_model_frame,
            values=self._model_values(
                self._global_provider_code(),
                self.global_model_var.get(),
            ),
            variable=self.global_model_var,
            state="normal",
        )
        self.global_model_combo.grid(row=0, column=0, sticky="ew")
        self.global_models_button = ctk.CTkButton(
            self.global_model_frame,
            text=self.t("button.refresh_models"),
            command=lambda: self._refresh_models("global"),
            width=118,
        )
        self.global_models_button.grid(row=0, column=1, padx=(8, 0))
        self.busy_sensitive_buttons.append(self.global_models_button)

        self.global_api_base_label = self._label(panel, 3, "label.api_base")
        self.global_api_base_frame = ctk.CTkFrame(panel, fg_color="transparent")
        self.global_api_base_frame.grid(
            row=3,
            column=1,
            padx=10,
            pady=8,
            sticky="ew",
        )
        self.global_api_base_frame.grid_columnconfigure(0, weight=1)
        self.global_api_base_entry = ctk.CTkEntry(
            self.global_api_base_frame,
            textvariable=self.global_api_base_var,
        )
        self.global_api_base_entry.grid(row=0, column=0, sticky="ew")
        self.global_local_test_button = ctk.CTkButton(
            self.global_api_base_frame,
            text=self.t("button.test_connection"),
            command=lambda: self._test_connection("global"),
            width=92,
        )
        self.global_local_test_button.grid(row=0, column=1, padx=(8, 0))
        self.busy_sensitive_buttons.append(self.global_local_test_button)
        self._entry(panel, 4, "label.temperature", self.global_temperature_var)
        self._entry(panel, 5, "label.max_tokens", self.global_max_tokens_var)
        ctk.CTkLabel(
            panel,
            text=self.t("settings.security_note"),
            justify="left",
            wraplength=700,
            text_color=("gray35", "gray70"),
        ).grid(row=6, column=0, columnspan=2, padx=10, pady=(12, 6), sticky="w")
        footer = ctk.CTkFrame(panel, fg_color="transparent")
        footer.grid(row=7, column=0, columnspan=2, padx=10, pady=16, sticky="ew")
        ctk.CTkLabel(
            footer,
            textvariable=self.global_status_var,
            font=ctk.CTkFont(weight="bold"),
        ).pack(side="left")
        ctk.CTkButton(
            footer,
            text=self.t("button.save_global_settings"),
            command=self._save_global_settings,
            height=38,
            width=210,
        ).pack(side="right")
        self._update_global_provider_fields()

    def _build_utilities_page(self) -> None:
        page = self.pages["utilities"]
        page.grid_columnconfigure((0, 1), weight=1)
        self._page_title(page, "utilities.title", "utilities.description", columns=2)
        utilities = [
            ("⬇", "button.export_markdown", lambda: self._export("md")),
            ("T", "button.export_txt", lambda: self._export("txt")),
            ("W", "button.export_docx", lambda: self._export("docx")),
            ("P", "button.export_pdf", lambda: self._export("pdf")),
            ("▣", "button.open_exports_folder", self._open_exports_folder),
            ("▤", "button.open_projects_folder", self._open_projects_folder),
        ]
        for index, (icon, key, command) in enumerate(utilities):
            frame = ctk.CTkFrame(page, corner_radius=14, border_width=1)
            frame.grid(
                row=1 + index // 2,
                column=index % 2,
                padx=8,
                pady=8,
                sticky="nsew",
            )
            ctk.CTkLabel(
                frame,
                text=icon,
                font=ctk.CTkFont(size=30, weight="bold"),
                width=62,
            ).pack(side="left", padx=(22, 8), pady=22)
            ctk.CTkButton(
                frame,
                text=self.t(key),
                command=command,
                height=38,
            ).pack(side="left", padx=(8, 22), pady=22, fill="x", expand=True)

    def _page_title(
        self,
        parent: ctk.CTkFrame,
        title_key: str,
        description_key: str,
        columns: int = 1,
    ) -> None:
        heading = ctk.CTkFrame(parent, fg_color="transparent")
        heading.grid(row=0, column=0, columnspan=columns, pady=(14, 6))
        ctk.CTkLabel(
            heading,
            text=self.t(title_key),
            font=ctk.CTkFont(size=27, weight="bold"),
        ).pack()
        ctk.CTkLabel(
            heading,
            text=self.t(description_key),
            text_color=("gray35", "gray70"),
        ).pack(pady=(4, 0))

    def _show_page(self, name: str, show_warning: bool = True) -> None:
        if self.is_busy and name != self.current_page:
            messagebox.showwarning(
                self.t("dialog.information_title"),
                self.t("error.busy"),
            )
            return
        for page_name, page in self.pages.items():
            if page_name == name:
                page.grid()
            else:
                page.grid_remove()
        self.current_page = name
        if name == "create":
            self._show_create_step(self.current_step, render_header=False)
            if show_warning and not self.global_configured:
                messagebox.showwarning(
                    self.t("dialog.global_settings_title"),
                    self.t("message.global_settings_missing"),
                )
        elif name == "maintenance":
            self._update_maintenance_summary()
        self._render_secondary_header()

    def _render_secondary_header(self) -> None:
        for child in self.secondary_header.winfo_children():
            child.destroy()
        self.secondary_header.grid_columnconfigure(0, weight=0)
        self.secondary_header.grid_columnconfigure(1, weight=1)
        self.secondary_header.grid_columnconfigure(2, weight=0)

        if self.current_page == "home":
            ctk.CTkLabel(
                self.secondary_header,
                text=self.t("home.navigation_hint"),
                text_color=("gray35", "gray70"),
            ).grid(row=0, column=1, pady=18)
            return

        ctk.CTkButton(
            self.secondary_header,
            text=self.t("button.back_home"),
            command=lambda: self._show_page("home"),
            width=110,
            fg_color="transparent",
            border_width=1,
            text_color=("gray15", "gray90"),
        ).grid(row=0, column=0, padx=18, pady=11, sticky="w")
        ctk.CTkLabel(
            self.secondary_header,
            text="",
            width=146,
        ).grid(row=0, column=2)

        if self.current_page != "create":
            ctk.CTkLabel(
                self.secondary_header,
                text=self.t(f"navigation.{self.current_page}"),
                font=ctk.CTkFont(size=17, weight="bold"),
            ).grid(row=0, column=1, pady=16)
            return

        stepper = ctk.CTkFrame(self.secondary_header, fg_color="transparent")
        stepper.grid(row=0, column=1, pady=8)
        self.step_buttons = {}
        for step in (1, 2, 3):
            if step > 1:
                ctk.CTkLabel(
                    stepper,
                    text="→",
                    font=ctk.CTkFont(size=19, weight="bold"),
                    text_color=("gray45", "gray60"),
                ).pack(side="left", padx=8)
            button = ctk.CTkButton(
                stepper,
                text=self._step_label(step),
                command=lambda selected=step: self._show_create_step(selected),
                width=190,
                height=38,
            )
            button.pack(side="left")
            self.step_buttons[step] = button
        self._refresh_flow_ui()

    def _show_create_step(
        self,
        step: int,
        render_header: bool = True,
    ) -> None:
        if step == 1 and self._project_is_locked():
            return
        if step == 2 and not self._project_has_outline():
            return
        if step == 3 and not self._project_is_locked():
            return
        self.current_step = step
        for number, frame in self.create_steps.items():
            if number == step:
                frame.grid()
            else:
                frame.grid_remove()
        if render_header:
            self._render_secondary_header()
        else:
            self._refresh_flow_ui()

    def _refresh_flow_ui(self) -> None:
        if not self.step_buttons:
            return
        has_outline = self._project_has_outline()
        locked = self._project_is_locked()
        for step, button in self.step_buttons.items():
            button.configure(text=self._step_label(step))
            enabled = (
                (step == 1 and not locked)
                or (step == 2 and has_outline and not locked)
                or (step == 3 and locked)
            )
            if step == 3 and self.book_generation_finished:
                button.configure(
                    state="disabled",
                    fg_color=("gray70", "gray25"),
                    hover_color=("gray70", "gray25"),
                )
            elif step == self.current_step:
                button.configure(
                    state="normal",
                    fg_color=("#1f6aa5", "#1f6aa5"),
                    hover_color=("#144870", "#144870"),
                )
            else:
                button.configure(
                    state="normal" if enabled else "disabled",
                    fg_color=("gray65", "gray30"),
                    hover_color=("gray55", "gray25"),
                )
        self.outline_back_button.configure(
            state="disabled" if locked or self.is_busy else "normal"
        )
        self.outline_text.configure(state="disabled" if locked else "normal")

    def _step_label(self, step: int) -> str:
        if step != 3:
            return self.t(f"wizard.step_{step}")
        if self.book_generation_finished:
            key = (
                "wizard.step_3_completed"
                if self.auto_export_completed
                else "wizard.step_3_finished"
            )
            return self.t(key)
        return self.t("wizard.step_3")

    def _project_has_outline(self) -> bool:
        return bool(self.project and self.project.chapters)

    def _project_is_locked(self) -> bool:
        if self.book_generation_started:
            return True
        if not self.project:
            return False
        if self.project.token_usage.book_tokens > 0:
            return True
        return any(
            chapter.content
            or chapter.status != "pending"
            or any(section.content or section.status != "pending" for section in chapter.sections)
            for chapter in self.project.chapters
        )

    def _project_is_finished(self) -> bool:
        if self.book_generation_finished:
            return True
        if not self.project or not self.project.chapters:
            return False
        return (
            all(
                chapter.status in {"completed", "failed"}
                for chapter in self.project.chapters
            )
            and any(
                chapter.content
                or chapter.status == "completed"
                or any(section.content for section in chapter.sections)
                for chapter in self.project.chapters
            )
        )

    def _start_new_book(self) -> None:
        if self.project and not messagebox.askyesno(
            self.t("dialog.new_project_title"),
            self.t("dialog.new_project_confirm"),
        ):
            return
        self.project = None
        self.book_generation_started = False
        self.book_generation_finished = False
        self.auto_export_completed = False
        self.project_file_path = None
        self.project_output_dir = None
        self.current_step = 1
        self._apply_global_to_project_form()
        self.title_var.set("")
        self.output_language_var.set(self.t("default.output_language"))
        self.tone_var.set(self.t("default.tone"))
        self.audience_var.set(self.t("default.target_audience"))
        self.chapter_count_var.set(self.t("default.chapter_count"))
        self.words_per_chapter_var.set(self.t("default.words_per_chapter"))
        self._reset_textbox(self.idea_text, "placeholder.main_idea")
        self._reset_textbox(
            self.instructions_text,
            "placeholder.additional_instructions",
        )
        self._reset_textbox(self.outline_text, "placeholder.outline")
        self._reset_textbox(self.preview_text, "placeholder.preview")
        self._reset_textbox(self.log_text, "placeholder.log")
        self.progress_bar.set(0)
        self.status_var.set(self.t("status.ready"))
        self.task_var.set(self.t("status.ready"))
        self._update_token_display()
        self._update_maintenance_summary()
        self._show_page("create")

    def _reset_textbox(self, textbox: ctk.CTkTextbox, placeholder_key: str) -> None:
        textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        self._set_placeholder(textbox, placeholder_key)

    def _continue_project(self) -> None:
        if not self.project:
            return
        self.current_step = 3 if self._project_is_locked() else 2
        self._show_page("create", show_warning=False)

    def _update_maintenance_summary(self) -> None:
        if not self.project:
            self.maintenance_title_var.set(self.t("maintenance.no_project"))
            self.maintenance_detail_var.set(
                self.t("maintenance.no_project_detail")
            )
            self.maintenance_continue_button.configure(state="disabled")
            self.maintenance_save_button.configure(state="disabled")
            return
        title = self.project.settings.title or self.t("default.untitled_book")
        completed = sum(
            chapter.status == "completed" for chapter in self.project.chapters
        )
        self.maintenance_title_var.set(title)
        self.maintenance_detail_var.set(
            self.t(
                "maintenance.project_detail",
                chapters=len(self.project.chapters),
                completed=completed,
                tokens=f"{self.project.token_usage.total_tokens:,}",
            )
        )
        self.maintenance_continue_button.configure(state="normal")
        self.maintenance_save_button.configure(state="normal")

    def _global_provider_code(self) -> str:
        return self.provider_labels[self.global_provider_var.get()]

    def _apply_global_provider_defaults(self, force: bool) -> None:
        provider = self._global_provider_code()
        spec = get_provider_spec(provider)
        if force or not self.global_model_var.get():
            self.global_model_var.set(spec.default_model)
        if force:
            self.global_api_key_var.set("")
        if spec.supports_api_base:
            if force or not self.global_api_base_var.get():
                self.global_api_base_var.set(spec.default_api_base or "")
        else:
            self.global_api_base_var.set("")
        self.global_model_combo.configure(
            values=self._model_values(provider, self.global_model_var.get())
        )
        self._update_global_provider_fields()

    def _update_global_provider_fields(self) -> None:
        provider = self._global_provider_code()
        if provider_requires_api_key(provider):
            self.global_api_key_label.grid()
            self.global_api_key_frame.grid()
            self.global_api_key_entry.configure(
                placeholder_text=self.t(
                    "placeholder.api_key_env",
                    env_var=provider_key_env_display(provider),
                )
            )
        else:
            self.global_api_key_label.grid_remove()
            self.global_api_key_frame.grid_remove()
        if provider_supports_api_base(provider):
            self.global_api_base_label.grid()
            self.global_api_base_frame.grid()
        else:
            self.global_api_base_label.grid_remove()
            self.global_api_base_frame.grid_remove()

    def _global_llm_settings_from_form(self) -> LLMSettings:
        provider = self._global_provider_code()
        api_key = self.global_api_key_var.get().strip() or None
        validated = validate_llm_input(
            model=self.global_model_var.get(),
            api_base=self.global_api_base_var.get(),
            temperature=self.global_temperature_var.get(),
            max_tokens=self.global_max_tokens_var.get(),
            api_key_required=provider_requires_api_key(provider),
            api_key_available=self._api_key_available(provider, api_key),
            api_base_required=provider_supports_api_base(provider),
        )
        return LLMSettings(
            provider=provider,
            model=validated.model,
            api_key=api_key,
            api_base=validated.api_base,
            temperature=validated.temperature,
            max_tokens=validated.max_tokens,
        )

    def _settings_for_scope(self, scope: str) -> LLMSettings:
        if scope == "project":
            return self._llm_settings_from_form()
        return self._global_llm_settings_from_form()

    @staticmethod
    def _api_key_available(provider: str, entered_key: str | None) -> bool:
        if entered_key:
            return True
        return any(
            os.getenv(name)
            for name in get_provider_spec(provider).api_key_env_vars
        )

    def _test_connection(self, scope: str) -> None:
        if self.is_busy:
            return
        try:
            settings = self._settings_for_scope(scope)
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_settings", exc)
            return
        self.status_var.set(self.t("status.testing_connection"))
        self.task_var.set(self.t("status.testing_connection"))

        def work() -> str:
            test_provider_connection(settings)
            return settings.provider

        self._start_worker(f"connection:{scope}", work)

    def _refresh_models(self, scope: str) -> None:
        if self.is_busy:
            return
        try:
            settings = self._settings_for_scope(scope)
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_settings", exc)
            return
        self.status_var.set(self.t("status.loading_models"))
        self.task_var.set(self.t("status.loading_models"))

        def work() -> tuple[str, list[str]]:
            models = list_provider_models(settings)
            if not models:
                raise ValueError(self.t("error.no_models_returned"))
            return settings.provider, models

        self._start_worker(f"models:{scope}", work)

    def _save_global_settings(self) -> None:
        try:
            settings = self._global_llm_settings_from_form()
        except (ValueError, ValidationError) as exc:
            self._show_error("error.invalid_settings", exc)
            return
        payload = settings.model_dump(exclude={"api_key"})
        self.preferences = self.settings_store.save_merged(
            {
                "language": self.translator.language,
                "appearance": self.appearance_labels[self.appearance_var.get()],
                "global_llm": payload,
            }
        )
        self.global_configured = True
        self.global_status_var.set(self.t("settings.configured"))
        messagebox.showinfo(
            self.t("dialog.information_title"),
            self.t("message.global_settings_saved"),
        )

    def _apply_global_to_project_form(self) -> None:
        self.provider_var.set(self.global_provider_var.get())
        self.model_var.set(self.global_model_var.get())
        self.api_key_var.set(self.global_api_key_var.get())
        self.api_base_var.set(self.global_api_base_var.get())
        self.temperature_var.set(self.global_temperature_var.get())
        self.max_tokens_var.set(self.global_max_tokens_var.get())
        self._apply_provider_defaults(force=False)

    def _open_exports_folder(self) -> None:
        self._open_folder(exports_directory())

    def _open_projects_folder(self) -> None:
        self._open_folder(projects_directory())

    def _open_folder(self, path: Any) -> None:
        folder = os.fspath(path)
        os.makedirs(folder, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)
        else:
            self._show_error(
                "error.open_folder_failed",
                RuntimeError(self.t("error.open_folder_unsupported")),
            )

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
        was_disabled = str(textbox._textbox.cget("state")) == "disabled"
        if was_disabled:
            textbox.configure(state="normal")
        textbox.delete("1.0", "end")
        textbox.insert("1.0", value)
        textbox.configure(text_color=("gray10", "gray90"))
        if was_disabled:
            textbox.configure(state="disabled")

    def _change_appearance(self, value: str) -> None:
        ctk.set_appearance_mode(self.appearance_labels[value])

    def _change_language(self, value: str) -> None:
        if self.is_busy:
            return

        language = self.language_labels[value]
        if language == self.translator.language:
            return

        provider = self._provider_code()
        global_provider = self._global_provider_code()
        appearance = self.appearance_labels[self.appearance_var.get()]
        current_page = self.current_page
        current_step = self.current_step
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
            "global_model": self.global_model_var.get(),
            "global_api_key": self.global_api_key_var.get(),
            "global_api_base": self.global_api_base_var.get(),
            "global_temperature": self.global_temperature_var.get(),
            "global_max_tokens": self.global_max_tokens_var.get(),
        }

        self.translator.set_language(language)
        self._refresh_locale_maps()
        self.title(self.t("app.title"))

        for child in self.winfo_children():
            child.destroy()
        self._recreate_variables(
            variable_state,
            provider=provider,
            global_provider=global_provider,
            appearance=appearance,
            language=language,
        )
        self._build_ui()
        self._restore_text_state(text_state)
        self._apply_provider_defaults(force=False)
        self._apply_global_provider_defaults(force=False)
        self._update_token_display()
        self.current_step = current_step
        self._show_page(current_page, show_warning=False)
        self._update_maintenance_summary()

    def _recreate_variables(
        self,
        state: dict[str, str],
        provider: str,
        global_provider: str,
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
        self.global_provider_var = ctk.StringVar(
            value=self._provider_label(global_provider)
        )
        self.global_model_var = ctk.StringVar(value=state["global_model"])
        self.global_api_key_var = ctk.StringVar(value=state["global_api_key"])
        self.global_api_base_var = ctk.StringVar(
            value=state["global_api_base"]
        )
        self.global_temperature_var = ctk.StringVar(
            value=state["global_temperature"]
        )
        self.global_max_tokens_var = ctk.StringVar(
            value=state["global_max_tokens"]
        )
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
        self.maintenance_title_var = ctk.StringVar(
            value=self.t("maintenance.no_project")
        )
        self.maintenance_detail_var = ctk.StringVar(
            value=self.t("maintenance.no_project_detail")
        )
        self.global_status_var = ctk.StringVar(
            value=(
                self.t("settings.configured")
                if self.global_configured
                else self.t("settings.not_configured")
            )
        )
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
        self.model_combo.configure(
            values=self._model_values(provider, self.model_var.get())
        )
        self._update_provider_fields()

    def _update_provider_fields(self) -> None:
        provider = self._provider_code()
        if provider_requires_api_key(provider):
            self.api_key_label.grid()
            self.api_key_frame.grid()
            self.api_key_entry.configure(
                placeholder_text=self.t(
                    "placeholder.api_key_env",
                    env_var=provider_key_env_display(provider),
                )
            )
        else:
            self.api_key_label.grid_remove()
            self.api_key_frame.grid_remove()

        if provider_supports_api_base(provider):
            self.api_base_label.grid()
            self.api_base_frame.grid()
        else:
            self.api_base_label.grid_remove()
            self.api_base_frame.grid_remove()

    def _llm_settings_from_form(
        self,
        require_api_key: bool = True,
    ) -> LLMSettings:
        provider = self._provider_code()
        api_key = self.api_key_var.get().strip() or None
        validated = validate_llm_input(
            model=self.model_var.get(),
            api_base=self.api_base_var.get(),
            temperature=self.temperature_var.get(),
            max_tokens=self.max_tokens_var.get(),
            api_key_required=(
                require_api_key and provider_requires_api_key(provider)
            ),
            api_key_available=self._api_key_available(provider, api_key),
            api_base_required=provider_supports_api_base(provider),
        )
        settings = LLMSettings(
            provider=provider,
            model=validated.model,
            api_key=api_key,
            api_base=validated.api_base,
            temperature=validated.temperature,
            max_tokens=validated.max_tokens,
        )
        return settings

    def _book_settings_from_form(self) -> BookSettings:
        idea = self._textbox_value(self.idea_text, "placeholder.main_idea")
        validated = validate_book_input(
            idea=idea,
            output_language=self.output_language_var.get(),
            tone=self.tone_var.get(),
            target_audience=self.audience_var.get(),
            chapter_count=self.chapter_count_var.get(),
            words_per_chapter=self.words_per_chapter_var.get(),
        )
        return BookSettings(
            title=self.title_var.get(),
            idea=validated.idea,
            output_language=validated.output_language,
            tone=validated.tone,
            target_audience=validated.target_audience,
            chapter_count=validated.chapter_count,
            words_per_chapter=validated.words_per_chapter,
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
        if self._project_is_locked():
            messagebox.showwarning(
                self.t("dialog.information_title"),
                self.t("message.outline_locked"),
            )
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
        self.book_generation_started = True
        self.current_step = 3
        self._show_page("create", show_warning=False)
        self.progress_bar.set(0)
        self.status_var.set(self.t("status.generating_book"))
        self.task_var.set(self.t("status.generating_book"))
        existing_output_dir = self.project_output_dir

        def progress(event: GenerationEvent) -> None:
            self.worker_queue.put(("progress", event))

        def work() -> tuple[
            BookProject,
            Path,
            dict[str, Path] | None,
            Exception | None,
        ]:
            service = BookGenerationService(project.llm_settings)
            generated_project = service.generate_book(
                project,
                progress_callback=progress,
                cancel_token=self.cancel_token,
            )
            self.worker_queue.put(("worker_status", "status.saving_outputs"))
            output_dir = existing_output_dir or create_project_directory(
                projects_directory(),
                generated_project.settings.title
                or self.t("default.untitled_book"),
                generated_project.created_at,
            )
            try:
                paths = export_project_bundle(generated_project, output_dir)
                export_error: Exception | None = None
            except Exception as exc:
                paths = None
                export_error = exc
            return generated_project, output_dir, paths, export_error

        self._start_worker("book", work)

    def _start_worker(self, operation: str, function: WorkerFunction) -> None:
        if operation in {"outline", "book"}:
            self._open_progress_dialog(operation)
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
                elif kind == "worker_status":
                    self._handle_worker_status(payload)
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
        self.progress_dialog_message.set(message)
        if self.progress_dialog_bar is not None:
            self.progress_dialog_bar.stop()
            self.progress_dialog_bar.set(event.fraction)
        total_tokens = event.params.get("total_tokens")
        if isinstance(total_tokens, int):
            self.token_var.set(f"{total_tokens:,}")
        self._append_log(message)

    def _handle_worker_status(self, message_key: str) -> None:
        message = self.t(message_key)
        self.task_var.set(message)
        self.progress_dialog_message.set(message)

    def _handle_success(self, operation: str, result: Any) -> None:
        self._close_progress_dialog()
        self._set_busy(False)
        if operation.startswith("connection:"):
            provider = str(result)
            self.status_var.set(self.t("status.connection_ok"))
            self.task_var.set(self.t("status.connection_ok"))
            messagebox.showinfo(
                self.t("dialog.information_title"),
                self.t(
                    "message.connection_ok",
                    provider=self.t(f"provider.{provider}"),
                ),
            )
            return
        if operation.startswith("models:"):
            scope = operation.split(":", 1)[1]
            provider, models = result
            cached_models = self.model_catalog_store.update(provider, models)
            if scope == "global":
                self.global_model_combo.configure(
                    values=self._model_values(
                        provider,
                        self.global_model_var.get(),
                    )
                )
            else:
                self.model_combo.configure(
                    values=self._model_values(provider, self.model_var.get())
                )
            self.status_var.set(self.t("status.models_updated"))
            self.task_var.set(self.t("status.models_updated"))
            messagebox.showinfo(
                self.t("dialog.information_title"),
                self.t(
                    "message.models_updated",
                    count=len(cached_models),
                    provider=self.t(f"provider.{provider}"),
                ),
            )
            return
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
            self.current_step = 2
            self._show_page("create", show_warning=False)
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
            project, output_dir, paths, export_error = result
            self.project = project
            self.project_output_dir = output_dir
            self.project_file_path = output_dir / "project.json"
            self._replace_text(self.preview_text, render_markdown(self.project))
            self.book_generation_started = True
            self.book_generation_finished = True
            self.auto_export_completed = export_error is None
            self.current_step = 3
            self._show_page("create", show_warning=False)
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
            if export_error is None and paths is not None:
                self._append_log(
                    self.t("log.auto_exported", path=output_dir)
                )
                messagebox.showinfo(
                    self.t("dialog.information_title"),
                    self.t("message.auto_exported", path=output_dir),
                )
            else:
                self._append_log(
                    self.t("log.auto_export_failed", error=str(export_error))
                )
                messagebox.showwarning(
                    self.t("dialog.information_title"),
                    self.t(
                        "error.auto_export_failed",
                        path=output_dir,
                        error=str(export_error),
                    ),
                )
        self._update_maintenance_summary()
        self._refresh_flow_ui()

    def _handle_cancelled(self) -> None:
        self._close_progress_dialog()
        self._set_busy(False)
        self.status_var.set(self.t("status.cancelled"))
        self.task_var.set(self.t("status.cancelled"))
        self._append_log(self.t("status.cancelled"))
        self._update_maintenance_summary()
        self._refresh_flow_ui()

    def _handle_worker_error(self, operation: str, error: Exception) -> None:
        self._close_progress_dialog()
        self._set_busy(False)
        self.status_var.set(self.t("status.failed"))
        self.task_var.set(self.t("status.failed"))
        self._append_log(self.t("log.error", error=str(error)))
        key = (
            "error.connection_test_failed"
            if operation.startswith("connection:")
            else (
                "error.model_refresh_failed"
                if operation.startswith("models:")
                else (
                    "error.generation_failed"
                    if operation in {"outline", "book"}
                    else "error.invalid_settings"
                )
            )
        )
        self._show_error(key, error)
        self._update_maintenance_summary()
        self._refresh_flow_ui()

    def _cancel_generation(self) -> None:
        if not self.is_busy:
            return
        self.cancel_token.cancel()
        self.status_var.set(self.t("status.cancelling"))
        self.task_var.set(self.t("status.cancelling"))
        self.progress_dialog_message.set(self.t("status.cancelling"))
        self._append_log(self.t("log.cancel_requested"))

    def _open_progress_dialog(self, operation: str) -> None:
        self._close_progress_dialog()
        dialog = ctk.CTkToplevel(self)
        dialog.title(self.t("dialog.generation_progress"))
        dialog.geometry("500x220")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.protocol("WM_DELETE_WINDOW", self._cancel_generation)
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(0, weight=1)

        card = ctk.CTkFrame(dialog, corner_radius=16)
        card.grid(row=0, column=0, padx=22, pady=22, sticky="nsew")
        card.grid_columnconfigure(0, weight=1)
        title_key = (
            "status.generating_outline"
            if operation == "outline"
            else "status.generating_book"
        )
        self.progress_dialog_message.set(self.t(title_key))
        ctk.CTkLabel(
            card,
            text=self.t("dialog.please_wait"),
            font=ctk.CTkFont(size=20, weight="bold"),
        ).grid(row=0, column=0, padx=20, pady=(24, 8))
        ctk.CTkLabel(
            card,
            textvariable=self.progress_dialog_message,
            wraplength=420,
            text_color=("gray35", "gray70"),
        ).grid(row=1, column=0, padx=20, pady=8)
        bar = ctk.CTkProgressBar(
            card,
            mode="indeterminate" if operation == "outline" else "determinate",
        )
        bar.grid(row=2, column=0, padx=28, pady=16, sticky="ew")
        if operation == "outline":
            bar.start()
        else:
            bar.set(0)
        ctk.CTkButton(
            card,
            text=self.t("button.cancel"),
            command=self._cancel_generation,
            width=120,
        ).grid(row=3, column=0, padx=20, pady=(4, 22))

        self.progress_dialog = dialog
        self.progress_dialog_bar = bar
        dialog.grab_set()
        dialog.lift()

    def _close_progress_dialog(self) -> None:
        if self.progress_dialog_bar is not None:
            self.progress_dialog_bar.stop()
        if self.progress_dialog is not None and self.progress_dialog.winfo_exists():
            try:
                self.progress_dialog.grab_release()
            except Exception:
                pass
            self.progress_dialog.destroy()
        self.progress_dialog = None
        self.progress_dialog_bar = None

    def _set_busy(self, busy: bool) -> None:
        self.is_busy = busy
        active_state = "disabled" if busy else "normal"
        for button in self.busy_sensitive_buttons:
            button.configure(state=active_state)
        for button in self.cancel_buttons:
            button.configure(state="normal" if busy else "disabled")
        self.language_menu.configure(state="disabled" if busy else "normal")
        self._refresh_flow_ui()

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
            initialdir=projects_directory(),
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
            self.project_file_path = Path(path)
            self.project_output_dir = self._output_directory_for_project_file(
                self.project_file_path
            )
            self.status_var.set(self.t("status.project_saved"))
            self._append_log(self.t("log.project_saved", path=path))
            self._update_maintenance_summary()
            messagebox.showinfo(
                self.t("dialog.information_title"),
                self.t("message.project_saved"),
            )
        except Exception as exc:
            self._show_error("error.save_failed", exc)

    def _load_project(self) -> None:
        path = filedialog.askopenfilename(
            title=self.t("dialog.load_project"),
            initialdir=projects_directory(),
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
            self.project_file_path = Path(path)
            self.project_output_dir = self._output_directory_for_project_file(
                self.project_file_path
            )
            self.book_generation_finished = False
            self.book_generation_started = self._project_is_locked()
            self.book_generation_finished = self._project_is_finished()
            self.auto_export_completed = all(
                (self.project_output_dir / filename).exists()
                for filename in ("book.md", "book.txt", "book.docx", "book.pdf")
            )
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
            global_provider = self._global_provider_code()
            self.api_key_var.set(
                self.global_api_key_var.get()
                if project.llm_settings.provider == global_provider
                else ""
            )
        else:
            self._apply_global_to_project_form()
        self._apply_provider_defaults(force=False)

        outline = BookOutline(
            title=settings.title or self.t("default.untitled_book"),
            subtitle=project.subtitle,
            chapters=project.chapters,
        )
        self._replace_text(self.outline_text, self._editable_outline_json(outline))
        self._replace_text(self.preview_text, render_markdown(project))
        self._update_token_display()
        self.current_step = 3 if self._project_is_locked() else 2
        self._show_page("create", show_warning=False)
        self._update_maintenance_summary()

    @staticmethod
    def _output_directory_for_project_file(path: Path) -> Path:
        if path.name.casefold() == "project.json":
            return path.parent
        return path.parent / path.stem

    def _export(self, format_name: str) -> None:
        if not self.project:
            messagebox.showwarning(
                self.t("dialog.information_title"),
                self.t("error.no_project"),
            )
            return
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
            "pdf": (
                "dialog.export_pdf",
                ".pdf",
                "dialog.pdf_files",
                export_pdf,
                "PDF",
            ),
        }
        title_key, extension, filetype_key, exporter, display_format = options[
            format_name
        ]
        filename = filedialog.asksaveasfilename(
            title=self.t(title_key),
            initialdir=exports_directory(),
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
        if isinstance(error, InputValidationError):
            messages = [
                self.t(
                    issue.message_key,
                    field=self.t(issue.field_key),
                    **issue.params,
                )
                for issue in error.issues
            ]
            messagebox.showerror(
                self.t("validation.title"),
                self.t("validation.intro")
                + "\n\n"
                + "\n".join(f"• {message}" for message in messages),
            )
            return
        if isinstance(error, ValidationError):
            messagebox.showerror(
                self.t("validation.title"),
                self.t("validation.generic"),
            )
            return
        if key == "error.invalid_outline":
            messagebox.showerror(
                self.t("dialog.error_title"),
                self.t("validation.outline"),
            )
            return
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
        updates: dict[str, Any] = {
            "language": self.translator.language,
            "appearance": self.appearance_labels[self.appearance_var.get()],
        }
        self.settings_store.save_merged(updates)
        self.destroy()


def run() -> None:
    """Launch the desktop application."""
    app = AIBookBatchWriterApp()
    app.mainloop()
