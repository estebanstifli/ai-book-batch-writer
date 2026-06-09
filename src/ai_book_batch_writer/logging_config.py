"""Application logging configuration."""

from __future__ import annotations

import logging
import os

from ai_book_batch_writer.config import user_data_dir


def configure_logging() -> None:
    """Configure file logging once for diagnostics."""
    log_level = os.getenv("AI_BOOK_WRITER_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(
                user_data_dir() / "ai_book_batch_writer.log",
                encoding="utf-8",
            )
        ],
        force=True,
    )

