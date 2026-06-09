"""JSON extraction, validation, and model-assisted repair."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from ai_book_batch_writer.models import BookChapter, BookOutline
from ai_book_batch_writer.prompts import build_json_repair_prompt

_FENCED_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _balanced_json_slice(text: str) -> str | None:
    start = next((index for index, char in enumerate(text) if char in "[{"), None)
    if start is None:
        return None

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char in pairs:
            stack.append(pairs[char])
        elif char in "}]":
            if not stack or char != stack.pop():
                return None
            if not stack:
                return text[start : index + 1]
    return None


def extract_json_from_text(text: str) -> str:
    """Extract the first complete JSON object or array from model output."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Model output was empty.")

    fenced = _FENCED_JSON.search(stripped)
    if fenced:
        return fenced.group(1).strip()

    candidate = _balanced_json_slice(stripped)
    if candidate:
        return candidate
    raise ValueError("No complete JSON object was found in model output.")


def parse_outline_document(text: str) -> BookOutline:
    """Extract and validate a complete outline document."""
    extracted = extract_json_from_text(text)
    try:
        payload: Any = json.loads(extracted)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc.msg}") from exc

    if isinstance(payload, list):
        payload = {"title": "Untitled Book", "subtitle": None, "chapters": payload}
    try:
        return BookOutline.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Outline schema validation failed: {exc}") from exc


def parse_outline_json(text: str) -> list[BookChapter]:
    """Extract and validate chapters from outline JSON."""
    return parse_outline_document(text).chapters


def repair_json_with_llm(text: str, llm: BaseChatModel) -> str:
    """Use a LangChain repair flow to convert malformed output into JSON."""
    chain = build_json_repair_prompt() | llm | StrOutputParser()
    return chain.invoke({"malformed_json": text}).strip()

