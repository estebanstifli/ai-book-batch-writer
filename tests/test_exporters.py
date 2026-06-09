from __future__ import annotations

from docx import Document

from ai_book_batch_writer.exporters import (
    export_docx,
    export_markdown,
    export_txt,
)


def test_markdown_export(tmp_path, sample_project) -> None:
    sample_project.token_usage.add_outline(100)
    sample_project.token_usage.add_book(900)
    path = tmp_path / "book.md"
    export_markdown(sample_project, path)
    content = path.read_text(encoding="utf-8")

    assert "# Reliable AI Systems" in content
    assert "## Chapter 1: Start With the Problem" in content
    assert "### Map the Workflow" in content
    assert "Document the current process" in content
    assert "**Total tokens:** 1000" in content


def test_txt_export(tmp_path, sample_project) -> None:
    path = tmp_path / "book.txt"
    export_txt(sample_project, path)
    content = path.read_text(encoding="utf-8")

    assert "RELIABLE AI SYSTEMS" not in content
    assert "CHAPTER 1: Start With the Problem" in content
    assert "Map the Workflow" in content


def test_docx_export_uses_heading_levels(tmp_path, sample_project) -> None:
    path = tmp_path / "book.docx"
    export_docx(sample_project, path)
    document = Document(path)
    headings = [
        (paragraph.text, paragraph.style.name)
        for paragraph in document.paragraphs
        if paragraph.style.name.startswith("Heading")
    ]

    assert ("Reliable AI Systems", "Heading 1") in headings
    assert ("Chapter 1: Start With the Problem", "Heading 2") in headings
    assert ("Map the Workflow", "Heading 3") in headings
