"""Export book projects to Markdown, plain text, and DOCX."""

from __future__ import annotations

from pathlib import Path

from docx import Document

from ai_book_batch_writer.models import BookProject
from ai_book_batch_writer.utils import ensure_parent_dir


def _title(project: BookProject) -> str:
    return project.settings.title or "Untitled Book"


def _metadata_lines(project: BookProject) -> list[tuple[str, str]]:
    settings = project.settings
    return [
        ("Language", settings.output_language),
        ("Tone", settings.tone),
        ("Target audience", settings.target_audience),
        ("Main idea", settings.idea),
        ("Outline tokens", str(project.token_usage.outline_tokens)),
        ("Book tokens", str(project.token_usage.book_tokens)),
        ("Total tokens", str(project.token_usage.total_tokens)),
    ]


def render_markdown(project: BookProject) -> str:
    """Render a project as Markdown."""
    lines = [f"# {_title(project)}"]
    if project.subtitle:
        lines.extend(["", f"*{project.subtitle}*"])

    lines.extend(["", "## Project Metadata", ""])
    lines.extend(f"- **{label}:** {value}" for label, value in _metadata_lines(project))
    lines.extend(["", "## Table of Contents", ""])
    lines.extend(
        f"{chapter.number}. [{chapter.title}](#chapter-{chapter.number})"
        for chapter in project.chapters
    )

    for chapter in project.chapters:
        lines.extend(
            [
                "",
                f'<a id="chapter-{chapter.number}"></a>',
                f"## Chapter {chapter.number}: {chapter.title}",
                "",
            ]
        )
        if chapter.content:
            lines.extend([chapter.content.strip(), ""])
        for section in chapter.sections:
            lines.extend([f"### {section.title}", ""])
            if section.content:
                lines.extend([section.content.strip(), ""])
            else:
                lines.extend([f"*{section.summary}*", ""])
    return "\n".join(lines).rstrip() + "\n"


def export_markdown(project: BookProject, path: str | Path) -> None:
    """Export a project to a UTF-8 Markdown file."""
    target = ensure_parent_dir(path)
    target.write_text(render_markdown(project), encoding="utf-8")


def export_txt(project: BookProject, path: str | Path) -> None:
    """Export a project to a readable UTF-8 plain-text file."""
    lines = [_title(project)]
    if project.subtitle:
        lines.append(project.subtitle)
    lines.append("=" * max(12, len(lines[0])))
    lines.append("")
    lines.extend(f"{label}: {value}" for label, value in _metadata_lines(project))
    lines.extend(["", "TABLE OF CONTENTS", ""])
    lines.extend(
        f"{chapter.number}. {chapter.title}" for chapter in project.chapters
    )

    for chapter in project.chapters:
        heading = f"CHAPTER {chapter.number}: {chapter.title}"
        lines.extend(["", heading, "-" * len(heading), ""])
        if chapter.content:
            lines.extend([chapter.content.strip(), ""])
        for section in chapter.sections:
            lines.extend([section.title, "~" * len(section.title), ""])
            lines.extend([(section.content or section.summary).strip(), ""])

    target = ensure_parent_dir(path)
    target.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def export_docx(project: BookProject, path: str | Path) -> None:
    """Export a project to DOCX with semantic heading levels."""
    document = Document()
    document.add_heading(_title(project), level=1)
    if project.subtitle:
        document.add_paragraph(project.subtitle, style="Subtitle")

    document.add_heading("Project Metadata", level=2)
    for label, value in _metadata_lines(project):
        paragraph = document.add_paragraph()
        paragraph.add_run(f"{label}: ").bold = True
        paragraph.add_run(value)

    document.add_heading("Table of Contents", level=2)
    for chapter in project.chapters:
        document.add_paragraph(
            f"{chapter.number}. {chapter.title}",
            style="List Number",
        )

    for chapter in project.chapters:
        document.add_heading(
            f"Chapter {chapter.number}: {chapter.title}",
            level=2,
        )
        if chapter.content:
            for paragraph_text in chapter.content.split("\n\n"):
                if paragraph_text.strip():
                    document.add_paragraph(paragraph_text.strip())
        for section in chapter.sections:
            document.add_heading(section.title, level=3)
            content = section.content or section.summary
            for paragraph_text in content.split("\n\n"):
                if paragraph_text.strip():
                    document.add_paragraph(paragraph_text.strip())

    target = ensure_parent_dir(path)
    document.save(target)
