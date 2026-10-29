"""Deterministic synthetic PDF builder for evaluation tests.

PyMuPDF is used (rather than a checked-in binary fixture) so the sample
documents are reproducible from source and easy to review. Text is wrapped at a
fixed width; with the project's 512-character chunk size each page produces
exactly one chunk.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Sequence

import pymupdf

from tests.evaluation.corpus import SampleDocument, SamplePage

FONT_SIZE = 10
LINE_HEIGHT = 14
MARGIN_X = 72
MARGIN_Y = 72
WRAP_WIDTH = 90


def build_pdf_bytes(pages: Sequence[SamplePage] | Sequence[str]) -> bytes:
    """Render page texts into a small, valid PDF and return its bytes."""

    document = pymupdf.open()
    try:
        for raw_page in pages:
            text = raw_page.text if isinstance(raw_page, SamplePage) else str(raw_page)
            page = document.new_page()
            y = MARGIN_Y
            for line in textwrap.wrap(
                text,
                width=WRAP_WIDTH,
                break_long_words=True,
                break_on_hyphens=False,
            ):
                page.insert_text((MARGIN_X, y), line, fontsize=FONT_SIZE)
                y += LINE_HEIGHT
        return document.tobytes()
    finally:
        document.close()


def write_pdf(path: Path, document: SampleDocument) -> Path:
    """Write a sample document to ``path`` and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(build_pdf_bytes(document.pages))
    return path
