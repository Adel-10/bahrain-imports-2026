"""Tests for reporting.py: number formatting and the markdown clean-up applied to every report.

    pytest tests/test_reporting.py

The report templates are wrapped to a readable width in the source code, and ``write_markdown``
joins those lines back into whole paragraphs. These tests make sure it joins only ordinary text
and never touches tables, headings, lists, images or code blocks, which would break the markdown.
"""

from __future__ import annotations

import pandas as pd

from reporting import and_list, join_paragraphs, md_table, pct, share_words, write_markdown


def test_number_formats():
    assert pct(-53.6) == "-54%"
    assert pct(7.9) == "+8%"
    assert pct(0.2) == "0%"
    assert share_words(66.4) == "two-thirds"
    assert share_words(58) == "58%"
    assert and_list(["a"]) == "a"
    assert and_list(["a", "b", "c"]) == "a, b and c"


def test_md_table():
    table = md_table(pd.DataFrame({"Year": ["2025"], "Rows": [324_816], "Share": [12.345]}), ".1f")
    assert table.splitlines() == ["| Year | Rows | Share |", "|---|---|---|", "| 2025 | 324,816 | 12.3 |"]


def test_paragraphs_are_joined_but_blocks_are_not():
    text = "\n".join([
        "# Heading",
        "First line of a paragraph",
        "continues here,",
        "and ends in 2026.",                   # a line starting with a year is text, not a numbered list
        "",
        "- a list item",
        "  wrapped onto two lines",
        "- another item",
        "1. numbered",
        "",
        "| a | b |",
        "|---|---|",
        "| 1 | 2 |",
        "![chart](figures/x.png)",
        "```",
        "code line one",
        "code line two",
        "```",
    ])
    out = join_paragraphs(text).splitlines()
    assert out[1] == "First line of a paragraph continues here, and ends in 2026."
    assert "- a list item wrapped onto two lines" in out
    assert "- another item" in out and "1. numbered" in out
    assert "| a | b |" in out and "|---|---|" in out and "| 1 | 2 |" in out
    assert "code line one" in out and "code line two" in out


def test_write_markdown_tidies_blank_lines(tmp_path):
    path = tmp_path / "report.md"
    write_markdown(path, "\n\n# Title\n\n\n\nText\nmore text\n\n\n")
    assert path.read_text(encoding="utf-8") == "# Title\n\nText more text\n"
