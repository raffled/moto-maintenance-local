"""
Unit tests for ingestion/index.py pure functions.
No PDF file or external dependencies required.
"""

import pytest
from ingestion.index import _phase_split, _split_page_by_section
from ingestion.parse import ParsedPage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_page(text: str, page_num: int = 1) -> ParsedPage:
    return ParsedPage(
        page_num=page_num,
        chapter_num=1,
        chapter_title="Test Chapter",
        section="",
        text=text,
    )


# ---------------------------------------------------------------------------
# _phase_split
# ---------------------------------------------------------------------------

class TestPhaseSplit:
    def test_no_phase_headers_returns_full(self):
        text = "Some procedure text without any phase headers."
        result = _phase_split(text)
        assert result == [("", text)]

    def test_single_preparatory(self):
        text = "Preparatory work\nStep one\nStep two"
        result = _phase_split(text)
        assert len(result) == 1
        assert result[0][0] == "preparatory"
        assert "Step one" in result[0][1]

    def test_all_three_phases(self):
        text = (
            "Preparatory work\nRemove the cover.\n"
            "Main work\nInstall the part.\n"
            "Reworking\nReinstall the cover."
        )
        result = _phase_split(text)
        assert len(result) == 3
        keys = [r[0] for r in result]
        assert keys == ["preparatory", "main", "reworking"]

    def test_phase_text_content(self):
        text = "Preparatory work\nStep A\nMain work\nStep B"
        result = _phase_split(text)
        assert "Step A" in result[0][1]
        assert "Step B" in result[1][1]

    def test_text_before_first_phase_included(self):
        # Pre-phase text (e.g. section heading) should be kept
        text = "7.2 Adjust the handlebar\nPreparatory work\nStep one"
        result = _phase_split(text)
        # The pre-phase fragment or it's merged into preparatory
        combined = " ".join(r[1] for r in result)
        assert "7.2 Adjust the handlebar" in combined

    def test_empty_text(self):
        result = _phase_split("")
        assert result == [("", "")]


# ---------------------------------------------------------------------------
# _split_page_by_section
# ---------------------------------------------------------------------------

class TestSplitPageBySection:
    def test_no_section_heading_returns_single_fragment(self):
        page = make_page("Some text with no section heading.")
        result = _split_page_by_section(page)
        assert result == [("", "Some text with no section heading.")]

    def test_section_heading_at_start(self):
        page = make_page("7.2 Adjusting the handlebar\nStep one\nStep two")
        result = _split_page_by_section(page)
        assert len(result) == 1
        assert result[0][0] == "7.2"
        assert "Step one" in result[0][1]

    def test_pre_heading_text_attributed_to_empty_key(self):
        page = make_page("End of previous section.\n7.2 New section\nNew content")
        result = _split_page_by_section(page)
        assert result[0][0] == ""
        assert "End of previous section" in result[0][1]
        assert result[1][0] == "7.2"

    def test_multiple_sections_on_one_page(self):
        page = make_page(
            "7.1 First section\nContent A\n7.2 Second section\nContent B"
        )
        result = _split_page_by_section(page)
        assert len(result) == 2
        assert result[0][0] == "7.1"
        assert result[1][0] == "7.2"
        assert "Content A" in result[0][1]
        assert "Content B" in result[1][1]

    def test_deep_section_number_parsed(self):
        page = make_page("18.4.15 Checking the valve seat\nProcedure text")
        result = _split_page_by_section(page)
        assert result[0][0] == "18.4.15"

    def test_measurement_value_not_parsed_as_section(self):
        # "1.0 mA" should not be treated as section "1.0" (no capital after number)
        page = make_page("Current: 1.0 mA at operating temperature")
        result = _split_page_by_section(page)
        assert result == [("", "Current: 1.0 mA at operating temperature")]
