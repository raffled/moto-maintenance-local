"""
Unit tests for ingestion/crossref.py pure functions.
No PDF file or external dependencies required.
"""

import pytest
from ingestion.crossref import _build_page_map, _section_for_page, annotate_chunks
from ingestion.index import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_TOC = [
    {"number": "5.3",    "title": "Starting the vehicle",       "page": 15},
    {"number": "6.9",    "title": "Removing the fork legs",     "page": 20},
    {"number": "6.10",   "title": "Disassembling the fork legs","page": 22},
    {"number": "6.11",   "title": "Disassembling the cartridge","page": 25},
    {"number": "7.2",    "title": "Adjusting the handlebar",    "page": 50},
    {"number": "7.3",    "title": "Adjusting the clutch lever", "page": 51},
]


def make_chunk(section: str, phase: str, text: str) -> Chunk:
    return Chunk(
        id=f"manual__{section}__{phase or 'full'}",
        manual="manual",
        chapter_num=1,
        chapter_title="Test",
        section=section,
        section_title="",
        phase=phase,
        text=text,
        torque_specs=[],
        figure_refs=[],
        page_nums=[1],
    )


# ---------------------------------------------------------------------------
# _build_page_map
# ---------------------------------------------------------------------------

class TestBuildPageMap:
    def test_returns_sorted_by_page(self):
        page_map = _build_page_map(SAMPLE_TOC)
        pages = [p for p, _ in page_map]
        assert pages == sorted(pages)

    def test_contains_all_entries(self):
        page_map = _build_page_map(SAMPLE_TOC)
        assert len(page_map) == len(SAMPLE_TOC)


# ---------------------------------------------------------------------------
# _section_for_page
# ---------------------------------------------------------------------------

class TestSectionForPage:
    @pytest.fixture
    def page_map(self):
        return _build_page_map(SAMPLE_TOC)

    def test_exact_start_page_returns_that_section(self, page_map):
        assert _section_for_page(22, page_map) == "6.10"

    def test_page_within_section_range(self, page_map):
        # Page 23 is between 6.10 (starts p.22) and 6.11 (starts p.25)
        assert _section_for_page(23, page_map) == "6.10"

    def test_page_before_first_entry_returns_empty(self, page_map):
        assert _section_for_page(5, page_map) == ""

    def test_page_at_section_boundary(self, page_map):
        # Page 25 is the exact start of 6.11
        assert _section_for_page(25, page_map) == "6.11"

    def test_page_beyond_last_entry_returns_last_section(self, page_map):
        # Page 100 is after all TOC entries — should return last section
        assert _section_for_page(100, page_map) == "7.3"


# ---------------------------------------------------------------------------
# annotate_chunks
# ---------------------------------------------------------------------------

class TestAnnotateChunks:
    def test_condition_extracted_as_prerequisite(self):
        chunk = make_chunk("9.9", "", "Condition: The shock absorber is removed.\nStep one.")
        annotate_chunks([chunk], SAMPLE_TOC)
        assert chunk.prerequisites == ["The shock absorber is removed."]

    def test_multiple_conditions_extracted(self):
        chunk = make_chunk("18.5", "", "Condition: Engine is cold.\nCondition: Oil drained.")
        annotate_chunks([chunk], SAMPLE_TOC)
        assert len(chunk.prerequisites) == 2

    def test_no_condition_empty_prerequisites(self):
        chunk = make_chunk("7.2", "", "Remove the handlebar clamp screws.")
        annotate_chunks([chunk], SAMPLE_TOC)
        assert chunk.prerequisites == []

    def test_page_refs_resolved_for_preparatory(self):
        text = "Preparatory work\n‒ Remove the fork legs.\n (p. 20)\n‒ Disassemble.\n (p. 22)"
        chunk = make_chunk("6.11", "preparatory", text)
        annotate_chunks([chunk], SAMPLE_TOC)
        assert "6.9" in chunk.references   # p.20 → section 6.9
        assert "6.10" in chunk.references  # p.22 → section 6.10

    def test_page_refs_suppressed_for_main(self):
        text = "Main work\nSpecial tool (T14015S)\n (p. 22)"
        chunk = make_chunk("6.11", "main", text)
        annotate_chunks([chunk], SAMPLE_TOC)
        assert chunk.references == []

    def test_page_refs_suppressed_for_full_phase(self):
        text = "Some procedure\n (p. 22)"
        chunk = make_chunk("6.11", "", text)
        annotate_chunks([chunk], SAMPLE_TOC)
        assert chunk.references == []

    def test_self_reference_excluded(self):
        # p.25 resolves to 6.11 — should not appear in 6.11's own references
        text = "Preparatory work\n‒ Check something.\n (p. 25)"
        chunk = make_chunk("6.11", "preparatory", text)
        annotate_chunks([chunk], SAMPLE_TOC)
        assert "6.11" not in chunk.references

    def test_references_deduplicated(self):
        # Two citations that resolve to the same section
        text = "Preparatory work\n‒ Step one.\n (p. 22)\n‒ Step two.\n (p. 23)"
        chunk = make_chunk("6.11", "preparatory", text)
        annotate_chunks([chunk], SAMPLE_TOC)
        assert chunk.references.count("6.10") == 1
