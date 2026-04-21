"""
Unit tests for agent/retrieval.py pure functions.
No ChromaDB or OpenAI required.

retrieve() and retrieve_from_section() require ChromaDB and are covered by
the smoke test (test_dependency_retrieval section).
"""

import json
import pytest
from agent.retrieval import _parse_result, retrieve_from_section, RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_metadata(
    *,
    manual: str = "test_manual",
    chapter_num: int = 6,
    chapter_title: str = "Fork",
    section: str = "6.12",
    section_title: str = "Disassembling the piston rod",
    phase: str = "preparatory",
    torque_specs: list = None,
    figure_refs: list = None,
    page_nums: list = None,
    image_paths: list = None,
    prerequisites: list = None,
    references: list = None,
) -> dict:
    return {
        "manual":        manual,
        "chapter_num":   chapter_num,
        "chapter_title": chapter_title,
        "section":       section,
        "section_title": section_title,
        "phase":         phase,
        "torque_specs":  json.dumps(torque_specs or []),
        "figure_refs":   json.dumps(figure_refs or []),
        "page_nums":     json.dumps(page_nums or [25]),
        "image_paths":   json.dumps(image_paths or []),
        "prerequisites": json.dumps(prerequisites or []),
        "references":    json.dumps(references or []),
    }


# ---------------------------------------------------------------------------
# _parse_result
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_scalar_fields_set_correctly(self):
        meta = make_metadata(section="6.12", phase="preparatory")
        chunk = _parse_result("my_id", meta, "some text")
        assert chunk.id == "my_id"
        assert chunk.section == "6.12"
        assert chunk.phase == "preparatory"
        assert chunk.text == "some text"

    def test_json_lists_deserialized(self):
        meta = make_metadata(
            torque_specs=[{"bolt": "M10", "nm": 40.0, "ftlbf": 29.5, "note": ""}],
            figure_refs=["W00322-10"],
            page_nums=[22, 23],
            image_paths=["data/images/manual/p0022_00.jpg"],
            prerequisites=["The fork legs have been removed"],
            references=["6.10", "6.11"],
        )
        chunk = _parse_result("id", meta, "text")
        assert chunk.torque_specs == [{"bolt": "M10", "nm": 40.0, "ftlbf": 29.5, "note": ""}]
        assert chunk.figure_refs == ["W00322-10"]
        assert chunk.page_nums == [22, 23]
        assert chunk.image_paths == ["data/images/manual/p0022_00.jpg"]
        assert chunk.prerequisites == ["The fork legs have been removed"]
        assert chunk.references == ["6.10", "6.11"]

    def test_default_depth_is_zero(self):
        chunk = _parse_result("id", make_metadata(), "text")
        assert chunk.depth == 0

    def test_explicit_depth_set(self):
        chunk = _parse_result("id", make_metadata(), "text", depth=2)
        assert chunk.depth == 2

    def test_empty_json_lists_give_empty_python_lists(self):
        meta = make_metadata()  # all list fields default to []
        chunk = _parse_result("id", meta, "text")
        assert chunk.torque_specs == []
        assert chunk.figure_refs == []
        assert chunk.image_paths == []
        assert chunk.prerequisites == []
        assert chunk.references == []

    def test_returns_retrieved_chunk_instance(self):
        chunk = _parse_result("id", make_metadata(), "text")
        assert isinstance(chunk, RetrievedChunk)

    def test_chapter_fields_passed_through(self):
        meta = make_metadata(chapter_num=9, chapter_title="Rear suspension")
        chunk = _parse_result("id", meta, "text")
        assert chunk.chapter_num == 9
        assert chunk.chapter_title == "Rear suspension"

    def test_manual_field_passed_through(self):
        meta = make_metadata(manual="FE_501s_2026_full_stem")
        chunk = _parse_result("id", meta, "text")
        assert chunk.manual == "FE_501s_2026_full_stem"


# ---------------------------------------------------------------------------
# retrieve_from_section — public API surface
# ---------------------------------------------------------------------------

class TestRetrieveFromSectionInterface:
    def test_is_callable(self):
        assert callable(retrieve_from_section)

    def test_accepts_section_and_collection_args(self):
        import inspect
        sig = inspect.signature(retrieve_from_section)
        params = list(sig.parameters)
        assert params[0] == "section"
        assert params[1] == "collection"

    def test_manual_param_is_optional(self):
        import inspect
        sig = inspect.signature(retrieve_from_section)
        manual_param = sig.parameters.get("manual")
        assert manual_param is not None
        assert manual_param.default is None
