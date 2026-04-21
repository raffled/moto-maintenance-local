"""
Unit tests for agent/planner.py pure functions.
No Claude API or ChromaDB required.
"""

import pytest
from agent.planner import _build_context, _collect_torque_specs, Plan
from agent.retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    section: str,
    phase: str = "",
    text: str = "Some procedure text.",
    depth: int = 0,
    torque_specs: list = None,
    image_paths: list = None,
    section_title: str = "",
) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"manual__1__{section}__{phase or 'full'}",
        manual="test_manual",
        chapter_num=7,
        chapter_title="Handlebar, controls",
        section=section,
        section_title=section_title,
        phase=phase,
        text=text,
        torque_specs=torque_specs or [],
        figure_refs=[],
        page_nums=[1],
        image_paths=image_paths or [],
        prerequisites=[],
        references=[],
        depth=depth,
    )


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_single_chunk_contains_section_number(self):
        chunk = make_chunk("7.2", text="Step one.\nStep two.")
        result = _build_context([chunk])
        assert "7.2" in result

    def test_section_title_included_when_present(self):
        chunk = make_chunk("7.2", section_title="Adjusting the handlebar")
        result = _build_context([chunk])
        assert "Adjusting the handlebar" in result

    def test_phase_label_included(self):
        chunk = make_chunk("7.2", phase="preparatory")
        result = _build_context([chunk])
        assert "preparatory" in result

    def test_depth_zero_labelled_as_target(self):
        chunk = make_chunk("7.2", depth=0)
        result = _build_context([chunk])
        assert "target" in result

    def test_depth_nonzero_labelled_as_prerequisite(self):
        chunk = make_chunk("6.10", depth=1)
        result = _build_context([chunk])
        assert "prerequisite" in result

    def test_chunk_text_included(self):
        chunk = make_chunk("7.2", text="Remove the handlebar clamp screws.")
        result = _build_context([chunk])
        assert "Remove the handlebar clamp screws." in result

    def test_multiple_chunks_all_present(self):
        chunks = [
            make_chunk("6.10", depth=1, text="Prerequisite text."),
            make_chunk("6.12", depth=0, text="Target text."),
        ]
        result = _build_context(chunks)
        assert "6.10" in result
        assert "6.12" in result
        assert "Prerequisite text." in result
        assert "Target text." in result

    def test_empty_chunks_returns_empty_string(self):
        assert _build_context([]) == ""


# ---------------------------------------------------------------------------
# _collect_torque_specs
# ---------------------------------------------------------------------------

class TestCollectTorqueSpecs:
    def test_single_spec_returned(self):
        chunk = make_chunk("7.2", torque_specs=[
            {"bolt": "M10", "nm": 40.0, "ftlbf": 29.5, "note": ""}
        ])
        specs = _collect_torque_specs([chunk])
        assert len(specs) == 1
        assert specs[0]["bolt"] == "M10"

    def test_duplicate_specs_deduplicated(self):
        spec = {"bolt": "M10", "nm": 40.0, "ftlbf": 29.5, "note": ""}
        c1 = make_chunk("7.2", torque_specs=[spec])
        c2 = make_chunk("7.3", torque_specs=[spec])
        specs = _collect_torque_specs([c1, c2])
        assert len(specs) == 1

    def test_distinct_specs_both_returned(self):
        c1 = make_chunk("7.2", torque_specs=[
            {"bolt": "M10", "nm": 40.0, "ftlbf": 29.5, "note": ""}
        ])
        c2 = make_chunk("7.3", torque_specs=[
            {"bolt": "M8", "nm": 20.0, "ftlbf": 14.8, "note": ""}
        ])
        specs = _collect_torque_specs([c1, c2])
        assert len(specs) == 2
        bolts = {s["bolt"] for s in specs}
        assert bolts == {"M10", "M8"}

    def test_same_bolt_different_nm_not_deduplicated(self):
        # Different torque values for the same bolt size are distinct specs
        c1 = make_chunk("7.2", torque_specs=[
            {"bolt": "M8", "nm": 20.0, "ftlbf": 14.8, "note": ""}
        ])
        c2 = make_chunk("7.3", torque_specs=[
            {"bolt": "M8", "nm": 25.0, "ftlbf": 18.4, "note": ""}
        ])
        specs = _collect_torque_specs([c1, c2])
        assert len(specs) == 2

    def test_no_specs_returns_empty(self):
        assert _collect_torque_specs([make_chunk("7.2")]) == []

    def test_empty_chunks_returns_empty(self):
        assert _collect_torque_specs([]) == []
