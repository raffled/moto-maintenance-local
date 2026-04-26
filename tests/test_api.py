"""
Unit tests for ui/main.py API endpoints.
Uses FastAPI's TestClient for in-process HTTP testing — no live server needed.
Tests cover request/response shapes and routing logic; actual Claude/OpenAI
calls are exercised by the smoke test.
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from ui.main import app, _group_by_chapter
from agent.retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(section: str, chapter_num: int, chapter_title: str, depth: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"manual__1__{section}__full",
        manual="test_manual",
        chapter_num=chapter_num,
        chapter_title=chapter_title,
        section=section,
        section_title="",
        phase="",
        text="Some text.",
        torque_specs=[],
        figure_refs=[],
        page_nums=[1],
        image_paths=[],
        prerequisites=[],
        references=[],
        depth=depth,
    )


# ---------------------------------------------------------------------------
# _group_by_chapter (pure helper)
# ---------------------------------------------------------------------------

class TestGroupByChapter:
    def test_single_chapter_returns_one_group(self):
        seeds = [make_chunk("7.2", 7, "Handlebar"), make_chunk("7.3", 7, "Handlebar")]
        groups = _group_by_chapter(seeds)
        assert len(groups) == 1

    def test_two_chapters_returns_two_groups(self):
        seeds = [make_chunk("18.3", 18, "Engine"), make_chunk("22.3", 22, "Lubrication")]
        groups = _group_by_chapter(seeds)
        assert len(groups) == 2

    def test_duplicate_section_counted_once(self):
        seeds = [
            make_chunk("7.2", 7, "Handlebar"),
            make_chunk("7.2", 7, "Handlebar"),  # duplicate
        ]
        groups = _group_by_chapter(seeds)
        assert sum(len(v) for v in groups.values()) == 1

    def test_empty_seeds_returns_empty(self):
        assert _group_by_chapter([]) == {}


# ---------------------------------------------------------------------------
# API endpoint routing (mocked Claude/OpenAI/ChromaDB)
# ---------------------------------------------------------------------------

MOCK_PLAN_RESPONSE = {
    "type": "plan",
    "query": "test query",
    "sections_used": ["7.2"],
    "text": "Step 1. Do something.",
    "torque_specs": [],
    "image_paths": [],
}

MOCK_CHUNKS_SINGLE = [make_chunk("7.2", 7, "Handlebar", depth=0)]
MOCK_CHUNKS_MULTI  = [
    make_chunk("18.3", 18, "Engine", depth=0),
    make_chunk("22.3", 22, "Lubrication", depth=0),
]


class TestAskEndpoint:
    def test_unambiguous_query_returns_plan_type(self):
        with (
            patch("ui.main._get_col",    return_value=MagicMock()),
            patch("ui.main._get_openai", return_value=MagicMock()),
            patch("ui.main._get_claude", return_value=MagicMock()),
            patch("ui.main.retrieve",    return_value=MOCK_CHUNKS_SINGLE),
            patch("ui.main._plan") as mock_plan,
        ):
            from agent.planner import Plan
            mock_plan.return_value = Plan(
                query="test", sections_used=["7.2"],
                text="Step 1.", torque_specs=[], image_paths=[],
            )
            client = TestClient(app)
            resp = client.post("/ask", json={"query": "adjust handlebar"})
        assert resp.status_code == 200
        assert resp.json()["type"] == "plan"

    def test_ambiguous_query_returns_disambiguation_type(self):
        with (
            patch("ui.main._get_col",    return_value=MagicMock()),
            patch("ui.main._get_openai", return_value=MagicMock()),
            patch("ui.main.retrieve",    return_value=MOCK_CHUNKS_MULTI),
        ):
            client = TestClient(app)
            resp = client.post("/ask", json={"query": "oil change"})
        assert resp.status_code == 200
        assert resp.json()["type"] == "disambiguation"

    def test_disambiguation_response_contains_candidates(self):
        with (
            patch("ui.main._get_col",    return_value=MagicMock()),
            patch("ui.main._get_openai", return_value=MagicMock()),
            patch("ui.main.retrieve",    return_value=MOCK_CHUNKS_MULTI),
        ):
            client = TestClient(app)
            body = client.post("/ask", json={"query": "oil change"}).json()
        candidates = body["candidates"]
        assert len(candidates) == 2
        chapter_nums = {c["chapter_num"] for c in candidates}
        assert chapter_nums == {18, 22}

    def test_disambiguation_response_echoes_query(self):
        with (
            patch("ui.main._get_col",    return_value=MagicMock()),
            patch("ui.main._get_openai", return_value=MagicMock()),
            patch("ui.main.retrieve",    return_value=MOCK_CHUNKS_MULTI),
        ):
            client = TestClient(app)
            body = client.post("/ask", json={"query": "oil and filter change"}).json()
        assert body["query"] == "oil and filter change"


class TestPlanEndpoint:
    def test_returns_plan_type(self):
        with (
            patch("ui.main._get_col", return_value=MagicMock()),
            patch("ui.main._get_claude", return_value=MagicMock()),
            patch("ui.main.retrieve_from_section", return_value=MOCK_CHUNKS_SINGLE),
            patch("ui.main._plan") as mock_plan,
        ):
            from agent.planner import Plan
            mock_plan.return_value = Plan(
                query="test", sections_used=["7.2"],
                text="Step 1.", torque_specs=[], image_paths=[],
            )
            client = TestClient(app)
            resp = client.post("/plan", json={"query": "adjust handlebar", "section": "7.2"})
        assert resp.status_code == 200
        assert resp.json()["type"] == "plan"

    def test_unknown_section_returns_404(self):
        with (
            patch("ui.main._get_col", return_value=MagicMock()),
            patch("ui.main._get_claude", return_value=MagicMock()),
            patch("ui.main.retrieve_from_section", return_value=[]),
        ):
            client = TestClient(app)
            resp = client.post("/plan", json={"query": "test", "section": "99.99"})
        assert resp.status_code == 404


class TestImagesEndpoint:
    def test_missing_file_returns_404(self):
        client = TestClient(app)
        resp = client.get("/images/nonexistent/path.jpg")
        assert resp.status_code == 404
