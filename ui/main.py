"""
FastAPI backend for the motorcycle repair assistant.

Endpoints
---------
POST /ask
    Takes {query, manual?}.  Runs semantic search and inspects the seed chapters.
    - If seeds span multiple chapters, returns a disambiguation response so the UI
      can present options to the user.
    - If seeds are all within one chapter, runs the full retrieve → plan pipeline
      immediately and returns the plan.

POST /plan
    Takes {query, section, manual?}.  Used after the user has chosen a specific
      section from a disambiguation prompt.  Runs retrieve_from_section → plan
      and returns the plan.

GET /images/{path}
    Serves image files from data/images/ so the frontend can render diagrams inline.

Run with:
    uvicorn ui.main:app --reload
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

from agent.planner import plan as _plan
from agent.retrieval import RetrievedChunk, open_collection, retrieve, retrieve_from_section

load_dotenv()

app = FastAPI(title="Moto Maintenance Assistant")

# Default manual stem — matches the indexed FE 501s PDF
_DEFAULT_MANUAL = "FE_501s_2026_US_en_Bundle_RM_069969-000001_05sq_m1du"

# Clients are initialised once on first use
_col    = None
_openai = None
_claude = None


def _get_col():
    global _col
    if _col is None:
        _col = open_collection()
    return _col


def _get_openai():
    global _openai
    if _openai is None:
        _openai = OpenAI()
    return _openai


def _get_claude():
    global _claude
    if _claude is None:
        _claude = anthropic.Anthropic()
    return _claude


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    query: str
    manual: str = _DEFAULT_MANUAL


class PlanRequest(BaseModel):
    query: str
    section: str
    manual: str = _DEFAULT_MANUAL


class SectionCandidate(BaseModel):
    section: str
    section_title: str


class ChapterGroup(BaseModel):
    chapter_num: int
    chapter_title: str
    sections: list[SectionCandidate]


class DisambiguationResponse(BaseModel):
    type: str = "disambiguation"
    query: str
    candidates: list[ChapterGroup]


class TorqueSpec(BaseModel):
    bolt: str
    nm: float
    ftlbf: float
    note: str


class PlanResponse(BaseModel):
    type: str = "plan"
    query: str
    sections_used: list[str]
    text: str
    torque_specs: list[TorqueSpec]
    image_paths: list[str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_by_chapter(seeds: list[RetrievedChunk]) -> dict[tuple, list[RetrievedChunk]]:
    """Group depth-0 seed chunks by (chapter_num, chapter_title), deduplicating sections."""
    by_chapter: dict[tuple, list[RetrievedChunk]] = defaultdict(list)
    seen_sections: set[str] = set()
    for c in seeds:
        if c.section not in seen_sections:
            seen_sections.add(c.section)
            by_chapter[(c.chapter_num, c.chapter_title)].append(c)
    return by_chapter


def _to_plan_response(query: str, chunks: list[RetrievedChunk]) -> PlanResponse:
    result = _plan(query, chunks, _get_claude())
    return PlanResponse(
        query=query,
        sections_used=result.sections_used,
        text=result.text,
        torque_specs=[TorqueSpec(**s) for s in result.torque_specs],
        image_paths=result.image_paths,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=DisambiguationResponse | PlanResponse)
def ask(req: AskRequest):
    """
    Entry point for a new repair query.

    Returns a DisambiguationResponse when the query matches sections from more
    than one chapter so the user can pick the right context.  Returns a
    PlanResponse directly when the results are unambiguous.
    """
    chunks = retrieve(req.query, _get_col(), _get_openai(), manual=req.manual, n_results=5)

    seeds      = [c for c in chunks if c.depth == 0]
    by_chapter = _group_by_chapter(seeds)

    if len(by_chapter) > 1:
        candidates = [
            ChapterGroup(
                chapter_num=ch_num,
                chapter_title=ch_title,
                sections=[
                    SectionCandidate(section=c.section, section_title=c.section_title)
                    for c in chapter_chunks
                ],
            )
            for (ch_num, ch_title), chapter_chunks in sorted(by_chapter.items())
        ]
        return DisambiguationResponse(query=req.query, candidates=candidates)

    return _to_plan_response(req.query, chunks)


@app.post("/plan", response_model=PlanResponse)
def plan_endpoint(req: PlanRequest):
    """
    Generate a plan from a specific section, used after disambiguation.
    """
    chunks = retrieve_from_section(req.section, _get_col(), manual=req.manual)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"Section {req.section!r} not found")
    return _to_plan_response(req.query, chunks)


@app.get("/images/{path:path}")
def serve_image(path: str):
    """Serve a diagram image by its relative path under data/images/."""
    img_path = Path("data/images") / path
    if not img_path.exists() or not img_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(img_path))
