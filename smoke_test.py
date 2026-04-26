"""
Smoke test for the ingestion pipeline.

Validates parse → chunk → index invariants against the FE 501s manual.
Does NOT re-embed; queries the existing ChromaDB collection.

Usage:
    python smoke_test.py
    python smoke_test.py --verbose
"""

import argparse
import sys
import traceback

import chromadb
import pypdf
from dotenv import load_dotenv
from openai import OpenAI

from ingestion.crossref import annotate_chunks
from ingestion.index import build_chunks
from ingestion.parse import parse_manual, parse_toc
from agent.retrieval import retrieve, retrieve_from_section, open_collection
from agent.planner import plan, Plan, _build_context, _collect_torque_specs
from fastapi.testclient import TestClient
from ui.main import app as fastapi_app
from pathlib import Path

load_dotenv()

PDF          = "manuals/FE_501s_2026_US_en_Bundle_RM_069969-000001_05sq_m1du.pdf"
MANUAL_STEM  = Path(PDF).stem
DB_PATH      = "data/chroma"
COLLECTION   = "manuals"
EMBED_MODEL  = "text-embedding-3-small"

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"

results: list[tuple[bool, str]] = []
verbose = False


def check(label: str, condition: bool, detail: str = "") -> bool:
    results.append((condition, label))
    status = PASS if condition else FAIL
    msg = f"  {status}  {label}"
    if detail and (verbose or not condition):
        msg += f"\n       {detail}"
    print(msg)
    return condition


# ---------------------------------------------------------------------------
# 1. Parse
# ---------------------------------------------------------------------------

def test_parse(pages, toc):
    print("\n── Parse ──────────────────────────────────────────────")

    check("Page count",
          len(pages) == 378,
          f"got {len(pages)}")

    check("TOC entry count",
          len(toc) == 350,
          f"got {len(toc)}")

    chapters = [p for p in pages if p.chapter_num is not None]
    check("Pages with chapter detected",
          len(chapters) >= 340,
          f"got {len(chapters)}")

    sections = [p for p in pages if p.section]
    check("Pages with section detected",
          len(sections) >= 220,
          f"got {len(sections)}")

    total_specs = sum(len(p.torque_specs) for p in pages)
    check("Torque specs extracted",
          total_specs >= 100,
          f"got {total_specs}")

    # Inline ref expansion — ofA → of (A)
    p50 = next((p for p in pages if p.page_num == 50), None)
    check("Inline refs expanded (of (A))",
          p50 is not None and "of (A)" in p50.text,
          "page 50 should contain 'of (A)'")

    check("Raw inline refs removed (ofA gone)",
          p50 is not None and "ofA" not in p50.text,
          "page 50 should not contain 'ofA'")

    check("Bolt specs untouched (M10)",
          p50 is not None and "M10" in p50.text,
          "page 50 should still contain 'M10'")

    toc_pages = [p for p in pages if "Table of contents" in p.text and p.page_num <= 10]
    check("TOC pages present and identifiable for filtering",
          len(toc_pages) >= 4,
          f"expected ≥4 TOC pages in first 10, got {len(toc_pages)}")


# ---------------------------------------------------------------------------
# 2. Chunk
# ---------------------------------------------------------------------------

def test_chunks(pages, toc):
    print("\n── Chunks ─────────────────────────────────────────────")

    chunks = build_chunks(pages, toc, MANUAL_STEM)
    annotate_chunks(chunks, toc)

    check("Chunk count",
          550 <= len(chunks) <= 650,
          f"got {len(chunks)}")

    # Section bleeding: 7.1 content must not appear in 7.2 chunk
    c71 = next((c for c in chunks if c.section == "7.1" and c.chapter_num == 7), None)
    c72 = next((c for c in chunks if c.section == "7.2" and c.chapter_num == 7), None)

    check("Section 7.1 chunk exists",
          c71 is not None)

    check("Section 7.2 chunk exists",
          c72 is not None)

    check("No section bleeding: 7.1 content absent from 7.2 chunk",
          c72 is not None and "7.1 Handlebar position" not in c72.text,
          "7.2 chunk should not contain 7.1 heading")

    # Cross-page torque spec: M8 handlebar clamp spec must appear in 7.2
    clamp = (c72 is not None and
             any("clamp" in t["description"].lower() and t["bolt"] == "M8"
                 for t in c72.torque_specs))
    check("Cross-page torque spec captured in 7.2 (M8 clamp)",
          clamp,
          "handlebar clamp screw spec from page 51 should be in section 7.2")

    # All chunk IDs must be unique
    ids = [c.id for c in chunks]
    check("All chunk IDs unique",
          len(ids) == len(set(ids)),
          f"{len(ids) - len(set(ids))} duplicate(s)")

    # Phase distribution sanity
    from collections import Counter
    phases = Counter(c.phase or "(none)" for c in chunks)
    check("Main-phase chunks present",
          phases["main"] >= 80,
          f"got {phases['main']}")
    check("Preparatory-phase chunks present",
          phases["preparatory"] >= 60,
          f"got {phases['preparatory']}")

    # Cross-reference annotations
    with_prereqs = [c for c in chunks if c.prerequisites]
    check("Prerequisites extracted",
          len(with_prereqs) >= 25,
          f"got {len(with_prereqs)} chunks with prerequisites")

    with_refs = [c for c in chunks if c.references]
    check("Page references extracted (preparatory only)",
          len(with_refs) >= 60,
          f"got {len(with_refs)} preparatory chunks with page references")

    check("Only preparatory chunks have references",
          all(c.phase == "preparatory" for c in with_refs),
          "non-preparatory chunk has references (tool-page false positive)")

    # 6.12 preparatory should reference both 6.10 and 6.11
    c612 = next((c for c in chunks if c.section == "6.12" and c.phase == "preparatory"), None)
    check("Section 6.12 preparatory references 6.10 and 6.11",
          c612 is not None and "6.10" in c612.references and "6.11" in c612.references,
          f"got {getattr(c612, 'references', None)}")

    return chunks


# ---------------------------------------------------------------------------
# 3. Index / retrieval
# ---------------------------------------------------------------------------

def test_retrieval():
    print("\n── Retrieval ──────────────────────────────────────────")

    try:
        chroma = chromadb.PersistentClient(path=DB_PATH)
        col    = chroma.get_collection(COLLECTION)
    except Exception as e:
        check("ChromaDB collection accessible", False, str(e))
        return

    count = col.count()
    check("ChromaDB collection not empty",
          count >= 500,
          f"got {count} — run `python ingestion/index.py <pdf>` to build")

    peek = col.get(limit=1, include=["metadatas"])["metadatas"]
    check("image_paths field present in metadata",
          bool(peek) and "image_paths" in peek[0],
          "re-index with --reset to populate image metadata")

    # At least some chunks should have images linked (pages with diagrams)
    sample = col.get(limit=200, include=["metadatas"])["metadatas"]
    import json as _json
    chunks_with_images = sum(
        1 for m in sample if _json.loads(m.get("image_paths", "[]"))
    )
    check("Some chunks have linked images",
          chunks_with_images > 0,
          f"0 of {len(sample)} sampled chunks have images — re-index with --reset")

    # Semantic retrieval spot-checks — expected section must appear in top 3
    client = OpenAI()

    def top_sections(query: str, n: int = 3) -> list[str]:
        emb = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
        res = col.query(query_embeddings=[emb], n_results=n)
        return [m["section"] for m in res["metadatas"][0]]

    cases = [
        ("remove the rear shock absorber",  "9.9"),
        ("valve clearance adjustment",       "24.2"),
        ("change engine oil and oil filter", "22.3"),
    ]
    for query, expected_section in cases:
        sections = top_sections(query)
        check(f'"{query[:40]}" → {expected_section} in top 3',
              expected_section in sections,
              f"top 3 sections: {sections}")


# ---------------------------------------------------------------------------
# 4. Dependency-aware retrieval (agent/retrieval.py)
# ---------------------------------------------------------------------------

def test_dependency_retrieval():
    print("\n── Dependency Retrieval ───────────────────────────────")

    try:
        col    = open_collection(DB_PATH)
        client = OpenAI()
    except Exception as e:
        check("open_collection succeeded", False, str(e))
        return

    # Query that should hit a preparatory chunk with references
    # Section 6.12 preparatory references 6.10 and 6.11
    chunks = retrieve(
        "disassembling the fork cartridge",
        col, client,
        manual=MANUAL_STEM,
        n_results=5,
    )

    check("retrieve() returns at least one chunk",
          len(chunks) >= 1,
          f"got {len(chunks)}")

    sections_returned = {c.section for c in chunks}
    check("Target section 6.12 in results",
          "6.12" in sections_returned,
          f"sections returned: {sorted(sections_returned)}")

    depths = [c.depth for c in chunks]
    check("Dependency walk fired (depth > 0 chunk present)",
          any(d > 0 for d in depths),
          f"all chunks at depth 0 — no prerequisites fetched")

    prereq_sections = {c.section for c in chunks if c.depth > 0}
    check("Prerequisites include section 6.10 or 6.11",
          bool(prereq_sections & {"6.10", "6.11"}),
          f"prerequisite sections: {sorted(prereq_sections)}")

    # Prerequisites must come first in the returned list
    ordered_depths = [c.depth for c in chunks]
    check("Result ordered with deepest prerequisites first",
          ordered_depths == sorted(ordered_depths, reverse=True),
          f"depth sequence: {ordered_depths}")

    # Image paths must be globally deduplicated across the result set
    all_paths = [p for c in chunks for p in c.image_paths]
    check("Image paths deduplicated across result set",
          len(all_paths) == len(set(all_paths)),
          f"{len(all_paths) - len(set(all_paths))} duplicate(s) found")

    # --- retrieve_from_section ---

    # Oil change query returns both Ch.18 (engine-out) and Ch.22 (maintenance)
    oil_chunks = retrieve(
        "oil and filter change", col, client, manual=MANUAL_STEM, n_results=5
    )
    seed_chapters = {c.chapter_num for c in oil_chunks if c.depth == 0}
    check("Disambiguation scenario: oil change seeds span multiple chapters",
          len(seed_chapters) > 1,
          f"chapters in seed set: {sorted(seed_chapters)}")

    # After user picks Ch.22, retrieve_from_section scopes the walk correctly
    scoped = retrieve_from_section("22.3", col, manual=MANUAL_STEM)
    scoped_chapters = {c.chapter_num for c in scoped}
    check("retrieve_from_section('22.3') returns chunks",
          len(scoped) >= 1,
          f"got {len(scoped)}")
    check("retrieve_from_section scoped to Ch.22 (no Ch.18 engine content)",
          18 not in scoped_chapters,
          f"chapters present: {sorted(scoped_chapters)}")
    check("retrieve_from_section result ordered correctly",
          [c.depth for c in scoped] == sorted([c.depth for c in scoped], reverse=True),
          f"depth sequence: {[c.depth for c in scoped]}")


# ---------------------------------------------------------------------------
# 5. Planner (agent/planner.py)
# ---------------------------------------------------------------------------

def test_planner():
    print("\n── Planner ────────────────────────────────────────────")

    try:
        col           = open_collection(DB_PATH)
        openai_client = OpenAI()
        claude        = __import__("anthropic").Anthropic()
    except Exception as e:
        check("Clients initialised", False, str(e))
        return

    chunks = retrieve_from_section("7.2", col, manual=MANUAL_STEM)
    check("Retrieved chunks for planner input",
          len(chunks) >= 1,
          f"got {len(chunks)}")

    result = plan("adjust the handlebar position", chunks, claude)

    check("plan() returns a Plan instance",
          isinstance(result, Plan))

    check("Plan text is non-empty",
          len(result.text) > 100,
          f"got {len(result.text)} chars")

    check("Plan covers section 7.2",
          "7.2" in result.sections_used,
          f"sections_used: {result.sections_used}")

    check("Plan text mentions torque or Nm",
          "Nm" in result.text or "torque" in result.text.lower(),
          "expected torque reference in plan text")

    check("Torque specs passed through to Plan",
          len(result.torque_specs) >= 1,
          f"got {len(result.torque_specs)} specs")

    check("plan() with no chunks returns gracefully",
          plan("anything", [], claude).text != "" and
          plan("anything", [], claude).sections_used == [])


# ---------------------------------------------------------------------------
# 6. API (ui/main.py)
# ---------------------------------------------------------------------------

def test_api():
    print("\n── API ────────────────────────────────────────────────")

    client = TestClient(fastapi_app)

    # Unambiguous query → PlanResponse directly (fork cartridge is cleanly Ch.6 only)
    resp = client.post("/ask", json={"query": "disassemble the fork cartridge"})
    check("POST /ask 200 for unambiguous query",
          resp.status_code == 200,
          f"status {resp.status_code}")
    body = resp.json()
    check("Unambiguous /ask returns plan type",
          body.get("type") == "plan",
          f"type={body.get('type')}")
    check("Plan response has non-empty text",
          len(body.get("text", "")) > 50,
          f"text length {len(body.get('text', ''))}")

    # Ambiguous query → DisambiguationResponse
    resp = client.post("/ask", json={"query": "oil and filter change"})
    check("POST /ask 200 for ambiguous query",
          resp.status_code == 200,
          f"status {resp.status_code}")
    body = resp.json()
    check("Ambiguous /ask returns disambiguation type",
          body.get("type") == "disambiguation",
          f"type={body.get('type')}")
    check("Disambiguation response has multiple chapter candidates",
          len(body.get("candidates", [])) > 1,
          f"candidates: {[c.get('chapter_num') for c in body.get('candidates', [])]}")

    # POST /plan after disambiguation
    resp = client.post("/plan", json={"query": "change the oil", "section": "22.3"})
    check("POST /plan 200 for section 22.3",
          resp.status_code == 200,
          f"status {resp.status_code}")
    check("POST /plan returns plan type",
          resp.json().get("type") == "plan",
          f"type={resp.json().get('type')}")

    # POST /plan with unknown section → 404
    resp = client.post("/plan", json={"query": "test", "section": "99.99"})
    check("POST /plan 404 for unknown section",
          resp.status_code == 404,
          f"status {resp.status_code}")

    # GET /images serves a known image file
    import json as _json
    col    = open_collection(DB_PATH)
    sample = col.get(limit=50, include=["metadatas"])["metadatas"]
    first_image = next(
        (p for m in sample for p in _json.loads(m.get("image_paths", "[]"))), None
    )
    if first_image:
        relative = str(Path(first_image).relative_to("data/images"))
        resp = client.get(f"/images/{relative}")
        check("GET /images serves a known image file",
              resp.status_code == 200,
              f"path={relative}, status={resp.status_code}")
    else:
        check("GET /images serves a known image file", False, "no images found in sample")

    # GET /images with missing path → 404
    resp = client.get("/images/nonexistent/file.jpg")
    check("GET /images 404 for missing file",
          resp.status_code == 404,
          f"status {resp.status_code}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global verbose
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    verbose = args.verbose

    print("Loading manual...")
    try:
        pages = parse_manual(PDF, extract_images=False)
        toc   = parse_toc(pypdf.PdfReader(PDF))
    except Exception:
        print(f"\n{FAIL}  Failed to load manual: {PDF}")
        traceback.print_exc()
        sys.exit(1)

    test_parse(pages, toc)
    test_chunks(pages, toc)
    test_retrieval()
    test_dependency_retrieval()
    test_planner()
    test_api()

    passed = sum(1 for ok, _ in results if ok)
    failed = sum(1 for ok, _ in results if not ok)
    total  = len(results)

    print(f"\n── Results ────────────────────────────────────────────")
    print(f"  {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} failed)")
        for ok, label in results:
            if not ok:
                print(f"    {FAIL}  {label}")
        sys.exit(1)
    else:
        print("  — all clear")


if __name__ == "__main__":
    main()
