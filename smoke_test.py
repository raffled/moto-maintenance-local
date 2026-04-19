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

from ingestion.index import build_chunks
from ingestion.parse import parse_manual, parse_toc

load_dotenv()

PDF        = "manuals/FE_501s_2026_US_en_Bundle_RM_069969-000001_05sq_m1du.pdf"
DB_PATH    = "data/chroma"
COLLECTION = "manuals"
EMBED_MODEL = "text-embedding-3-small"

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
          len(toc) == 254,
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

    chunks = build_chunks(pages, toc, "FE_501s_2026")

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
