"""
Build and populate the vector index from parsed manual pages.

Chunk hierarchy:
    Chapter > Section > Phase (Preparatory work | Main work | Reworking)

Each chunk is embedded with OpenAI text-embedding-3-small and stored in a
local ChromaDB collection. On Vertex AI migration, swap the embedding call
to Vertex AI Text Embeddings API — the ChromaDB interface stays the same.
"""

import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import chromadb
from openai import OpenAI

from ingestion.images import save_images
from ingestion.parse import ParsedPage, TorqueSpec, _extract_torque_specs, _FIGURE_REF, parse_manual, parse_toc
import pypdf


PHASE_HEADERS = ["Preparatory work", "Main work", "Reworking"]
PHASE_KEYS    = {"Preparatory work": "preparatory", "Main work": "main", "Reworking": "reworking"}
EMBED_MODEL   = "text-embedding-3-small"
COLLECTION    = "manuals"


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: str
    manual: str               # source filename stem
    chapter_num: int
    chapter_title: str
    section: str              # e.g. "7.2"
    section_title: str        # from TOC, e.g. "Adjusting the handlebar position"
    phase: str                # "preparatory" | "main" | "reworking" | ""
    text: str
    torque_specs: list[dict]  # serialised TorqueSpec dicts
    figure_refs: list[str]
    page_nums: list[int]
    image_paths: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _phase_split(text: str) -> list[tuple[str, str]]:
    """
    Split section text on phase headers.
    Returns list of (phase_key, text) tuples.
    If no phase headers are present, returns [("", text)].
    """
    pattern = r"(?=" + "|".join(re.escape(h) for h in PHASE_HEADERS) + r")"
    parts = re.split(pattern, text)
    result = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        matched_phase = ""
        for header in PHASE_HEADERS:
            if part.startswith(header):
                matched_phase = PHASE_KEYS[header]
                break
        result.append((matched_phase, part))
    return result or [("", text)]


def _toc_index(toc: list[dict]) -> dict[str, str]:
    """Build a section-number → title lookup from TOC entries."""
    return {e["number"]: e["title"] for e in toc}


# Matches section headings at the start of a line, e.g. "7.2 Adjusting..."
_SEC_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\s+[A-Z]", re.MULTILINE)


def _split_page_by_section(
    page: ParsedPage,
) -> list[tuple[str, str]]:
    """
    Split a single page's text at section heading boundaries.
    Returns list of (section_number, text_fragment) tuples.
    Text before the first heading is returned under section "" (belongs to
    the previous section and must be handled by the caller).
    """
    matches = list(_SEC_HEADING.finditer(page.text))
    if not matches:
        return [("", page.text)]

    fragments = []
    if matches[0].start() > 0:
        fragments.append(("", page.text[: matches[0].start()].strip()))

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page.text)
        fragments.append((m.group(1), page.text[m.start() : end].strip()))

    return fragments


def build_chunks(
    pages: list[ParsedPage],
    toc: list[dict],
    manual_stem: str,
) -> list[Chunk]:
    """
    Group ParsedPage records into Chunk objects using the
    chapter > section > phase hierarchy.

    Pages are split at section heading boundaries so that content appearing
    before the first heading on a page is attributed to the section that
    started on a previous page, not to whatever section ends the page.
    Torque specs and figure refs are re-extracted from the combined section
    text so that values that straddle a page join are not missed.
    """
    from collections import defaultdict

    toc_titles = _toc_index(toc)

    # buckets[(chapter_num, section)] = list of (text_fragment, page_num)
    buckets: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    # track the chapter for each section key
    chapter_of: dict[tuple, int] = {}
    chapter_title_of: dict[tuple, str] = {}

    current_key: tuple | None = None  # (chapter_num, section) of last heading seen

    for p in pages:
        if p.chapter_num is None or "Table of contents" in p.text:
            continue

        for sec_num, fragment in _split_page_by_section(p):
            if not fragment:
                continue
            if sec_num == "":
                # Pre-heading text belongs to the section that was active
                # before this page started
                if current_key:
                    buckets[current_key].append((fragment, p.page_num))
            else:
                key = (p.chapter_num, sec_num)
                current_key = key
                chapter_of[key] = p.chapter_num
                chapter_title_of[key] = p.chapter_title
                buckets[key].append((fragment, p.page_num))

    chunks: list[Chunk] = []
    for key in sorted(buckets):
        chapter_num = chapter_of[key]
        section     = key[1]
        frags       = buckets[key]

        combined_text   = "\n".join(text for text, _ in frags)
        page_nums       = list(dict.fromkeys(pn for _, pn in frags))
        chapter_title   = chapter_title_of[key]
        section_title   = toc_titles.get(section, "")

        # Re-extract from combined text to capture cross-page specs/refs
        all_torque      = [asdict(s) for s in _extract_torque_specs(combined_text)]
        all_figure_refs = list(dict.fromkeys(_FIGURE_REF.findall(combined_text)))

        phase_seen: dict[str, int] = {}
        for phase_key, phase_text in _phase_split(combined_text):
            k      = phase_key or "full"
            count  = phase_seen.get(k, 0)
            phase_seen[k] = count + 1
            suffix = f"_{count}" if count > 0 else ""
            chunk_id = f"{manual_stem}__{chapter_num}__{section}__{k}{suffix}"
            chunks.append(Chunk(
                id=chunk_id,
                manual=manual_stem,
                chapter_num=chapter_num,
                chapter_title=chapter_title,
                section=section,
                section_title=section_title,
                phase=phase_key,
                text=phase_text,
                torque_specs=all_torque,
                figure_refs=all_figure_refs,
                page_nums=page_nums,
            ))

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_chunks(chunks: list[Chunk], client: OpenAI) -> list[list[float]]:
    """Embed chunk texts in batches. Returns one embedding vector per chunk."""
    texts = [c.text for c in chunks]
    # ChromaDB / OpenAI batch limit is 2048 items; manual has ~300 chunks
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# ChromaDB upsert
# ---------------------------------------------------------------------------

def _chroma_metadata(chunk: Chunk) -> dict:
    """Flatten Chunk fields into ChromaDB-compatible metadata (str/int/float only)."""
    return {
        "manual":        chunk.manual,
        "chapter_num":   chunk.chapter_num,
        "chapter_title": chunk.chapter_title,
        "section":       chunk.section,
        "section_title": chunk.section_title,
        "phase":         chunk.phase,
        "figure_refs":   json.dumps(chunk.figure_refs),
        "torque_specs":  json.dumps(chunk.torque_specs),
        "page_nums":     json.dumps(chunk.page_nums),
        "image_paths":   json.dumps(chunk.image_paths),
    }


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
) -> None:
    collection.upsert(
        ids        =[c.id for c in chunks],
        documents  =[c.text for c in chunks],
        embeddings =embeddings,
        metadatas  =[_chroma_metadata(c) for c in chunks],
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def index_manual(
    pdf_path: str | Path,
    db_path: str | Path = "data/chroma",
    reset: bool = False,
) -> int:
    """
    Parse, chunk, embed, and index a single manual PDF.
    Returns the number of chunks written.

    reset=True drops and recreates the collection before indexing — use this
    after any chunking or parsing changes to avoid stale chunks accumulating.
    """
    pdf_path    = Path(pdf_path)
    manual_stem = pdf_path.stem
    images_dir  = Path(db_path).parent / "images"

    print(f"Parsing {pdf_path.name} ...")
    pages = parse_manual(pdf_path, extract_images=True)
    toc   = parse_toc(pypdf.PdfReader(str(pdf_path)))

    print(f"  {len(pages)} pages parsed, {len(toc)} TOC entries")

    chunks = build_chunks(pages, toc, manual_stem)
    print(f"  {len(chunks)} chunks built")

    print(f"Saving images to {images_dir} ...")
    page_images = save_images(pages, manual_stem, images_dir)
    total_imgs  = sum(len(v) for v in page_images.values())
    print(f"  {total_imgs} images saved across {len(page_images)} pages")

    for chunk in chunks:
        for pn in chunk.page_nums:
            chunk.image_paths.extend(page_images.get(pn, []))

    print(f"Embedding chunks ...")
    openai_client = OpenAI()
    embeddings = embed_chunks(chunks, openai_client)

    print(f"Writing to ChromaDB at {db_path} ...")
    chroma = chromadb.PersistentClient(path=str(db_path))

    if reset:
        try:
            chroma.delete_collection(COLLECTION)
            print(f"  Collection '{COLLECTION}' cleared.")
        except Exception:
            pass  # collection didn't exist yet

    collection = chroma.get_or_create_collection(COLLECTION)
    upsert_chunks(chunks, embeddings, collection)

    print(f"Done. {len(chunks)} chunks indexed.")
    return len(chunks)


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Index a repair manual PDF.")
    parser.add_argument("manual", help="Path to the PDF file")
    parser.add_argument("--db", default="data/chroma", help="ChromaDB directory")
    parser.add_argument("--reset", action="store_true",
                        help="Drop and recreate the collection before indexing")
    args = parser.parse_args()

    index_manual(args.manual, args.db, reset=args.reset)
