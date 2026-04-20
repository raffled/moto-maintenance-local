"""
Dependency-aware chunk retrieval from ChromaDB.

Query flow:
  1. Semantic search  — embed the query, find top-k matching chunks
  2. Dependency walk  — for preparatory chunks, follow `references` recursively
  3. Order            — deepest prerequisites first so the planner receives
                        chunks in the correct procedural sequence
  4. Image dedup      — image paths are deduplicated across the returned set
                        (pages that span section boundaries share images)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import chromadb
from openai import OpenAI

EMBED_MODEL = "text-embedding-3-small"
COLLECTION  = "manuals"


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    id: str
    manual: str
    chapter_num: int
    chapter_title: str
    section: str
    section_title: str
    phase: str
    text: str
    torque_specs: list[dict]
    figure_refs: list[str]
    page_nums: list[int]
    image_paths: list[str]
    prerequisites: list[str]
    references: list[str]
    depth: int = 0   # 0 = directly matched; N = N hops from a direct match


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_result(doc_id: str, metadata: dict, document: str, depth: int = 0) -> RetrievedChunk:
    """Unpack a single ChromaDB row into a RetrievedChunk."""
    return RetrievedChunk(
        id=doc_id,
        manual=metadata["manual"],
        chapter_num=metadata["chapter_num"],
        chapter_title=metadata["chapter_title"],
        section=metadata["section"],
        section_title=metadata["section_title"],
        phase=metadata["phase"],
        text=document,
        torque_specs=json.loads(metadata["torque_specs"]),
        figure_refs=json.loads(metadata["figure_refs"]),
        page_nums=json.loads(metadata["page_nums"]),
        image_paths=json.loads(metadata["image_paths"]),
        prerequisites=json.loads(metadata["prerequisites"]),
        references=json.loads(metadata["references"]),
        depth=depth,
    )


def _fetch_by_section(
    section: str,
    collection: chromadb.Collection,
    manual: str | None,
) -> list[RetrievedChunk]:
    """Fetch all chunks for an exact section number from the vector store."""
    if manual:
        where: dict = {"$and": [
            {"section": {"$eq": section}},
            {"manual":  {"$eq": manual}},
        ]}
    else:
        where = {"section": {"$eq": section}}

    results = collection.get(where=where, include=["documents", "metadatas"])
    return [
        _parse_result(doc_id, results["metadatas"][i], results["documents"][i])
        for i, doc_id in enumerate(results["ids"])
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    collection: chromadb.Collection,
    openai_client: OpenAI,
    manual: str | None = None,
    n_results: int = 5,
) -> list[RetrievedChunk]:
    """
    Return chunks relevant to *query*, including all transitive prerequisites.

    Parameters
    ----------
    query       : Natural-language question or task description.
    collection  : Open ChromaDB collection (call open_collection() to get one).
    openai_client: Authenticated OpenAI client for query embedding.
    manual      : Optional manual stem to restrict results to one source PDF.
    n_results   : Number of seed chunks from the semantic search step.

    Returns
    -------
    List of RetrievedChunk ordered with deepest prerequisites first, so the
    planner can sequence steps without further sorting.
    """
    # 1. Embed the query
    embedding = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[query],
    ).data[0].embedding

    # 2. Semantic search
    query_kwargs: dict = dict(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    if manual:
        query_kwargs["where"] = {"manual": {"$eq": manual}}

    raw = collection.query(**query_kwargs)

    seed_chunks = [
        _parse_result(
            raw["ids"][0][i],
            raw["metadatas"][0][i],
            raw["documents"][0][i],
            depth=0,
        )
        for i in range(len(raw["ids"][0]))
    ]

    # 3. BFS dependency walk — follow `references` on preparatory chunks
    seen_ids: set[str] = set()
    all_chunks: list[RetrievedChunk] = []
    queue: list[tuple[RetrievedChunk, int]] = [(c, 0) for c in seed_chunks]

    while queue:
        chunk, depth = queue.pop(0)
        if chunk.id in seen_ids:
            continue
        seen_ids.add(chunk.id)
        chunk.depth = depth
        all_chunks.append(chunk)

        if chunk.phase == "preparatory":
            dep_manual = manual or chunk.manual
            for ref_section in chunk.references:
                for dep in _fetch_by_section(ref_section, collection, dep_manual):
                    if dep.id not in seen_ids:
                        queue.append((dep, depth + 1))

    # 4. Sort: deepest prerequisites first; stable secondary sort by section
    all_chunks.sort(key=lambda c: (-c.depth, c.section))

    # 5. Deduplicate image paths across the set (preserving content-stream order)
    seen_images: set[str] = set()
    for chunk in all_chunks:
        deduped = []
        for path in chunk.image_paths:
            if path not in seen_images:
                seen_images.add(path)
                deduped.append(path)
        chunk.image_paths = deduped

    return all_chunks


def open_collection(
    db_path: str = "data/chroma",
    collection_name: str = COLLECTION,
) -> chromadb.Collection:
    """Open the local ChromaDB collection. Convenience wrapper for callers."""
    client = chromadb.PersistentClient(path=db_path)
    return client.get_collection(collection_name)
