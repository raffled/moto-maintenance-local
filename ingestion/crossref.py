"""
Detect cross-references and prerequisites within chunk text.

Two types are extracted and stored as chunk metadata:

  prerequisites : Raw text from "Condition:" lines — state that must be true
                  before executing the procedure.  The agent can use these as
                  semantic search queries to find the prerequisite sections.

  references    : Raw prose from "(see ...)" phrases.  Stored as-is rather than
                  resolved to section numbers — TOC titles lack enough unique
                  signal for reliable character-level fuzzy matching.  The
                  retrieval layer resolves them via semantic search at query time.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.index import Chunk


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# "(see Removing the cylinder head and cylinders section)"
_SEE_RE = re.compile(r'\(\s*see\s+([^)]{5,100})\)', re.IGNORECASE)

# "Condition: The engine is cold, ..."  (stops at newline)
_COND_RE = re.compile(r'Condition:\s*(.+?)(?:\n|$)', re.IGNORECASE)

# Strip trailing "section" / "chapter" for cleaner storage
_STRIP_NOISE = re.compile(r'\s+(section|chapter)\s*$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_chunks(chunks: list[Chunk], toc: list[dict]) -> None:
    """
    Populate prerequisites and references on each chunk in-place.
    Called after build_chunks(), before embedding.

    toc is accepted for API consistency and future use (e.g. exact-number
    matching if the manual edition uses numeric cross-references).
    """
    for chunk in chunks:
        chunk.prerequisites = [
            m.group(1).strip() for m in _COND_RE.finditer(chunk.text)
        ]
        chunk.references = [
            _STRIP_NOISE.sub("", m.group(1).replace("\n", " ")).strip()
            for m in _SEE_RE.finditer(chunk.text)
        ]
