"""
Detect cross-references and prerequisites within chunk text.

Two types are extracted and stored as chunk metadata:

  prerequisites : Raw text from "Condition:" lines — state that must be true
                  before executing the procedure.  The agent can use these as
                  semantic search queries to find the prerequisite sections.

  references    : Section numbers resolved from "(p. NN)" page citations.
                  These appear consistently in Preparatory work sections as the
                  machine-readable pointer to prerequisite procedure pages, e.g.:
                    ‒ Remove the fuel tank.
                     (p. 108)
                  Resolution uses a range lookup against the TOC so that any page
                  within a section maps to that section's number.

                  "(see <title>)" prose references also appear in the manual (3
                  occurrences) but are left in the raw chunk text for the agent
                  to read — character-level fuzzy matching against TOC titles
                  produces unreliable results for this manual.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.index import Chunk


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

_PAGE_REF_RE = re.compile(r'\(p\.?\s*(\d+)\)', re.IGNORECASE)
_COND_RE     = re.compile(r'Condition:\s*(.+?)(?:\n|$)', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Page → section range lookup
# ---------------------------------------------------------------------------

def _build_page_map(toc: list[dict]) -> list[tuple[int, str]]:
    """
    Build a sorted list of (start_page, section_number) from the TOC.
    Used for range lookups: the section for page N is the last entry
    whose start_page <= N.
    """
    return sorted(
        ((e["page"], e["number"]) for e in toc),
        key=lambda x: x[0],
    )


def _section_for_page(page_num: int, page_map: list[tuple[int, str]]) -> str:
    """Return the section number whose page range contains page_num."""
    result = ""
    for start_page, section in page_map:
        if start_page <= page_num:
            result = section
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def annotate_chunks(chunks: list[Chunk], toc: list[dict]) -> None:
    """
    Populate prerequisites and references on each chunk in-place.
    Called after build_chunks(), before embedding.
    """
    page_map = _build_page_map(toc)

    for chunk in chunks:
        chunk.prerequisites = [
            m.group(1).strip() for m in _COND_RE.finditer(chunk.text)
        ]

        # (p. NN) citations have two meanings depending on phase:
        #   preparatory → prerequisite procedure page  (resolve to section)
        #   main / full → special tools appendix page  (not a cross-reference)
        # Only extract from preparatory chunks to avoid false positives.
        if chunk.phase != "preparatory":
            chunk.references = []
            continue

        seen: set[str] = set()
        refs: list[str] = []
        for m in _PAGE_REF_RE.finditer(chunk.text):
            sec = _section_for_page(int(m.group(1)), page_map)
            if sec and sec != chunk.section and sec not in seen:
                seen.add(sec)
                refs.append(sec)
        chunk.references = refs
