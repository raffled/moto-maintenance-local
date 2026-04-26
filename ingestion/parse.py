"""
Parse a Husqvarna repair manual PDF into structured page records.

Output per page:
  page_num      : 1-based page number
  chapter_num   : int chapter number (None for front matter)
  chapter_title : str chapter title
  section       : str nearest section heading (e.g. "7.2")
  text          : full extracted text with header/license noise removed
  torque_specs  : list of {description, bolt, nm, ftlbf, note}
  figure_refs   : list of figure code strings (e.g. "W00322-10")
  images        : list of {key, data: bytes, width, height}
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypdf
from PIL import Image


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TorqueSpec:
    description: str
    bolt: str
    nm: float
    ftlbf: float
    note: str = ""          # e.g. "Loctite® 243"


@dataclass
class PageImage:
    key: str                # PDF XObject key, e.g. "/IMMtxAzPXR"
    data: bytes             # JPEG bytes for JPEG images; PNG bytes for all others
    width: int
    height: int
    format: str = "JPEG"    # "JPEG" or "PNG"


@dataclass
class ParsedPage:
    page_num: int           # 1-based
    chapter_num: Optional[int]
    chapter_title: str
    section: str            # nearest section heading, e.g. "7.2"
    text: str
    torque_specs: list[TorqueSpec] = field(default_factory=list)
    figure_refs: list[str] = field(default_factory=list)
    images: list[PageImage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TOC parsing
# ---------------------------------------------------------------------------

# Matches lines like:  "7.2 Adjusting the handlebar position .......... 50"
_TOC_ENTRY = re.compile(
    r"^(\d+(?:\.\d+)*)\s+(.+?)\s*\.{2,}\s*(\d+)\s*$"
)

# Matches the start of a wrapped entry, e.g. "18.3.3 Draining the engine oil and"
# (section number + title start, no dot leaders or page number yet)
_TOC_ENTRY_START = re.compile(r"^(\d+(?:\.\d+)+)\s+\w")


def parse_toc(pdf: pypdf.PdfReader) -> list[dict]:
    """Return list of {number, title, page} dicts from the Table of Contents.

    Some entries have titles that wrap to the next line, with the dot leaders
    and page number on the continuation line.  These are buffered and merged.
    """
    entries = []
    in_toc  = False
    pending = ""   # partial line waiting for its continuation

    for page in pdf.pages:
        text = page.extract_text() or ""
        if "Table of contents" in text:
            in_toc = True
        if not in_toc:
            continue
        if in_toc and "Table of contents" not in text and entries:
            toc_lines = [l for l in text.splitlines() if _TOC_ENTRY.match(l.strip())]
            if not toc_lines:
                break

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                pending = ""
                continue

            # Try matching the line alone first
            m = _TOC_ENTRY.match(stripped)
            if m:
                pending = ""
                entries.append({
                    "number": m.group(1),
                    "title":  m.group(2).strip(),
                    "page":   int(m.group(3)),
                })
                continue

            # If we have a buffered first line, try merging with this continuation
            if pending:
                merged = pending + " " + stripped
                m = _TOC_ENTRY.match(merged)
                if m:
                    entries.append({
                        "number": m.group(1),
                        "title":  m.group(2).strip(),
                        "page":   int(m.group(3)),
                    })
                    pending = ""
                    continue
                # New section number started before previous resolved — restart buffer
                if _TOC_ENTRY_START.match(stripped):
                    pending = stripped
                else:
                    pending = merged   # keep accumulating (rare 3-line wrap)
                continue

            # Start buffering a potential wrapped entry
            if _TOC_ENTRY_START.match(stripped):
                pending = stripped

    return entries


# ---------------------------------------------------------------------------
# Per-page parsing helpers
# ---------------------------------------------------------------------------

# Page header: "7 Handlebar, controls" or "Handlebar, controls 7"
_HEADER_LEFT  = re.compile(r"^(\d+)\s+(.+)$")
_HEADER_RIGHT = re.compile(r"^(.+)\s+(\d+)$")

# Section heading: "7.2" or "18.4.15" at start of a line
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+[A-Z]", re.MULTILINE)

# Figure reference codes embedded in text (e.g. W00322-10, V01736-10)
_FIGURE_REF = re.compile(r"\b([A-Z]\d{5}-\d{2})\b")

# Torque spec block (multiline):
#   <description line>
#   M<bolt> <val> Nm
#   (<val> ft[⋅·]lbf)
#   [optional Loctite/adhesive line]
_TORQUE_RE = re.compile(
    r"([^\n]+)\n"               # description
    r"(M\d+(?:x[\d.]+)?)\s+"     # bolt spec (x[\d.]+ handles decimal thread pitch e.g. M12x1.5)
    r"(\d+(?:\.\d+)?)\s*Nm\n"   # Nm value
    r"\(\s*(\d+(?:\.\d+)?)\s*ft[⋅·]lbf\)"  # ft·lbf value
    r"(?:\n(Loctite[^\n]+|[A-Z][a-z]+[^\n]{0,40}))?"  # optional note
)

# License watermark lines to strip
_LICENSE_RE = re.compile(
    r"Lizenziert für \| Licensed for:.*?raffled@gmail\.com,\s*\S+\n?",
    re.DOTALL,
)

# Inline diagram reference markers — a letter sequence ending in lowercase
# immediately followed by 1-2 uppercase letters OR digits, at a word boundary.
# Matches: screws1 → screws (1), ofA → of (A), distanceA → distance (A)
# Does NOT match: M8 (ends in uppercase), FE450 (ends in uppercase), WARNING (no marker)
_INLINE_REF_RE = re.compile(r"([a-zA-Z]+[a-z])([A-Z]{1,2}|\d+)\b")


def _clean_text(text: str) -> str:
    """Remove license watermark, expand inline diagram refs, normalize whitespace."""
    text = _LICENSE_RE.sub("", text)
    text = _INLINE_REF_RE.sub(r"\1 (\2)", text)
    # Collapse runs of blank lines to a single blank
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_chapter_header(lines: list[str]) -> tuple[Optional[int], str]:
    """
    Extract chapter number and title from the first non-empty lines.
    Headers appear as "7 Handlebar, controls" or "Handlebar, controls 7".
    Returns (chapter_num, chapter_title) or (None, "").
    """
    for line in lines[:4]:
        line = line.strip()
        if not line or line.isdigit():
            continue
        m = _HEADER_LEFT.match(line)
        if m and int(m.group(1)) < 100:
            return int(m.group(1)), m.group(2).strip()
        m = _HEADER_RIGHT.match(line)
        if m and int(m.group(2)) < 100:
            return int(m.group(2)), m.group(1).strip()
    return None, ""


def _find_sections(text: str) -> str:
    """Return the last section heading number seen on this page."""
    sections = _SECTION_RE.findall(text)
    return sections[-1] if sections else ""


def _extract_torque_specs(text: str) -> list[TorqueSpec]:
    specs = []
    for m in _TORQUE_RE.finditer(text):
        try:
            specs.append(TorqueSpec(
                description=m.group(1).strip(),
                bolt=m.group(2),
                nm=float(m.group(3)),
                ftlbf=float(m.group(4)),
                note=(m.group(5) or "").strip(),
            ))
        except (ValueError, AttributeError):
            continue
    return specs


_MIN_IMAGE_DIM = 100  # skip icons and decorative elements smaller than this


def _extract_images(page: pypdf.PageObject) -> list[PageImage]:
    """
    Extract images in content-stream order (preserves visual/reference ordering).
    Requires Pillow. Images smaller than _MIN_IMAGE_DIM on either side are skipped.
    """
    images = []
    for img_file in page.images:
        try:
            pil = img_file.image
            if pil.width < _MIN_IMAGE_DIM or pil.height < _MIN_IMAGE_DIM:
                continue
            orig_fmt = (pil.format or "").upper()
            if orig_fmt == "JPEG":
                data = img_file.data
                fmt = "JPEG"
            else:
                buf = io.BytesIO()
                pil.save(buf, format="PNG")
                data = buf.getvalue()
                fmt = "PNG"
            images.append(PageImage(
                key=img_file.name,
                data=data,
                width=pil.width,
                height=pil.height,
                format=fmt,
            ))
        except Exception:
            continue
    return images


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_manual(pdf_path: str | Path, extract_images: bool = True) -> list[ParsedPage]:
    """Parse a repair manual PDF and return a list of ParsedPage records."""
    pdf = pypdf.PdfReader(str(pdf_path))
    pages = []

    for i, pdf_page in enumerate(pdf.pages):
        raw = pdf_page.extract_text() or ""
        text = _clean_text(raw)
        lines = [l for l in text.splitlines() if l.strip()]

        chapter_num, chapter_title = _parse_chapter_header(lines)
        section = _find_sections(text)
        torque_specs = _extract_torque_specs(text)
        figure_refs = _FIGURE_REF.findall(text)
        images = _extract_images(pdf_page) if extract_images else []

        pages.append(ParsedPage(
            page_num=i + 1,
            chapter_num=chapter_num,
            chapter_title=chapter_title,
            section=section,
            text=text,
            torque_specs=torque_specs,
            figure_refs=figure_refs,
            images=images,
        ))

    return pages
