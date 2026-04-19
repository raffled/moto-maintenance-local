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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pypdf


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
    data: bytes
    width: int
    height: int


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


def parse_toc(pdf: pypdf.PdfReader) -> list[dict]:
    """Return list of {number, title, page} dicts from the Table of Contents."""
    entries = []
    in_toc = False
    for page in pdf.pages:
        text = page.extract_text() or ""
        if "Table of contents" in text:
            in_toc = True
        if not in_toc:
            continue
        # TOC ends when we hit non-TOC content
        if in_toc and "Table of contents" not in text and entries:
            # Check if this page still looks like a TOC page
            toc_lines = [l for l in text.splitlines() if _TOC_ENTRY.match(l.strip())]
            if not toc_lines:
                break
        for line in text.splitlines():
            m = _TOC_ENTRY.match(line.strip())
            if m:
                entries.append({
                    "number": m.group(1),
                    "title": m.group(2).strip(),
                    "page": int(m.group(3)),
                })
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
    r"(M\d+(?:x\d+)?)\s+"       # bolt spec
    r"(\d+(?:\.\d+)?)\s*Nm\n"   # Nm value
    r"\(\s*(\d+(?:\.\d+)?)\s*ft[⋅·]lbf\)"  # ft·lbf value
    r"(?:\n(Loctite[^\n]+|[A-Z][a-z]+[^\n]{0,40}))?"  # optional note
)

# License watermark lines to strip
_LICENSE_RE = re.compile(
    r"Lizenziert für \| Licensed for:.*?raffled@gmail\.com,\s*\S+\n?",
    re.DOTALL,
)


def _clean_text(text: str) -> str:
    """Remove license watermark and normalize whitespace."""
    text = _LICENSE_RE.sub("", text)
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


def _extract_images(page: pypdf.PageObject) -> list[PageImage]:
    images = []
    resources = page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    for key, obj in xobjects.items():
        if obj.get("/Subtype") != "/Image":
            continue
        try:
            data = obj.get_data()
            width = int(obj.get("/Width", 0))
            height = int(obj.get("/Height", 0))
            images.append(PageImage(key=key, data=data, width=width, height=height))
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
