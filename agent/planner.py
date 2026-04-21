"""
Generate a sequenced repair plan from retrieved chunks using the Claude API.

Entry point:  plan(query, chunks, anthropic_client)  → Plan

The planner receives chunks already sorted in prerequisite-first order by the
retrieval layer.  It formats them into a structured prompt, calls Claude, and
returns a Plan dataclass containing the generated text alongside the structured
metadata (torque specs, image paths, sections covered) pulled directly from
the chunks — no second Claude call needed for those.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anthropic

from agent.retrieval import RetrievedChunk

MODEL = "claude-sonnet-4-6"

_SYSTEM_PROMPT = """\
You are a motorcycle repair assistant. Generate a clear, sequential repair plan \
based on the manual sections provided.

Guidelines:
- Sections are given in prerequisite-first order — complete earlier sections before \
later ones.
- Use the section headings as plan headings.
- Number each step within a section. Keep the language direct and imperative \
("Remove the...", "Torque the...").
- Reference torque values inline where relevant, e.g. "Torque to 40 Nm (29.5 ft·lbf)".
- If a step requires a special tool, name it.
- Do not invent steps not present in the manual text.
- End with a brief summary of what was accomplished.\
"""


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    query: str
    sections_used: list[str]    # section numbers in the order presented to Claude
    text: str                   # full plan text as returned by Claude
    torque_specs: list[dict]    # deduplicated across all chunks
    image_paths: list[str]      # deduplicated by the retrieval layer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a labelled context block for the prompt."""
    parts = []
    for c in chunks:
        label = "prerequisite" if c.depth > 0 else "target"
        heading = f"=== Section {c.section}"
        if c.section_title:
            heading += f": {c.section_title}"
        if c.phase:
            heading += f" [{c.phase}]"
        heading += f" ({label}) ==="
        parts.append(f"{heading}\n{c.text.strip()}")
    return "\n\n".join(parts)


def _collect_torque_specs(chunks: list[RetrievedChunk]) -> list[dict]:
    """Return torque specs deduplicated by (bolt, nm) across all chunks."""
    seen: set[tuple] = set()
    specs: list[dict] = []
    for c in chunks:
        for s in c.torque_specs:
            key = (s["bolt"], s["nm"])
            if key not in seen:
                seen.add(key)
                specs.append(s)
    return specs


def _torque_block(specs: list[dict]) -> str:
    """Format torque specs as a compact reference block for the prompt."""
    if not specs:
        return ""
    lines = []
    for s in specs:
        line = f"  {s['bolt']}: {s['nm']} Nm ({s['ftlbf']} ft·lbf)"
        if s.get("note"):
            line += f" — {s['note']}"
        lines.append(line)
    return "\n\nTorque specifications:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan(
    query: str,
    chunks: list[RetrievedChunk],
    anthropic_client: anthropic.Anthropic,
    model: str = MODEL,
) -> Plan:
    """
    Generate a sequenced repair plan from retrieved chunks.

    Parameters
    ----------
    query            : The user's original repair goal.
    chunks           : Retrieved chunks in prerequisite-first order
                       (as returned by retrieve() or retrieve_from_section()).
    anthropic_client : Authenticated Anthropic client.
    model            : Claude model ID to use.

    Returns
    -------
    Plan with generated text, torque specs, and image paths.
    """
    if not chunks:
        return Plan(
            query=query,
            sections_used=[],
            text="No relevant manual sections found for this query.",
            torque_specs=[],
            image_paths=[],
        )

    torque_specs = _collect_torque_specs(chunks)
    # image_paths are already globally deduplicated by the retrieval layer
    image_paths = [p for c in chunks for p in c.image_paths]
    sections_used = list(dict.fromkeys(c.section for c in chunks))

    user_message = (
        f"Goal: {query}\n\n"
        f"Manual sections (prerequisite sections first):\n\n"
        f"{_build_context(chunks)}"
        f"{_torque_block(torque_specs)}"
    )

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return Plan(
        query=query,
        sections_used=sections_used,
        text=response.content[0].text,
        torque_specs=torque_specs,
        image_paths=image_paths,
    )
