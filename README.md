# Motorcycle Maintenance Agent

A Python-based AI agent for planning motorcycle repairs and upgrades. Upload a PDF service manual, ask a question about a procedure, and get a fully sequenced plan with torque specifications and reference images — including all prerequisite steps pulled from other chapters.

## Motorcycles
Planned motorycles (based on what's in my garage and what I have PDF service manuals for):
| Year | Make | Model |
|------|------|-------|
| 2024| Husqvarna | Norden 901 Expectition 
| 2025| Husqvarna | FE501s
| 2023| Royal Enfield | Hunter 350

## Architecture
![Architecture Diagram](img/Motorcycle%20Maintenance%20Agent%20%E2%80%94%20Conceptual%20Architecture.png)


### PDF Ingestion Pipeline
#### Ingestion Process
![Ingestion Pipeline](img/Ingestion%20pipeline%20architecture.png)

Service manuals are chunked by TOC section rather than fixed token windows. This preserves procedural coherence and maps naturally to how technicians reference the manual.

Chunks follow a three-level hierarchy driven by the manual's own structure:

```
Chapter  (e.g. 7 — Handlebar, controls)
└── Section  (e.g. 7.2 — Adjusting the handlebar position)
    └── Phase  (Preparatory work | Main work | Reworking)
```

Roughly 32% of sections contain multiple procedure phases under a single section number. The parser splits these on the manual's own phase headers rather than using arbitrary token windows.

#### Chunk Data Model

![Chunk Data Model](img/Chunk%20data%20model%20and%20dependency%20graph.png)

- Chapter number and title, section number, phase label
- Full text content
- Extracted torque specifications (bolt size, Nm, ft·lbf, adhesive notes)
- Procedure images saved to `data/images/` and linked by file path, in content-stream order
- Resolved cross-references to prerequisite sections (from `(p. NN)` citations in Preparatory work phases)

Cross-references are resolved at ingestion time and stored as explicit dependency edges, enabling the agent to retrieve the full prerequisite chain for any procedure.

### Dependency-Aware Retrieval

Retrieval runs in two phases:

1. **Semantic search** — the query is embedded with `text-embedding-3-small` and the top-k matching chunks are fetched from ChromaDB.
2. **Dependency walk** — any `preparatory` chunk in the seed set has its `references` list followed recursively (breadth-first search, BFS). Referenced sections are fetched by exact section number and added to the result set with an incremented `depth`.

Results are sorted with the deepest prerequisites first so the planner receives chunks in the correct procedural sequence. For example, querying "disassembling the fork cartridge" (section 6.12) automatically surfaces sections 6.10 and 6.11 as depth-1 prerequisites before the target section.

Image paths are deduplicated across the full result set at retrieval time — pages that span section boundaries would otherwise contribute the same images to multiple chunks.

#### Disambiguation

Some queries match procedures in more than one chapter with different assumptions (e.g. "oil and filter change" matches both Ch. 18 — engine assembly with the engine removed, and Ch. 22 — routine maintenance with the engine in the frame). When the seed set spans multiple chapters, the UI surfaces the candidates grouped by chapter and lets the user choose before running the full dependency walk. A second entry point, `retrieve_from_section(section, ...)`, seeds the BFS walk from a known section number rather than an embedding query, scoping the result to the user's chosen context.

### Agent

![Runtime](img/Runtime%20query%20flow.png)

Built directly on the Claude API (no agent framework). Given a repair or upgrade goal, the agent:
1. Identifies the target procedure and retrieves its section chunk
2. Traverses the dependency graph to collect all prerequisite sections
3. Generates a fully ordered, sequential plan
4. Annotates steps with torque values extracted from the manual
5. References relevant images and tables inline

### Web UI

FastAPI backend serving a lightweight frontend. The UI supports:
- Conversational planning interface
- Inline display of images and tables pulled from the PDF
- Torque spec callouts
- Printable step-by-step view

### Local → Cloud Migration Path

The project is designed to run locally first, then migrate to Google Cloud (Vertex AI) with minimal changes:

| Component | Local | Cloud (Vertex AI) |
|-----------|-------|-------------------|
| Vector store | ChromaDB (local) | Vertex AI Vector Search |
| PDF/image storage | Local filesystem | Cloud Storage |
| LLM | Claude API | Claude API (unchanged) |
| Serving | `uvicorn` local | Cloud Run |

The ingestion pipeline and agent logic are environment-agnostic; only the storage and serving layers change.

## Project Structure

```
moto-maintenance/
├── manuals/            # Raw PDF service manuals
├── ingestion/          # PDF parsing, chunking, embedding, indexing
│   ├── parse.py        # Extract text, images, torque specs by TOC section
│   ├── images.py       # Save extracted images to disk, build page→path index
│   ├── crossref.py     # Detect and resolve cross-references between sections
│   └── index.py        # Embed chunks and load into vector store
├── agent/              # Claude API integration and planning logic
│   ├── retrieval.py    # Semantic search + BFS dependency walk over prerequisite graph
│   └── planner.py      # Step sequencing and annotation
├── ui/                 # FastAPI backend + frontend
│   ├── main.py
│   └── static/
├── data/               # Local vector store and extracted assets
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in:
```
ANTHROPIC_API_KEY=...
```

## Ingesting a Manual

```bash
python -m ingestion.index manuals/your_manual.pdf
```

Use `--reset` to drop and rebuild the collection from scratch — required after any changes to chunking or parsing logic, since `upsert` leaves stale chunks from previous runs in place:

```bash
python -m ingestion.index manuals/your_manual.pdf --reset
```

Run the smoke test after re-indexing to confirm everything is healthy:

```bash
python smoke_test.py
```

This extracts sections, resolves cross-references, embeds chunks, and loads them into the local vector store.

## Usage

```bash
uvicorn ui.main:app --reload
```

Open `http://localhost:8000` and ask questions like:
- *"What are all the steps to replace the rear shock, starting from stock?"*
- *"Walk me through a full valve clearance check."*
- *"What torque do I need for the swingarm pivot bolt?"*

## Adding a New Manual

1. Place the PDF in `manuals/`
2. Add the bike details to the table at the top of this README
3. Run the ingestion pipeline

## Roadmap

- [x] PDF ingestion pipeline (parse, chunk by section, extract images/tables)
- [x] Cross-reference detection and dependency graph
- [x] Vector store setup (ChromaDB)
- [x] Dependency-aware retrieval (`agent/retrieval.py`)
- [ ] Claude API planner (`agent/planner.py`)
- [ ] FastAPI backend
- [ ] Web UI with inline image/table rendering
- [ ] Cloud migration: Cloud Storage + Vertex AI Vector Search + Cloud Run
