# Motorcycle Maintenance Agent

A Python-based AI agent for planning motorcycle repairs and upgrades. Upload a PDF service manual, ask a question about a procedure, and get a fully sequenced plan with torque specifications and reference images — including all prerequisite steps pulled from other chapters.

## Motorcycles

| Year | Make | Model | Manual |
|------|------|-------|--------|
| | | | |

## Architecture

### PDF Ingestion Pipeline

Service manuals are chunked by TOC section rather than fixed token windows. This preserves procedural coherence and maps naturally to how technicians reference the manual.

Each chunk stores:
- Section title and hierarchy (e.g. `Chassis > Rear Suspension > Removal`)
- Full text content
- Extracted images and tables with their positions in the section
- Cross-references to other sections (e.g. "See *Body Panel Removal*, Chapter 4")

Cross-references are resolved at ingestion time and stored as explicit dependency edges, enabling the agent to retrieve the full prerequisite chain for any procedure.

### Dependency-Aware Retrieval

When planning a multi-step procedure, the agent walks the dependency graph to determine the correct execution order before generating the plan. For example, removing the rear suspension requires:

1. Body panel removal (Chapter N)
2. Exhaust removal (Chapter N)
3. Rear wheel removal (Chapter N)
4. Rear suspension removal (Chapter N)

The retrieval layer surfaces all dependent sections so the agent has full context before sequencing steps.

### Agent

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
│   ├── parse.py        # Extract text, images, tables by TOC section
│   ├── crossref.py     # Detect and resolve cross-references between sections
│   └── index.py        # Embed chunks and load into vector store
├── agent/              # Claude API integration and planning logic
│   ├── retrieval.py    # Dependency-aware chunk retrieval
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
python ingestion/parse.py --manual manuals/your_manual.pdf
python ingestion/index.py
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

- [ ] PDF ingestion pipeline (parse, chunk by section, extract images/tables)
- [ ] Cross-reference detection and dependency graph
- [ ] Vector store setup (ChromaDB)
- [ ] Claude API integration and planner
- [ ] FastAPI backend
- [ ] Web UI with inline image/table rendering
- [ ] Cloud migration: Cloud Storage + Vertex AI Vector Search + Cloud Run
