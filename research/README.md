# Research Notebooks

Exploratory notebooks that validate design decisions and implementation details for the ingestion pipeline. Run them in order to walk through the reasoning behind each stage.

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [parse_validation.ipynb](parse_validation.ipynb) | Validates `ingestion/parse.py` against the FE 501s manual — confirms TOC parsing, chapter/section metadata extraction, torque spec counts, and embedded image extraction. |
| 2 | [chunking_strategy.ipynb](chunking_strategy.ipynb) | Documents the three-level chunking hierarchy (chapter → section → phase), validates phase header consistency across the full manual, estimates chunk sizes, and cross-checks the approach against the project README. |
| 3 | [index_validation.ipynb](index_validation.ipynb) | Validates `ingestion/index.py` — chunk distribution by phase, full metadata spot-check, reproduction of the three edge cases fixed during development, ChromaDB collection stats, and retrieval quality queries. |
| 4 | [image_indexing.ipynb](image_indexing.ipynb) | Walks through image extraction, storage, and metadata linking using sections 7.1–7.3 as a concrete example. Shows diagram callouts in context and illustrates how images are shared across sections that span the same page. |
| 5 | [crossref_validation.ipynb](crossref_validation.ipynb) | Validates prerequisite and cross-reference extraction. Shows all `Condition:` prerequisite chains and the 83 preparatory chunks with resolved `(p. NN)` page-number references. Explains why references are restricted to preparatory chunks (the same pattern in main work chunks points to the special tools appendix). |
