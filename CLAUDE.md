# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RelationRAG is a thesis research project: a graph-based RAG system that builds an entity–proposition–passage knowledge graph from a corpus via LLM extraction, then retrieves over proposition embeddings. It descends from HippoRAG/PropRAG (docstrings and structure still reference them).

## Commands

There is no test suite, linter, or build step. Dependencies are pinned in `requirements.txt` (a `.venv/` is used locally).

```bash
pip install -r requirements.txt

# Index a corpus (reads data/<dataset>_corpus.json, a list of {"title","text"})
python main.py --mode index --dataset 2wikimultihopqa

# Answer the question written in question.txt against an existing index
python main.py --mode query --dataset 2wikimultihopqa

# Rebuild instead of reusing cached artifacts
python main.py --mode index --force_index_from_scratch true --force_openie_from_scratch true
```

Other flags: `--llm_name` (default `gpt-4o-mini`), `--llm_base_url` (any OpenAI-compatible endpoint), `--embedding_name` (default `facebook/contriever`; only Contriever is wired up in `embedding_model/__init__.py`).

`OPENAI_API_KEY` must be set (loaded from `.env` via python-dotenv); `BaseConfig.__post_init__` raises without it, even in query mode. Run from the repo root — imports use the `src.relationrag...` package path, and the prompt template loader imports `src.relationrag.prompts.templates.<name>` by that path.

`data/` and `outputs/` are gitignored.

## Architecture

All orchestration lives in `src/relationrag/RelationRAG.py`; `main.py` is just a CLI that builds a `BaseConfig` (`utils/config_utils.py`) and calls `index()` or `answer()`.

**Indexing pipeline (`RelationRAG.index`)**
1. Passages (`"title\ntext"`) go into the chunk `EmbeddingStore`.
2. `EnhancedOpenIE` (`information_extraction/`) runs two LLM passes per chunk: NER, then entity-guided proposition extraction (`PropositionExtractor`), which jointly returns atomic propositions (each with `text` + `entities`) and intra-chunk proposition→proposition `relations` (`src`/`dst` indices, `type`, `symmetric`). Malformed JSON is repaired with the `fix_json` prompt.
3. OpenIE results are cached in `outputs/<dataset>/openie_results_ner_<llm>.json` and reused per chunk unless `force_openie_from_scratch`.
4. Entities and propositions are embedded into their own `EmbeddingStore`s.
5. Graph construction collects typed edges into `self.typed_edges`, then `augment_graph()` materializes nodes/edges in an igraph and writes `graph.graphml`. Edge types: `contains` (passage→proposition), `mentions` (proposition→entity), `synonym` (entity↔entity, string-based alias matching in `utils/entity_synonymy.py`), plus LLM-extracted proposition relation types normalized by `utils/relation_utils.py`.

**Retrieval (`retrieve` / `answer`)** — currently in progress: encode query → cosine search over proposition store only → top `seed_k` seeds, optionally MMR-reranked when `lambda_mmr < 1`. `generate()` is a stub that just prints/returns the seeds; graph traversal and answer prompting are not yet implemented.

**Key conventions**
- Node IDs everywhere are `compute_mdhash_id(text, prefix=...)` with prefixes `chunk-`, `entity-`, `proposition-`. Identical text ⇒ identical node, which is how dedup and incremental indexing work; graph builders skip nodes already present in the loaded graph.
- `EmbeddingStore` persists to `outputs/<dataset>/<llm>_<embedding>/<namespace>_embeddings/vdb_<namespace>.parquet` and only embeds new hashes.
- Prompts: each file in `prompts/templates/` defines a module-level `prompt_template` (string or list of chat messages); `PromptTemplateManager` auto-loads them keyed by filename and renders with `string.Template` (`${var}`) substitution.
- LLM access goes through `llm/openai_gpt.py::CacheOpenAI.infer`, which retries indefinitely on any error and accepts a `response_checker` kwarg to reject and retry responses.

## Plan de implementación

Antes de analizar, planear o implementar cualquier cambio significativo al proyecto, consulta `docs/plan.md`. Ahí están documentadas las decisiones de diseño ya aprobadas para cada parte (ej. retriever) que todavía está en desarrollo.
