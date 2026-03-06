# aiciblock-rag

RAG pipeline for medical patient documents. Retrieves relevant information from clinical guides and FAQs to answer patient questions about surgical procedures.

## Pipeline overview

```
markdown/*.md          Source documents (from PDFs, out of scope)
    |
    v
pipeline_chunks.py     Chunk + enrich (contextual prefix, LLM compact + doc2query)
    |
    v
chunks/*.json          Enriched chunk files
    |
    v
pipeline_embed.py      Embed (Nomic-Embed-v1.5) + store in Qdrant
    |
    v
qdrant_data/           Vector DB (local)
    |
    v
main_rag.py            Query -> retrieve -> generate answer (local LLM)
```

## 1. Markdown preparation

Source documents are converted from PDFs to markdown externally (out of scope). The chunking pipeline relies on formatting conventions documented below.

### Metadata tags

Place optional tags right after the H1 title. They are extracted into chunk metadata and stripped from the chunked text.

```markdown
# Document Title
Authors: Author Name
DOI: 10.xxxx/xxxxx
Date: 2024-01-15
Keywords: keyword1, keyword2
Procedure: hemorrhoidectomy
Doctype: faq

## First section...
```

| Tag | Description | Values |
|-----|-------------|--------|
| `Doctype:` | Determines splitting strategy | `faq` (paragraph-based) or `paper`/`guideline`/`educational` (header-based) |
| `Procedure:` | Medical procedure name (for metadata filtering) | Free text |
| `Authors:`, `DOI:`, `Date:`, `Keywords:` | Document metadata | Free text / comma-separated |

### Content cleanup rules

Before adding a markdown file, remove the following sections manually. They create noisy chunks that match many queries but contain no useful patient information:

| Remove | Why | Example |
|--------|-----|---------|
| Table of contents / index | Lists all section titles in one block; matches many queries with high similarity but has no actual content | `1. Introduccion \n 2. Preparacion...` |
| Author lists and affiliations | Names and institutions are not patient-relevant | `- Dr. X. Hospital Y. Ciudad Z.` |
| Acknowledgements / reviewers | Same as above | `## Agradecimientos` |
| Glossary / terminology annexes | Technical/clinical definitions create false positives for patient queries; relevant terms are already explained in the main text | `## Anexo 2. Glosario` |
| Citation blocks / editorial notes | Institutional boilerplate | `Esta guia debe citarse: ...` |
| Empty metadata tags | Tags with no value add noise | `Authors:\nDOI:\nProcedure:` |

### FAQ documents

Set `Doctype: faq`. These are split by double newline (paragraph blocks) instead of markdown headers.

- Separate each Q&A block with a blank line
- Keep each block self-contained
- Avoid very short blocks (< 100 chars) -- they get skipped

### General structure guidelines

- Use `##` and `###` headings to structure -- the chunker splits on these boundaries
- Avoid very long sections without subheadings (they exceed max_tokens and get split at arbitrary points)
- Keep numbered/bulleted lists within their parent section

## 2. Chunking

```bash
python pipeline_chunks.py                        # all markdown/*.md
python pipeline_chunks.py markdown/specific.md   # one file
```

### Splitting strategy

Two strategies selected automatically by `Doctype:` tag:

- **FAQ** (`Doctype: faq`): paragraph-based splitting (double newline boundaries), then token-based sub-split if oversized. Default: `max_tokens=256`, `overlap=0.10`.
- **Structured** (paper/guideline/educational): hierarchical split by markdown headers (H1/H2/H3), then token-based split within sections. Defaults vary by type.

The `Doctype:` tag is auto-detected per file. Fallback when missing: `paper`.

### Chunk size presets

| Type | max_tokens | overlap_ratio | Splitting |
|------|-----------|---------------|-----------|
| paper | 512 | 0.25 | header-based |
| guideline | 768 | 0.30 | header-based |
| educational | 512 | 0.25 | header-based |
| faq | 256 | 0.10 | paragraph-based |

These can be overridden globally in `pipeline_chunks.py` via `MAX_TOKENS` and `OVERLAP_RATIO` (set to `None` to use preset defaults). These are candidates for parameter sweeping in evaluation.

### Enrichment pipeline

Applied in order during chunking:

1. **Global metadata** -- extracted from document tags + source filename
2. **`page_content_original`** -- stored before any enrichment (true original text)
3. **Contextual prefix** (`CONTEXTUAL_PREFIX = True`) -- prepends `[{title} | {section}]` to chunk text. Improves retrieval by anchoring to document context. Zero cost.
4. **LLM enrichment** (`LLM_ENRICHMENT = True`, needs `OPENAI_API_KEY`) -- single OpenAI call per chunk (gpt-4o-mini):
   - `page_content_compact`: 40-60 word distilled summary (used as LLM prompt context, reduces CPU prefill time)
   - `doc2query_questions`: 5 synthetic patient questions in colloquial Spanish (appended to `page_content` for embedding/BM25, bridges clinical-to-patient language gap)

### Text variants per chunk

| Field | Content | Used for |
|-------|---------|----------|
| `page_content` | prefixed text + synthetic questions | embedding vectors, BM25 index |
| `metadata.page_content_original` | original chunk text (no prefix, no questions) | LLM prompt fallback |
| `metadata.page_content_compact` | LLM-distilled 40-60 word summary | LLM prompt (primary) |
| `metadata.doc2query_questions` | list of synthetic questions | inspection, debugging |

## 3. Embedding and indexing

```bash
python pipeline_embed.py
```

- **Embedding model:** Nomic-Embed-v1.5 (local, CPU, dim=384)
- **What gets embedded:** `page_content` (fully enriched: prefix + questions)
- **Vector DB:** Qdrant, local file storage (`./qdrant_data`), collection `medical_docs`, cosine distance
- **Re-ingestion:** drops and recreates collection on each run

## 4. Retrieval

All strategies implement `Retriever.retrieve(query, top_k) -> [(Document, score)]`.

| Strategy | How it works | Key properties |
|----------|-------------|----------------|
| `vector` | Embed query -> cosine search in Qdrant | Fast (~130ms), good precision |
| `bm25` | BM25Okapi on full corpus (scrolled from Qdrant) | Instant, exact keyword matching |
| `hybrid` | Vector + BM25 merged via RRF (k=60, 3x candidates) | Best overall precision when both signals are strong |
| `hybrid+rerank` | Hybrid -> cross-encoder reranking (BAAI/bge-reranker-v2-m3) | Best recall, ~30s/query on CPU |

```bash
# Set in main_rag.py or pass --strategy to eval scripts
RETRIEVAL_STRATEGY = "hybrid"
```

```python
from src.retrieval import build_retriever
retriever = build_retriever(client, "medical_docs", strategy="hybrid")
results = retriever.retrieve("que pasa si no me opero", top_k=5)
```

### Installing the reranker

```bash
uv sync --extra rerank
```

### BM25 tokenizers (`src/tokenizers.py`)

Three tokenizer options for BM25, selectable via `--tokenizer` flag or `build_retriever(tokenizer=...)`:

| Tokenizer | What it does | Use case |
|-----------|-------------|----------|
| `whitespace` | `text.lower().split()` | Baseline, no NLP dependencies |
| `whitespace+accent` | Whitespace + accent stripping (cirugía→cirugia) | Handles patient typos without NLP |
| `spacy` (default) | spaCy `es_core_news_sm`: lemmatize + remove stopwords + strip accents | Full Spanish NLP (comiendo→comer, removes de/la/en) |

The spaCy model must be installed: `uv pip install es_core_news_sm@https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl`

### Installing the reranker

```bash
uv sync --extra rerank
```

### Known gaps and next steps

**Metadata filtering** -- Qdrant supports payload filters. Tag each document with `Procedure: hemorrhoidectomy` and filter at query time to only search within relevant procedure documents. Same collection, just filtered. Planned as `FilteredRetriever` or filter param on `VectorRetriever`.

**Query preprocessing** -- spell-correction is risky in medical context (could "correct" valid medical terms). With good embeddings + doc2query, semantic search already handles typos. Accent normalization in BM25 tokenizer (already implemented) is the only preprocessing worth doing.

**Embedding model** -- currently Nomic-Embed-v1.5. Other candidates: `BAAI/bge-m3` (multilingual), `intfloat/multilingual-e5-large`. Dimension (384 vs 768) is an easy sweep; different models need changes to `src/embeddings.py`.

## 5. Generation

```bash
python main_rag.py "que pasa despues de operarme"
python main_rag.py   # interactive mode
```

- **LLM:** llama-cpp-python, CPU-only, GGUF models (default: Llama-3.2-1B-Instruct Q4_K_M)
- **Prompt text selection:** `page_content_compact` (primary) -> `page_content_original` (fallback) -> `page_content`
- **System prompt:** answers only from retrieved fragments, in Spanish

## 6. Evaluation

### Eval datasets

| Dataset | Queries | Description |
|---------|---------|-------------|
| `eval_dataset.json` | 14 | Clean, well-formed Spanish queries (baseline sanity check) |
| `eval_dataset_realistic.json` | 41 | Realistic patient queries: typos, colloquial language, 5 personas, 5 unanswerable |

Realistic queries are generated with an LLM using the prompt in `prompts/generate_eval_queries.md`. Each query includes: `intent`, `answerable`, `profile` (mayor/ansioso/joven/L2/baja_alfabetizacion), `difficulty` (easy/medium/hard), `relevant_sources`, `expected_keywords`, `category`.

### Retrieval metrics

| Metric | Description | Requires LLM |
|--------|-------------|:---:|
| Precision@K | Fraction of top-K results from relevant sources | No |
| Recall@K | Fraction of relevant sources found in top-K | No |
| MRR | Reciprocal rank of first relevant result | No |
| Keyword coverage | Expected keywords found in retrieved chunks | No |
| Faithfulness | Do chunks contain correct information? | Yes (OpenAI) |
| Relevance | Do chunks address the question? | Yes (OpenAI) |

### Running evaluations

```bash
# Single strategy
python eval_retrieval.py --strategy hybrid
python eval_retrieval.py --llm-judge              # + LLM scoring

# Compare strategies side-by-side
python compare_strategies.py                                          # all 4 strategies, default dataset
python compare_strategies.py vector hybrid                            # specific strategies
python compare_strategies.py --dataset eval_dataset_realistic.json    # realistic queries
python compare_strategies.py --top-k 10                               # change top_k
python compare_strategies.py --llm-judge                              # + LLM scoring
```

### Current baseline (realistic queries, 33 chunks, cleaned markdown)

```
Metric                    vector        hybrid
----------------------------------------------
P@3                      0.545         0.626 *
P@5                      0.541         0.566 *
R@5                      0.872 *       0.866
MRR                      0.783         0.825 *
keyword_cov              0.648         0.687 *
latency_ms_avg             137           133
```

### Parameters for sweeping

| Component | Parameter | Current value | Range to sweep |
|-----------|-----------|---------------|----------------|
| Chunking | `max_tokens` | 256-768 (by doctype) | 256, 384, 512, 768 |
| Chunking | `overlap_ratio` | 0.10-0.30 (by doctype) | 0.0, 0.10, 0.20, 0.30 |
| Chunking | `contextual_prefix` | True | True, False |
| Chunking | `llm_enrichment` | True | True, False |
| Retrieval | `strategy` | hybrid | vector, bm25, hybrid |
| Retrieval | `top_k` | 5 | 3, 5, 10 |
| Retrieval | BM25 tokenizer | spacy | whitespace, spacy |

Total: 960 configurations, 64 unique chunk configs (32 need OpenAI API for LLM enrichment).

### Running the parameter sweep

```bash
# List all configurations
python sweep.py --list

# Run a single config by index (for Slurm array jobs)
python sweep.py --config-index 42

# Run with explicit parameters
python sweep.py --max-tokens 512 --overlap 0.25 --strategy hybrid --tokenizer spacy

# Run all sequentially (for testing, not HPC)
python sweep.py --all

# Aggregate results into a ranked table
python sweep.py --aggregate
```

The sweep script handles chunking + embedding + evaluation. Chunk configs are reused across retrieval variants (same chunks, different strategies). Each config result is saved as a separate JSON file in `sweep_results/`.

For HPC: see `sweep.slurm` for a Slurm array job template. The recommended approach is two phases: (1) build all 64 chunk configs, (2) run all 960 evaluations (fast, reuses existing chunks).

## Project structure

```
markdown/               Source markdown documents
chunks/                 Chunked JSON files (generated)
qdrant_data/            Vector DB (generated)
models/                 GGUF model files
src/
  chunks.py             Chunking, metadata extraction, enrichment
  embeddings.py         Embedding model, Qdrant operations
  retrieval.py          Retriever strategies (vector, BM25, hybrid, rerank)
  tokenizers.py         BM25 tokenizers (whitespace, spaCy Spanish)
  evaluation.py         Metrics, LLM-as-judge, report generation
  llm.py                Local LLM inference
pipeline_chunks.py      Chunking pipeline entry point
pipeline_embed.py       Embedding pipeline entry point
main_rag.py             RAG query interface
eval_retrieval.py       Single-strategy evaluation
compare_strategies.py   Multi-strategy comparison
sweep.py                Full parameter sweep (chunk + embed + eval)
sweep.slurm             Slurm job template for HPC
prompts/                Reusable LLM prompts (eval query generation)
eval_dataset.json       Clean eval queries (14)
eval_dataset_realistic.json  Realistic eval queries (41)
queries.txt             Manual test queries
```
