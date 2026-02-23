# aiciblock-rag

RAG pipeline for medical documents. Implements chunking and embedding stages.

## Chunking pipeline

Converts markdown files into semantically chunked JSON documents ready for embedding.

**Strategy:** hybrid semantic (markdown headers) + token-based splitting with configurable overlap. FAQ documents use paragraph-based splitting (double newline) instead.

### Usage

```bash
python pipeline_chunks.py                        # all markdown/*.md
python pipeline_chunks.py markdown/specific.md   # one file
python pipeline_chunks.py markdown/*.md          # shell glob
```

Configuration is set at the top of `pipeline_chunks.py`: document type preset, output directory, and custom metadata.

### Document type presets

| Type          | max_tokens | overlap_ratio |
|---------------|------------|---------------|
| `paper`       | 512        | 0.25          |
| `guideline`   | 768        | 0.30          |
| `educational` | 512        | 0.25          |
| `faq`         | 256        | 0.10          |

### Preparing markdown for chunking

The chunking pipeline expects well-structured markdown. Follow these guidelines to get the best retrieval quality.

#### Metadata tags

Place optional tags at the top of the file, right after the H1 title. They are extracted and attached to each chunk's metadata.

```markdown
# Document Title
Authors: Author Name, Another Author
DOI: 10.xxxx/xxxxx
Date: 2024-01-15
Keywords: keyword1, keyword2, keyword3
Procedure: procedure_name
Doctype: paper

## Abstract

Abstract text here...

## Rest of content...
```

| Tag           | Description                              | Format                |
|---------------|------------------------------------------|-----------------------|
| `# Title`     | Document title (first H1)                | Markdown heading      |
| `Authors:`    | Author string                            | Free text             |
| `DOI:`        | DOI reference                            | `10.xxxx/xxxxx`       |
| `Date:`       | Publication date                         | Any format            |
| `Keywords:`   | Keywords                                 | Comma-separated       |
| `Procedure:`  | Medical procedure name                   | Free text             |
| `Doctype:`    | Document type (selects chunking preset)  | paper/guideline/educational/faq |
| `## Abstract` | Abstract section                         | Section content       |

Metadata tag lines between the H1 and the first content paragraph are automatically stripped from the chunked text (they remain in chunk metadata).

#### Remove tables of contents

Delete any table of contents or index sections before chunking. A TOC lists all section titles in one block, which makes it match many queries with high similarity but contains no actual content. This creates false positives that waste retrieval slots.

```markdown
<!-- REMOVE this kind of block: -->
1. Introducción
2. Preparación en domicilio
3. Durante su estancia en el hospital
4. Alta a domicilio
```

#### FAQ documents

Set `Doctype: faq` in the metadata. FAQ documents are split by double newline (paragraph blocks) instead of markdown headers. Each paragraph becomes one chunk.

To get clean FAQ chunks:
- Separate each question-answer block with a blank line
- Keep each block self-contained (don't rely on context from previous blocks)
- Avoid very short blocks (< 100 characters) — they'll be skipped by `min_chunk_size`

#### General guidelines

- Use `##` and `###` headings to structure the document — the chunker splits on these boundaries, preserving section context in metadata
- Avoid very long sections without subheadings — they produce large chunks that may exceed `max_tokens` and get split at arbitrary points
- Remove boilerplate that doesn't help answer patient questions: legal disclaimers, author affiliations, citation blocks, editorial notes
- Keep numbered/bulleted lists within their parent section — they chunk well as long as the section heading provides context

### Chunk enrichment

Two optional enrichment steps run during chunking (in this order). Both are configured in `pipeline_chunks.py`.

#### Contextual prefix (`CONTEXTUAL_PREFIX = True`)

Prepends `[{title} | {section}]` to each chunk's text. Improves retrieval by anchoring vectors and BM25 tokens to their document context. Zero runtime cost.

#### LLM enrichment (`LLM_ENRICHMENT = True`)

Uses a single OpenAI API call per chunk (gpt-4o-mini) to produce two things:

- **Compact summary** (40-60 words): dense, fact-preserving distillation used as the LLM prompt context. Keeps numbers, dosages, timeframes, procedure names. Reduces prefill time for CPU inference.
- **Doc2Query questions** (5 per chunk): synthetic patient-style questions in colloquial Spanish, appended to chunk text. Bridges the gap between clinical language in the source and how patients actually ask questions.

Requires `OPENAI_API_KEY` env var. Cost is ~$0.005 for 41 chunks.

After the full pipeline, each chunk has these text variants for different purposes:

| Field | Content | Used for |
|-------|---------|----------|
| `page_content` | prefixed text + synthetic questions | embedding vectors, BM25 index |
| `metadata.page_content_original` | original chunk text (no prefix, no questions) | LLM prompt fallback |
| `metadata.page_content_compact` | LLM-distilled 40-60 word summary | LLM prompt (primary) |
| `metadata.doc2query_questions` | list of synthetic questions | inspection, debugging |

The LLM prompt in `main_rag.py` picks the best text variant automatically: compact → original → page_content.

### Output format

JSON files in `chunks/`, one per input file:

```json
[
  {
    "page_content": "[Title | Section] chunk text...\n\nPreguntas frecuentes:\n¿pregunta 1?\n¿pregunta 2?",
    "metadata": {
      "source": "markdown/file.md",
      "title": "Document Title",
      "doctype": "paper",
      "section": "Introduction",
      "chunk_index": 0,
      "chunk_in_section": 0,
      "num_tokens": 342,
      "page_content_original": "chunk text...",
      "page_content_compact": "Compact summary here...",
      "doc2query_questions": ["¿pregunta 1?", "¿pregunta 2?"]
    }
  }
]
```

## Embedding pipeline

Reads chunk JSONs, generates embeddings with Nomic-Embed-v1.5 (local CPU), and stores them in Qdrant.

### Usage

```bash
python pipeline_embed.py
```

Configuration at the top of `pipeline_embed.py`: vector dimensions, Qdrant path, collection name.

### Components

- **Embedding model:** Nomic-Embed-v1.5 (local, CPU). Dimensions: 384 (default) or 768.
- **Vector DB:** Qdrant with local file storage (`./qdrant_data`).

Both are in `src/embeddings.py` and designed for easy swapping.

### Output metrics

```
41 embeddings in 78.9s (0.5 docs/s, 1924 ms/doc)
41 ingested in 0.17s (collection total: 41)
```

## Retrieval strategies

Modular retrieval via strategy pattern (`src/retrieval.py`). All strategies share a common `Retriever` interface and can be composed freely.

### Available strategies

| Strategy | Description | Dependencies |
|----------|-------------|--------------|
| `vector` | Dense cosine search via Qdrant (Nomic-Embed-v1.5) | *(default)* |
| `bm25` | Sparse keyword search (BM25Okapi), corpus loaded from Qdrant | `rank-bm25` |
| `hybrid` | Vector + BM25 combined via Reciprocal Rank Fusion (RRF, k=60) | `rank-bm25` |
| `hybrid+rerank` | Hybrid + cross-encoder reranking (BAAI/bge-reranker-v2-m3) | `rank-bm25`, `sentence-transformers` |

Set the strategy in `main_rag.py` via `RETRIEVAL_STRATEGY`, or pass `--strategy` to the test/eval scripts.

### Composition example

```python
from src.retrieval import build_retriever
from src.embeddings import _get_client

client = _get_client("./qdrant_data")
retriever = build_retriever(client, "medical_docs", strategy="hybrid+rerank")
results = retriever.retrieve("¿Qué riesgos tiene la cirugía?", top_k=3)
```

Retrievers can also be composed manually for custom setups:

```python
from src.retrieval import VectorRetriever, BM25Retriever, HybridRetriever, RerankedRetriever

hybrid = HybridRetriever(
    vector=VectorRetriever(client, "medical_docs"),
    keyword=BM25Retriever.from_qdrant(client, "medical_docs"),
)
retriever = RerankedRetriever(base=hybrid, candidates=10)
```
### Installing the reranker

The cross-encoder reranker is an optional dependency:

```bash
uv sync --extra rerank
```

## Retrieval test

Run queries against the vector store and inspect results.

```bash
python test_retrieval.py                                    # vector (default)
python test_retrieval.py --strategy hybrid                  # hybrid retrieval
python test_retrieval.py --strategy hybrid+rerank           # with reranker
python test_retrieval.py "¿pregunta concreta?"              # single ad-hoc query
python test_retrieval.py --strategy bm25 "¿pregunta?"       # combined
```

Shows per-query: top-K chunks with score, source file, section, token count, and text preview. Summary with score statistics.

Edit `queries.txt` to add/modify test queries (one per line, `#` for comments).

## Evaluation

Evaluate retrieval quality with metrics and optional LLM-as-judge.

```bash
python eval_retrieval.py                              # vector (default)
python eval_retrieval.py --strategy hybrid             # hybrid retrieval
python eval_retrieval.py --strategy hybrid+rerank      # with reranker
python eval_retrieval.py --llm-judge                   # + LLM scoring (needs OPENAI_API_KEY)
```

Generates `eval_report.md` with aggregated and per-query metrics.

### Metrics

| Metric | Description | Requires LLM |
|--------|-------------|:---:|
| Precision@K | Fraction of top-K results from relevant sources | No |
| Recall@K | Fraction of relevant sources found in top-K | No |
| MRR | Reciprocal rank of first relevant result | No |
| Keyword coverage | Expected keywords found in retrieved chunks | No |
| Faithfulness | Do chunks contain correct information for the query? | Yes |
| Relevance | Do chunks address the question directly? | Yes |

### Eval dataset

Ground truth in `eval_dataset.json`:

```json
{
  "query": "¿Cuáles son los riesgos de la cirugía de hemorroides?",
  "expected_keywords": ["dolor", "sangrado", "infección"],
  "relevant_sources": ["resumen-hemorroides.md"],
  "category": "riesgos"
}
```

LLM judge uses OpenAI API (`gpt-4o-mini` by default). Set `OPENAI_API_KEY` env var.
