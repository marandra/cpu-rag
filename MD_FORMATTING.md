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

