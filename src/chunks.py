"""
Markdown chunking for RAG pipeline.

Hybrid semantic + token-based chunking:
1. Semantic split by markdown headers (preserves document structure)
2. Token-based split within sections (respects context limits)
3. Paragraph-based split for FAQ documents
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# --- Document type presets ---

DOCUMENT_TYPE_PRESETS = {
    "paper": {"max_tokens": 512, "overlap_ratio": 0.25},
    "guideline": {"max_tokens": 768, "overlap_ratio": 0.30},
    "educational": {"max_tokens": 512, "overlap_ratio": 0.25},
    "faq": {"max_tokens": 256, "overlap_ratio": 0.10},
}

# Metadata tag pattern used for extraction and stripping
_METADATA_TAG_RE = re.compile(
    r"^(Authors?|DOI|Date|Procedure|Doctype|Docktype|Keywords?):\s*.*$",
    re.MULTILINE | re.IGNORECASE,
)


@dataclass
class ChunkConfig:
    """Chunking parameters."""

    max_tokens: int = 512
    overlap_ratio: float = 0.25
    min_chunk_size: int = 100
    tokenizer_model: str = "gpt-4o-mini"
    output_dir: str = "chunks"
    custom_metadata: dict | None = None
    contextual_prefix: bool = False
    llm_enrichment: bool = False
    llm_enrichment_model: str = "gpt-4o-mini"
    llm_enrichment_n_questions: int = 5

    @property
    def overlap_tokens(self) -> int:
        return int(self.max_tokens * self.overlap_ratio)

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError("overlap_ratio must be between 0 and 1")
        if self.max_tokens <= self.min_chunk_size:
            raise ValueError("max_tokens must be greater than min_chunk_size")

    @classmethod
    def from_document_type(cls, doc_type: str = "paper", **kwargs) -> ChunkConfig:
        if doc_type not in DOCUMENT_TYPE_PRESETS:
            raise ValueError(
                f"Unknown type: {doc_type}. Use: {list(DOCUMENT_TYPE_PRESETS.keys())}"
            )
        preset = DOCUMENT_TYPE_PRESETS[doc_type]
        return cls(
            max_tokens=preset["max_tokens"],
            overlap_ratio=preset["overlap_ratio"],
            **kwargs,
        )


# --- Token counting ---


def count_tokens(text: str, model_name: str = "gpt-4o-mini") -> int:
    """Count tokens using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# --- Metadata extraction ---


def extract_metadata(markdown_text: str) -> dict:
    """
    Extract metadata from markdown tags.

    Supported tags (all optional, case-insensitive):
        # Title              - First H1 heading
        Authors: ...         - Author string
        DOI: ...             - DOI reference
        Date: ...            - Document date
        Keywords: k1, k2     - Comma-separated keywords
        Procedure: ...       - Procedure name
        Doctype: ...         - paper | guideline | educational | faq
        ## Abstract          - Abstract section content
    """
    metadata: dict = {}

    # Title: first H1
    m = re.search(r"^#\s+(.+?)$", markdown_text, re.MULTILINE)
    if m:
        metadata["title"] = m.group(1).strip()

    # Simple tag patterns: "Tag: value"
    tag_fields = {
        "authors": r"^Authors?:\s*(.+?)$",
        "doi": r"^DOI:\s*(.+?)$",
        "date": r"^Date:\s*(.+?)$",
        "procedure": r"^Procedure:\s*(.+?)$",
        "doctype": r"^(?:Doctype|Docktype):\s*(.+?)$",
    }
    for field_name, pattern in tag_fields.items():
        m = re.search(pattern, markdown_text, re.MULTILINE | re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            # Drop values that are empty or look like another tag (cross-tag capture)
            if not value or re.match(r"^\w+:", value):
                continue
            if field_name == "doctype":
                value = value.lower()
            metadata[field_name] = value

    # Keywords: comma-separated list
    m = re.search(r"^Keywords?:\s*(.+?)$", markdown_text, re.MULTILINE | re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        if raw and not re.match(r"^\w+:", raw):
            keywords = [k.strip() for k in raw.split(",") if k.strip()]
            if keywords:
                metadata["keywords"] = keywords

    # Abstract: content after ## Abstract heading
    m = re.search(
        r"^##\s*Abstract\s*$\n\n(.+?)(?=\n\n##|\Z)",
        markdown_text,
        re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    if m:
        abstract = m.group(1).strip()
        if abstract and len(abstract) < 3000:
            metadata["abstract"] = abstract

    return metadata


def strip_metadata_block(markdown_text: str) -> str:
    """Remove metadata tag lines from the beginning of the document.

    Strips lines like 'Authors:', 'DOI:', 'Doctype:', etc. that appear
    between the H1 heading and the first real content paragraph/section.
    """
    lines = markdown_text.split("\n")
    result = []
    past_h1 = False
    past_metadata = False

    for line in lines:
        if not past_h1:
            result.append(line)
            if re.match(r"^#\s+", line):
                past_h1 = True
            continue

        if not past_metadata:
            # Skip metadata tag lines and blank lines in the metadata block
            if _METADATA_TAG_RE.match(line):
                continue
            if line.strip() == "":
                continue
            past_metadata = True

        result.append(line)

    return "\n".join(result)


# --- Chunking ---


def split_faq(markdown_text: str, config: ChunkConfig) -> list[Document]:
    """Split FAQ-style documents by double newline (paragraph blocks).

    Each paragraph block becomes one chunk. Blocks exceeding max_tokens
    are further split with the token splitter. Blocks below min_chunk_size
    are skipped.
    """
    blocks = re.split(r"\n\n+", markdown_text)

    avg_chars_per_token = 4
    max_chars = config.max_tokens * avg_chars_per_token
    overlap_chars = int(max_chars * config.overlap_ratio)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    global_index = 0

    for block in blocks:
        block = block.strip()
        if not block or len(block) < config.min_chunk_size:
            continue

        num_tokens = count_tokens(block, config.tokenizer_model)
        if num_tokens > config.max_tokens:
            subchunks = text_splitter.split_text(block)
        else:
            subchunks = [block]

        for subchunk in subchunks:
            subchunk = subchunk.strip()
            if not subchunk:
                continue
            chunks.append(
                Document(
                    page_content=subchunk,
                    metadata={
                        "chunk_index": global_index,
                        "num_tokens": count_tokens(subchunk, config.tokenizer_model),
                    },
                )
            )
            global_index += 1

    return chunks


def split_markdown(markdown_text: str, config: ChunkConfig, doctype: str | None = None) -> list[Document]:
    """
    Split markdown into chunks using hybrid semantic + token-based strategy.

    For FAQ documents, uses paragraph-based splitting instead.

    Returns list of Documents with metadata:
        title, section, subsection, chunk_in_section, chunk_index, num_tokens
    """
    if doctype == "faq":
        return split_faq(markdown_text, config)

    # 1. Semantic split by headers
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"),
            ("###", "subsection"),
        ],
        strip_headers=False,
    )
    sections = header_splitter.split_text(markdown_text)

    # 2. Token-based split within sections
    avg_chars_per_token = 4
    max_chars = config.max_tokens * avg_chars_per_token
    overlap_chars = int(max_chars * config.overlap_ratio)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    global_index = 0

    for section in sections:
        section_text = section.page_content.strip()
        if not section_text or len(section_text) < config.min_chunk_size:
            continue

        subchunks = text_splitter.split_text(section_text)

        for idx, subchunk in enumerate(subchunks):
            subchunk = subchunk.strip()
            if not subchunk:
                continue

            chunks.append(
                Document(
                    page_content=subchunk,
                    metadata={
                        **section.metadata,
                        "chunk_in_section": idx,
                        "chunk_index": global_index,
                        "num_tokens": count_tokens(subchunk, config.tokenizer_model),
                    },
                )
            )
            global_index += 1

    return chunks


# --- Contextual prefix ---


def _apply_contextual_prefix(doc: Document) -> str:
    """Build a contextual prefix from metadata and prepend to page_content.

    Format: [{title} | {section}] {content}
    Falls back gracefully when metadata fields are missing.
    """
    parts = []
    title = doc.metadata.get("title", "")
    if title:
        parts.append(title)
    section = doc.metadata.get("section", "")
    if section:
        parts.append(section)

    if parts:
        return f"[{' | '.join(parts)}] {doc.page_content}"
    return doc.page_content


def enrich_chunks(documents: list[Document]) -> list[Document]:
    """Apply contextual prefix to each chunk's page_content."""
    return [
        Document(
            page_content=_apply_contextual_prefix(doc),
            metadata=doc.metadata,
        )
        for doc in documents
    ]


# --- LLM chunk enrichment (compact + doc2query in one call) ---

_ENRICHMENT_PROMPT = """\
Eres un asistente especializado en documentación médica para pacientes.

Dado el siguiente fragmento de una guía médica, genera DOS cosas:

1. **compact**: Un resumen muy conciso en español (máximo 40-60 palabras) que \
preserve los datos clave del fragmento: cifras, dosis, duraciones, nombres de \
procedimientos, medicamentos y recomendaciones concretas. Elimina redundancias, \
explicaciones y texto de relleno. No añadas nada que no esté en el fragmento.

2. **questions**: Exactamente {n} preguntas distintas que un paciente podría \
hacer cuya respuesta esté en este fragmento. Usa lenguaje coloquial de paciente \
(no terminología médica), varía el tipo (¿qué?, ¿cuándo?, ¿por qué?, ¿cómo?, \
¿puedo...?).

Fragmento:
---
{chunk}
---

Responde SOLO con JSON válido (sin markdown):
{{"compact": "resumen aquí", "questions": ["pregunta 1", "pregunta 2"]}}"""


def generate_chunk_enrichment(
    documents: list[Document],
    n_questions: int = 5,
    model: str = "gpt-4o-mini",
) -> list[Document]:
    """Generate compact summary + synthetic questions per chunk via OpenAI.

    Single API call per chunk produces both outputs. Requires OPENAI_API_KEY.

    After enrichment, each Document has:
        page_content           — prefixed text + appended questions (for embedding & BM25)
        metadata["page_content_compact"]  — LLM-distilled summary (primary for LLM prompt)
        metadata["doc2query_questions"]   — questions list (for inspection)
    Note: metadata["page_content_original"] is set earlier in process_markdown,
    before contextual prefix, so it contains the true original chunk text.
    """
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    enriched = []
    for i, doc in enumerate(documents):
        prompt = _ENRICHMENT_PROMPT.format(n=n_questions, chunk=doc.page_content)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        compact = parsed.get("compact", "")
        questions = parsed.get("questions", [])

        enriched_content = (
            doc.page_content + "\n\nPreguntas frecuentes:\n" + "\n".join(questions)
        )
        enriched.append(
            Document(
                page_content=enriched_content,
                metadata={
                    **doc.metadata,
                    "page_content_compact": compact,
                    "doc2query_questions": questions,
                },
            )
        )
        n_q = len(questions)
        compact_len = len(compact)
        print(f"    [{i + 1}/{len(documents)}] +{n_q} questions, compact={compact_len} chars")

    return enriched


# --- Pipeline ---


def process_markdown(
    markdown_path: str | Path,
    config: ChunkConfig | None = None,
) -> list[Document]:
    """
    Full pipeline: markdown file -> chunked Documents saved as JSON.

    If no config is provided, auto-selects from doctype metadata in the file.
    """
    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    markdown_text = path.read_text(encoding="utf-8")

    # Extract metadata
    global_metadata = extract_metadata(markdown_text)

    # Auto-select config from doctype if not provided
    if config is None:
        doctype = global_metadata.get("doctype")
        if doctype and doctype in DOCUMENT_TYPE_PRESETS:
            config = ChunkConfig.from_document_type(doctype)
        else:
            config = ChunkConfig()

    # Strip metadata lines from content before chunking
    clean_text = strip_metadata_block(markdown_text)

    # Chunk (use doctype to select FAQ splitter)
    doctype = global_metadata.get("doctype")
    chunks = split_markdown(clean_text, config, doctype=doctype)

    # Enrich with global metadata + source
    documents = []
    for chunk in chunks:
        metadata = {"source": str(path), **global_metadata, **chunk.metadata}
        if config.custom_metadata:
            metadata.update(config.custom_metadata)
        documents.append(Document(page_content=chunk.page_content, metadata=metadata))

    # Store original page_content before any enrichment
    for doc in documents:
        doc.metadata["page_content_original"] = doc.page_content

    # Apply contextual prefix if configured
    if config.contextual_prefix:
        documents = enrich_chunks(documents)

    # LLM enrichment: compact summary + doc2query questions (single API call)
    if config.llm_enrichment:
        print(f"    LLM enrichment ({config.llm_enrichment_model})...")
        documents = generate_chunk_enrichment(
            documents,
            n_questions=config.llm_enrichment_n_questions,
            model=config.llm_enrichment_model,
        )

    # Save
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_chunks.json"

    docs_json = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in documents
    ]
    output_path.write_text(
        json.dumps(docs_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"  {path.name} -> {len(documents)} chunks -> {output_path}")
    return documents
