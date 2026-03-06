"""
Chunking pipeline: markdown files -> RAG-ready JSON chunks.

Usage:
    python pipeline_chunks.py                        # all markdown/*.md
    python pipeline_chunks.py markdown/specific.md   # one file
    python pipeline_chunks.py markdown/*.md          # shell glob

Each file's Doctype tag determines the splitting strategy:
    - faq: paragraph-based splitting (double newline boundaries)
    - paper/guideline/educational: header-based splitting (##/###)

If the file has no Doctype tag, defaults to DOC_TYPE_DEFAULT below.
Token sizes and overlap are configurable via MAX_TOKENS / OVERLAP_RATIO
overrides (set to None to use the per-doctype preset defaults).
"""

import sys
from glob import glob
from pathlib import Path

from src.chunks import ChunkConfig, extract_metadata, process_markdown, DOCUMENT_TYPE_PRESETS

# --- Configuration ---
OUTPUT_DIR = "chunks"
DOC_TYPE_DEFAULT = "paper"  # fallback when file has no Doctype tag
CONTEXTUAL_PREFIX = True  # prepend [title | section] to each chunk
LLM_ENRICHMENT = True  # compact summary + doc2query questions per chunk (needs OPENAI_API_KEY)

# Override chunk size/overlap (set to None to use doctype preset defaults)
MAX_TOKENS = None       # e.g. 512 — overrides preset for all files
OVERLAP_RATIO = None    # e.g. 0.25 — overrides preset for all files

CUSTOM_METADATA = {
    # "procedimiento": "colecistectomia",
    # "idioma": "es",
}

# --- Resolve input files ---
if len(sys.argv) > 1:
    files = []
    for arg in sys.argv[1:]:
        files.extend(glob(arg) if "*" in arg else [arg])
    md_files = [f for f in files if f.endswith((".md", ".markdown"))]
else:
    md_files = sorted(glob("markdown/*.md"))

if not md_files:
    print("No markdown files found.")
    sys.exit(1)

# --- Process (per-file doctype detection) ---
print(f"Processing {len(md_files)} file(s):")
total = 0
for f in md_files:
    # Detect doctype from file metadata
    text = Path(f).read_text(encoding="utf-8")
    meta = extract_metadata(text)
    doctype = meta.get("doctype", DOC_TYPE_DEFAULT)
    if doctype not in DOCUMENT_TYPE_PRESETS:
        doctype = DOC_TYPE_DEFAULT

    # Build config: start from doctype preset, apply overrides
    kwargs = dict(
        output_dir=OUTPUT_DIR,
        custom_metadata=CUSTOM_METADATA or None,
        contextual_prefix=CONTEXTUAL_PREFIX,
        llm_enrichment=LLM_ENRICHMENT,
    )
    if MAX_TOKENS is not None:
        kwargs["max_tokens"] = MAX_TOKENS
    if OVERLAP_RATIO is not None:
        kwargs["overlap_ratio"] = OVERLAP_RATIO

    config = ChunkConfig.from_document_type(doctype, **kwargs)
    print(f"  {Path(f).name}: doctype={doctype}, max_tokens={config.max_tokens}, overlap={config.overlap_ratio}")
    docs = process_markdown(f, config)
    total += len(docs)

print(f"Done: {total} chunks saved to {OUTPUT_DIR}/")
