"""
Chunking pipeline: markdown files -> RAG-ready JSON chunks.

Usage:
    python pipeline_chunks.py                        # all markdown/*.md
    python pipeline_chunks.py markdown/specific.md   # one file
    python pipeline_chunks.py markdown/*.md          # shell glob
"""

import sys
from glob import glob

from src.chunks import ChunkConfig, process_markdown

# --- Configuration ---
OUTPUT_DIR = "chunks"
DOC_TYPE = "paper"  # paper | guideline | educational | faq
CONTEXTUAL_PREFIX = True  # prepend [title | section] to each chunk
LLM_ENRICHMENT = True  # compact summary + doc2query questions per chunk (needs OPENAI_API_KEY)
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

# --- Process ---
config = ChunkConfig.from_document_type(
    DOC_TYPE,
    output_dir=OUTPUT_DIR,
    custom_metadata=CUSTOM_METADATA or None,
    contextual_prefix=CONTEXTUAL_PREFIX,
    llm_enrichment=LLM_ENRICHMENT,
)

print(f"Processing {len(md_files)} file(s):")
total = 0
for f in md_files:
    docs = process_markdown(f, config)
    total += len(docs)

print(f"Done: {total} chunks saved to {OUTPUT_DIR}/")
