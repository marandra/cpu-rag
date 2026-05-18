"""Count tokens of one or more markdown files using Ministral tokenizer.

Usage: uv run python tools/count_tokens.py path/to/file.md [path2.md ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import load_model

MODEL_PATH = "./models/Ministral-3-3B-Q4_K_M.gguf"


def main():
    if len(sys.argv) < 2:
        print("usage: count_tokens.py <file.md> [file2.md ...]")
        sys.exit(1)

    paths = [Path(p) for p in sys.argv[1:]]
    for p in paths:
        if not p.exists():
            print(f"MISSING: {p}")
            sys.exit(2)

    llm = load_model(path=MODEL_PATH, n_ctx=8192)

    for p in paths:
        text = p.read_text(encoding="utf-8")
        toks = llm.tokenize(text.encode("utf-8"), add_bos=False, special=False)
        print(f"{p}\t{len(toks)} tokens\t{len(text)} chars")


if __name__ == "__main__":
    main()
