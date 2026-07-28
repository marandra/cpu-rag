#!/usr/bin/env bash
# Fetch the v2 serving model (gemma-4-26B GGUF) from Hugging Face.
#
# Part of the v2 `init` stage (see docs/historial.md): the ~17 GB model is NOT
# baked into the container image, so it is downloaded on first install and
# then the snapshots are built over it (`python -m app.generate`).
#
# The source is the Unsloth redistribution repo, which is PUBLIC and NOT gated
# (confirmed 2026-07-24: anonymous HEAD -> 302 to CDN with user_id=public ->
# 200). No HF token, no huggingface_hub dependency — plain curl.
#
# Idempotent: if the destination already exists and its sha256 matches, this
# is a no-op. A partial/interrupted download is resumed (curl -C -), then the
# size and sha256 are verified before the file is accepted. A mismatch is
# fatal — a bad model would only surface as garbage generations later.
#
#   ./tools/fetch_model.sh              # download to ./models/<default name>
#   MODEL_DEST=/data/m.gguf ./tools/fetch_model.sh
set -euo pipefail

# Pinned to the exact quant v2 serves. Override MODEL_URL only to point at an
# internal mirror carrying the identical bytes (sha256 is still enforced).
MODEL_URL="${MODEL_URL:-https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"
MODEL_SHA256="${MODEL_SHA256:-f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f}"
MODEL_SIZE="${MODEL_SIZE:-16947541728}"
MODEL_DEST="${MODEL_DEST:-./models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf}"

command -v curl >/dev/null || { echo "ERROR: curl not found" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "ERROR: sha256sum not found" >&2; exit 1; }

verify() {
  # Returns 0 iff $MODEL_DEST exists with the expected size and sha256.
  [[ -f "$MODEL_DEST" ]] || return 1
  local size
  size="$(stat -c %s "$MODEL_DEST" 2>/dev/null || stat -f %z "$MODEL_DEST")"
  [[ "$size" == "$MODEL_SIZE" ]] || return 1
  echo "==> Verifying sha256 (this reads all ${MODEL_SIZE} bytes)..." >&2
  echo "${MODEL_SHA256}  ${MODEL_DEST}" | sha256sum -c --status
}

if verify; then
  echo "==> Model already present and verified: ${MODEL_DEST}"
  exit 0
fi

mkdir -p "$(dirname "$MODEL_DEST")"
echo "==> Downloading model (~16.9 GB) from:"
echo "    ${MODEL_URL}"
echo "    -> ${MODEL_DEST}  (resumes if partial)"
# -C - resumes a partial file; -L follows the CDN redirect; --fail turns an
# HTTP error (e.g. a future gating change -> 401) into a non-zero exit.
curl -L --fail --retry 5 --retry-delay 10 -C - -o "$MODEL_DEST" "$MODEL_URL"

if ! verify; then
  echo "ERROR: downloaded model failed size/sha256 verification." >&2
  echo "       Expected sha256 ${MODEL_SHA256}, size ${MODEL_SIZE}." >&2
  echo "       Leaving the file in place for inspection: ${MODEL_DEST}" >&2
  exit 1
fi
echo "==> Model downloaded and verified: ${MODEL_DEST}"
