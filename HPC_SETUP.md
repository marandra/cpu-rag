# HPC Setup — cpu-rag

Server: eurehpclogin02 | Project: `~/Projects/cpu-rag/`

## One-time install

```bash
# 1. Load Python 3.13
module load llm-fast/py313/torch280

# 2. Install uv (user-level, no admin needed)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc

# 3. Install all deps (CMAKE_ARGS required, see Issues below)
cd ~/Projects/cpu-rag
CMAKE_ARGS='-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON' uv sync

# 4. Place GGUF model in models/ (see src/llm.py for expected filename)
```

## Each session

```bash
module load llm-fast/py313/torch280
source ~/Projects/cpu-rag/.venv/bin/activate
```

## Issues

### llama-cpp-python build fails: `unsupported instruction 'vpdpbusd'`

- **Cause:** system binutils 2.35.2 (RHEL9 stock) can't assemble AVX-VNNI instructions;
  `-march=native` detects them and generates them anyway.
- **Fix:** disable native CPU detection, use AVX2 baseline:
  `CMAKE_ARGS='-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON' uv sync`
- Note: `-DGGML_AVX_VNNI=OFF` alone does NOT work; `-march=native` overrides it.
- **Admin fix:** provide binutils ≥ 2.36 or a gcc/13+ module.

## Admin requests

| Request | Reason |
|---------|--------|
| binutils ≥ 2.36 or gcc/13+ module | llama-cpp-python builds without workaround |
| system-wide uv | currently installed per-user via curl |
