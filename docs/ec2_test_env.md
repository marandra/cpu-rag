# EC2 test environment for the v2 deliverable

Spec + runbook for **our own** AWS EC2 that runs the v2 client bundle so the
client can trial it (their 134 questions + ad-hoc questions) before installing
anything, and so **we** can reproduce their results over the 134 questions.

> Provisioning happens in a **separate conversation** via Terraform (AWS creds +
> aws-cli handed over there). This file is the shared spec — both of us follow
> it. Do not provision from a conversation that lacks AWS access.

The bundle itself (`dist/rag-deliverable-v2/`) is the real production deliverable;
this EC2 is a trial/repro box, not the client's hosting (they arrange that
themselves, as they did with v1.1).

**One image, configured per use case — no demo/prod fork.** The bundle carries
the full pool (N replicas behind an nginx LB), parametrized and defaulted to
`N_REPLICAS=1` per profile. So the EC2 runs the *exact* production topology, just
with the numbers we choose. That is deliberate: it lets us (1) validate the real
deployment, and (2) measure the **speedup** on hardware closer to a client's
box (single-socket) than our dual-socket cluster — see §Speedup testing.

---

## 1. The multi-architecture story (read first — it drives the instance choice)

One `Dockerfile`, two image **flavors**:

| flavor | build | ISA | runs on | speed | role |
|---|---|---|---|---|---|
| **portable** | `docker build .` (default) | AVX2 + FMA + F16C, **AVX-512/AMX/AVX-VNNI explicitly OFF** | **any** modern x86_64 (Haswell 2013+) | baseline | **the client deliverable** |
| **native** | `tools/hpc/build_native_image.sh` | + AVX-512-VNNI + AMX-INT8 | **only** Sapphire/Emerald Rapids (Xeon SPR, EC2 m7i/c7i/r7i) | **prefill only** (gemma: ~0 % decode — see §4) | **ours** (HPC cluster; NOT for the gemma deliverable) |

> **Measured on gemma (2026-07-24, this EC2).** Native gives **~0 % decode gain**
> — gemma-4 is a bandwidth-bound MoE, and AMX/VNNI accelerate int8 *compute*,
> not the memory streaming that gates decode. It only speeds **prefill** (~17 %),
> and it **changes the answers** (int8 accumulation order shifts sampling: 83 vs
> 88 tokens on the same snapshot). The "+15–30 % / identical answers" claim was
> the dense **Ministral v1** and does **not** transfer. → **We ship portable only
> for the v2 gemma deliverable; native is not offered.** See the memory
> `gemma-native-no-decode-gain`.

The native image still **SIGILLs on any non-SPR CPU** by design (it emits
instructions those CPUs lack), so it is never what we ship to unknown hardware.

**Two build gotchas we already hit (don't regress):**
- pip caches the compiled llama-cpp wheel by *version*, not by build flags, so a
  wheel built once for one flavor gets silently reused by the other. The
  Dockerfile now forces a fresh source compile (`pip install . --no-cache-dir`).
  A naive portable build reused the native (AMX) wheel and SIGILL'd on a plain
  AVX2 laptop. See `docs/`… and the note in the Dockerfile.
- **Always smoke the portable image on a non-AVX-512 CPU** before shipping (the
  dev laptop is one): "runs there" ≈ "runs on the client's generic hardware".

**Why this matters for the EC2 choice:** m7i/c7i/r7i are all Sapphire Rapids, so
**one SPR EC2 can run BOTH flavors** side by side. That lets us (a) give the
client an honest, CPU-agnostic trial on the **portable** image, and (b) benchmark
portable-vs-native on the same box and **recommend** to the client whether the
native build is worth it *if* their production CPU turns out to be SPR.

---

## 2. Instance sizing

**The binding constraint is RAM, not CPU.** Each **replica** holds its own
~16 GB model resident (`N_REPLICAS × ~16 GB` per profile). At the default
`N_REPLICAS=1`, both profiles together need ~36–40 GB + KV/snapshots/OS. Scaling
replicas to benchmark throughput multiplies that — a 64 GB box fits ~3–4 gemma
replicas total, which is enough to see the scaling curve.

| goal | RAM floor | notes |
|---|---:|---|
| **both profiles at once** (the trial) | **~48–64 GB** | glucowise + aiciblock live together |
| one profile at a time (cheapest) | ~24–32 GB | start/stop each; awkward for a live demo |

CPU: serving wants `N_THREADS≈8` per running profile. Two profiles at nT=8 → 16
vCPU is comfortable; 8 vCPU works for a sequential trial (nT=4 each, or nT=8 when
only one is up).

### Recommended: `r7i.2xlarge` — 8 vCPU / 64 GB (SPR)
Cheapest instance that runs **both** profiles without RAM pressure **and** keeps
the native-image option (it's SPR). Memory-optimized, so you pay for the RAM you
actually need rather than cores you don't.

### More cores if the trial feels CPU-bound: `m7i.4xlarge` — 16 vCPU / 64 GB (SPR)
Same RAM, double the cores → nT=8 per profile simultaneously. ~1.5× the price.

### The cheap **limit** (know it, don't default to it)
- **Absolute floor for both profiles at once:** 64 GB is the practical RAM tier;
  below it (48 GB tiers barely exist, 32 GB is too little) you're forced to
  one-profile-at-a-time.
- **One profile at a time:** a 32 GB box (`m7i.2xlarge`, 8 vCPU / 32 GB, or
  `r7i.xlarge`, 4 vCPU / 32 GB) runs a single gemma. `r7i.xlarge` is the
  cheapest thing that serves *a* profile, but 4 vCPU is slow and you must stop
  one profile to test the other — only worth it for a quick check, not a client
  trial.
- Non-SPR is cheaper still (e.g. `r6i.2xlarge`, Ice Lake, 64 GB) and runs the
  **portable** image fine, but it **cannot** run our native image (no AMX), so we
  lose the side-by-side benchmark. Not worth the small saving.

---

## 3. Cost estimate (test EC2)

On-demand, us-east-1, **approximate — verify against the live AWS pricing page**;
prices drift and vary by region.

| instance | vCPU / RAM | ~$/hour | ~$/day (24 h) | ~$/8 h workday |
|---|---|---:|---:|---:|
| `r7i.2xlarge` (rec.) | 8 / 64 | ~$0.53 | ~$12.7 | ~$4.2 |
| `m7i.4xlarge` | 16 / 64 | ~$0.81 | ~$19.4 | ~$6.5 |
| `r7i.xlarge` (1 profile) | 4 / 32 | ~$0.26 | ~$6.3 | ~$2.1 |

Plus **EBS**: 50 GB gp3 ≈ $4/month (~$0.005/hour) — negligible. **Data transfer**:
the 17 GB model download is *inbound* (free); query traffic out is trivial.

**Cost hygiene:** this is bursty testing, not a 24/7 service. **Stop the instance
when idle** (you keep only the EBS charge) and destroy it when the trial is over.
A realistic trial is a handful of hours, so think **single-digit dollars**, not
the daily figure. Reserve/spot are not worth the hassle for something this
short-lived.

---

## 4. Speedup testing (why this box, not the cluster)

Our HPC cluster is **dual-socket** — its scaling is dominated by cross-socket
NUMA penalties that a client's **single-socket** server (or a single-NUMA EC2)
won't have. So the EC2 gives numbers closer to what the client will actually
see. Two things to measure, both on the same box.

> **Measured 2026-07-24 (r7i.2xlarge, 8 vCPU, portable, gemma glucowise/diabetes):**
> 1. **Native vs portable:** ~0 % decode gain, ~17 % faster prefill, answers change
>    — native is not worth offering for gemma (see §1).
> 2. **N_REPLICAS on single-socket does NOT raise throughput.** Aggregate is flat
>    (N=1×nT8 → 4.83 tok/s; N=2×nT4 → 4.75; N=3×nT2 → 4.35) while per-user drops
>    (5.83 → 3.23 → 2.23). One replica at full threads already saturates the single
>    memory bus; replicas trade latency for concurrency, not throughput. The
>    dual-socket cluster scales because it has 2 buses + 8× cores. Memory
>    `gemma-single-socket-bw-ceiling`.

1. **Per-request speed: portable vs native.** m7i/c7i/r7i are SPR, so load both
   image tars and compare `decode_tok_s` on identical questions. **Result above:
   for gemma, decode does not improve and the answers change** — the method
   stands, the outcome killed the native offer.
   ```bash
   # portable is the default; for native, retag and rebuild snapshots:
   RAG_IMAGE=cpu-rag-api:2.0.0-spr-native ./load_and_run.sh glucowise
   ```

2. **Throughput scaling: N_REPLICAS.** Concurrency == replicas, so drive
   concurrent load and watch aggregate tok/s as `N_REPLICAS` goes 1 → 2 → 3…
   (RAM-capped, ~3–4 total on 64 GB). **Result above: on single-socket the
   aggregate does NOT rise — it is flat/declining** (one memory bus, already
   saturated at N=1). Replicas buy concurrency, not throughput. The curve is
   still worth running per client CPU, but don't expect the cluster's scaling.
   ```bash
   N_THREADS=4 N_REPLICAS=2 ./load_and_run.sh glucowise    # keep nT*N <= cores
   # then hit :8001 with concurrent requests (tools/bench_model.py / a load tool)
   ```

Keep `N_THREADS * N_REPLICAS ≤ physical cores` so replicas don't oversubscribe.

## 5. Storage / OS / network

- **Root EBS:** 40–50 GB gp3 (17 GB model + ~2 GB snapshots + ~320 MB image + OS).
  A separate EBS for `models/` is optional.
- **OS + software:** Amazon Linux 2023 or Ubuntu 24.04, + Docker Engine + the
  compose plugin. Terraform user-data can install Docker and unpack the bundle.
- **Security group:**
  - Outbound HTTPS to `huggingface.co` (first-run model download, no token).
  - Inbound `8001` (glucowise/diabetes) + `8002` (aiciblock/hemorroides+cirugía),
    restricted to our IP and the client's IP.
  - Inbound `22` (SSH) to our IP only.
  - Auth is the `X-API-Key` header; TLS is out of scope for the trial.

---

## 6. Deploy runbook (on the box)

```bash
# 1. Copy the bundle over (or its tar.gz) and unpack.
#    e.g. scp -r dist/rag-deliverable-v2 ec2:/opt/rag
cd /opt/rag/rag-deliverable-v2

# 2. Configure.
cp .env.example .env
#    edit .env -> set a strong RAG_API_KEY

# 3. Bring up BOTH profiles (first run downloads the ~16.9 GB model once,
#    shared; builds snapshots per profile; starts serving).
./load_and_run.sh glucowise     # diabetes            -> :8001
./load_and_run.sh aiciblock     # hemorroides+cirugía -> :8002

# 4. Verify.
curl http://localhost:8001/health   # {"...","procedures":["diabetes"]}
curl http://localhost:8002/health   # {"...","procedures":["cirugia-abdominal","hemorroides"]}
```

**Optional — benchmark native vs portable on the same SPR box:** load the native
image tar too, retag `RAG_IMAGE=cpu-rag-api:2.0.0-spr-native`, regenerate
snapshots (the model fingerprint differs so they rebuild), and compare
`decode_tok_s` on the same questions. Answers should match; only speed changes.

## 7. Reproduce the client's results (our harness)

The replay script was deleted with the rest of `tools/audit_*.py` (2026-07-28):
write it when needed, it is ~30 lines. Read the 134 questions from
`reports/audit_questions.json`, POST each to `/query` with `RAG_API_KEY`, and save
one `eval/<arm>/<procedure>.json` per procedure with the answers under
`rows[].our_answer` — same shape as the existing runs. diabetes goes to port 8001
(glucowise), hemorroides and cirugia-abdominal to 8002 (aiciblock).

**There is no scoring step.** Correctness is settled by reading the 134 answers
against the served document and writing the verdict down, one per question, in a
report like `docs/auditoria_134_v22.md`.
