# Evaluation Framework for Medical Q&A Quality

Status: draft / formalization of existing practice — seed for a dedicated work line.

Scope: the **fulldoc** Q&A path of `cpu-rag`. Retrieval-specific evaluation
(precision@k, recall@k, MRR, chunk-config sweeps) lives in the sibling `gpu-rag`
repo and is intentionally out of scope here.

## 1. Purpose

In a medical Q&A system built on small models, answer quality is the bottleneck,
not infrastructure. Every change to the model, the prompt, or the distilled
document moves quality in non-obvious ways. Without a measurable, automatable
quality signal, those changes are evaluated by eyeballing a few answers.

This framework makes **answer quality a first-class, automatable metric** so it
can sit inside an ML workflow: a parameter sweep, a regression gate in CI, or a
model/prompt-version comparison produces a number, not an impression.

## 2. Evaluation datasets

### 2.1 Question sets (by what they test)

The corpus must be probed from three angles, each a separate dataset:

| Set | Purpose | Correct behaviour |
|-----|---------|-------------------|
| **Coverage** (`*_coverage.json`) | Answerable factual questions across the procedure document. | Answer, grounded, with the expected facts. |
| **Out-of-context** | Questions whose answer is simply not in the document. | Refuse cleanly ("No tengo información sobre eso"). |
| **Gray-zone** (`*_grayzone.json`) | Topically related to the document but the *specific* answer is absent. The hardest case and the most frequent source of hallucination. | Refuse — without hallucinating plausible detail. |

The gray-zone set is the discriminating one: a system that answers coverage
questions well but hallucinates on gray-zone questions is unsafe in a medical
context.

### 2.2 Per-question metadata schema

Each query is an object carrying enough metadata to score it automatically and
to slice results:

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The patient question, as asked. |
| `intent` | string | What the question is really after (one line). |
| `answerable` | bool | Whether the document contains the answer. |
| `profile` | enum | Demographic / persona variant (see 2.3). |
| `expected_keywords` | list | Facts that must appear in a correct answer; for refusals, the refusal phrase. |
| `category` | enum | e.g. `A_easy_factual`, `C_topic_related_no_answer`. |
| `test_focus` | string | What this query specifically stresses (e.g. `responde_sin_meta`, `rechazo_estricto`). |
| `difficulty` | enum | `easy` / `medium` / `hard`. |
| `procedure` | string | Procedure the question belongs to. |

### 2.3 Demographic / persona variants

The same `intent` is reworded for different patient profiles, so quality is
measured across the real user population, not only clean queries:

- `general` — clean, well-formed.
- `mayor` — elderly phrasing.
- `joven` — colloquial / informal.
- `ansioso` — anxious, over-detailed.
- `baja_alfabetizacion` — low-literacy phrasing, typos.
- `L2` — non-native Spanish speaker.

This turns each metric into a per-profile breakdown, exposing whether the system
degrades for the most vulnerable users.

### 2.4 Dataset generation

Datasets are generated with an LLM from a documented prompt (intent list +
profile + answerable target), then human-reviewed. The generation prompt is
itself versioned. Generation is part of the preparation pipeline, not a one-off.

## 3. Metrics

### 3.1 Deterministic (no LLM, fast, cheap)

- **Keyword coverage** — fraction of `expected_keywords` present in the answer.
- **Span coverage** — fraction of expected key facts found in a generated answer.
- **Refusal correctness** — for `answerable: false`, did the system refuse?

### 3.2 LLM-as-judge

A separate LLM call scores what keywords cannot:

- **Faithfulness** — is every claim in the answer supported by the fulldoc text?
- **Relevance** — does the answer address the question?
- **Refusal quality** — is the refusal clean (no hallucinated hedging)?

The judge model should differ from (or be stronger than) the system model, and
the judge prompt is versioned alongside the datasets.

### 3.3 Failure-mode taxonomy

Beyond scalar scores, answers are tagged against known failure modes (see
`prompt_versions.md`), so regressions are diagnosable:

- Open-list expansion (`"anticoagulantes, etc."` → invented drug names).
- Meta-comments about absent information ("la información no especifica…").
- Hybrid responses (mixing answer, refusal and meta in one reply).
- Cross-section drag (merging the answer with an adjacent elaboration).

## 4. Proposed work line

To turn current practice into a reusable framework:

1. **Unify the dataset schema** across procedures into a single versioned spec;
   validate with a schema check.
2. **Standardize the three set types** (coverage / out-of-context / gray-zone)
   for every procedure, with a target count and profile mix.
3. **Package the metrics** (deterministic + LLM-judge) as a single scoring
   library.
4. **Regression tracking** — store scored runs keyed by (model, prompt version,
   fulldoc version) and diff against the previous baseline.
5. **CI gate** — fail a change that drops span coverage or refusal correctness
   below a threshold.
6. **Judge calibration** — periodically check LLM-judge scores against a small
   human-labelled set.
