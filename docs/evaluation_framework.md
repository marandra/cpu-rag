# Evaluation Framework for Medical Q&A Quality

Status: formalization of existing practice — a methodology we run, share, and
iterate. This is a protocol + recommendations, not a spec for elaborate infra.

Scope: the **fulldoc** Q&A path of `cpu-rag` — one distilled markdown per
procedure, sent in full as `INFORMACIÓN`, no retrieval. Retrieval evaluation
(P@k, R@k, MRR, chunk-config sweeps) lives in the sibling `gpu-rag` repo and is
**out of scope here** — see §7 for what carries over and what does not.

---

## 0. Honest framing — is this worth it?

Yes, but right-size it. In a medical Q&A system on a small CPU model, the infra
is solved and **answer quality is the bottleneck**. Every change to model,
prompt, or distilled doc moves quality in non-obvious ways; without a
measurable, automatable signal those changes are judged by eyeballing a handful
of answers.

But the scale is small: 3 procedures, ~109 questions, one CPU model. The goal is
a methodology that is **reliable, reproducible, and cheap to run** — not an
evaluation MLOps platform. Two consequences run through this whole document:

1. **Keyword coverage is a fast gate, not the safety metric.** A keyword can be
   present in a wrong or hallucinated answer, and a correct paraphrase can miss
   the literal keyword. Faithfulness (every claim grounded in the fulldoc) is
   the metric that actually protects patients — and it needs a judge (§3.2).
2. **N is small, so deltas are noisy.** With ~20 questions per set, one question
   is 5%. Report this and do not chase ±1-question movements as if they were
   signal (§5).

---

## 1. Evaluation datasets

### 1.1 Question sets (by what they test)

The corpus is probed from three angles, each a separate set:

| Set | File | Purpose | Correct behaviour |
|-----|------|---------|-------------------|
| **Coverage** | `*_coverage.json` | Answerable factual questions across the document. | Answer, grounded, with the expected facts. |
| **Out-of-context** | (in `*_grayzone.json`, cat. `D`/`E`/`F`) | Answer simply not in the document (off-domain, administrative, meta). | Refuse cleanly. |
| **Gray-zone** | `*_grayzone.json` (cat. `C`) | Topically related but the *specific* answer is absent. | Refuse — **without** hallucinating plausible detail. |

The gray-zone set is the discriminating one: a system that answers coverage
questions well but hallucinates on gray-zone questions is unsafe. This is the
classic *answerable vs. unanswerable* distinction (SQuAD 2.0 lineage): the hard
frontier is "related-but-absent", not "obviously off-topic".

### 1.2 Per-question metadata schema

Each query carries enough metadata to score it automatically and to slice
results. Fields currently populated across the cpu-rag datasets:

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The patient question, as asked. |
| `intent` | string | What the question is really after (one line). |
| `answerable` | bool | Whether the document contains the answer. |
| `profile` | enum | Persona variant (see §1.3). Currently all `general`. |
| `relevant_sources` | list | Source doc(s); `[]` when `answerable: false`. |
| `expected_keywords` | list | Facts that must appear in a correct answer; for refusals, the refusal phrase. |
| `category` | enum | `A_easy_factual`, `B_specific_present`, `C_topic_related_no_answer`, `D_out_of_domain`, `E_administrative`, `F_meta_assistant`, `G_conversational_edge`. |
| `test_focus` | string | What this query specifically stresses (e.g. `responde_sin_meta`, `rechazo_estricto`). |
| `difficulty` | enum | `easy` / `medium` / `hard`. |
| `procedure` | string | Procedure the question belongs to. |

Optional, inherited from gpu-rag and useful if added later:

- `relevant_spans` — exact text spans the correct answer should rest on. In
  fulldoc this is a cheap, deterministic faithfulness proxy: *does the answer
  reuse the span actually present in the doc?* Stronger than keyword presence.

### 1.3 Persona / robustness variants

The same `intent` is reworded for different patient profiles, so quality is
measured across the real user population, not only clean queries:

- `general` — clean, well-formed.
- `mayor` — elderly phrasing, no accents, short sentences.
- `joven` — colloquial / WhatsApp style, no opening `¿¡`.
- `ansioso` — anxious, several doubts crammed into one long question.
- `baja_alfabetizacion` — low-literacy phrasing, frequent misspellings.
- `L2` — non-native Spanish (gender/preposition errors, limited vocabulary).

**Frame this as invariance testing** (à la CheckList, Ribeiro et al. 2020):
rewording the question for a different profile **must not** change whether the
answer is correct or whether a refusal is correct. The metric is then *robustness
= the drop in coverage/faithfulness when the profile varies while the intent is
held fixed.* It directly exposes whether the system degrades for the most
vulnerable users — a relevant equity signal, and a publishable angle.

**Cost discipline.** Do not expand 109 × 6 profiles up front. Pick an
**invariance subset**: ~10 core intents × 4 profiles, each profile a reworded
twin of the same `general` intent, and measure the robustness drop. If the drop
is significant, that is a finding; if not, you did not spend on 600 questions.
Currently `profile` is 100% `general` — this is the dimension with the most
upside still unexercised.

### 1.4 Dataset generation

Datasets are LLM-generated from a documented, **versioned** prompt (intent list +
profile + answerable target), then human-reviewed. The generation prompt that
seeds realistic patient phrasing — misspellings, no `¿¡`, colloquial, vague — is
maintained alongside the datasets (see gpu-rag `docs/generate_eval_queries.md`
for the lineage). LLMs tend to write "too clean" even when asked to be messy:
review and rewrite artificial-feeling queries. As real user-testing queries
appear, fold them in — they beat synthetic ones.

---

## 2. Outcome taxonomy

Beyond scalar scores, each answer is bucketed into a qualitative outcome
(inherited from gpu-rag, where it proved diagnosable):

**Answerable questions**

| Outcome | Meaning |
|---------|---------|
| `GOOD` | Correct answer, grounded in the doc. |
| `PARTIAL` | Relevant but incomplete. |
| `MISS` | Wrong or does not address the question. |
| `FALSE_REFUSAL` | Refuses although the information was present. |

**Non-answerable questions**

| Outcome | Meaning |
|---------|---------|
| `OK_REF` | Refuses correctly ("No tengo información sobre eso"). |
| `LEAK` | Hallucinates an answer from model knowledge. |

`LEAK` on the gray-zone set is the single most important number for medical
safety.

---

## 3. Metrics

### 3.1 Deterministic (no LLM — fast, cheap, the CI gate)

Implemented today in `tools/run_eval.py`:

- **Keyword coverage** — fraction of `expected_keywords` present in the answer.
  A *gate*, not the safety metric (see §0).
- **Refusal correctness** — for `answerable: false`, did the system refuse?

**Known weakness to fix (gap #2).** Refusal detection is currently an exact
substring match on `"no tengo información sobre eso"`. This is brittle in both
directions: a correct refusal phrased differently scores as `FAIL`, and a
hallucination that happens to contain the phrase scores as `OK`. Replace with a
small refusal/answer/hybrid classifier (regex set or the judge), so the metric
on the most safety-critical set is trustworthy.

If `relevant_spans` are added, **span coverage** (does the answer reuse a span
actually present in the doc?) is a stronger deterministic faithfulness proxy
than raw keyword presence.

### 3.2 LLM-as-judge (gap #1 — the safety metric)

A separate, stronger LLM scores what keywords cannot. gpu-rag already ships a
judge (`src/evaluation.py`, gpt-4o-mini, rubric-anchored 0.0–1.0 with JSON
output) — but there it scores the **retrieved chunks**. In fulldoc the "context"
is always the whole document, so the judge must score the **generated answer
against the fulldoc**:

- **Faithfulness / groundedness** — is every claim in the answer supported by
  the fulldoc text? (The TruLens "groundedness" / RAGAS "faithfulness" idea,
  applied answer-vs-doc.)
- **Relevance** — does the answer address the question?
- **Refusal quality** — for non-answerable questions, is the refusal clean (no
  hallucinated hedging, no invented detail)?

Design rules, grounded in the LLM-judge literature (MT-Bench / *Judging
LLM-as-a-Judge*, Zheng et al. 2023; G-Eval, Liu et al. 2023):

- **Judge ≥ system model.** Use a stronger external model (e.g. gpt-4o-mini or
  better) — never the CPU model judging itself (self-enhancement bias).
- **Pointwise with an anchored rubric**, not pairwise. Pairwise adds position
  bias for marginal benefit at this scale. Reuse the 0.0/0.4/0.7/1.0 anchors
  from gpu-rag's prompts.
- **Watch verbosity bias** — judges reward longer answers; the rubric must
  reward grounding, not length.
- **Version the judge prompt** alongside the datasets. A judge prompt change
  invalidates comparison across runs just like a model change does.

### 3.3 Failure-mode tags

Tag answers against known failure modes (see `docs/prompt_versions.md`) so
regressions are diagnosable, not just visible as a dropped scalar:

- Open-list expansion (`"anticoagulantes, etc."` → invented drug names).
- Meta-comments about absent information ("la información no especifica…").
- Hybrid responses (mixing answer, refusal and meta in one reply).
- Cross-section drag (merging the answer with an adjacent elaboration).

---

## 4. The protocol — three layers

A reliable, automatable methodology at this scale is three layers, run at
different cadences:

1. **Deterministic gate** (seconds, no LLM) — keyword coverage + robust refusal
   classification. Runs on every change. This is the cheap CI signal.
2. **LLM-judge** (rubric-anchored, versioned) — faithfulness, relevance,
   refusal-quality on a 1–5 / 0–1 scale with chain-of-thought. Runs before
   "promoting" a prompt or model version. One strong judge model.
3. **Human gold set** (~20–30 items, labelled once) — the calibration anchor.
   Measure judge↔human agreement (Cohen's κ / % agreement). **If κ < ~0.6 the
   judge is not trustworthy and its number is not used.** Re-check periodically
   and whenever the judge prompt or judge model changes.

Calibration is a **requirement**, not a "periodic nicety": an uncalibrated judge
score is decorative.

---

## 5. Statistical reliability (gap #3)

- With ~20 questions per set, **one question = 5%**. Treat small deltas between
  prompts/models as noise, not signal. Report the N alongside every aggregate.
- Generation is near-deterministic at `temperature=0.1`; confirm run-to-run
  stability once, or run N times and report variance. State explicitly which.
- Prefer per-category and per-outcome breakdowns over a single headline number —
  a stable aggregate can hide a regression concentrated in gray-zone `LEAK`.

---

## 6. Regression tracking (defer the heavy infra)

Versioned comparison matters; a CI gate and a dashboard do **not yet** pay off
at this cadence. Minimum viable:

- Store each scored run keyed by **(model, prompt version, fulldoc version,
  date)** — encode it in the report filename under `reports/` and diff manually.
- A CI gate that fails a change dropping faithfulness or refusal correctness
  below a threshold is the natural next step **once iteration is frequent** — not
  before.

---

## 7. Relationship to `gpu-rag` (lineage)

This framework descends from the retrieval-RAG evaluation defined in `gpu-rag`.
What carries over and what does not:

| From gpu-rag | In fulldoc (cpu-rag) |
|--------------|----------------------|
| Dataset schema, persona profiles, generation prompt | ✅ Carried over (this doc). |
| Outcome taxonomy GOOD/PARTIAL/MISS/… , OK_REF/LEAK | ✅ Adopted (§2). |
| LLM-judge (faithfulness, relevance) | ✅ Adapted — score answer-vs-fulldoc, **not** chunks (§3.2). |
| Retrieval metrics: P@3/P@5, R@5, MRR, chunk precision/recall | ❌ N/A — no retrieval; no chunks to rank. |
| RAGAS/TruLens context-precision / context-recall | ❌ N/A — context is always the whole doc; only faithfulness + answer-relevance survive. |
| `relevant_spans` exact-span matching | ◐ Optional — repurposed as a deterministic faithfulness proxy (§3.1). |

---

## 8. Work line — concrete next steps

In priority order (value / effort):

1. **Robust refusal classification** (gap #2) — replace the exact-substring match
   in `run_eval.py`. *High value / low effort.* Fixes the most safety-critical
   metric.
2. **Faithfulness judge** (gap #1) — adapt gpu-rag's `llm_judge` to score the
   generated answer against the fulldoc; version the judge prompt. *High value /
   medium effort.*
3. **Report N and variance** (gap #3) — print N and per-category/per-outcome
   breakdowns; settle determinism. *High value / ~zero effort.*
4. **Human gold set + κ** — label ~20–30 answers once; wire judge calibration.
5. **Invariance subset** — ~10 intents × 4 profiles; measure the robustness drop.
6. **Regression tracking** — `(model, prompt, fulldoc, date)` run keys; CI gate
   only once iteration is frequent.

## 9. References

- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* (2023)
  — judge biases: position, verbosity, self-enhancement.
- Liu et al., *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*
  (2023) — chain-of-thought, rubric-anchored judging.
- Ribeiro et al., *Beyond Accuracy: Behavioral Testing of NLP Models with
  CheckList* (ACL 2020) — invariance tests (the persona/robustness framing).
- Rajpurkar et al., *Know What You Don't Know: Unanswerable Questions for SQuAD*
  (SQuAD 2.0, 2018) — answerable vs. unanswerable; abstention.
- RAGAS / TruLens — RAG triad (context precision/recall, groundedness,
  answer relevance); only the answer-side half applies to fulldoc.
