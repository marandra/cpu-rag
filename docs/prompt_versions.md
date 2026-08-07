# System Prompt — state & lessons

What is served today, and the lessons that still constrain prompt work. Canonical
text: `app/prompt.py::SYSTEM_PROMPT_TEMPLATE`; the variant machinery
(`get_system_prompt(procedure, variant)`, `_replacing`, `_plus_example`) lives in
the same file, and each variant carries its own rationale in a comment above it.

## What is served

**V13 base + the `d1c-tu` abstention literal** (`prompt_variant` in
`app/config.py`). 915 tokens, procedure-agnostic, temperature 0.1, fulldoc
single-document mode with `INFORMACIÓN` framing.

Structure, in order:

1. **Role** — "asistente médico … lector que solo dice lo que el texto dice".
2. **REGLA** — binary decision: information present → answer `lo justo, empezando
   por el hecho, sin elaborar`; otherwise the abstention literal, exactly and
   without additions.
3. **PROHIBIDO**, 5 bullets — inventing (including unit equivalences), completing
   from general knowledge, expanding lists/categories, commenting on what the text
   does *not* say, meta preambles.
4. **6 examples** —
   - Ej 1: positive answer + anti-equivalence note (mg/dl not converted to mmol/L)
   - Ej 2: refusal when the topic is similar but the answer is absent
   - Ej 3: out-of-scope / non-medical refusal
   - Ej 4: category without enumeration — generic question reproduces the
     category, concrete question ("frutas concretas") refuses
   - Ej 5: answer only what is there; do not comment on absence
   - Ej 6: do not drag adjacent topics from the same passage

**User template**: `INFORMACIÓN:\n{context}\n\nPREGUNTA: {query}`.

The `d1c-tu` literal is short and inverted, and it tutea because **the corpus
does**: register is set by the documents, not by the prompt, so the literal has to
turn with them. It reaches all 42 abstentions, 24 of them emotional, with no loss
of accuracy and 18% fewer output tokens than the previous phrasing.

Quality of the served combination is measured by reading the 134 questions, not
from this document — see `docs/auditoria_134_v22.md` and `docs/estado.md`.

## Two limits of the prompt as a lever

Both measured on gemma-4-26B, and they bound what any future variant can achieve:

- **Over-refusal and invention are one threshold.** Pushing the model to answer
  more makes it invent more. No A/B variant escaped this.
- **Shortening is not free.** A shorter prompt ties on decision and saves 229
  tokens, but drives hemorroides telegraphic answers from 23% to 58%.

## Lessons that still hold

Derived across V6→V13 on Ministral-3-3B-Q4_K_M and not re-measured one by one on
gemma; treat them as priors for prompt work, not as current measurements. The two
limits above *were* re-measured on gemma and are stated separately for that
reason.

1. **Few-shot beats declarative for behaviour shaping.** Adding declarative rules
   ("don't do X") to fix a failure mode rarely works; modelling the right
   behaviour in an example does. Validated repeatedly.

2. **String menus in PROHIBIDO become repertoire.** Listing literal forbidden
   phrases trains the model to produce them. One version listed "según la
   información…" and the model leaked it 5×; removing the literal string removed
   the leak.

3. **Pick a source noun whose residual leak reads naturally.** With `INFORMACIÓN`,
   a meta-leak ("Según la información disponible…") is nearly imperceptible, where
   "según los fragmentos" was jarring. Choose vocabulary for graceful failure.

4. **Anti-X rule words alone do not activate concepts.** PROHIBIDO listing
   "equivalencias" did not stop "(7 mmol/L)" appearing after mg/dl. Only rewriting
   Ej 1 with an explicit anti-equivalence note changed it.

5. **The generic↔concrete distinction needs explicit modelling.** The same
   category ("fruta") demands different correct responses for "¿qué fruta?"
   (reproduce the category) and "¿qué frutas concretas?" (refuse). Ej 4 models both
   branches in one example.

6. **Brevity instructions drift the exact refusal phrase.** Asking for brief,
   topically relevant answers makes the model rewrite the abstention literal to
   name the topic. Counter it explicitly in REGLA.

7. **Cross-section drag is hard to eliminate.** When the answer is in section A
   and a related elaboration in section B, the model often merges both. Ej 6 covers
   time-separated drag (pre/post op) but not categorical drag. Accepted as residual.

8. **Prompt size is a real lever for generation speed**, but with a floor: below
   ~1K tokens the returns flatten and wording quality starts to pay for it. See
   the shortening limit above for what that costs downstream.

9. **Conversational edges ("hola", "gracias") are stateless by design.** No prompt
   change fully fixes them, because the system is single-turn. They need a UX layer
   or a stateful session, not prompt work.

10. **One document beats multiple chunks where it fits.** Fulldoc + distilled
    markdown made the "FRAGMENTOS plural" framing obsolete and removed the
    ranking/threshold tuning that consumed earlier rounds.

Pre-fulldoc prompt versions (V1–V8) targeted a retrieval-mode multi-chunk pipeline
that this repo no longer has. That history is in git; do not re-derive it for the
fulldoc path. The retrieval-mode literature belongs to `gpu-rag-deprecated` — the
old retrieval app, not the v3.0 GPU service that reused the `gpu-rag` name in
2026-07.
