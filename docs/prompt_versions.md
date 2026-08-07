# System Prompt — State & Lessons

Current: **V13** (2026-05-12, fulldoc single-document mode, INFORMACIÓN framing).
SP size: 915 tokens. Canonical text: `app/prompt.py::SYSTEM_PROMPT_TEMPLATE`.

---

## V13 — current state

**Goal**: extractive Q&A over a single distilled procedure document (fulldoc
bypass, no retrieval). One model, one document, one query → one answer.

**Structure** (in order):
1. Role: "asistente médico ... lector que solo dice lo que el texto dice".
2. REGLA: binary decision — info literal → respond `lo justo, empezando por el
   hecho, sin elaborar`; otherwise EXACTAMENTE y sin añadidos
   `"No tengo información sobre eso."`.
3. PROHIBIDO (5 bullets): inventar (incluye equivalencias), completar con
   conocimiento general, expandir listas/categorías, comentar lo que NO dice,
   preámbulos meta.
4. 6 EJEMPLOS:
   - Ej 1: positive answer + anti-equivalence note (mg/dl no convertir a mmol/L)
   - Ej 2: refusal when topic similar but answer absent
   - Ej 3: OOS / non-medical refusal
   - Ej 4: category w/o enumeration — generic question → reproduce category;
     concrete question ("frutas concretas") → refusal
   - Ej 5: respond only what's there, no commenting on absence
   - Ej 6: don't drag adjacent topics from same passage

**User template**: `INFORMACIÓN:\n{context}\n\nPREGUNTA: {query}`.

**Eval results** (diabetes, distilled `_3429.md`, dataset 38 queries):
- 36/38 correct. Q15 minor drift (refusal includes the topic noun "dosis de
  insulina"). Q35 partial meta-sneak + adjacent-section drag.
- Gen speed 4.2–4.6 tok/s, TTFT <1s, warmup ~80s for a 3,429t doc.

---

## Lessons learned (recurring patterns confirmed across V6→V13)

Load-bearing insights for any future prompt work on 3B-class models
(Ministral-3-3B-Q4_K_M):

1. **Few-shot beats declarative for behavior shaping.** Adding/expanding
   declarative rules ("don't do X") to fix a failure mode rarely works; modeling
   the right behavior in an example does. Validated multiple times.

2. **String menus in PROHIBIDO become repertoire.** Listing literal forbidden
   phrases ("según los fragmentos…", "no se menciona…") trains the model to
   produce them. V10 listed "según la información…" and the model leaked it 5×;
   V11 removed the literal string and the leak disappeared.

3. **Pick a source noun whose residual leak reads naturally.** Renaming
   `FRAGMENTOS`→`DOCUMENTO`→`INFORMACIÓN` means that when a meta-leak does occur
   ("Según la información disponible…") it is almost imperceptible vs the cold
   "según los fragmentos". Choose vocabulary for graceful failure.

4. **Anti-X rule words alone don't activate concepts.** PROHIBIDO listing
   "equivalencias" did not stop the model inserting "(7 mmol/L)" after mg/dl.
   Only rewriting Ej 1 with an explicit anti-equivalence note changed it.

5. **Generic↔concrete distinction needs explicit modeling.** The same category
   ("fruta") yields different correct responses for "¿qué fruta?" (reproduce
   category) vs "¿qué frutas concretas?" (refusal). Ej 4 V13 models both branches
   in one example.

6. **Brevity instructions drift the exact refusal phrase.** Asking for brief
   answers + topical relevance → model rewrites "No tengo información sobre eso."
   as "No tengo información sobre marcas/dosis…". Counter with an explicit REGLA
   "sin añadidos … No precises el tema en la frase de rechazo".

7. **Cross-section drag is hard to eliminate.** When the doc has the answer in
   section A but a related elaboration in section B, the model often merges both.
   Ej 6 covers time-separated drag (pre/post op) but not categorical drag.
   Accepted as a residual.

8. **Prompt size is a real lever for gen speed.** V8 1,204t → V13 915t correlated
   with measurable speedup (~3.0→~4.3 tok/s). Below ~1K SP tokens marginal
   returns flatten — don't optimize past the point where wording quality suffers.

9. **Conversational edges ("hola"/"gracias") are stateless-by-design.** No prompt
   change fully fixes them because the system is single-turn. Needs a UX layer
   (frontend conversation manager) or stateful session, not prompt work.

10. **Use a single doc, not multiple chunks, where possible.** Adopting fulldoc +
    distilled markdown made the "FRAGMENTOS plural" framing obsolete and
    simplified model behavior. Removing retrieval also eliminated the
    ranking/threshold tuning that consumed earlier rounds.

---

## Trajectory (fulldoc era)

| Ver | Key idea | SP toks | Status |
|---|---|---|---|
| V9  | First fulldoc-aware: `FRAGMENTOS`→`DOCUMENTO`, +category few-shot | 1,331 | Won Q16, leaked "según el documento" |
| V10 | `DOCUMENTO`→`INFORMACIÓN` + global compaction | 1,011 | Leak reads natural |
| V11 | Drop string menu from anti-meta + (Nota) anchor in Ej 1 + prune examples | 845 | Leak resolved; output more verbose |
| V12 | Brevity push in REGLA + new Ej 1 anti-equivalence | 858 | Brevity win, drift on exact refusal |
| **V13** | "Sin añadidos" in REGLA + Ej 4 genérico↔concreto contrast | 915 | **36/38; current** |

Pre-fulldoc versions (V1–V8) targeted the retrieval-mode multi-chunk pipeline.
V8 (2026-05-06) was the last retrieval-mode prompt and the base from which the
fulldoc-mode V9 forked. That history lives in git; do not re-derive it for the
cpu-rag fulldoc path — the retrieval-mode literature belongs to
`gpu-rag-deprecated` (the old retrieval app, not the v3.0 GPU service that reused
the `gpu-rag` name in 2026-07).
