"""Score a prompt variant against the baseline, over the 134 audit questions.

    uv run python tools/audit_score.py --run v13=eval/audit_replay
    uv run python tools/audit_score.py \
        --run v13=eval/audit_ab/v13 --run v14a=eval/audit_ab/v14a \
        --baseline v13 --out reports/audit_score.md

Why this is not tools/audit_triage.py
-------------------------------------

The triage carries the verdicts of the base replay *nailed into code*, one per
question, each citing the fulldoc line that decides it. That is the right shape
for reasoning about the baseline and the wrong shape for scoring a new run: it
cannot say whether a different prompt got the same question right.

So this reads the verdicts as ground truth and applies them to whatever passes
it is given:

    MUST REFUSE  = the 35 FN + {8, 52, 111, 117}   = 39
    MUST ANSWER  = the rest                        = 95

The four extras are DEF answers that were produced out of invented material
where the fulldoc supports no answer, so refusing them would have been right.
Baseline: 106/134 = 79%.

What it reports, and why each part is there
-------------------------------------------

1. Decision accuracy over all 134, as **mean and spread across seeds**, not a
   number. At t=0.1 the seed alone flips 30% of the argued questions, so a
   single draw cannot tell a prompt effect from a sampling one.
2. A **paired** diff against the baseline, gains and regressions kept apart.
   Unpaired means comparing V14's average to V13's average, which throws away
   half the power; and an aggregate hides composition — cirugia already showed
   a refusal count that stayed at 4 across nine seeds while *which* four moved.
3. Rule-boundary probes on the breaks that survive the seed sweep. Those are
   the reproducible target: a prompt change lands on them cleanly.
4. Guardrails, so a variant cannot "win" by turning chatty: refusal rate and
   share of telegraphic answers per procedure.
5. Stability: for each question, how many conditions agree on the decision.
   This is an acceptance criterion and not just a power problem — a question
   that flips is defective even when its average answer is fine.

Input shapes
------------

A run is a directory of `<procedure>.json`, in either of the two shapes we
already produce, so the scorer can be checked against the baseline today:

  * tools/audit_seed_sweep.py  — many conditions per question ("temp|seed")
  * eval/audit_replay/         — the single served draw, condition "replay"
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from audit_triage import (
    REPLAY_DIR,
    PROCEDURES,
    TELEGRAPHIC_CHARS,
    TRIAGE,
    abstained,
    deferred,
    refused,  # noqa: F401  (histórico; puntuar con abstained)
)

# DEF answers built on material the fulldoc does not carry: there was no answer
# to give, so refusing was the right call and they score as must-refuse.
ANSWERED_WITHOUT_GROUND = {8, 52, 111, 117}

MUST_REFUSE = {i for i, v in TRIAGE.items() if v[0] == "FN"} | ANSWERED_WITHOUT_GROUND
MUST_ANSWER = set(TRIAGE) - MUST_REFUSE

BASELINE_CORRECT = 106          # what the served replay scores, 79%
BASELINE_FLIP_RATE = 0.30       # argued questions that flip on seed alone, t=0.1

# Rule-boundary probes: (what a break looks like, regex that finds it, regex
# that excuses it). Only the breaks the seed sweep found stable are worth
# probing — 4, 22 and 89 did not reproduce there, so a probe on them would
# measure the build as much as the prompt. 26 is in at 3/9 because it is the
# one break that is arithmetically checkable, and its intermittency is the
# point.
#
# These match the *break*, not the question: a hit means the answer welded,
# de-scoped or re-subjected a rule. The third element is what keeps the probe
# honest — for 67 the break is not "promises premedication", it is "promises
# it without the condition", and its control pair 86 promises it *with* the
# condition. A probe that cannot tell those apart would score a fix as a
# no-change. See CONTROL_NEGATIVE and --self-check.
BOUNDARY_PROBES: dict[int, tuple[str, str, str | None]] = {
    # The `unless` used to require "regional o general" adjacent, and gemma
    # writes "anestesia regional o *anestesia* general" — the repeated noun sat
    # between the two words and the probe read a preserved disjunction as a
    # closed one. Same shape as the 108 markdown bug below: an intervening
    # token where the probe assumed adjacency. Corrected 2026-07-27.
    105: ("cierra la disyunción «regional o general» en una sola",
          r"anestesia\s+regional",
          r"regional\s+[oy]\s+(la\s+)?(anestesia\s+)?general"
          r"|general\s+[oy]\s+(la\s+)?(anestesia\s+)?regional"),
    108: ("manda al paciente ajustar el anticoagulante, sin sujeto clínico",
          r"\b(debes|tienes que|deber[áa]s)\s+(ajustar|dejar|suspender)",
          r"(el\s+)?(equipo|m[ée]dico|cirujano|an?estesi[óo]log[oa])[^.]{0,60}indicar"),
    67: ("promete la premedicación sin la condición que la acota",
         r"(dar[áa]n?|darte|dar|dan)[^.]{0,30}(medicaci[óo]n|pastilla)",
         r"(si|cuando|en casos? de)[^.]{0,40}(ansiedad|temor|nervios)"),
    87: ("traslada el «con ayuda» de sentarse a caminar",
         r"ayuda[^.]{0,40}\bcaminar|caminar[^.]{0,40}\bayuda", None),
    # Had no `unless` at all, so it fired on any mention of laparoscopy — the
    # topic, not the break. The defect is answering "what is minimally
    # invasive" *with* the description of laparoscopy and nothing else; naming
    # laparoscopy as a case after stating the genus is correct. The exemption
    # is the genus statement (not opening the cavities). Corrected 2026-07-27.
    84: ("responde el género (mínimamente invasiva) con la especie (laparoscopia)",
         r"laparoscopia|laparosc[óo]pica",
         r"(evita|evitar|sin)[^.]{0,40}abrir|no\s+se\s+abren?[^.]{0,30}cavidad"),
    29: ("convierte el criterio de alarma (>39 °C) en umbral de tratamiento",
         r"39[^.]{0,60}paracetamol|paracetamol[^.]{0,60}39", None),
    26: ("prescripción imposible: 150 semanales en sesiones de 30-45 diarios",
         r"150[^.]{0,160}30\s*[-a]\s*45", None),
}

# The other member of each control pair: same material, boundary kept. A probe
# that fires here is matching the topic instead of the break.
CONTROL_NEGATIVE = {67: 86, 87: 88}


_MARKUP = re.compile(r"[*_`]+")


def _plain(text: str) -> str:
    """Drop markdown emphasis before matching.

    Not cosmetic. The 108 probe was `debes\\s+ajustar` and the model writes
    "debes **ajustar tu medicación**", so the asterisks sat between the two
    words and the probe read a variant that had not changed the defect at all
    as having fixed it 9/9. Emphasis is formatting; a probe that sees it is
    measuring the wrong thing. `--self-check` now asserts this invariant.
    """
    return _MARKUP.sub("", text)


def broke_boundary(qid: int, answer: str) -> bool:
    what, pattern, unless = BOUNDARY_PROBES[qid]
    answer = _plain(answer)
    if unless and re.search(unless, answer, re.I):
        return False
    return bool(re.search(pattern, answer, re.I))


def telegraphic(text: str) -> bool:
    # Deferrals are short by design and are not corpus fragments, so counting
    # them here would make G2 look telegraphic for doing the thing it was
    # built to do.
    return (not abstained(text) and not deferred(text)
            and len(text.strip()) < TELEGRAPHIC_CHARS)


def load_run(root: Path, exclude: tuple[str, ...] = ()) -> dict[str, dict[int, str]]:
    """-> {condition: {qid: answer}}. A condition is one seed at one temperature.

    Both shapes collapse to the same thing: a set of passes over the questions.
    The replay is just a run with a single condition.

    `exclude` drops whole procedures. It exists for one specific, defensible
    cut: hemorroides is not our corpus and not production corpus — it arrived as
    a 1.1 KB "RESUMEN", was never distilled by us, and has no source document
    (verified 2026-07-23). Scoring without it answers "how does the system do on
    a production-grade corpus", which is a different and legitimate question.
    It is NOT a licence to report the better number without that label: dropping
    the worst document and publishing the total is cherry-picking, and the only
    thing that makes this cut honest is saying which corpus it covers.
    """
    passes: dict[str, dict[int, str]] = {}
    files = sorted(root.glob("*.json"))
    if not files:
        raise SystemExit(f"ERROR: no <procedure>.json under {root}")
    for path in files:
        if path.stem not in PROCEDURES or path.stem in exclude:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "questions" in payload:                       # seed-sweep shape
            for row in payload["questions"]:
                for cond, answer in row["answers"].items():
                    passes.setdefault(cond, {})[row["id"]] = answer or ""
        elif "rows" in payload:                          # replay shape
            for row in payload["rows"]:
                passes.setdefault("replay", {})[row["id"]] = row.get("our_answer") or ""
        else:
            raise SystemExit(f"ERROR: {path} is neither a sweep nor a replay")
    return passes


def correct(qid: int, answer: str) -> bool:
    """Three-way, because A3's G2 arm emits a third kind of answer.

    A deferral ("eso depende de tu caso, coméntalo con tu equipo") is neither a
    refusal nor an answer. Scored as: right where the document supports no
    answer — it fabricates nothing, and it is the outcome the auditors' own
    `Sense resposta` code asks for — and wrong where the document does answer,
    because there the system had the material and dodged.

    Arms that never defer are unaffected: `deferred` is false for every answer
    in every run measured before 2026-07-23, so no published number moves.
    """
    if deferred(answer):
        return qid in MUST_REFUSE
    return abstained(answer) == (qid in MUST_REFUSE)


def procedure_of(qid: int) -> str:
    if qid <= 55:
        return "diabetes"
    return "cirugia-abdominal" if qid <= 103 else "hemorroides"


def score_conditions(passes: dict[str, dict[int, str]]) -> dict[str, tuple[int, int]]:
    """-> {condition: (correct, asked)}."""
    return {
        cond: (sum(1 for q, a in answers.items() if correct(q, a)), len(answers))
        for cond, answers in passes.items()
    }


def paired_diff(base: dict[str, dict[int, str]],
                variant: dict[str, dict[int, str]]) -> tuple[list, list, list[str]]:
    """Gains and regressions per (question, condition), over shared conditions.

    Only conditions both runs actually ran are paired, which is what makes the
    frozen seed safe to include. It comes out of the snapshot pickle, so one
    might expect it to differ per variant — it does not: the stored seed is
    `LLAMA_DEFAULT_SEED` chained by warmup order *within the process*, and every
    A/B cell is a fresh process warming one procedure, so all of them land on
    position 1 and share generation seed 1894574933 (measured, job 7289). It
    therefore pairs like any other condition. If a future harness warms several
    procedures per process, the stored seeds diverge by position and the
    intersection here silently drops the unmatched ones rather than comparing
    two different seeds as if they were one.
    """
    shared = sorted(set(base) & set(variant))
    gains, regressions = [], []
    for cond in shared:
        for qid, answer in variant[cond].items():
            if qid not in base[cond]:
                continue
            was, now = correct(qid, base[cond][qid]), correct(qid, answer)
            if now and not was:
                gains.append((qid, cond))
            elif was and not now:
                regressions.append((qid, cond))
    return gains, regressions, shared


def stability(passes: dict[str, dict[int, str]]) -> dict[int, tuple[int, int]]:
    """-> {qid: (conditions agreeing with the majority decision, conditions)}."""
    per_q: dict[int, list[bool]] = {}
    for answers in passes.values():
        for qid, answer in answers.items():
            per_q.setdefault(qid, []).append(abstained(answer))
    out = {}
    for qid, decisions in per_q.items():
        majority = max(decisions.count(True), decisions.count(False))
        out[qid] = (majority, len(decisions))
    return out


def emit(runs: dict[str, dict[str, dict[int, str]]], baseline: str) -> list[str]:
    L: list[str] = []
    add = L.append

    add("# Puntuación de variantes de prompt\n")
    add(f"Verdad de terreno: **{len(MUST_REFUSE)} rechazables** "
        f"(los {len(MUST_REFUSE) - len(ANSWERED_WITHOUT_GROUND)} FN + "
        f"{sorted(ANSWERED_WITHOUT_GROUND)}), **{len(MUST_ANSWER)} respondibles**. "
        f"Línea base servida: {BASELINE_CORRECT}/134 = "
        f"{BASELINE_CORRECT / 134:.0%}.\n")

    # --- 1. decision accuracy ------------------------------------------------
    add("## 1. Acierto de decisión\n")
    add("| variante | condiciones | preguntas | acierto medio | min–max |")
    add("| --- | ---: | ---: | ---: | --- |")
    for label, passes in runs.items():
        scored = score_conditions(passes)
        rates = [c / n for c, n in scored.values() if n]
        asked = max(n for _, n in scored.values())
        spread = (f"{min(rates):.1%}–{max(rates):.1%}" if len(rates) > 1 else "—")
        mean = statistics.fmean(rates)
        note = ""
        if asked == len(TRIAGE) and len(rates) == 1:
            note = f"  ({round(mean * asked)}/{asked})"
        add(f"| {label} | {len(rates)} | {asked} | {mean:.1%}{note} | {spread} |")
    add("")
    if any(len(p) == 1 for p in runs.values()):
        add("> Una sola condición es **una tirada**: a t=0,1 el seed por sí solo "
            "voltea el 30 % de las preguntas discutidas. No comparar variantes así.\n")

    # --- 2. paired diff ------------------------------------------------------
    add(f"## 2. Diff emparejado contra `{baseline}`\n")
    for label, passes in runs.items():
        if label == baseline:
            continue
        gains, regressions, shared = paired_diff(runs[baseline], passes)
        if not shared:
            add(f"**{label}**: sin condiciones comunes con `{baseline}` — "
                f"nada que emparejar.\n")
            continue
        add(f"**{label}** — {len(shared)} condiciones comunes de "
            f"{len(set(runs[baseline]) | set(passes))}; las no compartidas se "
            f"descartan en vez de emparejar seeds distintos.\n")
        add(f"- gana **{len(gains)}** (pregunta, condición): "
            f"{sorted({q for q, _ in gains})}")
        add(f"- rompe **{len(regressions)}**: {sorted({q for q, _ in regressions})}")
        add(f"- neto: **{len(gains) - len(regressions):+d}**\n")

        # Split by direction, because the net hides the mechanism. A variant
        # that gains on answerable questions and loses on refusable ones in
        # roughly equal measure has not learned to tell them apart: it has
        # moved a single threshold, and the net is then an accident of how the
        # 95/39 split happens to fall.
        g_ref = sum(1 for q, _ in gains if q in MUST_REFUSE)
        r_ref = sum(1 for q, _ in regressions if q in MUST_REFUSE)
        add(f"|  | responde menos de lo debido | rechaza menos de lo debido |")
        add(f"| --- | ---: | ---: |")
        add(f"| gana | {len(gains) - g_ref} | {g_ref} |")
        add(f"| rompe | {len(regressions) - r_ref} | {r_ref} |")
        ratio = (len(gains) - g_ref) / r_ref if r_ref else float("inf")
        add(f"\nGana {len(gains) - g_ref} sobre-rechazos y paga {r_ref} rechazos "
            f"correctos: **{ratio:.1f}×**. Cerca de 1× es mover el umbral, no "
            f"discriminar mejor.\n")
        add("Una variante que gana 6 y pierde 5 no es progreso. Leer el texto de "
            "las que se mueven; el diff es pequeño a propósito.\n")

    # --- 3. boundary probes --------------------------------------------------
    add("## 3. Fronteras rotas (probes)\n")
    add("| id | qué mide | " + " | ".join(runs) + " |")
    add("| ---: | --- | " + " | ".join("---:" for _ in runs) + " |")
    for qid, (what, _, _) in BOUNDARY_PROBES.items():
        cells = []
        for passes in runs.values():
            hits = tot = 0
            for answers in passes.values():
                if qid in answers:
                    tot += 1
                    hits += broke_boundary(qid, answers[qid])
            cells.append(f"{hits}/{tot}" if tot else "—")
        add(f"| {qid} | {what} | " + " | ".join(cells) + " |")
    add("\nMenos es mejor: cada celda es en cuántas condiciones el probe ve la "
        "frontera rota. Un probe es una regex — confirma leyendo antes de "
        "cantar victoria.\n")

    # --- 4. guardrails -------------------------------------------------------
    add("## 4. Guardarraíles\n")
    add("| variante | procedimiento | rechazo | derivación | telegráficas |")
    add("| --- | --- | ---: | ---: | ---: |")
    for label, passes in runs.items():
        for proc in PROCEDURES:
            n = ref = dfr = tel = 0
            for answers in passes.values():
                for qid, answer in answers.items():
                    if procedure_of(qid) != proc:
                        continue
                    n += 1
                    ref += abstained(answer)
                    dfr += deferred(answer)
                    tel += telegraphic(answer)
            if n:
                add(f"| {label} | {proc} | {ref / n:.0%} | {dfr / n:.0%} "
                    f"| {tel / n:.0%} |")
    add(f"\nTelegráfica = respuesta no rechazo de menos de {TELEGRAPHIC_CHARS} "
        "caracteres. Sirve para pillar la variante que «mejora» volviéndose "
        "charlatana, o la que responde con una línea del documento.\n")
    add("Derivación = la tercera salida («eso depende de tu caso»). Es el "
        "guardarraíl de G2: derivar es correcto donde el documento no responde "
        "y es una escapatoria donde sí responde, así que una tasa alta en "
        "**todos** los procedimientos significa que la variante ha aprendido a "
        "esquivar, no a distinguir. Léela junto a la columna de acierto, nunca "
        "sola.\n")

    # --- 5. stability --------------------------------------------------------
    add("## 5. Estabilidad frente al seed\n")
    add("| variante | preguntas con >1 condición | voltean | tasa |")
    add("| --- | ---: | ---: | ---: |")
    for label, passes in runs.items():
        st = stability(passes)
        multi = {q: v for q, v in st.items() if v[1] > 1}
        flipped = [q for q, (agree, tot) in multi.items() if agree < tot]
        rate = f"{len(flipped) / len(multi):.0%}" if multi else "—"
        add(f"| {label} | {len(multi)} | {len(flipped)} | {rate} |")
    add(f"\nLínea base a batir: **{BASELINE_FLIP_RATE:.0%}** de volteo sobre las "
        "preguntas discutidas (diabetes 33 %, cirugía 13 %, hemorroides 42 %). "
        "Subir el acierto medio bajando la estabilidad no vale.\n")
    for label, passes in runs.items():
        st = stability(passes)
        unstable = sorted(q for q, (agree, tot) in st.items() if tot > 1 and agree < tot)
        if unstable:
            add(f"- `{label}` voltea: {unstable}")
    add("")
    return L


def scorecard(replays: Path, exclude: tuple[str, ...] = ()) -> int:
    """XX: what share of the 134 the delivered system answers acceptably.

    Two standards, because they answer different questions and the gap between
    them is the honest part:

      A. **correctness** — right given the corpus (verdict OK) plus refusing
         what the corpus cannot support (FN, actually refused). This is the
         headline, and the one that answers what the auditors asked.
      B. **presentable** — the same, minus answers that are correct but
         telegraphic. Whether a one-line answer is acceptable to a patient is a
         judgement; this reports both rather than choosing silently.

    Their audit scored 9% acceptable against ADA/ERAS/ASA/ASCRS. The whole gap
    is the anchor, and it is defensible question by question because every
    verdict in audit_triage.TRIAGE cites the corpus line that decides it.

    Limit, and it must be stated wherever this number is: the decision layer is
    measured over 9 seeds, but this one is a **single draw**. "Correct given the
    corpus" is a judgement on a specific text, not a computable predicate, so it
    cannot be re-derived for a different seed without re-reading. With ~30% of
    argued questions flipping on the seed, XX has a band this does not show.
    """
    passes = load_run(replays, exclude)
    (cond,) = passes
    answers = passes[cond]

    rows = []
    for proc, lo, hi in (("diabetes", 1, 55), ("cirugia-abdominal", 56, 103),
                         ("hemorroides", 104, 134)):
        if proc in exclude:
            continue
        ids = [i for i in TRIAGE if lo <= i <= hi]
        ok = [i for i in ids if TRIAGE[i][0] == "OK"]
        ref = [i for i in ids if TRIAGE[i][0] == "FN" and abstained(answers[i])]
        tel = [i for i in ok if telegraphic(answers[i])]
        rows.append((proc, len(ids), len(ok), len(ref), len(tel)))

    tot = [sum(r[i] for r in rows) for i in range(1, 5)]
    print(f"XX sobre las {tot[0]} preguntas de la auditoría, "
          f"veredictos de audit_triage.TRIAGE\n")
    print(f"{'':>18} {'n':>4} {'correctas':>10} {'rechazos ok':>12} "
          f"{'telegráficas':>13} {'A corrección':>13} {'B presentable':>14}")
    for proc, n, ok, ref, tel in rows:
        print(f"{proc:>18} {n:>4} {ok:>10} {ref:>12} {tel:>13} "
              f"{(ok + ref) / n:>12.0%} {(ok - tel + ref) / n:>13.0%}")
    n, ok, ref, tel = tot
    print(f"{'TOTAL':>18} {n:>4} {ok:>10} {ref:>12} {tel:>13} "
          f"{(ok + ref) / n:>12.0%} {(ok - tel + ref) / n:>13.0%}")
    print(f"\nA) {ok + ref}/{n} = {(ok + ref) / n:.0%} — correcta según el corpus "
          f"o correctamente rechazada.")
    print(f"B) {ok - tel + ref}/{n} = {(ok - tel + ref) / n:.0%} — además no "
          f"telegráfica (<{TELEGRAPHIC_CHARS} car).")
    print("\nUna sola tirada: la calidad no es un predicado computable y no se "
          "puede promediar sobre seeds sin releer.")
    return 0


def self_check(replays: Path) -> int:
    """Score the served replay and assert it reproduces what we already know.

    Everything here is a number this repo has argued about elsewhere, so a
    silent drift in the ground truth, the refusal detector or a probe regex
    shows up as a failure instead of as a plausible-looking variant delta.
    """
    passes = load_run(replays)
    (cond,) = passes
    answers = passes[cond]
    fails = []

    got = sum(1 for q, a in answers.items() if correct(q, a))
    print(f"decision baseline: {got}/{len(answers)} (want {BASELINE_CORRECT}/134)")
    if (got, len(answers)) != (BASELINE_CORRECT, 134):
        fails.append("baseline decision accuracy moved")

    for qid in BOUNDARY_PROBES:
        hit = broke_boundary(qid, answers[qid])
        # 26 is the intermittent one (3/9 across seeds), but the served draw is
        # the one the triage quotes, so it must fire here too.
        print(f"probe {qid:>3}: {'HIT' if hit else 'miss'} on the break")
        if not hit:
            fails.append(f"probe {qid} does not see the break it documents")

    # Emphasis must not change a verdict. The model bolds the key phrase often,
    # and a probe that matches across a word boundary silently stops firing.
    emphasise = lambda t: re.sub(r"(\w+)", r"**\1**", t)  # noqa: E731
    for qid in BOUNDARY_PROBES:
        plain = broke_boundary(qid, answers[qid])
        bold = broke_boundary(qid, emphasise(answers[qid]))
        print(f"probe {qid:>3}: {'stable' if plain == bold else 'MOVES'} under markdown")
        if plain != bold:
            fails.append(f"probe {qid} changes verdict when the answer is bolded")

    for qid, control in CONTROL_NEGATIVE.items():
        hit = broke_boundary(qid, answers[control])
        print(f"probe {qid:>3}: {'HIT' if hit else 'miss'} on control {control}")
        if hit:
            fails.append(f"probe {qid} also fires on its control pair {control} "
                         f"— it is matching the topic, not the break")

    if fails:
        print("\nFAILED:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nself-check ok")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scorecard", action="store_true",
                   help="XX: share of the 134 answered acceptably, by the two "
                        "standards. Reads the served replay.")
    p.add_argument("--self-check", action="store_true",
                   help="Score the served replay and assert the numbers this "
                        "repo already argues about. Run it after touching a "
                        "probe or the ground truth.")
    p.add_argument("--run", action="append", metavar="LABEL=DIR",
                   help="A run to score. Repeat. The first is the baseline "
                        "unless --baseline says otherwise.")
    p.add_argument("--baseline", default=None)
    # `action="append"` and not a plain option: repeating the flag used to let
    # the last one win silently, so asking to drop two procedures dropped one
    # and reported a number for a corpus nobody had asked for. Both spellings
    # now accumulate — repeated flags and comma-separated lists.
    p.add_argument("--exclude-procedure", action="append", default=[],
                   help="Procedure to drop, e.g. 'hemorroides' to score only "
                        "the production-grade corpus. Repeatable, and accepts "
                        "a comma-separated list. Always report which corpus "
                        "the number covers.")
    p.add_argument("--out", default=None, help="Write markdown here too.")
    args = p.parse_args()

    exclude = tuple(dict.fromkeys(
        s.strip()
        for chunk in args.exclude_procedure
        for s in chunk.split(",")
        if s.strip()
    ))
    for name in exclude:
        if name not in PROCEDURES:
            raise SystemExit(f"ERROR: --exclude-procedure {name!r} is not one "
                             f"of {list(PROCEDURES)}")

    if args.scorecard:
        return scorecard(Path(REPLAY_DIR), exclude)
    if args.self_check:
        if exclude:
            raise SystemExit("ERROR: --self-check asserts the full-134 numbers; "
                             "it cannot run with --exclude-procedure.")
        return self_check(Path(REPLAY_DIR))
    if not args.run:
        raise SystemExit("ERROR: need at least one --run LABEL=DIR "
                         "(or --self-check / --scorecard)")

    runs: dict[str, dict[str, dict[int, str]]] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"ERROR: --run wants LABEL=DIR, got {spec!r}")
        label, _, path = spec.partition("=")
        runs[label] = load_run(Path(path), exclude)

    baseline = args.baseline or next(iter(runs))
    if baseline not in runs:
        raise SystemExit(f"ERROR: --baseline {baseline!r} is not one of {list(runs)}")

    report = "\n".join(emit(runs, baseline))
    print(report)
    if args.out:
        dest = Path(args.out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(report, encoding="utf-8")
        print(f"\nWrote {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
