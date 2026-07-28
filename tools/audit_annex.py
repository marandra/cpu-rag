"""Genera los anexos que se mandan al cliente.

    python3 tools/audit_annex.py

  1. `docs/auditoria_134_evaluacion.md` — SU lista (sus preguntas, la respuesta
     que auditaron, su puntuación) con nuestra evaluación añadida en paralelo.
  2. `docs/auditoria_134_v2.md` — las mismas 134 respondidas por la v2 (modelo
     + corpus nuevos), con nuestra evaluación.
  3. `docs/auditoria_134_v22.md` — las mismas 134 respondidas por la **v2.2**,
     que es la que se entrega: corpus y abstención en tú. Este es el que va
     adjunto al correo; el de la v2 se mantiene como historia.

Un solo criterio en los dos: **correcta / incorrecta**. Correcta = responde lo
que se pregunta apoyada en el documento, sin inventar y sin fundir ni des-acotar
una regla; o se abstiene donde el documento no da material.

Sin lógica de puntuación propia. Los veredictos ya existen: los de la v1.1 en
`audit_triage.TRIAGE` (leídos a mano en su día), los de la v2 en
`audit_hand.HAND` y los de la v2.2 en `audit_hand_v22.READ`. Esto los junta con
las preguntas y los formatea, saneando la jerga interna — los ficheros salen
fuera.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_hand import verdicts  # noqa: E402
from audit_hand_v22 import V22_DIR  # noqa: E402
from audit_hand_v22 import verdicts as verdicts_v22  # noqa: E402
from audit_score import MUST_REFUSE, load_run  # noqa: E402
from audit_triage import PROCEDURES, TRIAGE, refused  # noqa: E402

BLOCKS = (("diabetes", "Diabetes", 1, 55),
          ("cirugia-abdominal", "Cirugía abdominal", 56, 103),
          ("hemorroides", "Hemorroides", 104, 134))

# Las 9 donde la respuesta que registrasteis y la que produjo nuestra
# reproducción no coinciden: la versión auditada no era determinista.
NONDET = {13, 22, 23, 39, 40, 42, 111, 115, 117}

# Motivos reescritos para salir fuera: los originales se refieren al cliente en
# tercera persona o comparan con nuestra propia reproducción.
OVERRIDE = {
    22: "Cruza dos categorías de §Grupos de alimentos: el documento dice «moderar: "
        "edulcorantes» y la respuesta manda evitarlos, contradiciéndose dentro de la "
        "misma frase.",
    23: "Responde con la individualización del §Tratamiento farmacológico.",
    42: "Añade «No tengo información sobre eso» después de haber respondido.",
    67: "Pierde la condición: §Premedicación dice «cuando el grado de ansiedad y temor "
        "sea elevado, le darán medicación», y la respuesta promete la premedicación sin "
        "condicionarla.",
    84: "Responde «qué es mínimamente invasiva» con la descripción de la laparoscopia; "
        "el documento las distingue como género y caso.",
    96: "Añade «la ansiedad es normal» antes de la premedicación.",
    115: "Responde desde §Riesgos, «dolor al defecar».",
}


def clean(text: str) -> str:
    """Saca la jerga interna del motivo antes de que salga del equipo."""
    text = re.sub(r"\s*Reclasificada desde \w+ por el barrido de fronteras\.", "", text)
    text = text.replace("fulldoc", "documento").replace("**", "")
    # Los veredictos OK empiezan por «Correcta;», que en el anexo va delante como
    # marca y quedaría repetido.
    text = re.sub(r"^Correcta( y completa)?[;,]?\s*", "", text.strip())
    return (text[:1].upper() + text[1:]).strip().rstrip(".")


def load_meta() -> dict[int, dict]:
    meta = {}
    for proc in PROCEDURES:
        for row in json.loads(Path(f"eval/ec2/{proc}.json").read_text())["rows"]:
            meta[row["id"]] = row
    return meta


def old_verdict(qid: int, answer: str) -> tuple[bool, str]:
    """Nuestra lectura de la respuesta que auditaron."""
    kind, _, evidence = TRIAGE[qid]
    correcta = kind == "OK" or (kind == "FN" and refused(answer))
    return correcta, clean(OVERRIDE.get(qid, evidence))


def head(title: str, ok: int, n: int, marginales: int = 0) -> list[str]:
    """Solo criterio y resultado. Las aclaraciones viven en el correo."""
    L = [
        f"# {title}", "",
        "**Criterio.** Correcta = responde lo que se pregunta apoyada en el documento del "
        "procedimiento, sin inventar y sin fundir ni des-acotar una regla; o se abstiene "
        "donde el documento no da material. Abstenerse bien cuenta como correcta: es la "
        "conducta diseñada.",
        "",
        f"**Resultado: {ok} de {n} correctas ({ok / n:.0%}).**",
    ]
    if marginales:
        # El filo es parte del resultado: si se descuentan, sale el extremo bajo.
        L += [
            "",
            f"De esas {ok}, **{marginales} quedan «en el filo»**: apoyadas en el documento y "
            f"sin inventar, pero sin responder del todo lo que se preguntaba. Van marcadas una "
            f"a una. Si se descuentan todas, el resultado es {ok - marginales} de {n} "
            f"({(ok - marginales) / n:.0%}); la horquilla es "
            f"{(ok - marginales) / n:.0%}–{ok / n:.0%}.",
        ]
    return L + ["", "---", ""]


def emit(path: Path, title: str, answers: dict[int, str], answer_label: str,
         verdict, meta: dict[int, dict], with_score: bool) -> None:
    ids = sorted(TRIAGE)
    ok = sum(1 for q in ids if verdict(q, answers[q])[0])
    marg = sum(1 for q in ids
               if verdict(q, answers[q])[0] and q in getattr(verdict, "marginales", ()))

    L = head(title, ok, len(ids), marg)
    for proc, label, lo, hi in BLOCKS:
        block = [q for q in ids if lo <= q <= hi]
        good = sum(1 for q in block if verdict(q, answers[q])[0])
        L += [f"## {label}", "",
              f"*{good} de {len(block)} correctas ({good / len(block):.0%}).*", ""]
        for qid in block:
            row = meta[qid]
            correcta, why = verdict(qid, answers[qid])
            marca = "Correcta" if correcta else "Incorrecta"
            if correcta and qid in getattr(verdict, "marginales", ()):
                marca = "Correcta, en el filo"
            L += [f"### {qid}", "",
                  f"**Pregunta.** {row['question']}", "",
                  f"**{answer_label}.** {(answers[qid] or '').strip()}", ""]
            if with_score:
                L += [f"**Vuestra puntuación.** {row.get('score', '?')}/10, "
                      f"{row.get('verdict', '?')}.", ""]
            L += [f"**Nuestra evaluación.** {marca}. {why}.", ""]

    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"escrito {path} — {ok}/{len(ids)} = {ok / len(ids):.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args()
    out = Path(args.outdir)

    meta = load_meta()
    new = load_run(Path("eval/ec2"))["replay"]
    hand = verdicts(new)

    def hand_verdict(qid: int, _answer: str) -> tuple[bool, str]:
        correcta, _marginal, why = hand[qid]
        return correcta, clean(why)

    hand_verdict.marginales = {q for q in hand if hand[q][0] and hand[q][1]}

    emit(
        out / "auditoria_134_evaluacion.md",
        "Las 134 preguntas, con nuestra evaluación",
        {q: (meta[q].get("their_answer") or "") for q in TRIAGE},
        "Respuesta de la v1.1",
        old_verdict, meta, with_score=True,
    )

    emit(
        out / "auditoria_134_v2.md",
        "Las 134 preguntas respondidas por la v2",
        new, "Respuesta de la v2", hand_verdict, meta, with_score=False,
    )

    v22 = load_run(Path(V22_DIR))["replay"]
    hand22 = verdicts_v22(v22)

    def v22_verdict(qid: int, _answer: str) -> tuple[bool, str]:
        correcta, _marginal, why = hand22[qid]
        return correcta, clean(why)

    v22_verdict.marginales = {q for q in hand22 if hand22[q][0] and hand22[q][1]}

    emit(
        out / "auditoria_134_v22.md",
        "Las 134 preguntas respondidas por la v2.2",
        v22, "Respuesta de la v2.2", v22_verdict, meta, with_score=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
