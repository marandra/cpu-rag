"""Genera el anexo que se manda al cliente: las 134 comentadas una a una.

    python3 tools/audit_annex.py --out docs/auditoria_134_comentadas.md

Sin lógica de puntuación propia: sólo junta lo que ya existe —la pregunta y la
crítica del auditor (`eval/ec2/*.json`), la respuesta de la versión auditada
(`eval/audit_replay/`), la de la versión actual (`eval/ec2/`) y el veredicto
leído a mano (`tools/audit_hand.py`)— y lo formatea. Si cambia el run o la
lectura, se regenera.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_hand import verdicts  # noqa: E402
from audit_score import MUST_REFUSE, load_run, telegraphic  # noqa: E402
from audit_triage import PROCEDURES, TRIAGE, refused  # noqa: E402

PROC_LABEL = {"diabetes": "Diabetes", "cirugia-abdominal": "Cirugía abdominal",
              "hemorroides": "Hemorroides"}


def load_meta(root: Path) -> dict[int, dict]:
    meta = {}
    for proc in PROCEDURES:
        for row in json.loads((root / f"{proc}.json").read_text())["rows"]:
            meta[row["id"]] = row
    return meta


def esc(text: str | None) -> str:
    return (text or "—").replace("\n", "\n> ").strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="eval/audit_replay")
    ap.add_argument("--new", default="eval/ec2")
    ap.add_argument("--out", default="docs/auditoria_134_comentadas.md")
    args = ap.parse_args()

    old = load_run(Path(args.old))["replay"]
    new = load_run(Path(args.new))["replay"]
    meta = load_meta(Path(args.new))
    hand = verdicts(new)

    ids = sorted(TRIAGE)
    ok = [q for q in ids if hand[q][0]]
    strict = [q for q in ok if not hand[q][1]]
    tel = [q for q in ok if telegraphic(new[q])]
    tel_s = [q for q in strict if telegraphic(new[q])]
    n = len(ids)

    L = [
        "# Las 134 preguntas, comentadas una a una",
        "",
        "Anexo a la respuesta a la auditoría. Para cada pregunta: vuestra puntuación y",
        "vuestra crítica, la respuesta de la **versión que auditasteis (v1.1)**, la de la",
        "**versión actual** y nuestro veredicto con el motivo.",
        "",
        "## Cómo lo hemos puntuado",
        "",
        "Dos criterios separados, porque miden cosas distintas:",
        "",
        "- **Corrección** — responde lo que se pregunta, apoyada en el documento, sin",
        "  inventar y sin fundir ni des-acotar una regla; **o** se abstiene donde el",
        "  documento no da material. Abstenerse bien cuenta como correcta: es la conducta",
        "  diseñada.",
        "- **Presentable** — además, no es telegráfica (menos de 80 caracteres). Una",
        "  respuesta puede ser correcta y aun así no ser enseñable a un paciente.",
        "",
        "El veredicto es **una lectura a mano** de las respuestas de la versión actual. 97",
        "de las 134 están leídas una a una; las otras 37 se trasladan sin leer porque ambas",
        "versiones se abstienen sobre lo mismo y el veredicto es idéntico. Seis respuestas",
        "quedan marcadas **(en el filo)**: apoyadas y sin invención, pero sin responder del",
        "todo lo preguntado. De ahí sale la horquilla.",
        "",
        "## Resultado global",
        "",
        "| sobre las 134 | corrección | presentable |",
        "| --- | ---: | ---: |",
        f"| **versión actual** | **{len(strict)}–{len(ok)}/{n} = "
        f"{len(strict) / n:.0%} – {len(ok) / n:.0%}** | "
        f"**{len(strict) - len(tel_s)}–{len(ok) - len(tel)}/{n} = "
        f"{(len(strict) - len(tel_s)) / n:.0%} – {(len(ok) - len(tel)) / n:.0%}** |",
        "| versión auditada (v1.1) | 84/134 = 63 % | 67/134 = 50 % |",
        "| vuestra evaluación | — | 9 % |",
        "",
        "Por documento, en la versión actual:",
        "",
        "| documento | n | corrección | presentable |",
        "| --- | ---: | ---: | ---: |",
    ]

    for proc, lo, hi in (("diabetes", 1, 55), ("cirugia-abdominal", 56, 103),
                         ("hemorroides", 104, 134)):
        pid = [q for q in ids if lo <= q <= hi]
        pok = [q for q in pid if hand[q][0]]
        pst = [q for q in pok if not hand[q][1]]
        pt = [q for q in pok if telegraphic(new[q])]
        pts = [q for q in pst if telegraphic(new[q])]
        m = len(pid)
        L.append(f"| {PROC_LABEL[proc]} | {m} | "
                 f"{len(pst) / m:.0%} – {len(pok) / m:.0%} | "
                 f"{(len(pst) - len(pts)) / m:.0%} – {(len(pok) - len(pt)) / m:.0%} |")

    L += [
        "",
        "La telegrafía está concentrada: de las "
        f"{len(tel)} respuestas telegráficas que quedan, la mayoría son del documento de",
        "hemorroides (28 líneas, 164 palabras). Sin ese documento, corrección y",
        "presentabilidad prácticamente coinciden — es decir, cuando el sistema acierta",
        "sobre un documento bien redactado, la respuesta ya es enseñable.",
        "",
        "---",
        "",
    ]

    for proc, lo, hi in (("diabetes", 1, 55), ("cirugia-abdominal", 56, 103),
                         ("hemorroides", 104, 134)):
        L += [f"## {PROC_LABEL[proc]}", ""]
        for qid in [q for q in ids if lo <= q <= hi]:
            row = meta[qid]
            correcta, marginal, why = hand[qid]
            tag = "✅ correcta" if correcta else "❌ incorrecta"
            if correcta and marginal:
                tag = "🟡 correcta (en el filo)"
            pres = ("sí" if correcta and not telegraphic(new[qid]) else
                    "no — telegráfica" if correcta else "—")
            want = "abstenerse" if qid in MUST_REFUSE else "responder"
            lista = row.get("profile", "").split("—")[0].strip()

            L += [
                f"### {qid}. {row['question']}",
                "",
                f"*{lista} · lo correcto aquí es **{want}** · vuestra puntuación: "
                f"{row.get('score', '?')}/10 ({row.get('verdict', '?')})*",
                "",
                f"**Vuestra crítica.** {esc(row.get('critique'))}",
                "",
                f"**Respuesta v1.1 (la que auditasteis).** {esc(old.get(qid))}",
                "",
                f"**Respuesta actual.** {esc(new.get(qid))}",
                "",
                f"**Nuestro veredicto: {tag}** · presentable: {pres}",
                f"— {why}.",
                "",
            ]

    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"escrito {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
