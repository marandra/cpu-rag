"""Volcado lado a lado de las preguntas que se mueven entre dos runs.

    uv run python tools/audit_movers.py --base eval/audit_replay --new eval/ec2
    uv run python tools/audit_movers.py --ids 3,5,21 --base ... --new ...

Existe porque `audit_score.py` cierra la **decisión** (responder/rechazar) de
forma automática, pero corrección y presentable no se computan: su columna `ok`
es el set fijo de veredictos de `audit_triage.TRIAGE`, leídos a mano sobre el
run de Ministral, y no acredita a un modelo nuevo por acertar lo que el viejo
fallaba. Cerrar corrección/presentable exige leer las respuestas nuevas — y
para leerlas hacen falta las dos lado a lado, con la crítica del auditor y el
veredicto de terreno delante. Eso es lo único que hace este script.

Sin `--ids` saca exactamente los movers: las preguntas donde la decisión
cambia respecto al baseline, separadas en ganancias y roturas.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_score import MUST_REFUSE, load_run  # noqa: E402
from audit_triage import PROCEDURES, TRIAGE, VERDICT_LABEL, deferred, refused  # noqa: E402


def load_meta(root: Path) -> dict[int, dict]:
    """La pregunta, la respuesta del cliente y la crítica del auditor."""
    meta: dict[int, dict] = {}
    for proc in PROCEDURES:
        path = root / f"{proc}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text())["rows"]:
            meta[row["id"]] = row
    return meta


def decision(answer: str | None) -> str:
    if refused(answer):
        return "RECHAZA"
    if deferred(answer):
        return "DERIVA"
    return "RESPONDE"


def wrap(text: str | None, indent: str = "    ") -> str:
    import textwrap
    if not text:
        return indent + "(vacío)"
    return "\n".join(
        textwrap.fill(p, width=96, initial_indent=indent, subsequent_indent=indent)
        for p in text.splitlines() if p.strip()
    )


def emit(qid: int, meta: dict, base: str | None, new: str | None,
         base_label: str, new_label: str) -> list[str]:
    verdict, subtype, evidence = TRIAGE.get(qid, ("?", "", ""))
    row = meta.get(qid, {})
    want = "RECHAZAR" if qid in MUST_REFUSE else "RESPONDER"

    out = [
        "",
        "=" * 100,
        f"ID {qid}  [{row.get('procedure', '?')}]  debe: {want}   "
        f"terreno: {verdict} {subtype}".rstrip(),
        "=" * 100,
        "",
        "PREGUNTA",
        wrap(row.get("question")),
        "",
        f"VEREDICTO DE TERRENO — {VERDICT_LABEL.get(verdict, verdict)}",
        wrap(evidence),
        "",
        f"AUDITOR (score {row.get('score', '?')}/10 — {row.get('verdict', '?')})",
        wrap(row.get("critique")),
        "",
        "AUDITOR — qué debería incluir",
        wrap(row.get("should_include")),
        "",
        f"--- {base_label}  [{decision(base)}] ---",
        wrap(base),
        "",
        f"--- {new_label}  [{decision(new)}] ---",
        wrap(new),
        "",
        "VEREDICTO A MANO (rellenar): correcta=?  presentable=?  nota=",
    ]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="eval/audit_replay")
    ap.add_argument("--new", default="eval/ec2")
    ap.add_argument("--base-label", default="BASE (Ministral V13 servido)")
    ap.add_argument("--new-label", default="NEW (gemma v2 EC2)")
    ap.add_argument("--ids", help="lista separada por comas; por defecto, los movers")
    ap.add_argument("--out", help="fichero de salida (por defecto stdout)")
    args = ap.parse_args()

    base_root, new_root = Path(args.base), Path(args.new)
    base = load_run(base_root)["replay"]
    new = load_run(new_root)["replay"]
    meta = load_meta(new_root) | load_meta(base_root)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
        groups = [("SELECCIÓN", ids)]
    else:
        gains, breaks = [], []
        for qid in sorted(TRIAGE):
            if qid not in base or qid not in new:
                continue
            want_refuse = qid in MUST_REFUSE
            ok_base = refused(base[qid]) == want_refuse
            ok_new = refused(new[qid]) == want_refuse
            if ok_new and not ok_base:
                gains.append(qid)
            elif ok_base and not ok_new:
                breaks.append(qid)
        groups = [("ROTURAS", breaks), ("GANANCIAS", gains)]

    lines: list[str] = []
    for title, ids in groups:
        lines += ["", "#" * 100, f"# {title}  ({len(ids)}): {ids}", "#" * 100]
        for qid in ids:
            lines += emit(qid, meta, base.get(qid), new.get(qid),
                          args.base_label, args.new_label)

    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"escrito {args.out} ({len(text)} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
