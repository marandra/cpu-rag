"""Extract the third-party audit spreadsheet into a plain JSON payload.

Runs on the laptop, not the cluster: openpyxl is not installed in the cluster's
`.venv-native`, and shipping a JSON is simpler than shipping a dependency.

    uv run --with openpyxl python tools/audit_extract.py \
        --xlsx Auditoria_critica_RAG_preguntes_pacients.xlsx \
        --out reports/audit_questions.json

The `Àrea` column is what tells us which procedure — and therefore which pool —
each question belongs to; the audit never records it directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

AREA_TO_PROCEDURE = {
    "Diabetes": "diabetes",
    "Cirugía abdominal": "cirugia-abdominal",
    "Hemorroides": "hemorroides",
}

# Spreadsheet header -> the short key we carry around everywhere downstream.
COLUMNS = {
    "ID": "id",
    "Àrea": "area",
    "Perfil de pregunta": "profile",
    "Pregunta": "question",
    "Resposta RAG": "their_answer",
    "Correcció (0-2)": "s_correccio",
    "Completesa/utilitat (0-2)": "s_completesa",
    "Claredat/adequació (0-2)": "s_claredat",
    "Contextualització (0-2)": "s_contextualitzacio",
    "Seguretat/empatia (0-2)": "s_seguretat",
    "Puntuació total (0-10)": "score",
    "Valoració": "verdict",
    "Prioritat de correcció": "priority",
    "Tipus de mancances": "fault_type",
    "Avaluació crítica": "critique",
    "Què hauria d’incloure": "should_include",
}

SCORE_KEYS = ("id", "score", "s_correccio", "s_completesa", "s_claredat",
              "s_contextualitzacio", "s_seguretat")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="Auditoria_critica_RAG_preguntes_pacients.xlsx")
    p.add_argument("--sheet", default="Avaluació completa")
    p.add_argument("--out", default="reports/audit_questions.json")
    args = p.parse_args()

    import openpyxl

    wb = openpyxl.load_workbook(args.xlsx, data_only=True)
    rows = list(wb[args.sheet].iter_rows(values_only=True))
    header = list(rows[0])

    missing = [c for c in COLUMNS if c not in header]
    if missing:
        raise SystemExit(f"Sheet {args.sheet!r} is missing columns: {missing}")
    index = {col: header.index(col) for col in COLUMNS}

    records = []
    for raw in rows[1:]:
        if raw[index["ID"]] is None:
            continue
        rec = {}
        for col, key in COLUMNS.items():
            val = raw[index[col]]
            rec[key] = val if val is None else str(val).strip()
        for key in SCORE_KEYS:
            rec[key] = int(rec[key]) if rec[key] is not None else None
        area = rec["area"]
        if area not in AREA_TO_PROCEDURE:
            raise SystemExit(f"Row {rec['id']}: unknown area {area!r}")
        rec["procedure"] = AREA_TO_PROCEDURE[area]
        records.append(rec)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    by_proc: dict[str, int] = {}
    for r in records:
        by_proc[r["procedure"]] = by_proc.get(r["procedure"], 0) + 1
    print(f"Wrote {dest}  ({len(records)} questions)")
    for proc, n in sorted(by_proc.items()):
        print(f"  {proc:20s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
