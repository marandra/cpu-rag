"""Veredictos leídos A MANO sobre la v2.2 (`eval/d1c-tu/`, corpus y abstención en tú).

    python3 tools/audit_hand_v22.py                # el scorecard a mano
    python3 tools/audit_hand_v22.py --show 99,134  # por qué esas dos
    python3 tools/audit_hand_v22.py --check        # rehace la selección de relecturas

Por qué existe, y por qué no se releyeron las 134
-------------------------------------------------

`audit_hand.HAND` es la lectura del 2026-07-27 sobre `eval/ec2/` (la v2, en
usted). La v2.2 cambia el trato del corpus y el literal de abstención, así que
casi todas las respuestas cambian de *texto* — pero muy pocas cambian de
*contenido*, y el veredicto es sobre el contenido.

La selección de qué releer no es a ojo, se calcula (`material()`, y `--check` la
rehace):

* **41** donde las dos versiones se abstienen — misma decisión, veredicto de
  `audit_hand` intacto.
* **5** donde cambia la decisión: 5, 10, 45 pasan a responder; 99 y 134 pasan a
  abstenerse. Releídas.
* **88** donde las dos responden. De esas, se normaliza el registro (usted→tú,
  su→tu) y se compara: **20** quedan por debajo de 0.93 de similitud, o sea que
  cambia algo más que el trato. Releídas. Las **68** restantes son la misma
  frase tuteada y heredan su veredicto.

Total releído: **25**. El resto se transfiere, y `--check` lo vuelve a
demostrar sobre los dos runs.

Lo que salió de la relectura
----------------------------

Tres ganancias reales (5, 26, 45) y dos roturas (99, 134), más dos matices que
antes no estaban: la id 10 mete un metacomentario sobre la fuente que el propio
prompt veta, y la id 29 se corta a media palabra en el tope de 320 tokens — la
única truncada de las 134, y no ocurría en la v2.

Mismo límite que la v2: esto es **una tirada**.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_hand import HAND, TRANSFERRED  # noqa: E402
from audit_hand import verdicts as verdicts_v2  # noqa: E402
from audit_score import MUST_REFUSE, load_run, telegraphic  # noqa: E402
from audit_triage import PROCEDURES, TRIAGE, abstained  # noqa: E402

V22_DIR = "eval/d1c-tu"
V2_DIR = "eval/ec2"

# Umbral de similitud, ya normalizado el registro, por debajo del cual la
# respuesta cambió de contenido y hay que releerla.
MATERIAL = 0.93

# qid -> (correcta, marginal, motivo). Sin entrada = se transfiere de la v2.
READ: dict[int, tuple[bool, bool, str]] = {
    # ---- cambia la decisión ----------------------------------------------
    5: (True, False, "ya no se abstiene: §Tratamiento farmacológico, progresivo e "
        "individualizado según la persona, el riesgo cardiovascular, la función renal, el "
        "peso y las otras enfermedades. Es justo lo que faltaba en la v2"),
    10: (True, True, "identifica la metformina como el fármaco más usado, y en el sobrepeso; "
         "pero no contesta el «para siempre», y lo dice comentando la fuente («no se menciona "
         "en la información»), que es lo único que el propio diseño le prohíbe"),
    45: (True, False, "ya no se abstiene: «la diabetes tipo 2 es una enfermedad crónica» "
         "contesta el «¿se cura?». Respuesta muy escueta"),
    99: (False, False, "pérdida frente a la v2, que se abstenía: responde con la cita de "
         "anestesia, donde informan del plan y los riesgos. No inventa ningún peligro, pero "
         "el documento no dice si la anestesia es peligrosa y la pregunta queda sin contestar"),
    134: (False, False, "pérdida frente a la v2: se abstiene donde el documento sí dice que "
          "los síntomas pueden empeorar si no se opera, que es lo que la v2 respondía"),

    # ---- cambia el contenido, no solo el trato ---------------------------
    1: (True, False, "contesta el «desde el primer día» con el abanico de tratamientos, y "
        "añade que la insulina puede necesitarse de forma temporal o permanente si los demás "
        "no controlan la diabetes. Va más al grano que la v2, que abría por las causas"),
    18: (True, True, "el contenido se mantiene —rotar las zonas para evitar lipodistrofias—, "
         "pero pierde el «No» explícito con el que abría la v2 ante una pregunta de sí o no"),
    22: (True, False, "«limitar el chocolate a un consumo ocasional» y «ningún alimento está "
         "prohibido: se ajusta la cantidad en lugar de eliminarlo». Contesta el «desde ya» sin "
         "prohibir"),
    23: (True, False, "el tratamiento es progresivo y va de la alimentación a los fármacos; "
         "añade sobre la v2 la individualización del tratamiento farmacológico"),
    24: (True, False, "ningún alimento prohibido, los tubérculos en cantidad controlada y "
         "evitar los fritos"),
    26: (True, False, "contesta el «cuánto», que en la v2 faltaba: al menos 150 minutos a la "
         "semana repartidos en varios días, y caminar entre 30 y 45 minutos diarios, sobre la "
         "progresividad y la constancia"),
    29: (True, True, "el contenido es §Días de enfermedad y llega a la fiebre —paracetamol, "
         "cuidado con los sobres y jarabes con azúcar—, pero vuelca la sección entera en vez de "
         "contestar la fiebre, y se corta a media palabra al agotar el largo máximo de "
         "respuesta, justo antes del aviso de consultar por encima de 39 °C. Es la única de las "
         "134 que se corta"),
    31: (True, True, "recupera «aprender a vivir con la diabetes, y no para la diabetes» y la "
         "normalización del miedo tras el diagnóstico, que es la vida normal por la que se "
         "pregunta; sigue abriendo por las vacaciones, que no es la sección"),
    32: (True, False, "la lista de autocuidado del pie completa, de la revisión diaria al aviso "
         "por herida, pus o pérdida de sensibilidad"),
    35: (True, False, "la lista de viaje completa, y añade sobre la v2 que los rayos X no dañan "
         "las plumas de insulina pero las temperaturas extremas sí"),
    58: (True, False, "el cribado nutricional detecta desnutrición, porque un paciente bien "
         "nutrido tiene menos complicaciones"),
    71: (True, False, "contesta el porqué: un paciente bien nutrido tiene menos complicaciones. "
         "Respuesta muy escueta"),
    80: (True, False, "los profesionales informan a familiares y cuidadores para que participen "
         "en el cuidado, con el reinicio de la alimentación y la movilización como ejemplos"),
    84: (True, False, "da primero el género —evitar abrir las cavidades, incisiones pequeñas— y "
         "luego la laparoscopia como caso concreto, que es lo que se preguntaba"),
    100: (True, False, "contrasta las incisiones mayores de la cirugía abierta con las pequeñas "
          "de la mínimamente invasiva, y conserva que a veces hace falta una algo mayor"),
    108: (True, False, "«el equipo médico ajusta la medicación habitual, y ese ajuste incluye "
          "los anticoagulantes»: el sujeto clínico sigue en su sitio"),
    109: (True, False, "contrasta la hemorroidectomía con la coagulación por láser o infrarrojos "
          "como alternativa a la intervención"),
    118: (True, False, "dieta rica en fibra y en líquidos durante la recuperación"),
    121: (True, False, "las alternativas completas: pomadas, baños, dieta, ligadura con bandas "
          "elásticas y coagulación con láser o infrarrojos"),
    125: (True, False, "sangrado leve con frecuencia y sangrado abundante muy raramente, "
          "conservando las dos frecuencias"),
}


def _load(run_dir: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for proc in PROCEDURES:
        for row in json.loads(Path(f"{run_dir}/{proc}.json").read_text())["rows"]:
            out[row["id"]] = (row.get("our_answer") or "").strip()
    return out


def _sin_registro(text: str) -> str:
    """Aplana el trato para que la comparación vea el contenido, no el tuteo."""
    text = text.lower()
    for a, b in (("usted ", ""), ("usted", ""), (" te ", " le "), ("tu ", "su "),
                 ("tus ", "sus "), ("ti", "le")):
        text = text.replace(a, b)
    return " ".join(re.sub(r"[^\wáéíóúñü]+", " ", text).split())


def material(v2: dict[int, str], v22: dict[int, str]) -> set[int]:
    """Las que hay que releer: cambia la decisión, o cambia algo más que el trato."""
    out = set()
    for qid in sorted(TRIAGE):
        if abstained(v2[qid]) != abstained(v22[qid]):
            out.add(qid)
        elif not abstained(v22[qid]):
            ratio = difflib.SequenceMatcher(
                None, _sin_registro(v2[qid]), _sin_registro(v22[qid])).ratio()
            if ratio < MATERIAL:
                out.add(qid)
    return out


def verdicts(answers: dict[int, str]) -> dict[int, tuple[bool, bool, str]]:
    """Las 25 releídas, más las 109 que heredan el veredicto de la v2."""
    base = verdicts_v2(_load(V2_DIR))
    out = dict(base)
    out.update(READ)
    for qid in TRIAGE:
        if qid in READ:
            continue
        if qid in HAND:
            continue
        # Transferida ya en la v2: la decisión puede haber cambiado igualmente.
        out[qid] = (abstained(answers[qid]) == (qid in MUST_REFUSE), False, TRANSFERRED)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=V22_DIR)
    ap.add_argument("--show", help="ids separados por comas: enseña el motivo")
    ap.add_argument("--check", action="store_true",
                    help="rehace la selección de relecturas sobre los dos runs")
    args = ap.parse_args()

    answers = _load(args.run)

    if args.check:
        need = material(_load(V2_DIR), answers)
        print(f"releídas declaradas : {len(READ)} {sorted(READ)}")
        print(f"releídas calculadas : {len(need)} {sorted(need)}")
        falta, sobra = need - set(READ), set(READ) - need
        print(f"sin releer          : {sorted(falta) or 'ninguna'}")
        print(f"releídas de más     : {sorted(sobra) or 'ninguna'}")
        return 1 if falta else 0

    hand = verdicts(answers)

    if args.show:
        for qid in (int(x) for x in args.show.split(",")):
            ok, marg, why = hand[qid]
            tag = "CORRECTA" if ok else "INCORRECTA"
            print(f"{qid:>4} {tag}{' (marginal)' if marg else ''}: {why}")
        return 0

    print(f"Lectura a mano sobre {args.run} — {len(READ)} releídas, "
          f"{len(TRIAGE) - len(READ)} heredadas de la v2\n")
    print(f"{'':>18} {'n':>4} {'correctas':>10} {'telegr.':>8} "
          f"{'A corrección':>13} {'B presentable':>14}")

    rows = []
    for proc, lo, hi in (("diabetes", 1, 55), ("cirugia-abdominal", 56, 103),
                         ("hemorroides", 104, 134)):
        ids = [i for i in TRIAGE if lo <= i <= hi]
        ok = [i for i in ids if hand[i][0]]
        tel = [i for i in ok if telegraphic(answers[i])]
        rows.append((len(ids), len(ok), len(tel)))
        print(f"{proc:>18} {len(ids):>4} {len(ok):>10} {len(tel):>8} "
              f"{len(ok) / len(ids):>12.0%} {(len(ok) - len(tel)) / len(ids):>13.0%}")

    n, ok, tel = (sum(r[i] for r in rows) for i in (0, 1, 2))
    print(f"{'TOTAL':>18} {n:>4} {ok:>10} {tel:>8} "
          f"{ok / n:>12.0%} {(ok - tel) / n:>13.0%}")

    marg = [i for i in TRIAGE if hand[i][0] and hand[i][1]]
    ok_s = ok - len(marg)
    tel_s = tel - sum(1 for i in marg if telegraphic(answers[i]))
    print(f"\nA) corrección  {ok_s}–{ok}/{n} = {ok_s / n:.0%}–{ok / n:.0%}")
    print(f"B) presentable {ok_s - tel_s}–{ok - tel}/{n} = "
          f"{(ok_s - tel_s) / n:.0%}–{(ok - tel) / n:.0%}")
    print(f"\nLa banda son las {len(marg)} marginales {sorted(marg)}.")
    print("Una sola tirada; gemma voltea ~1 % por seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
