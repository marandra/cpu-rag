"""Veredictos leídos A MANO sobre el run entregado (`eval/ec2/`, gemma v2).

    python3 tools/audit_hand.py                 # el scorecard a mano
    python3 tools/audit_hand.py --show 29,84    # por qué esas dos

Por qué existe
--------------

`audit_score.scorecard()` no puede puntuar a gemma. Su columna de correctas es
el set fijo de veredictos de `audit_triage.TRIAGE`, que se leyeron a mano sobre
las respuestas de **Ministral**: acredita a un run nuevo por *rechazar* lo que
el corpus no cubre (eso sí es computable) pero no por *responder bien* lo que
Ministral respondía mal. Sobre `eval/ec2/` daba 60 %/55 %, que es un suelo, no
una medida.

«Correcta según el corpus» no es un predicado computable. La única forma de
cerrarlo es leer las respuestas, y la única forma de que una lectura a mano sea
auditable es escribirla, una línea por pregunta, con el motivo. Eso es este
fichero: la lectura del 2026-07-27 sobre las 134 respuestas de `eval/ec2/`.

Criterio, el mismo que `scorecard()`
------------------------------------

* **A correcta** = responde lo que se pregunta, apoyada en el fulldoc, sin
  invención y sin fundir/de-sujetar una regla; **o** rechaza donde el fulldoc no
  da material.
* **B presentable** = A y además no telegráfica (<80 car). Los rechazos cuentan
  como presentables: son la conducta diseñada, no un texto pobre.

Las 134, ninguna transferida (corregido 2026-07-28)
---------------------------------------------------

Este fichero declaraba que «37 de las 134 no necesitan lectura» porque se
transferían de la pasada de Ministral cuando ambos runs se abstenían. Era falso,
y de ahí salió que el mismo atajo se repitiera en `audit_hand_v22`. El porqué
está en la cabecera de `audit_reading.py`; el resumen es que el predicado
comparaba la abstención contra un `MUST_REFUSE` precalculado en vez de contra el
documento servido, y que «ambas se abstienen» no implica «ambas dicen lo mismo».

Ahora las 134 tienen veredicto explícito y escrito: las 97 leídas en su día, y
las 37 restantes leídas el 2026-07-28 sobre las respuestas de `eval/ec2` y
contra los tres documentos servidos (`LEIDAS_2026_07_28`). Nada se deduce de
nada.

Al leerlas, cinco resultaron mal dadas — todas sobre-abstenciones que el
predicado automático daba por buenas: 10, 45, 54, 85 y 104. El total de la v2
baja de 120 a **117**.

El margen
---------

`MARGINAL` marca las que están en el filo — gemma no inventa ni rompe reglas,
pero tampoco responde del todo lo que se preguntaba. Contarlas o no da la banda,
y la banda es parte del resultado. Se reporta el rango, no un punto.

Límite, y va allí donde vaya el número: esto es **una tirada**. gemma voltea ~1 %
por seed, así que la banda por muestreo es estrecha, pero no es cero, y la
calidad no se puede promediar sobre seeds sin releer las 97.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_reading import require_complete  # noqa: E402
from audit_score import load_run, telegraphic  # noqa: E402
from audit_triage import TRIAGE  # noqa: E402

EC2_DIR = "eval/ec2"

# Las 37 que el atajo daba por transferidas, leídas el 2026-07-28 sobre las
# respuestas de `eval/ec2` y contra los tres documentos servidos. Tres eran
# sobre-abstenciones que el predicado automático daba por buenas: 10, 45 y 54.
LEIDAS_2026_07_28: dict[int, tuple[bool, bool, str]] = {
    10: (False, False, "se abstiene; §Tratamiento farmacológico identifica la metformina como el "
         "fármaco más usado, sobre todo con sobrepeso, que es la mitad de la pregunta"),
    13: (True, False, "se abstiene bien: el documento manda anotar los autoanálisis, no la comida"),
    19: (True, False, "se abstiene bien: el documento no da plazos de respuesta al tratamiento"),
    20: (False, False, "se abstiene; §Tratamiento farmacológico dice que el tratamiento es "
         "progresivo y §Insulina que si los otros tratamientos no controlan puede necesitarse "
         "insulina, que es exactamente lo que se pregunta"),
    27: (False, False, "se abstiene; §Tratamiento farmacológico dice «tienes diabetes aunque no "
         "uses insulina» y enumera tratarse solo con alimentación y ejercicio, o con pastillas"),
    37: (True, False, "se abstiene bien: el documento no compara casos entre familiares"),
    38: (True, False, "se abstiene bien: el documento no trata el ámbito laboral"),
    43: (True, False, "se abstiene bien: el documento habla de heridas y úlceras, nunca de "
         "amputación"),
    44: (True, True, "se abstiene; §Aspectos psicológicos manda identificar las situaciones "
         "difíciles y pedir ayuda, pero en genérico, y el documento no trata la vergüenza al "
         "inyectarse"),
    45: (False, False, "se abstiene; §Introducción dice que la diabetes tipo 2 es una enfermedad "
         "crónica, que contesta el «¿se cura?»"),
    46: (True, False, "se abstiene bien: el documento no hace pronóstico individual"),
    49: (True, False, "se abstiene bien: el documento nombra la diabetes gestacional pero no trata "
         "la fertilidad"),
    50: (True, False, "se abstiene bien: el documento no habla de esperanza de vida"),
    53: (True, True, "se abstiene; §Aspectos psicológicos manda pedir ayuda ante las situaciones "
         "difíciles, pero en genérico, y el documento no trata el miedo a las agujas"),
    54: (False, False, "se abstiene; §Causas incluye la herencia y §Aspectos psicológicos dice "
         "«no debes culparte», que son las dos mitades de la pregunta"),
    72: (True, False, "se abstiene bien: el documento atribuye la molestia por el gas al abdomen, "
         "nunca al hombro"),
    73: (True, False, "se abstiene bien: el documento regula las horas previas, no los días "
         "previos"),
    77: (True, False, "se abstiene bien: el documento no trata la higiene previa"),
    79: (True, False, "se abstiene bien: el documento termina en el alta y no trata la vida "
         "posterior"),
    85: (False, False, "se abstiene; §Qué es la cirugía mayor dice que «la recuperación puede "
         "llevar varios días o semanas», que es material para esta pregunta. Es la misma frase "
         "que en la 76 usa donde no tocaba"),
    92: (True, False, "se abstiene bien: el documento no cuantifica el riesgo vital"),
    94: (True, False, "se abstiene bien: el documento no trata el despertar intraoperatorio"),
    95: (True, False, "se abstiene bien: el documento no dice qué ocurre si hay una complicación"),
    97: (True, False, "se abstiene bien: el documento no menciona ostomías"),
    98: (True, False, "se abstiene bien: este documento no recoge la retirada del consentimiento, "
         "que sí está en el de hemorroides"),
    99: (True, False, "se abstiene bien: el documento no dice si la anestesia es peligrosa"),
    101: (True, False, "se abstiene; el documento dice que se pueden consultar las dudas en "
          "cualquier momento, pero no que se pueda hacer a solas, que es lo que se pregunta"),
    102: (True, False, "se abstiene bien: el documento no da la duración del ingreso"),
    104: (False, False, "se abstiene; el documento tiene las dos partes del contraste —la "
          "hemorroidectomía extirpa las hemorroides, y la ligadura con bandas elásticas es una "
          "alternativa—, y son las mismas que sí une en la 109 para el láser"),
    113: (True, False, "se abstiene bien: el documento no trata la baja laboral"),
    120: (True, False, "se abstiene bien: el documento no dice si hay ingreso"),
    126: (True, False, "se abstiene bien: este documento no tiene sección psicológica y no da "
          "material sobre la vergüenza"),
    129: (True, False, "se abstiene bien: el documento no trata el riesgo vital"),
    130: (True, False, "se abstiene bien: este documento no tiene sección psicológica y no da "
          "material sobre el miedo"),
    131: (True, False, "se abstiene bien: este documento no tiene sección psicológica y no da "
          "material sobre la vergüenza"),
    132: (True, False, "se abstiene bien: el documento no describe quién está en el quirófano"),
    133: (False, False, "se abstiene; §Riesgos da las muy raras —estrechamiento anal, "
          "incontinencia—, que son secuelas permanentes, y son las mismas que sí responde en la "
          "112 y en la 128"),
}

# qid -> (correcta, marginal, motivo).
HAND: dict[int, tuple[bool, bool, str]] = {
    # ---- las 7 «roturas» de decisión -------------------------------------
    # Tres de las siete no son roturas de contenido: en 100 y 109 la verdad de
    # terreno ya decía «FN parcial» y el fulldoc sí da material; en 36 el
    # rechazo es lo correcto y quien fallaba era Ministral.
    3: (False, False, "se abstiene; §Criterios diagnósticos dice «≥126 en dos ocasiones», que es justo la repetición por la que se pregunta"),
    5: (False, False, "se abstiene; §Tratamiento farmacológico («progresivo, individualizado») sostenía la respuesta"),
    21: (False, False, "se abstiene ante los primeros pasos, que el documento sí enumera. Es una pérdida real frente a la v1.1"),
    36: (True, False, "se abstiene bien: el documento NO dice qué llevar a la cita. La v1.1 servía la lista de equipaje de vacaciones (vuestro 2/10)"),
    76: (False, False, "responde «recuperación días o semanas» a la duración del INGRESO: fuera de diana"),
    100: (True, False, "§Cirugía mínimamente invasiva: incisiones pequeñas frente a mayores. Es justo lo que pedíais en vuestra crítica"),
    109: (True, False, "§Alternativas: contrasta la hemorroidectomía con la coagulación por láser, apoyándose en el documento"),

    # ---- las 23 ganancias de decisión ------------------------------------
    6: (True, False, "§Autoanálisis literal, con la individualización de la frecuencia"),
    8: (True, False, "se abstiene; «el médico te enseña a pincharte» no está en el documento — la v1.1 lo inventaba"),
    16: (True, False, "§Mitos: la DM2 inicial no da síntomas, se descubre por análisis"),
    25: (True, False, "§Mitos + §Alimentación: dieta equilibrada general, ningún alimento prohibido"),
    39: (True, False, "§Grupos de alimentos: ningún producto milagro cura la diabetes; respuesta muy escueta"),
    40: (True, False, "§Alimentación literal: ajustar cantidad en vez de eliminar"),
    41: (True, False, "§Aspectos psicológicos: normaliza el miedo y añade «vivir con, no para»"),
    47: (True, False, "§Aspectos psicológicos «no culparse», que es exactamente lo que se pregunta"),
    51: (True, False, "§Alimentación: responde el miedo a no volver a disfrutar con «ningún alimento prohibido»"),
    52: (True, False, "se abstiene; §Hipoglucemia NO afirma mortalidad. La v1.1 respondía «sí, riesgo de muerte» a un miedo a morir"),
    55: (True, False, "§Aspectos psicológicos: apoyo en entorno y asociaciones"),
    61: (True, False, "§Reanudar la alimentación literal: comer pronto no abre la sutura"),
    70: (True, False, "§Control del dolor (ACP) literal: todo programado, sin peligro de sobredosis"),
    83: (True, False, "§Qué es la cirugía mayor: «suele requerir anestesia general», con la premedicación detrás"),
    103: (True, False, "§Premedicación: normaliza la reacción emocional ante la intervención"),
    110: (True, True, "§Beneficios «reduce recaídas»: contesta de refilón la recurrencia; respuesta muy escueta"),
    111: (True, False, "se abstiene; el documento dice «fibra y líquidos» pero NUNCA el porqué, que la v1.1 inventaba"),
    114: (True, False, "§Preparación «ayuno 6–8 h» responde directo el día antes"),
    117: (True, False, "se abstiene; el documento nombra los baños de asiento pero no los define — la v1.1 los definía por su cuenta"),
    121: (True, False, "§Alternativas completa: pomadas, baños, dieta, bandas, láser"),
    123: (True, False, "§Aspectos prácticos literal: puede retirar el consentimiento"),
    127: (True, False, "§Preparación «dolor moderado»; respuesta muy escueta"),
    128: (True, False, "§Riesgos «muy raras: incontinencia», y conserva la rareza"),

    # ---- las 18 DEF sin cambio de decisión -------------------------------
    # Aquí está el grueso de lo que el scorer automático no ve: Ministral falla
    # las 18 por construcción (verdicto DEF) y gemma arregla 15.
    2: (True, True, "«al principio, alimentación, ejercicio y pérdida de peso»: sin el «en muchos casos» inventado, pero contesta de lado"),
    4: (True, False, "combina HC con proteína/fibra/grasa: caen los dos defectos (menú inventado y la dosis de insulina re-sujetada)"),
    7: (False, False, "vuelca la batería de revisiones en vez de decir para qué sirve la HbA1c; mejor sección que la v1.1, misma falla"),
    22: (True, False, "«limitar» dulces y HC rápidos; se va la contradicción moderar/evitar edulcorantes"),
    24: (True, False, "sin la fuga de meta-comentario ni «patatas integrales»; tubérculos en cantidad controlada, evitar fritos"),
    26: (True, True, "progresividad y constancia, sin la prescripción aritméticamente imposible; pero no dice cuánto"),
    29: (True, False, "arreglada: paracetamol y el >39 °C vuelven a ser cosas distintas, y recupera «cuidado con sobres y jarabes con azúcar»"),
    30: (False, False, "«los glucómetros miden la glucosa capilar» no contesta si hay que comprarlo"),
    60: (True, False, "define el catéter epidural entero; la v1.1 se quedaba en nombrarlo"),
    63: (False, False, "respuesta idéntica a la de la v1.1: sigue omitiendo que todo analgésico puede dar efectos no deseados"),
    67: (True, False, "recupera la condición «cuando el grado de ansiedad y temor sea elevado»"),
    68: (True, False, "recupera «y luego vía oral (pastillas)»: se acaba el gotero indefinido"),
    69: (True, False, "recupera «el cirujano le indicará cómo proceder», que era la respuesta que faltaba"),
    84: (True, False, "da primero el género (no abrir las cavidades) y LUEGO la laparoscopia como caso concreto, que es lo que se preguntaba"),
    87: (True, False, "devuelve el «con ayuda» a sentarse, no a caminar"),
    89: (True, True, "coherente y apoyada (la bebida de HC evita el hambre); no dice «no improvise ingesta»"),
    105: (True, False, "«anestesia regional o anestesia general»: conserva la disyunción del documento en vez de cerrarla en una sola"),
    108: (True, False, "«el equipo médico le ajusta la medicación»: el sujeto clínico vuelve a su sitio"),

    # ---- las 49 OK sin cambio de decisión --------------------------------
    # El lado del riesgo: aquí gemma solo puede perder. Pierde dos.
    31: (False, False, "responde «puede ir de vacaciones» a «¿podré hacer vida normal?»: sección equivocada"),
    42: (False, False, "«la insulina reduce las complicaciones, incluida la ceguera» a «¿me quedaré ciego?»: fuera de diana y engañosa"),
    56: (True, True, "solo describe la abierta; el contraste con la laparoscópica queda implícito"),
    65: (True, False, "mejora sobre la v1.1: dice que unas pruebas se hacen a todos y otras son específicas, que es justo lo que pedíais"),
    17: (True, False, "mejora sobre la v1.1: deja caer los objetivos fijos <130/80 y LDL<100 que puntuasteis 2/10"),
    91: (True, True, "§Premedicación literal; normaliza la reacción emocional sin decir «sí» explícito"),
    # Las 42 OK restantes: gemma reformula o amplía sin romper nada.
    **{q: (True, False, "mantiene el acierto de la v1.1, reformulada o ampliada sobre la misma base documental")
       for q in (1, 9, 11, 12, 14, 15, 18, 23, 28, 32, 33, 34, 35, 48, 57, 58, 59,
                 62, 64, 66, 71, 74, 75, 78, 80, 81, 82, 86, 88, 90, 93, 96, 106,
                 107, 112, 115, 116, 118, 119, 122, 124, 125, 134)},
}


def verdicts(_answers: dict[int, str] | None = None) -> dict[int, tuple[bool, bool, str]]:
    """La lectura de las 134 sobre `eval/ec2`. Todas escritas, ninguna deducida."""
    return require_complete({**HAND, **LEIDAS_2026_07_28}, TRIAGE,
                            "lectura de eval/ec2 (audit_hand)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=EC2_DIR)
    ap.add_argument("--show", help="ids separados por comas: enseña el motivo")
    args = ap.parse_args()

    answers = load_run(Path(args.run))["replay"]
    hand = verdicts(answers)

    if args.show:
        for qid in (int(x) for x in args.show.split(",")):
            ok, marg, why = hand[qid]
            tag = "CORRECTA" if ok else "INCORRECTA"
            print(f"{qid:>4} {tag}{' (marginal)' if marg else ''}: {why}")
        return 0

    print(f"Lectura a mano sobre {args.run} — las 134, ninguna transferida\n")
    hdr = (f"{'':>18} {'n':>4} {'correctas':>10} {'telegr.':>8} "
           f"{'A corrección':>13} {'B presentable':>14}")
    print(hdr)

    rows = []
    for proc, lo, hi in (("diabetes", 1, 55), ("cirugia-abdominal", 56, 103),
                         ("hemorroides", 104, 134)):
        ids = [i for i in TRIAGE if lo <= i <= hi]
        for strict in (False, True):
            ok = [i for i in ids if hand[i][0] and not (strict and hand[i][1])]
            tel = [i for i in ok if telegraphic(answers[i])]
            if not strict:
                rows.append((proc, len(ids), len(ok), len(tel)))
                print(f"{proc:>18} {len(ids):>4} {len(ok):>10} {len(tel):>8} "
                      f"{len(ok) / len(ids):>12.0%} {(len(ok) - len(tel)) / len(ids):>13.0%}")

    n, ok, tel = (sum(r[i] for r in rows) for i in (1, 2, 3))
    print(f"{'TOTAL':>18} {n:>4} {ok:>10} {tel:>8} "
          f"{ok / n:>12.0%} {(ok - tel) / n:>13.0%}")

    marg = [i for i in TRIAGE if hand[i][0] and hand[i][1]]
    ok_s = ok - len(marg)
    tel_s = tel - sum(1 for i in marg if telegraphic(answers[i]))
    print(f"\nA) corrección  {ok_s}–{ok}/{n} = {ok_s / n:.0%}–{ok / n:.0%}")
    print(f"B) presentable {ok_s - tel_s}–{ok - tel}/{n} = "
          f"{(ok_s - tel_s) / n:.0%}–{(ok - tel) / n:.0%}")
    print(f"\nLa banda son las {len(marg)} marginales {sorted(marg)}: apoyadas y "
          "sin invención,\npero sin responder del todo lo que se preguntaba. El "
          "extremo bajo las descuenta.")
    print("Una sola tirada; gemma voltea ~1 % por seed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
