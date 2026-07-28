"""Veredictos leídos A MANO sobre la v2.2 (`eval/d1c-tu/`, corpus y abstención en tú).

    python3 tools/audit_hand_v22.py                # el scorecard a mano
    python3 tools/audit_hand_v22.py --show 99,134  # por qué esas dos
    python3 tools/audit_hand_v22.py --diff-v2      # qué se mueve respecto de la v2

Las 134, una a una
------------------

A diferencia de `audit_hand` —que leyó 97 y transfirió 37 por regla automática—
aquí están **las 134 leídas contra el texto de la v2.2 y contra los tres
documentos servidos**. Sin herencia: `READ` tiene una entrada por pregunta.

La primera versión de este fichero releyó solo 25 (las que cambiaban de decisión
o de contenido respecto de la v2) y heredó 109. Ese atajo es correcto para
*seguir* un número, pero no para publicarlo: 34 de las heredadas no las había
leído nadie nunca, venían de la transferencia automática de la pasada de
Ministral. La lectura completa movió cuatro veredictos que la herencia daba por
buenos, todos en preguntas cuyo texto no había cambiado:

* **85** («¿cuánto dura mi recuperación en casa?») se abstiene, y §Qué es la
  cirugía mayor sí da «varios días o semanas» — la misma frase que en la 76 usa
  donde no tocaba. Pasa a incorrecta.
* **104** («¿en qué se diferencia de las bandas elásticas?») se abstiene con las
  dos partes del contraste en el documento, y las une sin problema en la 109
  para el láser. Pasa a incorrecta.
* **63**, **44**, **53** y **54** pasan a marginales: apoyadas, pero dejan fuera
  algo que el documento sí tenía.

Criterio, el mismo que `audit_hand`
-----------------------------------

* **A correcta** = responde lo que se pregunta, apoyada en el documento, sin
  invención y sin fundir/de-sujetar una regla; **o** se abstiene donde el
  documento no da material.
* **B presentable** = A y además no telegráfica (<80 car).
* `MARGINAL` (el segundo campo) marca el filo. Se reporta la banda, no un punto.

Límite, y va allí donde vaya el número: esto es **una tirada**. gemma voltea
~1 % por seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_reading import require_complete  # noqa: E402
from audit_score import telegraphic  # noqa: E402
from audit_triage import PROCEDURES, TRIAGE  # noqa: E402

V22_DIR = "eval/d1c-tu"
V2_DIR = "eval/ec2"


# qid -> (correcta, marginal, motivo). Las 134, ninguna heredada.
READ: dict[int, tuple[bool, bool, str]] = {
    # ------------------------------------------------------------------ diabetes
    1: (True, False, "contesta el «desde el primer día»: puedes tratarte solo con alimentación "
        "y ejercicio, o con pastillas, o con insulina, y la insulina llega si los otros "
        "tratamientos no controlan"),
    2: (True, True, "«al principio, alimentación, ejercicio y pérdida de peso» es lo que dice "
        "§Tratamiento farmacológico, pero contesta de lado: no dice si con eso se evita la "
        "medicación"),
    3: (False, False, "se abstiene; §Diagnóstico dice «≥126 mg/dl en dos ocasiones», que es "
        "justo la repetición de análisis por la que se pregunta"),
    4: (True, False, "§Hidratos de carbono: combinarlos con proteína, grasa saludable y fibra "
        "reduce los picos, y hay que distribuirlos a lo largo del día"),
    5: (True, False, "§Tratamiento farmacológico: el tratamiento es progresivo y se individualiza "
        "según la persona, el riesgo cardiovascular, la función renal, el peso y las otras "
        "enfermedades. Contesta que la pauta no marca gravedad"),
    6: (True, False, "§Autoanálisis: es especialmente importante con insulina o con fármacos con "
        "riesgo de hipoglucemia, y el equipo individualiza la frecuencia"),
    7: (False, False, "vuelca la batería de revisiones en vez de decir para qué sirve la HbA1c; "
        "el documento la nombra en tres sitios pero nunca explica qué mide"),
    8: (True, False, "se abstiene bien: el documento no dice quién enseña a inyectarse"),
    9: (True, False, "§Objetivos del tratamiento literal: no hay diabetes «buena» ni «mala», sino "
        "bien o mal controlada"),
    10: (True, True, "identifica la metformina como el fármaco más usado, y en el sobrepeso; pero "
         "no contesta el «para siempre», y lo dice comentando la fuente («no se menciona en la "
         "información»), que es lo único que su propio diseño le prohíbe"),
    11: (True, False, "§Alimentación: perder entre un 5 % y un 10 % del peso mejora la glucemia, "
         "la tensión, los lípidos y el riesgo cardiovascular. Contesta el «cuánto» exacto"),
    12: (True, False, "§Causas: da la combinación entera y sitúa la obesidad abdominal como uno de "
         "los principales factores, sin convertirla en la causa única"),
    13: (True, False, "se abstiene bien: el documento manda anotar los autoanálisis, no la comida"),
    14: (True, False, "§Alcohol: aporta calorías y favorece las hipoglucemias; con moderación, "
         "nunca en ayunas y evitando las de alta graduación"),
    15: (True, False, "§Pie diabético: la diabetes altera la sensibilidad y la circulación de los "
         "pies, y eso aumenta el riesgo de heridas, infecciones y úlceras"),
    16: (True, False, "§Síntomas: la diabetes tipo 2 puede no dar síntomas durante años y se "
         "descubre habitualmente en un análisis"),
    17: (True, False, "contesta que sí y por qué: el tratamiento es integral y se controlan a la "
         "vez glucosa, tensión, colesterol, peso y tabaco, con las revisiones detrás"),
    18: (True, True, "el contenido está —rotar las zonas para evitar lipodistrofias—, pero pierde "
         "el «No» explícito ante una pregunta de sí o no"),
    19: (True, False, "se abstiene bien: el documento no da plazos de respuesta al tratamiento"),
    20: (False, False, "se abstiene; §Tratamiento farmacológico dice que el tratamiento es "
         "progresivo y §Insulina que si los otros tratamientos no controlan puede necesitarse "
         "insulina, que es exactamente lo que se pregunta"),
    21: (False, False, "se abstiene ante los primeros pasos, que §Tratamiento farmacológico sí "
         "enumera: alimentación, ejercicio y pérdida de peso"),
    22: (True, False, "limitar el chocolate a un consumo ocasional, y ningún alimento prohibido, "
         "se ajusta la cantidad. Contesta el «desde ya» sin prohibir"),
    23: (True, False, "§Tratamiento farmacológico entero: progresivo, de la alimentación a los "
         "fármacos, con las cuatro combinaciones posibles y la individualización"),
    24: (True, False, "ningún alimento prohibido, y los tubérculos en cantidad controlada evitando "
         "los fritos. Contesta por el pan y por la patata"),
    25: (True, False, "§Alimentación: la misma dieta equilibrada que la población general, "
         "reduciendo calorías si hay exceso de peso, y ningún alimento prohibido"),
    26: (True, False, "§Ejercicio físico: al menos 150 minutos a la semana repartidos en varios "
         "días, y caminar entre 30 y 45 minutos diarios, con la progresividad delante"),
    27: (False, False, "se abstiene; §Tratamiento farmacológico dice «tienes diabetes aunque no "
         "uses insulina» y enumera tratarse solo con alimentación y ejercicio, o con pastillas"),
    28: (True, False, "§Alcohol: con moderación, nunca en ayunas y evitando las bebidas de alta "
         "graduación"),
    29: (True, True, "el contenido es §Días de enfermedad y llega a la fiebre —paracetamol, "
         "cuidado con los sobres y jarabes con azúcar—, pero vuelca la sección entera en vez de "
         "contestar la fiebre, y se corta a media palabra al agotar el largo máximo de respuesta, "
         "justo antes del aviso de consultar por encima de 39 °C. Es la única de las 134 que se "
         "corta"),
    30: (False, False, "«los glucómetros miden la glucosa capilar» no contesta si hay que "
         "comprarlo; §Autoanálisis sostenía la respuesta"),
    31: (True, True, "recupera «aprender a vivir con la diabetes, y no para la diabetes» y la "
         "normalización del miedo, que es la vida normal por la que se pregunta; pero sigue "
         "abriendo por las vacaciones, que no es la sección"),
    32: (True, False, "§Pie diabético completo, de la revisión diaria al aviso por herida, pus, "
         "cambio de color o pérdida de sensibilidad"),
    33: (True, False, "§Síntomas: puede no notarse nada durante años, con la lista de síntomas "
         "posibles"),
    34: (True, False, "§Seguimiento sanitario: revisiones periódicas para detectar las "
         "complicaciones antes de que den síntomas"),
    35: (True, False, "§Viajes y vacaciones completo, incluida la medicación en el equipaje de "
         "mano y las temperaturas extremas"),
    36: (True, False, "se abstiene bien: el documento no dice qué llevar a la cita"),
    37: (True, False, "se abstiene bien: el documento no compara casos entre familiares"),
    38: (True, False, "se abstiene bien: el documento no trata el ámbito laboral"),
    39: (True, False, "§Grupos de alimentos: ningún suplemento y ningún producto milagro cura la "
         "diabetes. Respuesta muy escueta"),
    40: (True, False, "§Alimentación literal: ningún alimento prohibido, se ajusta la cantidad en "
         "lugar de eliminarlo"),
    41: (True, False, "§Aspectos psicológicos: normaliza el miedo tras el diagnóstico y añade "
         "«vivir con, no para»"),
    42: (False, False, "responde «la insulina reduce las complicaciones, incluida la ceguera» a "
         "«¿me quedaré ciego?»: fuera de diana, y le nombra la ceguera a quien teme quedarse "
         "ciego. §Complicaciones oculares, con la retinopatía y las revisiones, era la sección"),
    43: (True, False, "se abstiene bien: el documento habla de heridas y úlceras, nunca de "
         "amputación"),
    44: (True, True, "se abstiene; §Aspectos psicológicos manda identificar las situaciones "
         "difíciles y pedir ayuda, pero en genérico, y el documento no trata la vergüenza al "
         "inyectarse"),
    45: (True, False, "§Introducción: la diabetes tipo 2 es una enfermedad crónica, que contesta "
         "el «¿se cura?». Respuesta muy escueta"),
    46: (True, False, "se abstiene bien: el documento no hace pronóstico individual"),
    47: (True, False, "§Aspectos psicológicos «no debes culparte», que es exactamente lo que se "
         "pregunta"),
    48: (True, False, "§Aspectos psicológicos: es normal sentir preocupación, negación, "
         "frustración o miedo tras el diagnóstico"),
    49: (True, False, "se abstiene bien: el documento nombra la diabetes gestacional pero no trata "
         "la fertilidad"),
    50: (True, False, "se abstiene bien: el documento no habla de esperanza de vida"),
    51: (True, False, "§Alimentación: ningún alimento prohibido, que contesta el miedo a no volver "
         "a disfrutar"),
    52: (True, False, "se abstiene bien: §Hipoglucemia describe síntomas y tratamiento pero no "
         "afirma mortalidad"),
    53: (True, True, "se abstiene; §Aspectos psicológicos manda pedir ayuda ante las situaciones "
         "difíciles, pero en genérico, y el documento no trata el miedo a las agujas"),
    54: (True, True, "«la herencia es una de las causas» acota el «seguro» sin afirmarlo, pero "
         "deja sin contestar la culpa, que §Aspectos psicológicos sí cubre, y remata con la "
         "abstención después de haber respondido"),
    55: (True, False, "§Aspectos psicológicos: buscar apoyo en la familia, en el entorno y en las "
         "asociaciones"),

    # --------------------------------------------------------- cirugía abdominal
    56: (True, True, "la cirugía abierta usa incisiones mayores, que pueden dar más dolor y "
         "alargar la recuperación; contesta por un lado del contraste y deja el otro implícito"),
    57: (True, False, "§Cirugía mínimamente invasiva: el gas insuflado puede dar molestia "
         "abdominal 1 o 2 días y desaparece al absorberse"),
    58: (True, False, "§Cribado nutricional: detecta desnutrición, porque un paciente bien nutrido "
         "tiene menos complicaciones"),
    59: (True, False, "§Bebidas con carbohidratos: evita los efectos del ayuno preoperatorio, "
         "incomodidad, hambre y sed"),
    60: (True, False, "§Control del dolor: define el catéter epidural entero, un tubo fino y "
         "flexible en la columna conectado a una bomba que bloquea los nervios del dolor"),
    61: (True, False, "§Reanudar la alimentación literal: beber y comer a las pocas horas es "
         "seguro y no aumenta el riesgo de que se abra la sutura"),
    62: (True, False, "§Reanudar la alimentación: la primera defecación suele producirse 2 o 3 "
         "días después de reiniciar la alimentación"),
    63: (True, True, "da los efectos que el documento atribuye a la epidural —vértigo o debilidad "
         "en las piernas, pasajeros—, pero se deja el marco de esa misma frase: que todos los "
         "fármacos para el dolor pueden producir efectos no deseados"),
    64: (True, False, "§Levantarse de la cama: caminar pronto, por el riesgo de coágulos, "
         "debilidad muscular y neumonía"),
    65: (True, False, "§Cribado nutricional: unas pruebas se hacen a todos los pacientes y otras "
         "son específicas según la cirugía"),
    66: (True, False, "§Bebidas con carbohidratos: líquidos sin riesgo hasta 2 horas antes"),
    67: (True, False, "§Premedicación con su condición intacta, «cuando el grado de ansiedad y "
         "temor sea elevado», y las dos pastillas"),
    68: (True, False, "§Control del dolor completo: pauta fija más rescate, intravenoso las "
         "primeras 24 o 48 horas y luego oral, la bomba de ACP y la epidural"),
    69: (True, False, "§Bebidas con carbohidratos: con diabetes, el cirujano indica cómo proceder, "
         "con el control de glucosa previo"),
    70: (True, False, "§Control del dolor: la bomba de ACP está toda programada y no hay peligro "
         "de sobredosis"),
    71: (True, False, "§Cribado nutricional: un paciente bien nutrido tiene menos complicaciones. "
         "Respuesta muy escueta"),
    72: (True, False, "se abstiene bien: el documento atribuye la molestia por el gas al abdomen, "
         "nunca al hombro"),
    73: (True, False, "se abstiene bien: el documento regula las horas previas, no los días "
         "previos"),
    74: (True, False, "§Control del dolor: la intensidad es máxima en las primeras 24 horas y "
         "después disminuye"),
    75: (True, False, "§Levantarse de la cama: el mismo día, sentarse en un sillón con ayuda; al "
         "día siguiente, cortos paseos"),
    76: (False, False, "responde «la recuperación puede llevar varios días o semanas» a la "
         "duración del INGRESO: el documento no da la estancia hospitalaria y esa frase es de "
         "otra cosa"),
    77: (True, False, "se abstiene bien: el documento no trata la higiene previa"),
    78: (True, False, "§Reanudar la alimentación: lo antes posible, preferiblemente en las "
         "primeras 24 horas, y de forma progresiva"),
    79: (True, False, "se abstiene bien: el documento termina en el alta y no trata la vida "
         "posterior"),
    80: (True, False, "§Colaboración de familiares: los profesionales les informan para que "
         "participen, y la respuesta conserva el sujeto en tercera persona donde el documento "
         "habla del paciente"),
    81: (True, False, "§Quién informa completo: la consulta de cirugía, la cita con anestesia y la "
         "información escrita de enfermería"),
    82: (True, False, "§Quién informa: se decide sobre el tratamiento y se firma el consentimiento "
         "escrito"),
    83: (True, False, "§Qué es la cirugía mayor: «suele requerir anestesia general», sin cerrarlo "
         "en una certeza, con la premedicación detrás"),
    84: (True, False, "da primero el género —evitar abrir las cavidades, incisiones pequeñas— y "
         "luego la laparoscopia como caso concreto"),
    85: (False, False, "se abstiene; §Qué es la cirugía mayor dice que «la recuperación puede "
         "llevar varios días o semanas», que es material para esta pregunta. Es la misma frase "
         "que en la 76 usa donde no tocaba"),
    86: (True, False, "§Premedicación con su condición: cuando el grado de ansiedad y temor sea "
         "elevado"),
    87: (True, False, "§Levantarse de la cama: el «con ayuda» queda donde el documento lo pone, en "
         "sentarse, no en caminar"),
    88: (True, False, "§Levantarse de la cama: no solo no es peligroso, es lo indicado, y da el "
         "porqué"),
    89: (True, True, "la bebida con hidratos evita el hambre del ayuno, que es coherente y está "
         "apoyado; pero no dice que no se deba improvisar ninguna ingesta"),
    90: (True, False, "§Reanudar la alimentación: 2 o 3 días después de reiniciar la alimentación"),
    91: (True, True, "§Premedicación: normaliza la reacción emocional, aunque sin decir «sí, es "
         "normal» de forma explícita"),
    92: (True, False, "se abstiene bien: el documento no cuantifica el riesgo vital"),
    93: (True, False, "§Control del dolor: máxima en las primeras 24 horas y después disminuye"),
    94: (True, False, "se abstiene bien: el documento no trata el despertar intraoperatorio"),
    95: (True, False, "se abstiene bien: el documento no dice qué ocurre si hay una complicación"),
    96: (True, False, "§Premedicación: comunicar la sensación al equipo, y la pastilla para dormir "
         "la noche antes"),
    97: (True, False, "se abstiene bien: el documento no menciona ostomías"),
    98: (True, False, "se abstiene bien: este documento no recoge la retirada del consentimiento, "
         "que sí está en el de hemorroides"),
    99: (False, False, "responde con la cita de anestesia, donde informan del plan y los riesgos; "
         "no inventa ningún peligro, pero el documento no dice si la anestesia es peligrosa y la "
         "pregunta queda sin contestar. La v2 se abstenía aquí"),
    100: (True, False, "contrasta las incisiones mayores de la abierta con las pequeñas de la "
          "mínimamente invasiva, y conserva que a veces hace falta una algo mayor"),
    101: (True, False, "se abstiene; el documento dice que se pueden consultar las dudas en "
          "cualquier momento, pero no que se pueda hacer a solas, que es lo que se pregunta"),
    102: (True, False, "se abstiene bien: el documento no da la duración del ingreso"),
    103: (True, False, "§Premedicación: cualquier intervención provoca alguna reacción emocional"),

    # --------------------------------------------------------------- hemorroides
    104: (False, False, "se abstiene; el documento tiene las dos partes del contraste —la "
          "hemorroidectomía extirpa las hemorroides, y la ligadura con bandas elásticas es una "
          "alternativa—, y son las mismas que sí une en la 109 para el láser"),
    105: (True, False, "«anestesia regional o anestesia general»: conserva la disyunción del "
          "documento en vez de cerrarla en una sola"),
    106: (True, False, "§En qué consiste: entre 30 y 60 minutos"),
    107: (True, False, "§Riesgos: las muy raras, estrechamiento anal, incontinencia y sangrado "
          "abundante, conservando la rareza"),
    108: (True, False, "§Preparación: el equipo médico ajusta la medicación habitual y ese ajuste "
          "incluye los anticoagulantes; el sujeto clínico sigue en su sitio"),
    109: (True, False, "contrasta la hemorroidectomía con la coagulación por láser o infrarrojos "
          "como alternativa a la intervención"),
    110: (True, True, "§Beneficios «reduce las recaídas» contesta de refilón la recurrencia; "
          "respuesta muy escueta"),
    111: (True, False, "se abstiene bien: el documento manda fibra y líquidos pero nunca dice por "
          "qué"),
    112: (True, False, "§Riesgos: la incontinencia, conservando el «muy raramente»"),
    113: (True, False, "se abstiene bien: el documento no trata la baja laboral"),
    114: (True, False, "§Preparación: el ayuno de 6 a 8 horas contesta directo por el día antes"),
    115: (True, False, "§Riesgos: el dolor al defecar, con su frecuencia"),
    116: (True, False, "§Cuidados después: de 2 a 4 semanas"),
    117: (True, False, "se abstiene bien: el documento nombra los baños de asiento pero no los "
          "define"),
    118: (True, False, "§Cuidados después: dieta rica en fibra y en líquidos durante la "
          "recuperación"),
    119: (True, False, "§En qué consiste: entre 30 y 60 minutos"),
    120: (True, False, "se abstiene bien: el documento no dice si hay ingreso"),
    121: (True, False, "§Alternativas completas: pomadas, baños, dieta, ligadura con bandas "
          "elásticas y coagulación con láser o infrarrojos"),
    122: (True, False, "§Qué ocurre si no te operas: los síntomas pueden empeorar"),
    123: (True, False, "§Tu consentimiento: se puede retirar en cualquier momento antes de la "
          "cirugía"),
    124: (True, False, "§Preparación: entre 6 y 8 horas de ayuno"),
    125: (True, False, "§Riesgos: sangrado leve con frecuencia y sangrado abundante muy raramente, "
          "con las dos frecuencias"),
    126: (True, False, "se abstiene bien: este documento no tiene sección psicológica y no da "
          "material sobre la vergüenza"),
    127: (True, False, "§Cuidados después: dolor moderado. Respuesta muy escueta"),
    128: (True, False, "§Riesgos: la incontinencia, y conserva el «muy raramente» ante un miedo al "
          "«para siempre»"),
    129: (True, False, "se abstiene bien: el documento no trata el riesgo vital"),
    130: (True, False, "se abstiene bien: este documento no tiene sección psicológica y no da "
          "material sobre el miedo"),
    131: (True, False, "se abstiene bien: este documento no tiene sección psicológica y no da "
          "material sobre la vergüenza"),
    132: (True, False, "se abstiene bien: el documento no describe quién está en el quirófano"),
    133: (False, False, "se abstiene; §Riesgos da las muy raras —estrechamiento anal, "
          "incontinencia—, que son secuelas permanentes, y son las mismas que sí responde en la "
          "112 y en la 128"),
    134: (False, False, "se abstiene; §Qué ocurre si no te operas dice que los síntomas pueden "
          "empeorar, que es la misma frase que sí sirve en la 122. La v2 respondía aquí"),
}

require_complete(READ, TRIAGE, "lectura de eval/d1c-tu (audit_hand_v22)")


def _load(run_dir: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for proc in PROCEDURES:
        for row in json.loads(Path(f"{run_dir}/{proc}.json").read_text())["rows"]:
            out[row["id"]] = (row.get("our_answer") or "").strip()
    return out


def verdicts(_answers: dict[int, str] | None = None) -> dict[int, tuple[bool, bool, str]]:
    """La lectura completa. El argumento se acepta por simetría con `audit_hand`."""
    return require_complete(dict(READ), TRIAGE, "lectura de eval/d1c-tu")


def _scorecard(answers: dict[int, str], hand: dict[int, tuple[bool, bool, str]],
               title: str) -> None:
    print(f"{title}\n")
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
    print(f"\nLa banda son las {len(marg)} marginales {sorted(marg)}: apoyadas y sin invención,\n"
          "pero sin responder del todo lo que se preguntaba. El extremo bajo las descuenta.")
    print("Una sola tirada; gemma voltea ~1 % por seed.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", help="ids separados por comas: enseña el motivo")
    ap.add_argument("--diff-v2", action="store_true",
                    help="compara con la v2 bajo esta misma lectura")
    args = ap.parse_args()

    answers = _load(V22_DIR)
    hand = verdicts()

    if args.show:
        for qid in (int(x) for x in args.show.split(",")):
            ok, marg, why = hand[qid]
            tag = "CORRECTA" if ok else "INCORRECTA"
            print(f"{qid:>4} {tag}{' (marginal)' if marg else ''}: {why}")
        return 0

    if args.diff_v2:
        # Las dos lecturas están completas y escritas, así que comparar es
        # comparar veredictos. Sin umbrales ni predicados de por medio.
        from audit_hand import verdicts as verdicts_v2

        v2 = verdicts_v2()
        n_v2 = sum(v2[q][0] for q in TRIAGE)
        n_v22 = sum(hand[q][0] for q in TRIAGE)
        print(f"v2   (eval/ec2, leído)     : {n_v2}/134")
        print(f"v2.2 (eval/d1c-tu, leído)  : {n_v22}/134")
        gana = sorted(q for q in TRIAGE if hand[q][0] and not v2[q][0])
        pierde = sorted(q for q in TRIAGE if not hand[q][0] and v2[q][0])
        print(f"\nla v2.2 gana {gana}")
        print(f"la v2.2 pierde {pierde}")
        return 0

    _scorecard(answers, hand, "Lectura a mano sobre eval/d1c-tu — las 134, una a una")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
