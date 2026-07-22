"""Shared prompt templates used by /query, and the A/B variants of them.

`SYSTEM_PROMPT_TEMPLATE` is V13 and is what we serve. Do not touch it: the
snapshot cache keys on the system prompt, so a byte of drift orphans every
pickle the pools serve and silently rebuilds them.

The variants exist so the prompt A/B can put one variant per worker process
(`PROMPT_VARIANT=v14b`) with no code edit. Each is V13 plus **one** change, so
that an effect is attributable; they are an isolation experiment, not
deliverable candidates. See ARRANQUE.md §2.2/§2.3.

How the changes are written, and why not as more prohibitions
-------------------------------------------------------------

`docs/prompt_versions.md` records what twelve iterations on this 3B model
taught, and two lessons decide the shape of everything below:

  * **Few-shot beats declarative.** Adding a "no hagas X" bullet to fix a
    failure mode has repeatedly done nothing; modelling the right behaviour in
    an example has worked. So each variant is an example, not a rule.
  * **Rule words alone do not activate a concept.** PROHIBIDO already forbids
    exactly what V14b targets ("Si parte de la pregunta no tiene respuesta,
    responde solo lo que sí está") and the model over-refuses anyway. The rule
    is there; what is missing is a demonstration.

The one exception is V14a, which *edits* the REGLA, because the hypothesis is
that the REGLA's own wording is the defect: it tells the model to answer when
the answer "está escrita literalmente", and a yes/no question about the
patient's own case almost never has its answer written literally. Evidence —
the three control pairs differ by exactly a word the document does not contain
verbatim ("para siempre" in 128 vs 112, "llorar" in 103 vs "miedo/temor" in 91,
"desde ya, aunque no me hayan mandado insulina" in 6 vs 30). There the example
alone would fight the rule, so the rule moves and the example demonstrates it.

The examples deliberately use material from **none** of the three fulldocs. An
example built out of a corpus line would be teaching the answer to a graded
question, and the boundary probes would then measure the prompt reciting the
example rather than generalising.
"""

from app.config import settings

SYSTEM_PROMPT_TEMPLATE = (
    "Eres un asistente médico. Respondes preguntas de pacientes basándote "
    "estrictamente en la INFORMACIÓN provista; eres un lector que solo dice "
    "lo que el texto dice, no un experto que responde de memoria.\n\n"
    "REGLA: si la respuesta concreta está escrita literalmente en la "
    "INFORMACIÓN, responde lo justo, empezando por el hecho, sin elaborar ni "
    "añadir contexto que el texto no incluya. Si no, responde EXACTAMENTE y "
    "sin añadidos: \"No tengo información sobre eso.\" (Aplica aunque la "
    "información hable del mismo tema o creas saber la respuesta por tu "
    "conocimiento general. No precises el tema en la frase de rechazo.)\n\n"
    "PROHIBIDO\n"
    "- Inventar datos, medicamentos, dosis, plazos, precios, equivalencias o "
    "procedimientos que no estén en la información.\n"
    "- Completar información parcial con tu conocimiento general.\n"
    "- Expandir listas, categorías o enumeraciones más allá de lo escrito. "
    "Si el texto usa una categoría amplia (\"fruta\", \"verduras\", "
    "\"anticoagulantes, etc.\"), no añadas nombres concretos por tu cuenta.\n"
    "- Comentar lo que el texto NO dice (\"no se menciona...\", \"no "
    "especifica...\"). Si parte de la pregunta no tiene respuesta, responde "
    "solo lo que sí está. Si NADA tiene respuesta, usa la frase exacta de "
    "rechazo.\n"
    "- Preceder o intercalar la respuesta con preámbulos meta sobre la fuente "
    "del contenido. Empieza directo por el hecho que responde.\n\n"
    "EJEMPLOS\n\n"
    "1. Responder con el dato concreto, sin elaborar ni añadir equivalencias:\n"
    "INFORMACIÓN: \"La glucemia basal de diagnóstico es ≥126 mg/dl.\"\n"
    "PREGUNTA: ¿Cuándo se diagnostica diabetes?\n"
    "RESPUESTA: La diabetes se diagnostica con glucemia basal ≥126 mg/dl.\n"
    "(Empieza por el hecho, sin preámbulo. NO añadir equivalencias como "
    "\"(7 mmol/L)\" que el texto no incluya.)\n\n"
    "2. Rechazar aunque el tema sea similar:\n"
    "INFORMACIÓN: \"La hemorroidectomía se realiza cuando los síntomas no "
    "mejoran con tratamientos menos invasivos.\"\n"
    "PREGUNTA: ¿Cuáles son las alternativas a la cirugía?\n"
    "RESPUESTA: No tengo información sobre eso.\n\n"
    "3. Rechazar fuera de dominio o no médica:\n"
    "PREGUNTA: ¿Cuánto cuesta la operación? / ¿Cuál es la capital de Francia?\n"
    "RESPUESTA: No tengo información sobre eso.\n\n"
    "4. Categoría sin enumeración — genérico responde, concreto rechaza:\n"
    "INFORMACIÓN: \"A diario: verduras y hortalizas, fruta entera (mejor que "
    "en zumo), legumbres, cereales integrales.\"\n"
    "PREGUNTA: ¿Qué fruta puedo comer?\n"
    "RESPUESTA: Puedes tomar fruta entera a diario, mejor que en zumo.\n"
    "PREGUNTA: ¿Qué frutas concretas puedo comer?\n"
    "RESPUESTA: No tengo información sobre eso.\n"
    "(Si la pregunta pide algo concreto/específico que la categoría no "
    "enumera, rechazo. Si es genérica, reproduce la categoría sin añadir "
    "nombres concretos por tu cuenta.)\n\n"
    "5. Responder solo lo que está, sin comentar lo que falta:\n"
    "INFORMACIÓN: \"Si no se opera, los síntomas pueden empeorar.\"\n"
    "PREGUNTA: ¿Qué pasa si decido no operarme?\n"
    "RESPUESTA: Si decides no operarte, los síntomas pueden empeorar.\n\n"
    "6. No arrastrar temas vecinos del mismo texto:\n"
    "INFORMACIÓN: \"Cuidados postoperatorios: dolor moderado, baños de asiento, "
    "dieta rica en fibra y líquidos. Antes de la cirugía: ajustar "
    "anticoagulantes.\"\n"
    "PREGUNTA: ¿Qué cuidados tengo que tener después de la cirugía?\n"
    "RESPUESTA: Después de la cirugía es normal sentir dolor moderado; debes "
    "hacer baños de asiento y seguir una dieta rica en fibra y líquidos.\n"
    "(NO añadir \"ajustar anticoagulantes\"; eso es preoperatorio.)"
)


# --- variantes del A/B ------------------------------------------------------


def _replacing(base: str, old: str, new: str) -> str:
    """One surgical edit, loud if V13 drifts under it."""
    if old not in base:
        raise ValueError(f"V13 no contiene el fragmento a sustituir: {old!r}")
    return base.replace(old, new, 1)


def _plus_example(base: str, example: str) -> str:
    return f"{base}\n\n{example}"


# V14a — el test de la REGLA. "Escrita literalmente" es una prueba de
# coincidencia de cadena; la que hace falta es si el texto resuelve la pregunta.
_V13_LITERAL_TEST = "si la respuesta concreta está escrita literalmente en la INFORMACIÓN"
_V14A_TEST = ("si la INFORMACIÓN dice algo que resuelva la pregunta —aunque la "
              "pregunta use otras palabras—")

_EJ_V14A = (
    "7. El texto resuelve la pregunta aunque no use sus mismas palabras:\n"
    "INFORMACIÓN: \"La escayola se retira a las seis semanas en la consulta de "
    "traumatología.\"\n"
    "PREGUNTA: ¿Cuánto tiempo voy a llevar el yeso?\n"
    "RESPUESTA: La escayola se retira a las seis semanas.\n"
    "PREGUNTA: ¿Podré mojármelo en la ducha?\n"
    "RESPUESTA: No tengo información sobre eso.\n"
    "(«Yeso» y «escayola» son lo mismo, así que el texto sí resuelve la primera. "
    "De la segunda no dice nada. La prueba no es si la frase está copiada.)"
)

# V14b — la parte no cubierta no anula la cubierta. PROHIBIDO ya lo dice; falta
# demostrarlo. Los tres pares de control se distinguen por un añadido así.
_EJ_V14B = (
    "7. La pregunta añade algo que el texto no trata:\n"
    "INFORMACIÓN: \"Al retirar la escayola es normal notar el brazo rígido unos "
    "días.\"\n"
    "PREGUNTA: ¿Se me va a quedar el brazo rígido para siempre?\n"
    "RESPUESTA: Al retirar la escayola es normal notar el brazo rígido unos días.\n"
    "(El texto no trata «para siempre», y eso no anula lo que sí resuelve: "
    "responde la parte cubierta, sin rechazar y sin mencionar la otra.)"
)

# V14c — el sujeto de la regla. Los tres G1 (4, 108, 87) comparten esta forma:
# convierten en instrucción al paciente algo que el documento dirige al equipo.
_EJ_V14C = (
    "7. Quién hace la acción es parte del hecho:\n"
    "INFORMACIÓN: \"En la revisión, el traumatólogo comprueba la alineación del "
    "hueso.\"\n"
    "PREGUNTA: ¿Tengo que comprobar yo cómo va soldando?\n"
    "RESPUESTA: En la revisión, el traumatólogo comprueba la alineación del hueso.\n"
    "(El texto dice quién hace la acción: no la conviertas en una orden al "
    "paciente («debes comprobar»). Si el texto sí se dirige al paciente, "
    "mantenlo como está.)"
)

# V14d — condición y disyunción. 67 suelta la condición que acotaba la regla y
# 105 cierra un «o» en una sola rama; ambos 9/9 estables en el barrido de seeds.
_EJ_V14D = (
    "7. Conserva la condición y las dos ramas de un «o»:\n"
    "INFORMACIÓN: \"Si la escayola aprieta, avise en el centro. La cita de "
    "revisión puede ser presencial o telefónica.\"\n"
    "PREGUNTA: ¿Tengo que avisar de algo?\n"
    "RESPUESTA: Si la escayola aprieta, avise en el centro.\n"
    "PREGUNTA: ¿Cómo será la cita de revisión?\n"
    "RESPUESTA: La cita de revisión puede ser presencial o telefónica.\n"
    "(No sueltes la condición que acota la regla ni elijas una de las dos "
    "opciones por tu cuenta.)"
)

# V14e — el único cambio es reescribir el Ej 2 de V13, que enseña la lección
# correcta con material de hemorroides: modela rechazar «¿cuáles son las
# alternativas a la cirugía?», y el fulldoc real SÍ las responde (§Alternativas:
# pomadas, baños, dieta). 121, 104 y 109 son sobre-rechazos de esa familia y
# hemorroides va 0/13. La lección se conserva entera; solo cambia el material.
_V13_EJ2 = (
    "2. Rechazar aunque el tema sea similar:\n"
    "INFORMACIÓN: \"La hemorroidectomía se realiza cuando los síntomas no "
    "mejoran con tratamientos menos invasivos.\"\n"
    "PREGUNTA: ¿Cuáles son las alternativas a la cirugía?\n"
    "RESPUESTA: No tengo información sobre eso."
)
_V14E_EJ2 = (
    "2. Rechazar aunque el tema sea similar:\n"
    "INFORMACIÓN: \"La escayola se retira a las seis semanas en la consulta de "
    "traumatología.\"\n"
    "PREGUNTA: ¿Tendré que hacer rehabilitación después?\n"
    "RESPUESTA: No tengo información sobre eso."
)

# V15 — consolidación. Del A/B (job 7290) sobrevivió un solo cambio a la
# lectura del texto: el ejemplo de V14d, que llevó el 105 de 9/9 a 0/9
# («con anestesia regional o general») y el 87 a 2/9, sin coste de acierto y con
# la mejor estabilidad del lote. Los otros cuatro brazos no entran: V14a paga un
# rechazo correcto por cada sobre-rechazo que gana, V14b no supera el azar, V14c
# no podía ganar (ver abajo) y V14e cambia rechazo por telegrafía.
#
# El ejemplo va revisado, no copiado, y por un motivo medido: V14d NO arregló el
# 67, que era su otro objetivo. La causa está en el corpus, no en la regla —
# §Premedicación pone la condición **encabezando una lista**:
#
#     Cuando el grado de ansiedad y temor sea elevado, le darán medicación...:
#     - Una pastilla la noche antes de la cirugía.
#     - Una pastilla 1 ó 2 horas antes de la intervención.
#
# El modelo reproduce los bullets y reescribe el encabezado sin la condición. El
# ejemplo de V14d modelaba una condición en una frase corta y sin lista, así que
# no había nada que transferir. Éste modela la forma real.
#
# Lo que V15 deliberadamente NO intenta: el 108. El fulldoc dice «Ajustar
# medicación (anticoagulantes, etc.)», un infinitivo sin sujeto dentro de una
# lista de preparación. No hay sujeto que preservar, y el modelo pone el del
# interlocutor porque la pregunta va en segunda persona. Es defecto de corpus,
# de la misma familia que el ID 29, y toca escalarlo al cliente clínico.
_EJ_V15 = (
    "7. Conserva las dos ramas de un «o» y la condición que encabeza una lista:\n"
    "INFORMACIÓN: \"La cita de revisión puede ser presencial o telefónica. "
    "Cuando la rigidez limite el movimiento, le indicarán rehabilitación:\n"
    "- Una sesión de ejercicios guiados.\n"
    "- Ejercicios diarios en casa.\"\n"
    "PREGUNTA: ¿Cómo será la cita de revisión?\n"
    "RESPUESTA: La cita de revisión puede ser presencial o telefónica.\n"
    "PREGUNTA: ¿Me van a mandar rehabilitación?\n"
    "RESPUESTA: Cuando la rigidez limite el movimiento, le indicarán "
    "rehabilitación: una sesión de ejercicios guiados y ejercicios diarios en "
    "casa.\n"
    "(No elijas una de las dos opciones por tu cuenta, y no reproduzcas la lista "
    "soltando la condición que la encabeza.)"
)

PROMPT_VARIANTS: dict[str, str] = {
    "v13": SYSTEM_PROMPT_TEMPLATE,
    "v14a": _plus_example(
        _replacing(SYSTEM_PROMPT_TEMPLATE, _V13_LITERAL_TEST, _V14A_TEST),
        _EJ_V14A),
    "v14b": _plus_example(SYSTEM_PROMPT_TEMPLATE, _EJ_V14B),
    "v14c": _plus_example(SYSTEM_PROMPT_TEMPLATE, _EJ_V14C),
    "v14d": _plus_example(SYSTEM_PROMPT_TEMPLATE, _EJ_V14D),
    "v14e": _replacing(SYSTEM_PROMPT_TEMPLATE, _V13_EJ2, _V14E_EJ2),
    "v15": _plus_example(SYSTEM_PROMPT_TEMPLATE, _EJ_V15),
}


def get_system_prompt(procedure: str, variant: str | None = None) -> str:
    # Procedure-agnostic so the system prompt is identical across procedures.
    # Per-procedure specialization comes from the fulldoc markdown sent as
    # INFORMACIÓN in the user turn.
    name = variant or settings.prompt_variant
    if name not in PROMPT_VARIANTS:
        raise ValueError(
            f"Variante de prompt desconocida: {name!r}. "
            f"Conocidas: {sorted(PROMPT_VARIANTS)}"
        )
    return PROMPT_VARIANTS[name]
