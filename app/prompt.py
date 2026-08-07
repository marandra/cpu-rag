"""Shared prompt templates used by /query, and the A/B variants of them.

`SYSTEM_PROMPT_TEMPLATE` is V13 and is what we serve. Do not touch it: the
snapshot cache keys on the system prompt, so a byte of drift orphans every
pickle the pools serve and silently rebuilds them.

The variants exist so the prompt A/B can put one variant per worker process
(`PROMPT_VARIANT=v14b`) with no code edit. Each is V13 plus **one** change, so
that an effect is attributable; they are an isolation experiment, not
deliverable candidates. See docs/estado.md.

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


def _replacing_all(base: str, old: str, new: str) -> str:
    """Como `_replacing`, para un literal que V13 repite en regla y ejemplos."""
    if old not in base:
        raise ValueError(f"V13 no contiene el fragmento a sustituir: {old!r}")
    return base.replace(old, new)


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

# --- serie G: el A/B sobre gemma-4 (A3, 2026-07-23) -------------------------
#
# Por qué una serie nueva y no más v14: aquella se corrió sobre Ministral y
# atacaba defectos que **gemma ya no comete** — 0 invenciones y 0 fusiones en las
# 134, con 1 % de volteo por seed frente al 22 %. La pregunta cambia: no es «¿qué
# prompt reduce el sobre-rechazo?» sino **qué parte de V13 sigue haciendo falta y
# qué le falta para la capa de calidad**, que es la que A2 midió (84 % / 78 %).
#
# Se mantienen las dos reglas de método de la serie v14: un cambio por variante,
# y el ejemplo por delante de la prohibición declarativa.

# G1 — ¿es peso muerto el andamiaje anti-invención? V13 dedica tres viñetas de
# PROHIBIDO y un ejemplo entero a un fallo que gemma no comete. Si quitarlo no
# cuesta nada, sobran ~200 t de prefijo **en cada petición**. Se quita el bloque
# entero —regla y su demostración— porque dejar el ejemplo enseñaría igualmente
# la conducta y el test no mediría nada.
_V13_ANTI_INVENCION = (
    "- Inventar datos, medicamentos, dosis, plazos, precios, equivalencias o "
    "procedimientos que no estén en la información.\n"
    "- Completar información parcial con tu conocimiento general.\n"
    "- Expandir listas, categorías o enumeraciones más allá de lo escrito. "
    "Si el texto usa una categoría amplia (\"fruta\", \"verduras\", "
    "\"anticoagulantes, etc.\"), no añadas nombres concretos por tu cuenta.\n"
)
_V13_EJ4 = (
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
)

# G2 — la tercera salida. Es la aportación más directamente accionable de los
# auditores: su código `Sense resposta` penaliza abstenerse de algo que «es
# podria contestar **o derivar**», y en eso tienen razón. Dianas leídas en A2:
# 99 (¿es peligrosa la anestesia?), 100 (¿cicatrices grandes?) y 76 (¿cuánto
# estaré ingresado?), las tres contadas hoy como fallo de decisión.
#
# Aquí la regla se mueve **además** del ejemplo, por el mismo motivo que en
# V14a: la REGLA solo ofrece dos salidas, así que un ejemplo que enseñe una
# tercera pelearía contra ella.
_V13_DOS_SALIDAS = (
    "Si no, responde EXACTAMENTE y sin añadidos: \"No tengo información sobre "
    "eso.\""
)
_G2_TRES_SALIDAS = (
    "Si no, y la INFORMACIÓN tampoco trata el tema, responde EXACTAMENTE y sin "
    "añadidos: \"No tengo información sobre eso.\" Si la INFORMACIÓN trata el "
    "tema pero no resuelve tu caso concreto, responde EXACTAMENTE y sin "
    "añadidos: \"Eso depende de tu caso; coméntalo con tu equipo médico.\""
)
_EJ_G2 = (
    "7. Tercera salida: el tema sí está, tu caso no:\n"
    "INFORMACIÓN: \"La escayola se retira a las seis semanas en la consulta de "
    "traumatología.\"\n"
    "PREGUNTA: A mí me la pusieron hace cuatro, ¿ya me la pueden quitar?\n"
    "RESPUESTA: Eso depende de tu caso; coméntalo con tu equipo médico.\n"
    "PREGUNTA: ¿Cuánto cuesta la operación?\n"
    "RESPUESTA: No tengo información sobre eso.\n"
    "(La primera es del mismo tema que el texto pero pide una decisión sobre tu "
    "caso concreto: deriva. La segunda el texto no la trata en absoluto: "
    "rechazo. No uses la derivación cuando el texto sí responde.)"
)

# G3 — la forma de la respuesta, que es la capa que A2 destapó. El sistema
# devuelve el apunte del corpus tal cual: «Ayuno 6–8 h.» a «¿puedo comer el día
# antes?». Este brazo prueba si eso lo arregla el prompt o solo el corpus, y por
# eso **se corre contra el corpus servido**, no contra el v4: si se cambian las
# dos cosas a la vez no se puede atribuir nada.
_EJ_G3 = (
    "7. Del apunte telegráfico, una frase completa:\n"
    "INFORMACIÓN: \"Escayola: 6 semanas. Revisión: consulta de traumatología.\"\n"
    "PREGUNTA: ¿Cuánto tiempo llevaré la escayola?\n"
    "RESPUESTA: Llevarás la escayola 6 semanas.\n"
    "(El texto está en apuntes sin verbo; tu respuesta no. Devuelve una frase "
    "completa que reenuncie lo que se pregunta, nunca el apunte suelto "
    "(«6 semanas.»). Completar la frase no es añadir: no metas ningún dato que "
    "el apunte no traiga.)"
)

# G4 — respuesta parcial explícita. Dianas de A2: 110, 128 y 133 preguntan por
# **permanencia** («¿para siempre?») y el sistema contesta **frecuencia** («muy
# rara»), que no es la pregunta. Choca de frente con la viñeta de PROHIBIDO que
# veta comentar lo que el texto no dice, así que esa viñeta es el cambio.
#
# Riesgo conocido y por eso se mide: esa prohibición existe porque el modelo
# filtraba meta-comentario al paciente (defectos 24 y 26). La viñeta nueva veta
# la fórmula meta y permite solo acotar el hueco.
_V13_NO_COMENTAR = (
    "- Comentar lo que el texto NO dice (\"no se menciona...\", \"no "
    "especifica...\"). Si parte de la pregunta no tiene respuesta, responde "
    "solo lo que sí está. Si NADA tiene respuesta, usa la frase exacta de "
    "rechazo.\n"
)
_G4_ACOTAR = (
    "- Comentar la fuente o el documento (\"no se menciona...\", \"el texto no "
    "especifica...\"). Si parte de la pregunta no tiene respuesta, responde lo "
    "que sí está y acota en media frase qué parte no puedes responder, sin "
    "nombrar el documento. Si NADA tiene respuesta, usa la frase exacta de "
    "rechazo.\n"
)
_EJ_G4 = (
    "7. Responde lo cubierto y acota lo que no:\n"
    "INFORMACIÓN: \"Al retirar la escayola es normal notar el brazo rígido unos "
    "días.\"\n"
    "PREGUNTA: ¿Se me va a quedar el brazo rígido para siempre?\n"
    "RESPUESTA: Al retirar la escayola es normal notar el brazo rígido unos "
    "días; sobre si esa rigidez es permanente no tengo información.\n"
    "(Da primero lo que sí está. El hueco se acota en media frase, sin nombrar "
    "el documento y sin fórmulas como «no se especifica».)"
)


# D1 — el texto de la abstención. Hoy 45 preguntas reciben "No tengo información
# sobre eso."; 37 se abstienen con razón y 21 de esas son emocionales («me da
# vergüenza pincharme delante de otros», «tengo miedo a las agujas»). El sistema
# acierta al no tener nada que decir y lo dice de la peor forma posible.
#
# Por qué esto y no la derivación de G2: G2 se midió (9 semillas) y solo convirtió
# 4 de las 21 emocionales, a cambio de derivar 7 preguntas que el documento sí
# responde. Reescribir el literal alcanza a las 37 y **no puede desplazar una
# respuesta correcta**, porque solo cambia lo que ya era una abstención: la REGLA
# sigue ofreciendo dos salidas y el test para elegirlas no se toca.
#
# El literal se sustituye en la regla y en los tres ejemplos a la vez; dejar los
# ejemplos con la frase vieja enseñaría la frase vieja.
#
# Registro: usted, y con él los tres ejemplos de respuesta de V13, que tuteaban.
# Medido 2026-07-27 sobre `eval/ec2`: de 89 respuestas no-rechazo, **0** salen en
# tú y 56 en usted — el corpus (363 marcas de usted, 0 de tú) manda y el tuteo del
# prompt no se cuela. Así que unificar aquí no cambia nada por sí solo; se hace
# porque D1 ya paga el rebuild del snapshot y salir con dos tratos en el mismo
# prompt es incoherente. El día que el corpus pase a tú, esto se gira con él.
_V13_RECHAZO = "No tengo información sobre eso."
_D1_RECHAZO = (
    "Esto es algo que conviene comentar con su equipo sanitario, que podrá "
    "orientarle teniendo en cuenta su caso. En la información de la que dispongo "
    "no se trata este tema."
)

# Los tres RESPUESTA del template que tuteaban. Solo se giran las que habla el
# paciente: las instrucciones al modelo («no añadas… por tu cuenta») siguen en tú.
_D1_TUTEO = [
    ("RESPUESTA: Puedes tomar fruta entera a diario, mejor que en zumo.",
     "RESPUESTA: Puede tomar fruta entera a diario, mejor que en zumo."),
    ("RESPUESTA: Si decides no operarte, los síntomas pueden empeorar.",
     "RESPUESTA: Si decide no operarse, los síntomas pueden empeorar."),
    ("RESPUESTA: Después de la cirugía es normal sentir dolor moderado; debes "
     "hacer baños de asiento y seguir una dieta rica en fibra y líquidos.",
     "RESPUESTA: Después de la cirugía es normal sentir dolor moderado; debe "
     "hacer baños de asiento y seguir una dieta rica en fibra y líquidos."),
]


def _d1() -> str:
    p = _replacing_all(SYSTEM_PROMPT_TEMPLATE, _V13_RECHAZO, _D1_RECHAZO)
    for old, new in _D1_TUTEO:
        p = _replacing(p, old, new)
    return p


# D1b — solo el literal de abstención; los ejemplos se quedan como en V13.
#
# D1 agrupó dos cambios y no se pudo atribuir nada. Medido en diabetes (55 preg.):
# 18 abstenciones adoptan el texto nuevo, 0 respuestas se vuelven abstención, 3
# abstenciones pasan a responder —las tres marcadas Deficient por los auditores,
# la 45 una ganancia limpia— pero **19 respuestas cambian** y 4 pierden el «Sí/No»
# inicial (1, 17, 18, 47) sin que ninguna lo gane. Las marcas de «usted» se
# duplican (25→50): la huella del giro de registro. Ese brazo se descarta —coste
# medible, beneficio nulo (0 de 89 respuestas tuteaban ya)— y aquí queda el
# cambio solo, que es lo que se puede evaluar.
#
# Corrige de paso la premisa con la que se planificó esto («el literal no puede
# desplazar una respuesta correcta»): es falsa. Vive en el system prompt, y tocarlo
# cambia el prefijo KV entero y perturba todas las respuestas, no solo las
# abstenciones. Por eso este brazo también se lee entero, no solo en las 45.
#
# El texto va en usted a propósito: el modelo lo emite tal cual junto a respuestas
# que el corpus fuerza en usted. Manda el registro de la SALIDA, no el de los
# ejemplos (que siguen tuteando, como en V13).
def _d1b() -> str:
    return _replacing_all(SYSTEM_PROMPT_TEMPLATE, _V13_RECHAZO, _D1_RECHAZO)


# D1c — mismo cambio que D1b, mejor redactado. Dos cosas, ambas salidas de leer
# las 44 abstenciones de D1b:
#
#   1. Orden invertido. D1b termina en la parte fría («…no se trata este tema»);
#      cerrar en la derivación deja al paciente con la acción útil, no con el
#      límite del sistema.
#   2. Más corto: 99 caracteres frente a 168. No es cosmética — el literal se
#      emite en 44 de las 134, así que a ~6 tok/s son ~3 s menos en un tercio de
#      las preguntas. La longitud del prefijo casi no cuenta (el snapshot la
#      cachea una vez); la de la salida se paga en cada petición.
#
# Se mantiene «según su caso»: en las 24 preguntas emocionales es lo que hace que
# la frase reconozca a la persona. Se evita reabrir con «No tengo información
# sobre…», que era justo el arranque que motivó D1 (ahorraría 11 caracteres).
_D1C_RECHAZO = (
    "Esto no lo recoge la información que tengo, pero su equipo sanitario podrá "
    "orientarle según su caso."
)


def _d1c() -> str:
    return _replacing_all(SYSTEM_PROMPT_TEMPLATE, _V13_RECHAZO, _D1C_RECHAZO)


# D1c-tu — D1c con el literal tuteado, para servir el corpus convertido a tú
# (`*-tu.md`). El registro lo fija el corpus, no el prompt: medido sobre la v2,
# 0 de 89 respuestas tuteaban con un corpus en usted. Así que el literal tiene
# que girar CON el corpus o el paciente recibe las dos formas mezcladas.
#
# Los ejemplos del template ya tutean (V13 siempre lo hizo), así que aquí, por
# primera vez, prompt y corpus coinciden en trato.
_D1C_TU_RECHAZO = (
    "Esto no lo recoge la información que tengo, pero tu equipo sanitario podrá "
    "orientarte según tu caso."
)


def _d1c_tu() -> str:
    return _replacing_all(SYSTEM_PROMPT_TEMPLATE, _V13_RECHAZO, _D1C_TU_RECHAZO)


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
    # serie G — A3, sobre gemma-4
    "g1": _replacing(
        _replacing(SYSTEM_PROMPT_TEMPLATE, _V13_ANTI_INVENCION, ""),
        _V13_EJ4, ""),
    "g2": _plus_example(
        _replacing(SYSTEM_PROMPT_TEMPLATE, _V13_DOS_SALIDAS, _G2_TRES_SALIDAS),
        _EJ_G2),
    "g3": _plus_example(SYSTEM_PROMPT_TEMPLATE, _EJ_G3),
    "g4": _plus_example(
        _replacing(SYSTEM_PROMPT_TEMPLATE, _V13_NO_COMENTAR, _G4_ACOTAR),
        _EJ_G4),
    # D1 — texto de la abstención + registro en los ejemplos. Medido y DESCARTADO
    # por el segundo cambio; se conserva porque `eval/d1/` es su medición.
    "d1": _d1(),
    # D1b — solo el texto de la abstención. Bate a la v2 (125/134 vs 122).
    "d1b": _d1b(),
    # D1c — D1b con la frase invertida y a la mitad de largo. Es la SERVIDA (v2.1).
    "d1c": _d1c(),
    # D1c-tu — la misma, tuteada, para el corpus convertido a tú.
    "d1c-tu": _d1c_tu(),
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
