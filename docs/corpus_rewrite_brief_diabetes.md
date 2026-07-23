# Encargo: reescribir la guía de diabetes tipo 2 para pacientes

> Brief para una sesión independiente. Quien lo aplica **no debe conocer las
> preguntas con las que se evaluará el resultado**, o estaría escribiendo hacia
> el test. Las reglas no son estilo: cada una sale de un experimento medido.

## Contexto

Un sistema de preguntas y respuestas responde dudas de pacientes usando **un
único documento** por procedimiento. El sistema tiene prohibido usar
conocimiento propio: solo puede decir lo que el documento dice, y si el
documento no resuelve la pregunta debe responder «No tengo información sobre
eso».

Eso convierte la redacción del documento en la variable que decide la calidad
del servicio. Un documento correcto pero mal redactado produce tres fallos
medidos: el sistema **rechaza** preguntas que el documento sí resuelve; las
**responde con una frase que comparte vocabulario** con la pregunta pero no la
contesta; o **reasigna el sujeto** de una regla y convierte en instrucción al
paciente algo dirigido al equipo clínico.

## Tu encargo

Reescribir el documento adjunto abajo aplicando las reglas de redacción, y
producir `corpus/markdown/diabetes.v4.md`.

**Exactamente los mismos hechos que el original. Ni uno más, ni uno menos.**
Solo cambia la forma. Esta versión aísla el efecto de la redacción, así que
cualquier hecho añadido la invalida como experimento.

Reformular no es añadir: convertir «Ayuno 6–8 h» en «Usted debe permanecer en
ayunas entre 6 y 8 horas antes de la intervención» explicita un sujeto elidido y
es correcto. Añadir una recomendación que no está en el original **no** lo es,
por muy razonable que sea clínicamente.

**Reordenar y mover contenido de sección sí está permitido, y a veces es
obligatorio** (ver R4). Mover un hecho de sitio no es añadirlo.

Cuando dudes de si algo es reformulación o añadido, **déjalo fuera** y anótalo
en el informe.

## Reglas de redacción

Cada una está medida, y varias contradicen la intuición. En particular: **«usa
frases completas» por sí solo NO funciona** — se probó sobre este mismo corpus y
empeoró el resultado.

**R1. Autosuficiencia.** Cada frase debe llevar dentro su condición, su actor y
su alcance, de forma que reformularla no pueda perder nada. Si una regla y la
condición que la acota viven en frases distintas, el sistema conserva una y
suelta la otra.

**R2. El contenido accionable va en la oración principal.** Ésta es la regla más
importante y la menos obvia. El sistema conserva la oración principal y suelta lo
subordinado. Si escribes «Es su médico quien decide cómo ajustar su medicación»,
lo que queda en la principal es «su médico decide» y el ajuste se pierde. Escribe
el acto en la principal y cuelga de él la condición, el actor y el alcance.

**R3. Cada pregunta previsible necesita una frase que la responda.** Que la
respuesta sea *deducible* del documento no basta: tiene que existir una frase que
la afirme o la niegue directamente. Deriva esas preguntas del tema y del sentido
común clínico, y comprueba que para cada una hay una frase, no un razonamiento.
Si no hay material para responderla, **déjala sin responder y anótala** — no la
rellenes.

**R4. El contenido clínico va en su sección temática.** Nada importante puede
vivir únicamente en una sección de mitos, creencias, FAQ o anexos. El sistema
recupera por vocabulario: si una palabra clave solo aparece en una lista de
mitos, responderá con esa línea aunque no venga a cuento, y sonará absurda.
**Este documento tiene una sección de «Creencias frecuentes (mitos)» y una de
«Preguntas frecuentes»: cada hecho clínico que viva solo ahí debe pasar a su
sección temática.** Enuncia cada hecho una sola vez, en el sitio que le toca.

**R5. Sujeto explícito en toda acción.** Debe quedar claro quién hace cada cosa:
el paciente, el equipo sanitario o el sistema. Nada de infinitivos sueltos ni de
imperativos sin destinatario: el sistema le asigna como sujeto a quien pregunta,
y convierte en instrucción al paciente algo dirigido al equipo clínico. Es el
defecto más grave que produce este corpus.

**R6. Frases completas con verbo conjugado.** Nada de viñetas nominales
telegráficas («HbA1c ≥6,5 %.», «Síntomas: sed, cansancio.»). Pero ojo: esto es
necesario y **no suficiente** — R1, R2 y R3 son las que deciden.

**R7. No compenses con longitud.** No repitas ideas, no añadas fórmulas de
cortesía ni encuadres genéricos. Cada frase debe aportar un hecho o responder una
pregunta.

**Cuando R1 y R7 choquen, gana R1.** Está medido: entre tres reescrituras de otro
documento, la más natural y menos autosuficiente quedó la peor, y la que repetía
el anclaje en cada frase ganó por el doble. **Un texto que se lee peor sirve
mejor.** R7 prohíbe *frases* de relleno, no *palabras* repetidas.

## Aviso clínico

Este documento contiene reglas de días de enfermedad que **no debes corregir ni
matizar** aunque te parezcan discutibles clínicamente. Tu encargo es la forma.
Cualquier duda clínica que detectes, anótala en el informe y **no la arregles en
el texto**.

## Qué entregar

1. `corpus/markdown/diabetes.v4.md`
2. `eval/corpus_ab_v4/BLIND_INFORME_diabetes.md`, con:
   - las preguntas de paciente que has previsto (R3) y dónde las responde el
     texto; las que no tengan material, listadas aparte;
   - qué decidiste dejar fuera por dudar entre reformulación y añadido;
   - qué contenido moviste de sección por R4 y de dónde a dónde;
   - cualquier punto donde las reglas chocaran y cómo lo resolviste;
   - las dudas clínicas que hayas detectado y no tocado.

## Lo que NO debes hacer

- **No abras ni busques el conjunto de preguntas de evaluación.** En concreto, no
  leas `eval/`, `reports/`, `ARRANQUE.md`, `docs/corpus_guidelines.md`,
  `tools/audit_*.py` ni ningún fichero `.xlsx` del repositorio. Si te topas con
  ellos, no los abras: invalidarían el experimento.
- No optimices para preguntas concretas que creas que se van a evaluar.
- No añadas contenido clínico de tu propio conocimiento.
- No cambies ninguna cifra, ningún umbral ni ningún nombre de fármaco.

## Documento original a reescribir

El fichero es `corpus/markdown/diabetes.md`. Léelo de ahí (es el único fichero
del corpus que necesitas abrir) y escribe tu versión al lado, como
`diabetes.v4.md`. No modifiques el original.
