# Encargo: reescribir un documento informativo para pacientes

> Este fichero es el brief que se entrega a una sesión independiente, sin
> contexto de este proyecto, para que reescriba un documento del corpus. Se
> escribe así a propósito: quien lo aplica **no debe conocer las preguntas con
> las que se evaluará el resultado**, o estaría escribiendo hacia el test.
>
> Las reglas de la sección «Reglas de redacción» no son estilo: cada una sale de
> un experimento medido sobre este corpus. Son el germen del documento de
> guidelines (tarea 6).

## Contexto

Un sistema de preguntas y respuestas responde dudas de pacientes usando **un
único documento** por procedimiento. El sistema tiene prohibido usar
conocimiento propio: solo puede decir lo que el documento dice, y si el
documento no resuelve la pregunta debe responder «No tengo información sobre
eso».

Eso convierte la redacción del documento en la variable que decide la calidad
del servicio. Un documento correcto pero mal redactado produce dos fallos
distintos: el sistema rechaza preguntas que el documento sí podría responder, o
las responde con una frase que comparte vocabulario con la pregunta pero no la
contesta.

## Tu encargo

Reescribir el documento de **cirugía de hemorroides** que se adjunta abajo,
aplicando las reglas de redacción. Debes producir **dos versiones**:

### Versión A — `hemorroides.vA.md`

**Exactamente los mismos hechos que el original. Ni uno más, ni uno menos.**
Solo cambia la forma. Esta versión aísla el efecto de la redacción, así que
cualquier hecho añadido la invalida como experimento.

Reformular no es añadir: convertir «Ayuno 6–8 h» en «Usted debe permanecer en
ayunas entre 6 y 8 horas antes de la intervención» explicita un sujeto elidido y
es correcto. Añadir «no debe suspender su medicación por su cuenta» **no** lo
es, por muy razonable que sea clínicamente: esa frase no está en el original.

Cuando dudes de si algo es reformulación o añadido, **déjalo fuera** y anótalo
en el informe final.

### Versión B — `hemorroides.vB.md`

Las mismas reglas, pero además **puedes ampliar el contenido**, con un límite
estricto: solo con material que esté en los documentos fuente que se te indiquen.
**No puedes usar tu conocimiento clínico propio, ni siquiera para cosas que
sepas ciertas.** El documento resultante es material de prueba y no se desplegará
sin validación clínica, pero aun así no debe contener una sola afirmación cuya
procedencia no puedas señalar.

Por cada bloque que añadas, anota de qué fichero y de qué sección sale.

Esta versión mide algo distinto de la A: cuánto del problema es *forma* y cuánto
es que el documento original es demasiado breve para las preguntas que recibe.

## Reglas de redacción

Cada una está medida, y varias contradicen la intuición. En particular: **«usa
frases completas» por sí solo NO funciona** — se probó y empeoró el resultado.

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
la afirme o la niegue directamente. Piensa qué preguntará un paciente ante este
procedimiento y comprueba que para cada una hay una frase, no un razonamiento.
Deriva esas preguntas del procedimiento y del sentido común clínico.

**R4. El contenido clínico va en su sección temática.** Nada importante puede
vivir únicamente en una sección de mitos, creencias, FAQ o anexos. El sistema
recupera por vocabulario: si una palabra clave solo aparece en una lista de
mitos, responderá con esa línea aunque no venga a cuento.

**R5. Sujeto explícito en toda acción.** Debe quedar claro quién hace cada cosa:
el paciente, el equipo médico o el sistema sanitario. Nada de infinitivos sueltos
(«Ajustar medicación») ni de imperativos sin destinatario: el sistema le asigna
como sujeto a quien pregunta, y convierte en instrucción al paciente algo
dirigido al equipo clínico. Es el defecto más grave que produce este corpus.

**R6. Frases completas con verbo conjugado.** Nada de viñetas nominales
telegráficas («Ayuno 6–8 h.», «Duración: 30–60 minutos.»). Pero ojo: esto es
necesario y **no suficiente** — R1, R2 y R3 son las que deciden.

**R7. No compenses con longitud.** No repitas, no añadas fórmulas de cortesía ni
encuadres genéricos («toda cirugía tiene riesgos»). Cada frase debe aportar un
hecho o responder una pregunta.

## Documento original a reescribir

```markdown
# RESUMEN Cirugía de hemorroides

1. Procedimiento
- La hemorroidectomía es la cirugía para extirpar las hemorroides internas o externas cuando producen sangrado, dolor o molestias importantes y no han mejorado con tratamientos menos invasivos.
- Se realiza en quirófano, con anestesia regional o general.
- Duración: 30–60 minutos.

2. Preparación y cuidados posteriores
- Ayuno 6–8 h.
- Ajustar medicación (anticoagulantes, etc.).
- Tras la cirugía: dolor moderado, baños de asiento, dieta rica en fibra y líquidos, recuperación en 2–4 semanas.

3. Beneficios
- Elimina sangrado, dolor y molestias.
- Reduce recaídas.

4. Riesgos y complicaciones
- Frecuentes: dolor al defecar, sangrado leve.
- Menos frecuentes: infección, retraso en la cicatrización.
- Muy raras: estrechamiento anal, incontinencia, sangrado abundante.

5. Alternativas
- Tratamientos médicos (pomadas, baños, dieta).
- Ligadura con bandas elásticas, coagulación con láser o infrarrojos.

6. Aspectos prácticos y legales
- Si no se opera, los síntomas pueden empeorar.
- Puede retirar su consentimiento en cualquier momento antes de la cirugía.
```

## Fuentes permitidas para la versión B

Solo estas, en `corpus/sources/` del repositorio:

- `via-clinica-cirugia-adulto-rica-2021-paciente.md`
- `gpc_555_cma_iacs_compl-pacientes.md`
- `resumen-fisura-anal.md` — es otro procedimiento, pero comparte la vía
  perianal y el circuito quirúrgico. Úsalo solo para lo que sea claramente
  común, y márcalo.

No hay documento fuente propio de hemorroides: el «RESUMEN» de arriba es todo el
material específico que existe.

## Qué entregar

1. `corpus/markdown/hemorroides.vA.md`
2. `corpus/markdown/hemorroides.vB.md`
3. Un informe corto con:
   - las preguntas de paciente que has previsto (R3) y dónde las responde cada
     versión;
   - qué decidiste dejar fuera de la A por dudar entre reformulación y añadido;
   - la procedencia de cada bloque añadido en la B;
   - cualquier punto donde las reglas chocaran entre sí y cómo lo resolviste.

## Lo que NO debes hacer

- No busques ni pidas el conjunto de preguntas de evaluación. Si lo encuentras
  en el repositorio, **no lo abras**: invalidaría el experimento.
- No optimices para preguntas concretas que creas que se van a evaluar.
- No añadas contenido clínico de tu propio conocimiento en ninguna de las dos
  versiones.
