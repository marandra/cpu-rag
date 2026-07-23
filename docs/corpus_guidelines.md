# Guidelines de redacción del corpus

> Cómo debe estar escrito un documento informativo para pacientes para que el
> sistema de preguntas y respuestas pueda usarlo.
>
> Estado: **validado experimentalmente** el 2026-07-23 (job 7329) sobre el
> documento de hemorroides. Cada regla sale de una medición, no de un criterio
> de estilo. Origen: `docs/corpus_rewrite_brief.md`, que es la versión operativa
> que se entrega a quien reescribe.

## 0. ⚠️ Alcance: esto repara material telegráfico, no mejora cualquier texto

**Medido el 2026-07-23 (jobs 7341/7342), y es la corrección más importante de
este documento.** Las siete reglas se aplicaron a ciegas a los otros dos
documentos del corpus y el resultado se parte en dos según **cómo estuviera
escrito el original**:

| documento de partida | decisión | telegráficas |
| --- | ---: | ---: |
| **diabetes** — viñetas nominales, 13 KB | 83,4 % → **85,5 %** | 11 % → **7 %** |
| **cirugía abdominal** — ya en prosa, 7,6 KB | 93,8 % → **88,7 %** | 2 % → **10 %** |

**En el documento que ya estaba bien escrito, aplicar las reglas lo empeoró por
los dos lados**, y la telegrafía se multiplicó por cinco.

La explicación es la propia R1. Trocear en una-frase-un-hecho autosuficiente es
lo que salva un documento de viñetas, porque cada viñeta pasa a ser recuperable
entera. Pero un párrafo de prosa **ya era** una unidad recuperable, y rica: al
trocearlo, el sistema recupera **una** frase corta donde antes devolvía un
párrafo con contexto. La autosuficiencia le quitó al modelo el material que
tenía alrededor.

**Consecuencia práctica, y es la regla que manda sobre todas las demás:**

> Aplica estas reglas al material que llega **telegráfico** —viñetas sin verbo,
> «RESUMEN de X», listas nominales—. Si un documento ya está en prosa con frases
> completas y párrafos temáticos, **no lo reescribas entero**: aplica solo R3
> (que exista una frase por pregunta previsible), R4 (nada viviendo solo en
> mitos o FAQ) y R5 (sujeto explícito), que no fragmentan, y **deja los párrafos
> como están**.

R1, R2 y R6 son la cirugía para material roto. En un texto sano son iatrogénicas.

## 1. Para quién es este documento

Para **quien redacta o entrega el material clínico**, dentro o fuera del equipo.

No es (solo) una nota interna sobre cómo destilamos: es un **requisito sobre el
material que se nos entrega**. El documento de hemorroides que peor se comporta
de todo el corpus —1,1 KB— **no lo destilamos nosotros**: llegó así, en formato
«RESUMEN Cirugía de X», con seis secciones de viñetas sin verbo. Los dos defectos
más persistentes de todo el proyecto (los IDs 105 y 108) están **ya en el
material de origen**, literalmente:

    - Se hace en quirófano, con anestesia regional o general.
    - Ajustar medicación habitual.

Esa forma —el infinitivo sin sujeto, la disyunción dentro de una viñeta— es la
que rompe al modelo. Por eso estas reglas se aplican tanto al material propio
como al recibido.

## 2. Por qué la redacción decide la calidad

El sistema responde con **un único documento** por procedimiento y tiene
prohibido usar conocimiento propio: solo puede decir lo que el documento dice, y
si el documento no resuelve la pregunta debe responder «No tengo información
sobre eso».

Eso convierte la redacción en la variable que decide la calidad del servicio. Un
documento **clínicamente correcto pero mal redactado** produce dos fallos
distintos, y los dos se han medido:

- **rechaza** preguntas que el documento sí podría responder (sobre-rechazo);
- **responde** con una frase que comparte vocabulario con la pregunta pero no la
  contesta.

Y hay un tercer fallo, el más grave, que es puramente de forma: el sistema
**reasigna el sujeto** de una regla y convierte en instrucción al paciente algo
que el documento dirige al equipo clínico.

Dos advertencias antes de las reglas, porque las dos contradicen la intuición y
las dos están medidas:

- **«Escribe frases completas» por sí solo NO funciona.** Se probó (corpus v2) y
  el resultado **empeoró**: acierto 78,1 % → 76,5 %.
- **«Nombra al actor en la frase» tampoco basta** si el actor se lleva la oración
  principal. «Es su médico quien decide cómo ajustar su medicación» hizo que el
  sistema pasara de dar una instrucción errónea a **rechazar la pregunta**.

## 3. Las reglas

Se conserva la numeración `R1`–`R7` del brief, que es la que citan los
experimentos. La columna de prioridad es lo que se ha aprendido después.

| # | Regla | Prioridad |
| --- | --- | --- |
| **R3** | Cada pregunta previsible necesita una frase que la responda | **decisiva** |
| **R2** | El contenido accionable va en la oración principal | **decisiva** |
| **R1** | Autosuficiencia de cada frase | **decisiva** |
| **R5** | Sujeto explícito en toda acción | alta (evita el defecto más grave) |
| **R4** | El contenido clínico va en su sección temática | alta |
| **R6** | Frases completas con verbo conjugado | necesaria, no suficiente |
| **R7** | No compenses con longitud | cede ante R1 |

---

### R3 · Cada pregunta previsible necesita una frase que la responda

Que la respuesta sea **deducible** del documento no basta: tiene que existir una
frase que la afirme o la niegue **directamente**.

El sistema es un excelente recuperador de frases y un mal sintetizador. Medido
sobre diabetes (55 preguntas): responde bien **si y solo si** una frase del corpus
ya afirma o niega lo que se pregunta.

- Contesta perfectamente «tengo miedo, ¿mi vida cambia para siempre?», «estoy
  agobiado, ¿es normal?» y «me siento solo» — las tres tienen su frase literal
  («Tras el diagnóstico es normal sentir negación, frustración o miedo»).
- Falla en «¿se cura?», «¿puedo hacer vida normal?» y «¿es culpa mía?» — las
  tres exigen una inferencia o negar una premisa. Comprobado: **«se cura» no
  aparece nunca** en el documento.

**Cómo aplicarla.** Deriva del procedimiento y del sentido común clínico la lista
de preguntas que hará un paciente, y comprueba **una por una** que existe una
frase, no un razonamiento. Las que no tengan material se dejan **deliberadamente
sin cubrir** —para que el sistema pueda decir «no tengo información»— y se
**listan** como carencia del corpus (ver §5).

### R2 · El contenido accionable va en la oración principal

La regla menos obvia y la que explica más fallos. **El sistema conserva la
oración principal y suelta lo subordinado.**

Medido con tres ediciones de una línea, cada una en un documento distinto:

    67   principal = la medicación y sus dos pastillas; la restricción, dentro  -> SE CONSERVA
    108  principal = «es su equipo médico quien decide»; el ajuste, subordinado -> RECHAZA
    29   principal = la consulta por >39 °C; el paracetamol, aparte            -> LO SUELTA

Escribe el acto en la principal y **cuelga de él** la condición, el actor y el
alcance:

| ✗ | ✓ |
| --- | --- |
| Es su médico quien decide cómo ajustar su medicación | El equipo médico le ajusta la medicación antes de la operación, incluidos los anticoagulantes |
| Consulte si la fiebre supera los 39 °C. (y aparte) Si fiebre: paracetamol | Puede tomar paracetamol si tiene fiebre, sea cual sea la temperatura |

⚠️ **Efecto secundario de R2 que hay que vigilar: puede ascender una
recomendación a obligación.** «Hoy se recomienda beber y comer lo antes posible»
tiene el acto en una subordinada, así que R2 obliga a reescribirlo; pero toda
forma que conserve «se recomienda» lo deja subordinado, y la salida natural es
«usted debe beber y comer lo antes posible». Eso ya **no** es la misma modalidad:
una recomendación se ha convertido en una instrucción.

Apareció espontáneamente en la reescritura ciega de cirugía abdominal
(2026-07-23), y quien la escribió la declaró como su única desviación de
modalidad. **Regla: cuando R2 obligue a mover el acto a la principal y la fuente
sea una recomendación, hay que preservar la modalidad de forma explícita**
(«Usted puede…», «Su equipo le recomienda que…»), no dejar que se convierta en
un deber. Y anotarlo, porque es un cambio de contenido, no de forma.

### R1 · Autosuficiencia

Cada frase debe llevar **dentro** su condición, su actor y su alcance, de forma
que reformularla no pueda perder nada. Si una regla y la condición que la acota
viven en frases distintas, el sistema conserva una y suelta la otra.

El caso que la confirmó (ID 67): plegar la lista **sin** hacer la frase
autosuficiente no cambió nada (8/9 → 8/9); reescribirla como «La medicación **no
se da a todos los pacientes**: solo se administra cuando…» la arregló del todo
(6/9 → 0/9, con la condición conservada en las 9 tiradas).

### R5 · Sujeto explícito en toda acción

Debe quedar claro quién hace cada cosa: el paciente, el equipo médico o el
sistema sanitario. **Nada de infinitivos sueltos** («Ajustar medicación») ni de
imperativos sin destinatario: el sistema le asigna como sujeto **a quien
pregunta**, porque la pregunta va en segunda persona, y convierte en instrucción
al paciente algo dirigido al equipo clínico.

Es el defecto de peor consecuencia clínica que produce este corpus (familia G1) y
el único que **sobrevivió a todo**: seis variantes de prompt, dos ediciones de
corpus y un cambio de modelo. Cayó solo al reescribir el documento aplicando
R2 + R5 juntas:

    origen   «Ajustar medicación (anticoagulantes, etc.).»
    servido  «Debes ajustar la medicación (anticoagulantes, etc.).»   <- al paciente
    reescrito «El equipo médico le ajusta la medicación …,
               incluidos los anticoagulantes.»                        <- arreglado, 0/9

Dato que da confianza en la regla: **dos autores independientes**, trabajando a
ciegas desde el brief, dieron con **la misma estructura**.

### R4 · El contenido clínico va en su sección temática

Nada importante puede vivir únicamente en una sección de mitos, creencias, FAQ o
anexos. El sistema **recupera por vocabulario**: si una palabra clave solo
aparece en una lista de mitos, responderá con esa línea aunque no venga a cuento.

El caso limpio: a «¿me voy a quedar ciego?» el sistema contesta «la insulina
inyectada reduce complicaciones, incluida la ceguera» —que no responde y encima
sugiere que el paciente necesita insulina—. Motivo: **«ceguera» aparece una sola
vez en todo el documento, en la lista de MITOS**, mientras que la sección de
complicaciones oculares habla de retinopatía, cataratas y glaucoma.

Corolario: **cada hecho se enuncia una sola vez**, en la sección que le
corresponde. La sección vecina no lo repite ni lo referencia.

### R6 · Frases completas con verbo conjugado

Nada de viñetas nominales telegráficas: «Ayuno 6–8 h.», «Duración: 30–60
minutos.», «Frecuentes: dolor al defecar, sangrado leve.»

Necesaria y **no suficiente**: R1, R2 y R3 son las que deciden. Un documento que
solo cumple R6 mide peor que el original.

### R7 · No compenses con longitud

No repitas ideas, no añadas fórmulas de cortesía ni encuadres genéricos («toda
cirugía tiene riesgos»). Cada frase debe aportar un hecho o responder una
pregunta.

**R7 cede ante R1 — y esto está medido.** Hacer cada frase autosuficiente obliga
a repetir el anclaje («de la operación de hemorroides») y el sujeto («usted») en
frases contiguas, lo que a ojo humano se lee mal. En el A/B compitieron tres
reescrituras: la más natural y menos autosuficiente quedó **la peor de las tres**,
y la que repite el sintagma unas quince veces ganó **por el doble**. Un texto que
se lee peor **sirve mejor**.

R7 se aplica como prohibición de **frases** de relleno, no de **palabras**
repetidas. Se puede alternar el sintagma («la hemorroidectomía», «esta cirugía»,
«durante la recuperación») para no repetir literalmente, pero no se puede quitar
el anclaje.

## 4. Antes y después, sobre el documento completo

Original recibido (viñetas, sin sujeto, sin verbo):

```markdown
2. Preparación y cuidados posteriores
- Ayuno 6–8 h.
- Ajustar medicación (anticoagulantes, etc.).
- Tras la cirugía: dolor moderado, baños de asiento, dieta rica en fibra y líquidos, recuperación en 2–4 semanas.
```

Reescrito aplicando las siete reglas, **sin añadir ni un hecho**:

```markdown
## Preparación antes de la intervención

Usted debe permanecer en ayunas entre 6 y 8 horas antes de la operación de hemorroides.

El equipo médico le ajusta a usted, antes de la operación de hemorroides, la
medicación que usted toma habitualmente, y ese ajuste incluye los anticoagulantes.

## Cuidados después de la intervención

Usted tiene un dolor moderado después de la operación de hemorroides.

Usted se hace baños de asiento durante la recuperación de la operación de hemorroides.

Usted sigue una dieta rica en fibra y en líquidos durante la recuperación de la
operación de hemorroides.

Usted se recupera de la operación de hemorroides en un plazo de 2 a 4 semanas.
```

Resultado medido, 31 preguntas × 9 seeds, mismo modelo y mismo prompt:

| | corpus servido | reescrito (solo forma) | reescrito y ampliado |
| --- | ---: | ---: | ---: |
| respuestas telegráficas | **56 %** | **23 %** | **16 %** |
| ID 108 (sujeto reasignado) | 8/9 roto | **0/9** | **0/9** |
| tamaño | 1,1 KB | 2,6 KB | 8,1 KB |

La reescritura **de forma sola** —cero hechos nuevos— corta las respuestas
telegráficas a menos de la mitad y arregla el defecto clínico que no arregló
nada más.

## 5. Qué NO arregla la redacción

Estas reglas son necesarias y no bastan. Hay tres cosas que ninguna reescritura
puede resolver, y que hay que **escalar en vez de intentar arreglar**:

1. **Hueco de contenido.** Si el material no lo dice, no hay forma de escribirlo.
   En hemorroides quedan **8 preguntas previsibles** que ninguna versión puede
   responder ni siquiera ampliando desde las fuentes disponibles —vuelta al
   trabajo, conducir, deporte, ingreso y días, puntos y cura de la herida, cómo
   hacer los baños de asiento, señales de alarma, analgésico en casa, laxantes,
   ir acompañado—. Es **el postoperatorio domiciliario entero**.
2. **Defecto clínico del contenido.** El sistema reproduce el documento
   fielmente; si el documento está equivocado, la respuesta lo estará. Requiere
   decisión clínica, no edición.
3. **Contradicción entre fuentes.** Cuando dos documentos permitidos se
   contradicen, la reescritura solo puede conservar ambas o elegir una; ninguna
   de las dos salidas es una decisión de redacción.

Por eso el entregable de una reescritura **incluye la lista de lo que no se pudo
cubrir**. Esa lista es tan valiosa como el texto.

## 6. Procedimiento de aplicación

1. **Derivar las preguntas previsibles** del procedimiento, antes de escribir.
2. **Escribir** aplicando las siete reglas.
3. **Comprobar R3 una por una**: para cada pregunta, señalar la frase que la
   responde. Las que no tengan frase, a la lista de carencias.
4. **Entregar, junto al documento, un informe corto** con:
   - la tabla de preguntas previsibles y dónde las responde el texto;
   - lo que se dejó fuera por dudar entre reformulación y añadido;
   - la procedencia de cada bloque añadido, si se amplió;
   - los puntos donde las reglas chocaron y cómo se resolvió.

Ese informe es lo que hace auditable la reescritura. El del job 7329 está en
`eval/corpus_ab/BLIND_REWRITE_INFORME.md` y sirve de plantilla.

### Reglas de higiene del experimento

Si la reescritura se va a evaluar, quien la escribe **no debe conocer el conjunto
de preguntas de evaluación**, o estaría escribiendo hacia el test. En el job 7329
se ejecutó así —dos versiones a ciegas más una mía como control de
contaminación— y el control salió limpio: conocer las preguntas no dio ventaja.

Y si se amplía el contenido, se amplía **solo desde fuentes existentes**, con la
procedencia anotada línea a línea. El resultado es **material de prueba, no
corpus desplegable**: no entra en producción sin validación clínica.

## 7. Cómo NO medir el resultado

Tres instrumentos que fallaron y conviene no volver a usar:

- **El guardarraíl automático de «respuesta telegráfica» (<80 caracteres) es mal
  predictor de calidad.** Marca respuestas que contestan perfectamente en una
  línea y no ve respuestas malas de 65 caracteres. Lo que predice la calidad es
  si la frase responde la pregunta, no su longitud. Sirve como **señal agregada**
  (el 56 % → 23 % de arriba), no como veredicto por pregunta.
- **Ningún probe por regex vale sin leer la respuesta.** Cinco veces en una
  semana un probe cantó un arreglo falso porque el modelo cambió la redacción lo
  justo para esquivar la expresión regular, o un fallo falso por la misma razón.
- **El acierto agregado no mide una edición pequeña.** Cambiar tres líneas de
  tres documentos tiene un efecto esperado de ~2 puntos sobre 134 preguntas, que
  es exactamente el suelo de ruido. El agregado sirve de **guardarraíl** (¿se ha
  roto algo global?), no de evidencia de mejora.

Y una trampa de método específica de los A/B de corpus: **la verdad de terreno
sobre qué preguntas son respondibles se deriva del corpus**. Un corpus reescrito
hace respondibles preguntas que antes no lo eran, y un scorer con la verdad
antigua las cuenta como **regresión** cuando son mejoras. Hay que re-derivar la
respondibilidad después de cada reescritura.

---

## Apéndice — procedencia de cada regla

| Regla | De dónde sale |
| --- | --- |
| R1 | A/B de corpus v3 (job 7303): el ID 67 se arregla al hacer la frase autosuficiente; el v2, que solo plegó la lista, no movió nada |
| R2 | Comparación 67 / 108 / 29 en los jobs 7300 y 7303: el modelo conserva la principal y suelta lo subordinado |
| R3 | Pase cualitativo de diabetes con gemma-4 (2026-07-23): responde bien sii existe una frase que lo afirma o lo niega |
| R4 | Mismo pase: «ceguera» solo aparece en la lista de mitos y el sistema responde desde ahí |
| R5 | Barrido de reglas fundidas (2026-07-21) y severidad clínica: los tres G1 reasignan el sujeto. Confirmada por el job 7329, que arregla el 108 |
| R6 | A/B de corpus v2 (job 7300): necesaria pero insuficiente — sola, empeora |
| R7 | Job 7329: la versión más natural y menos autosuficiente quedó la peor de las tres |
| §5 | Informe de la reescritura ciega: 8 preguntas sin material posible |
| §7 | Jobs 7290, 7299, 7303, 7329 — probes falsos y la obsolescencia de `MUST_REFUSE` |
