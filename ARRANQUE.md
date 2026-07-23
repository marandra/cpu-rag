# ARRANQUE — gemma adoptado; A1, A2 y A4 cerradas; falta A3 y el email

> Fichero de arranque de sesión. El usuario dirá solo "lee ARRANQUE.md y continuamos
> desde ahí". Bórralo cuando la tarea acabe.
>
> **Última sesión, 2026-07-23 (tarde). Empezar por aquí:**
>
> - **A1 HECHA** → `docs/corpus_guidelines.md`. Lee su **§0** antes que nada: las
>   guidelines **no ayudan en general**, reparan material telegráfico y **empeoran
>   un texto que ya está en prosa**. Es el hallazgo de la sesión.
> - **A2 HECHA** → el XX del email es **84 % corrección / 78 % presentable**.
>   Veredicto por pregunta en `eval/audit_quality/gemma4_verdicts.json`.
> - **A4 HECHA** (jobs 7341/7342). Diabetes v4 mejora, **cirugía v4 empeora y no
>   entra**. Reescrituras hechas por **sesiones ciegas**, no por mí.
> - **A3 CONSTRUIDA, SIN LANZAR.** Las 4 variantes `g1`-`g4` están en
>   `app/prompt.py`, `prompt_ab.sbatch` ya acepta `VENV`/`MODEL`, y el scorer ya
>   distingue la derivación. El comando está en §A3. **Es lo siguiente.**
> - **Borrador del email vivo** → `docs/respuesta_auditoria_borrador.md`. Le falta
>   lo que solo puede aportar el usuario: las «ideas sueltas» de otra conversación.
>
> **Estado del repo (commiteado 2026-07-23, rama `audit/rule-boundary-sweep`):**
>
> - Todo el trabajo de la sesión está en git **menos el `.xlsx` de la auditoría**,
>   que se deja fuera a propósito: es material del cliente y meterlo en el
>   historial no se deshace. Sigue sin trackear en la raíz.
> - `corpus/markdown/` está en `.gitignore`, así que las reescrituras **no** están
>   ahí. Copiadas a `eval/corpus_ab_v4/corpora/` y `eval/corpus_ab/corpora/`, que
>   sí se versionan. **Para volver a correr un A/B hay que copiarlas de vuelta a
>   `corpus/markdown/` primero.**

## Dónde estamos

Un tercero auditó glucowise (diabetes) y aiciblock (cirugía abdominal + hemorroides)
con 134 preguntas propias y entregó `Auditoria_critica_RAG_preguntes_pacients.xlsx`
(raíz del repo, sin trackear).

**Hecho:** reproducción contra los pools reales, triaje de las 134, verificación de
las observaciones que nos pasaron, evaluación propia con nuestros criterios, el
**barrido de reglas fundidas**, la **medición de inestabilidad**, el **barrido de
seeds**, el **orden por consecuencia clínica**, y (2026-07-22) el **método del A/B
de prompts** y el arreglo del layout de snapshots.

**Pendiente, en este orden:**

1. ~~Reescribir `tools/hpc/seed_sweep.sbatch`~~ **HECHO 2026-07-22.** Ver
   «El vehículo — HECHO» abajo. De paso quedó **medida la causa real** del
   desajuste con el replay, y no eran los hilos: era la **procedencia del
   snapshot**.
2. ~~Construir `tools/audit_score.py`~~ **HECHO 2026-07-22.** Ver §2.1.
3. ~~Escribir las variantes y correr el A/B~~ **HECHO 2026-07-22** (job 7290,
   6 brazos × 134 × 9 seeds, 60 min). **Ninguna variante mejora la decisión.**
   Resultado y por qué, abajo: «RESULTADO DEL A/B».
4. ~~Decidir V15~~ **HECHO y RECHAZADO 2026-07-22** (job 7299). Candidato a
   entregable: **V14d**, no V15.
5. ~~A/B de corpus~~ **HECHO 2026-07-22**: v2 (gramática, job 7300) refutado;
   **v3 (autosuficiencia, job 7303) gana** — 79,4 % y estabilidad 16 %, la mejor
   medida. Ver «A/B DE CORPUS v3».
6. ~~Verificar el pin de llama-cpp~~ **HECHO 2026-07-22**: el pin era falso,
   subido a 0.3.34. Ver «LLAMA-CPP — EL PIN CAE».
7. ~~Barrido de modelos, criba 1~~ **HECHO 2026-07-22** (jobs 7314 y 7316).
   **Los dos Qwen3.5 caen por arquitectura, no por velocidad.** Ver «CRIBA DE
   MODELOS» abajo.
8. ~~Criba 2 (decisión)~~ **HECHA 2026-07-22, job 7320. gemma-4-26B gana fuerte:
   91 % vs 78 %.** Ver «CRIBA 2» abajo.
9. ~~Decidir modelo~~ **HECHO 2026-07-23: se adopta gemma-4-26B.** Decisión del
   usuario — «claramente es un mejor modelo para nuestro uso y podemos pagarlo».
   El fichero de 17 GB **no es bloqueante**: se resuelve con una **etapa `init`
   posterior a instalar el contenedor** que baja el modelo y genera los
   snapshots. `gemma + V13` es la línea base desde la que se trabaja.
10. **ES LO SIGUIENTE**: el plan de 10 tareas de §PLAN 2026-07-23.

## PLAN DE CIERRE — acordado 2026-07-23 al final de la sesión. EMPEZAR AQUÍ

Las tareas 1-4 del plan de más abajo están **hechas**. Lo que queda se agrupa en
tres fases; A y C se solapan, B depende de A2.

### FASE A — cerrar lo que aún no sabemos

| # | tarea | estado |
| --- | --- | --- |
| **A1** | **Guidelines de documentación** → `docs/corpus_guidelines.md`. Base: `docs/corpus_rewrite_brief.md` (7 reglas) + lo que midió el job 7329 | **HECHA 2026-07-23**, ver §A1 abajo |
| **A2** | **Recalcular el XX con gemma sobre las 134** | **HECHA 2026-07-23: A 84 % · B 78 %.** Ver §A2 |
| **A3** | **Prueba sistemática de la influencia de los prompts** sobre gemma (era la tarea 8). Pedida por el usuario 2026-07-23, **después de A4**. Diseño en §A3 | tras A4 |
| **A4** | **Aplicar las guidelines a diabetes y cirugía** | **EN CURSO**, ver §A4 |

Nota para A2: la verdad de terreno (`MUST_REFUSE`) se derivó del corpus original.
Si A4 cambia diabetes o cirugía, **hay que re-derivar la respondibilidad** o el
scorer contará mejoras como regresiones (pasó con el 109).

### FASE B — producto. Es un v2, no un v1.2

Cambia el modelo (17 GB), el empaquetado, el corpus y la config de servicio.
Llamarlo v1.2 vende corto lo que es.

| # | tarea |
| --- | --- |
| **B1** | Decidir el número de versión y el alcance |
| **B2** | **Etapa `init`**: descarga el modelo y genera snapshots tras instalar el contenedor. Medido: warm **7,7 s** por procedimiento a nT=64 (45 s a nT=8) → el init debe usar todos los cores |
| **B3** | Empaquetado y entregable (rompe la imagen portable de un fichero de v1.1) |
| **B4** | Config de servicio: **`nT=8 N=8` confirmado por D4** — 8 usuarios a 8,60 tok/s |

### FASE C — comunicación

| # | tarea |
| --- | --- |
| **C1** | **Escalado clínico** (era la tarea 7). Cuatro puntos, ver abajo |
| **C2** | **Borrador de la respuesta a los auditores**, a completar según lleguen datos |
| **C3** | Email final: entregable + que se bajen la última versión + que reevalúen con los criterios detallados |

**La condición que bloqueaba C2 ya se cumple.** [[audit-response-plan]] decía «no
redactar hasta que el trabajo de corpus esté hecho». Las guidelines están
validadas (job 7329), así que **procede empezar el borrador** — como borrador
vivo, no como envío.

### A3 — PRUEBA SISTEMÁTICA DE PROMPTS SOBRE GEMMA. Diseño, pedida 2026-07-23

Va **después de A4**. La diferencia con el A/B de julio es que aquel se corrió
sobre Ministral y contra defectos que **gemma ya no comete**: V13 está optimizado
contra invención y fusión, y gemma no inventa (0 invenciones en las 134) ni
funde. Así que el eje ya no es «¿qué prompt reduce el sobre-rechazo?» sino **qué
parte de V13 sigue haciendo falta y qué le falta para la capa de calidad**.

**TODO EL ANDAMIAJE ESTÁ CONSTRUIDO 2026-07-23. Falta solo lanzarlo**, cuando
liberen nodo 7341/7342:

    VARIANTS="g1 g2 g3 g4" VENV=.venv-latest \
      MODEL=gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
      OUT_ROOT=eval/audit_ab_g sbatch tools/hpc/prompt_ab.sbatch

    python3 tools/audit_score.py --baseline g0 \
        --run g0=eval/model_ab/gemma4 --run g1=eval/audit_ab_g/g1 ...

1. **`prompt_ab.sbatch` parametrizado** con `VENV`/`MODEL` — estaba pineado a
   `.venv-native` (llama-cpp 0.3.19, **sin arquitectura `gemma4`**), o sea que no
   podía correr el modelo que servimos. Los defaults dejan el comportamiento
   viejo byte a byte.
2. **Las 4 variantes escritas** en `app/prompt.py`, con `v13` intacto (la clave
   de caché de snapshots hashea el system prompt; tocarlo huerfaniza los pickles
   de los pools). Coste de prefijo medido:

   | | chars | ~tok | vs v13 |
   | --- | ---: | ---: | ---: |
   | v13 | 3341 | 928 | — |
   | **g1** | 2424 | **673** | **−255 t en cada petición** |
   | g2 | 4102 | 1139 | +211 |
   | g3 | 3786 | 1051 | +123 |
   | g4 | 3871 | 1075 | +147 |

3. ⚠️ **La trampa de medición de G2, desactivada antes de correr nada.**
   `refused()` solo casa «no tengo informaci», así que una derivación contaba
   como **respuesta** y G2 habría salido convirtiendo rechazos en respuestas en
   cada pregunta que esquivara — el sexto falso positivo de la semana, y éste
   habría llegado hasta el email. Arreglado:
   - `deferred()` **nuevo** en `audit_triage.py`, **sin tocar `refused()`** para
     no reescribir en silencio todo número ya publicado;
   - `audit_score.correct()` pasa a tres vías: derivar es **correcto donde el
     documento no responde** (no inventa, y es lo que pide su propia rúbrica) y
     **fallo donde sí responde** (tenía el material y lo esquivó);
   - `telegraphic()` deja de contar derivaciones, que son cortas por diseño;
   - columna nueva de **derivación** en los guardarraíles, con la advertencia de
     leerla junto al acierto: tasa alta en los tres procedimientos = ha aprendido
     a esquivar, no a distinguir.

   **Verificado que no mueve nada**: ministral 78,1 % y gemma4 91,0 %, neto +155,
   ratio 6,7× — idénticos a lo publicado.

Cinco brazos, y el control es gratis (`eval/model_ab/gemma4`, ya medido):

| brazo | qué cambia | qué hipótesis mata |
| --- | --- | --- |
| **G0** | V13 tal cual | control, ya medido |
| **G1** | V13 **menos** el andamiaje anti-invención y anti-fusión | si no pierde nada, esas ~200-300 t de prefijo son **peso muerto** con gemma, y el prefijo se paga en cada petición |
| **G2** | **la tercera salida**: permiso explícito para «esto depende de tu caso, coméntalo con tu equipo» cuando el tema está cubierto y la respuesta concreta no | es **la aportación más accionable de los auditores** y su código `Sense resposta`. Diana: 99, 100, 76 |
| **G3** | regla de forma: la respuesta debe ser una frase completa que reenuncie lo preguntado; nunca un fragmento suelto | separa **prompt de corpus** en la capa presentable. Diana: 114, 124 («Ayuno 6–8 h.») |
| **G4** | respuesta parcial explícita: dar lo que el documento sí dice **y** decir qué parte no cubre | Diana: 110, 128, 133 — preguntan permanencia y el sistema contesta frecuencia |

**G3 es el brazo con más valor de método**, gane o pierda: si una regla de prompt
arregla el fragmento telegráfico, entonces la telegrafía **no** era solo del
corpus y hay que matizar el punto 5 del email; si no lo arregla, el punto 5 queda
reforzado por una segunda vía. Correrlo **contra el corpus servido**, no contra
el v4, o se confunden las dos variables.

**Cómo se mide, con lo aprendido en A2:** acierto de decisión y estabilidad
automáticos sobre 134 × 9 seeds; guardarraíl de telegrafía **como señal
agregada, no como veredicto**; probes dirigidos a los IDs diana; y **lectura a
mano solo de las que se mueven**. Ningún número del apartado de probes vale sin
leer la respuesta — van cinco falsos positivos esta semana.

### A4 — GUIDELINES APLICADAS A DIABETES Y CIRUGÍA, 2026-07-23

**Método, corregido por el usuario:** las reescrituras **no las hago yo**. Dos
**sesiones ciegas** independientes, una por documento, desde
`docs/corpus_rewrite_brief_diabetes.md` y `docs/corpus_rewrite_brief_cirugia.md`
— briefs escritos a propósito **sin ejemplos derivados del set de evaluación** (el
`docs/corpus_guidelines.md` NO se les pasa: cita preguntas de la auditoría) y con
prohibición explícita de abrir `eval/`, `reports/`, `ARRANQUE.md`, `tools/audit_*`
y los `.xlsx`. Informes en `eval/corpus_ab_v4/`.

Mi propia versión de diabetes, escrita antes de esa corrección, se conserva como
**control de contaminación** en `corpus/markdown/diabetes.v4-mio.md`, igual que
`mio` en el job 7329.

**Diabetes — HECHA.** `corpus/markdown/diabetes.v4.md`, 13,0 → **22,4 KB**.

- **Fidelidad verificada**: **cero cifras del original perdidas, cero cifras
  nuevas**; metformina, glucagón, paracetamol, acetona, sacarina/aspartamo/
  ciclamato, HbA1c y LDL intactos.
- **R4 se aplicó a fondo**: disuelve **«Preguntas frecuentes» y «Creencias
  frecuentes (mitos)»** enteras y reparte los diez mitos a sus secciones
  temáticas. Es exactamente lo que predijo el pase cualitativo — el «¿me voy a
  quedar ciego?» fallaba porque «ceguera» solo existía en la lista de mitos.
- **Convergencia inesperada**: la ciega sale 22,4 KB y la mía contaminada 20,7 KB.
  Dos autores, mismo brief, mismo factor de crecimiento (~1,6-1,7×).

⚠️ **Coste que hay que medir, no dar por bueno:** 13 → 22,4 KB es ~1,7× de
prefijo. [[kv-fulldoc-bench]] dice que el tamaño de contexto fija la velocidad de
decodificación, y [[distillation-method]] fijó el objetivo en 2-4K tokens
justamente por eso. **Las guidelines empujan en dirección contraria al objetivo
de destilación**, y el A/B tiene que reportar warm y tok/s, no solo acierto. Es
un intercambio nuevo que ningún experimento anterior tocaba: hemorroides creció
de 1,1 a 2,6 KB, que no le costaba nada a nadie.

**Y el informe ciego corrobora el escalado clínico por una vía independiente.**
Sin acceso a la auditoría ni a nuestras notas, lista 12 dudas clínicas y **la
número 1 es el ID 29**: «Nunca suspenda la medicación de la diabetes» en días de
enfermedad, señalando que con deshidratación, vómitos o diarrea la práctica es la
contraria para algunos antidiabéticos orales. **Es el mismo defecto que los
auditores marcaron como «la resposta més perillosa del bloc», encontrado a
ciegas y desde el documento.** Eso es un argumento para el email: el proceso de
guidelines **encuentra solo** el defecto clínico que ellos encontraron con ADA.

Otras dudas nuevas y buenas del informe ciego, para C1: el objetivo de glucemia
en ayunas llega a **130 mg/dl** mientras el criterio diagnóstico es **≥126**
(solape que puede confundir); el título anuncia **prediabetes** y el cuerpo no la
trata; la hipoglucemia no lleva recontrol a los 15 min ni hidrato lento
posterior; y el **glucagón** aparece en el bloque de viaje pero no en el de
inconsciencia.

**Cirugía abdominal — HECHA.** `corpus/markdown/cirugia-abdominal.v4.md`,
7,6 → **11,9 KB** (×1,56, mismo factor que diabetes). Fidelidad verificada igual:
**cero cifras alteradas**. La sesión inventarió **84 hechos atómicos** y verificó
correspondencia 1:1. Trabajo principal, como se predijo, en R1/R2/R5 y no en R6:
elimina **toda** estructura de lista, parte los 8 párrafos de 4-7 hechos
encadenados, y quita los impersonales («será trasladado», «se insufla», «el dolor
se trata»). La lista de premedicación —el defecto del 67— pasa a que **cada
pastilla repita la condición entera**.

⚠️ **Tres desviaciones declaradas, y hay que tenerlas presentes al puntuar:**

1. **Añade una frase que el original solo dejaba deducible**: «la mínimamente
   invasiva puede producirle menos dolor y una recuperación más corta que la
   abierta». Es el contrapositivo de lo que sí dice el original, pero enunciado
   en afirmativo. **Toca directamente el 56 y el 84.** Lo que salva la
   atribución: los dos **ya salían A✓ B✓** en A2, así que la frase no tiene
   margen que cosechar — solo puede empatar o empeorar.
2. **Asciende una recomendación a obligación**: «hoy se recomienda beber y comer»
   → «usted debe beber y comer». Es un cambio de **modalidad**, o sea de
   contenido. Lo declaró como su única desviación de ese tipo.
3. Explicita el término de comparación: «menos complicaciones **que un paciente
   desnutrido**».

**La (2) es un hallazgo, no solo un defecto**, y ya está incorporada a
`docs/corpus_guidelines.md`: **aplicar R2 a una recomendación puede convertirla
en un deber**, porque toda forma que conserve «se recomienda» deja el acto
subordinado. La regla nueva es preservar la modalidad explícitamente («usted
puede…», «su equipo le recomienda que…»). Salió sola de la sesión ciega, que es
la mejor manera de encontrarla.

Sin material, listadas por la sesión: **16 preguntas**, entre ellas ayuno de
sólidos, medicación habitual, cuidado de la herida, duración de la operación,
duración del ingreso, alta y signos de alarma. Dudas clínicas anotadas: el
posible choque entre «puede beber hasta 2 horas antes» y «unas horas antes le
darán la bebida»; el «no hay peligro de sobredosis» sin matiz; y dos usos
distintos de la bomba de ACP sin aclarar si son excluyentes.

**Cómo se corre el A/B cuando estén las dos** (`corpus_ab.sbatch` ya está
parametrizado; el brazo de control **no se re-corre**, es `eval/model_ab/gemma4`):

    PROCEDURE=diabetes PROFILE_FOR=glucowise OUT_ROOT=eval/corpus_ab_v4 \
      CORPORA="v4:diabetes.v4.md mio:diabetes.v4-mio.md" \
      sbatch tools/hpc/corpus_ab.sbatch

    PROCEDURE=cirugia-abdominal PROFILE_FOR=aiciblock OUT_ROOT=eval/corpus_ab_v4 \
      CORPORA="v4:cirugia-abdominal.v4.md" \
      sbatch tools/hpc/corpus_ab.sbatch

### RESULTADO A4 — jobs 7341/7342, HECHOS 2026-07-23. **Se parte en dos**

Salidas en `eval/corpus_ab_v4/`. Control = `eval/model_ab/gemma4` (no se re-corre).

| | decisión | telegráficas |
| --- | ---: | ---: |
| diabetes control | 83,4 % | 11 % |
| **diabetes v4 (ciego)** | **85,5 %** | **7 %** |
| diabetes mio (contaminado) | 83,6 % | 8 % |
| **cirugía control** | **93,8 %** | **2 %** |
| cirugía v4 | 88,7 % | **10 %** |

**El hallazgo, y es el que contesta la pregunta del usuario:** las guidelines
**no ayudan en general**. Mejoran los dos ejes en diabetes (viñetas nominales) y
**empeoran los dos en cirugía** (ya en prosa), donde la telegrafía se multiplica
por cinco.

Causa, y es la propia R1: trocear en una-frase-un-hecho salva un documento de
viñetas porque cada viñeta pasa a ser recuperable entera; pero **un párrafo de
prosa ya era una unidad recuperable y rica**, y al trocearlo el sistema devuelve
una frase corta donde antes daba un párrafo con contexto. La autosuficiencia le
quita al modelo el contexto que tenía alrededor.

**Ya incorporado a `docs/corpus_guidelines.md` como §0 (alcance)**, con la regla
que manda sobre las demás: en material telegráfico, las siete; en prosa sana,
**solo R3, R4 y R5**, que no fragmentan. R1/R2/R6 son cirugía para material roto
y son iatrogénicas en un texto sano.

**Segunda confirmación de que la contaminación no da ventaja**: en diabetes la
versión ciega (85,5 %) bate a la mía contaminada (83,6 %), igual que en
hemorroides. Van dos de dos.

Estabilidad intacta en los tres brazos (1 %, 1 %, 0 %).

⚠️ **Bug de tooling detectado**: `--exclude-procedure` **no acumula** — el último
gana, porque no es `action="append"`. Pedir dos exclusiones da un resultado
silenciosamente equivocado. Los números por procedimiento de arriba están
calculados aparte, no con ese flag.

**Pendiente de A4**: leer a mano las que se mueven (v4 gana 31, 39, 47, 100 y
rompe 5, 10, 85, 101, 103) y decidir si el corpus v4 de diabetes entra en el
entregable. Cirugía v4 **no entra**: está medido peor que el original.

### Re-derivación de `MUST_REFUSE` — HECHA, y sale **sin cambios**

Era el paso que podía invalidar la puntuación (pasó con el 109 en hemorroides).
Revisadas las **28 rechazables** de diabetes (13) y cirugía (15) contra los dos
textos v4. Ocho dieron hit de keyword; **leídas las frases una a una, las ocho
son material que ya estaba en el original en forma equivalente**:

| id | qué disparó el probe | veredicto |
| ---: | --- | --- |
| 8 | «aprender sobre… autoanálisis» | ya estaba; ninguno dice **quién enseña** a pincharse |
| 19 | «efectos secundarios» | no es «cuánto tarda en hacer efecto» |
| 49 | «embarazo» (diabetes gestacional) | definición, no fertilidad |
| 52 | «estabilidad de día y de noche» | es de las insulinas basales, no de morir durmiendo |
| 53 | «agujas» | solo la lista de equipaje del viaje |
| 73, 79 | «días antes», «alta» | literales del original |
| **37** | «individualizable» | **el único a vigilar**: R1 hace que v4 lo repita en las 5 viñetas de objetivos en vez de una vez en el título. Más superficie, pero sigue sin decir que el curso de una persona difiera del de un familiar |

**Conclusión: la verdad de terreno no se mueve y el A/B es puntuable tal cual.**
Y de paso es una **segunda confirmación independiente** de que las sesiones
ciegas respetaron «ni un hecho más»: si hubieran añadido contenido, la
respondibilidad habría cambiado.

Vigilar el 37 leyendo, no por probe.

### A2 — EL XX CON GEMMA, HECHA 2026-07-23. **A 84 % · B 78 %**

Leídas a mano las 79 que faltaban (cirugía 48 + hemorroides 31) contra el corpus
**servido** —hemorroides = el RESUMEN de 1,1 KB, no las reescrituras—, sobre la
respuesta mayoritaria de los 9 seeds de `eval/model_ab/gemma4/`. Veredicto por
pregunta, con nota, en **`eval/audit_quality/gemma4_verdicts.json`**.

| | n | A · corrección | B · presentable |
| --- | ---: | ---: | ---: |
| diabetes | 55 | 39 · 71 % | 36 · 65 % |
| cirugia-abdominal | 48 | **44 · 92 %** | **43 · 90 %** |
| hemorroides | 31 | 30 · 97 % | 25 · 81 % |
| **TOTAL** | **134** | **113 · 84 %** | **104 · 78 %** |

**Frente a Ministral V13: 63 % → 84 % y 50 % → 78 %.** Y frente al 9 % de ellos.

**Cuadra con la decisión publicada, y eso es la comprobación de que la lectura no
se ha ido:** los fallos de decisión salen 3 en cirugía (76, 99, 100) + 9 en
diabetes = 12 → **122/134 = 91,0 %**, el número del job 7320 al dígito. Los otros
8 puntos que pierde A son preguntas con **decisión correcta y contenido malo**:
65 y 108, más 7 en diabetes.

⚠️ **El 97 % de hemorroides no es lo que parece y no debe citarse solo.** 11 de
sus 31 son rechazables y las 11 se rechazan bien: puntos gratis. Mirando solo lo
que hay que responder, el orden se invierte:

| respondibles | A | B |
| --- | ---: | ---: |
| cirugia-abdominal (33) | 32 · 97 % | **31 · 94 %** |
| diabetes (42) | 26 · 62 % | — |
| hemorroides (20) | 19 · 95 % | **14 · 70 %** |

O sea: hemorroides **acierta el hecho y falla la forma** —es el único
procedimiento donde A y B se separan 25 puntos—, y diabetes falla el hecho.

Lo que pierde cada procedimiento, en una línea:

- **cirugía**: 65 (no-respuesta circular), 76 (responde el ingreso con la línea de
  la recuperación), 83 (pega premedicación no preguntada porque «dormir del todo»
  engancha «pastilla para dormir»), 99 y 100 (**derivaciones legítimas** contadas
  como fallo por nuestra verdad de terreno, más estricta que la de ellos).
- **hemorroides**: el **108** es el único fallo de contenido, y es el G1 de
  siempre. Los 5 de forma son 110, 114, 124, 128 y 133.
- **par de control nuevo, e invertido**: el **76** («¿cuánto tiempo voy a estar
  ingresado?») lo responde mal y el **102** (la misma pregunta con marco
  emocional) lo rechaza bien. Con Ministral el sesgo iba al revés.
- **128 y 133** comparten defecto: preguntan por **permanencia** («para siempre»)
  y el sistema contesta **frecuencia** («muy rara»). El corpus no dice nada de
  permanencia — es hueco de contenido, no de forma.

Dos salvedades de método, para no sobrevender el número:

- **Diabetes entra como agregado.** Sus 55 se leyeron en la sesión anterior y
  **no se persistió el veredicto por pregunta**, solo los totales. Si el número
  va a salir del equipo, conviene re-registrarlas con el mismo formato.
- **Si A4 reescribe diabetes o cirugía hay que re-derivar la respondibilidad**
  antes de volver a puntuar, o las mejoras contarán como regresiones (pasó con el
  109, que vA y vB responden bien y el scorer puntúa como fallo).

### A1 — GUIDELINES, HECHA 2026-07-23 → `docs/corpus_guidelines.md`

Documento entregable, no nota interna: está escrito **para quien redacta o
entrega el material clínico**, dentro o fuera del equipo, porque la corrección
(1) de abajo demuestra que la forma que rompe al modelo llega ya en el material
de origen. Contenido:

- Las **7 reglas R1-R7** del brief, cada una con la medición que la sostiene y
  ejemplos antes/después reales (no inventados).
- **Prioridad medida**, que el brief no daba: R3, R2 y R1 son las decisivas; R6
  es necesaria y no suficiente (sola empeora: 78,1 → 76,5 %); **R7 cede ante R1**
  (el job 7329 lo zanjó — la versión más legible fue la peor de las tres).
- Un **antes/después del documento entero** de hemorroides con la tabla de
  resultado (telegráficas 56 → 23 → 16 %, el 108 a 0/9).
- **§5 Qué NO arregla la redacción** — hueco de contenido, defecto clínico,
  contradicción entre fuentes. Es la bisagra con C1: lo que no se puede escribir
  se escala.
- **§6 Procedimiento** — derivar preguntas, escribir, comprobar R3 una por una, y
  entregar el informe. Plantilla: `eval/corpus_ab/BLIND_REWRITE_INFORME.md`. Con
  la higiene del experimento (quien escribe no ve el set de evaluación).
- **§7 Cómo NO medir** — el guardarraíl de <80 car es mal predictor, ningún probe
  vale sin leer, el acierto agregado no mide una edición pequeña, y
  `MUST_REFUSE` hay que re-derivarlo tras cada reescritura.
- Apéndice de **procedencia regla → job**, para que nada quede sin respaldo.

### C1 — qué escalar al cliente clínico, actualizado 2026-07-23

Cambió respecto a la versión anterior: **el 108 ya NO es «irresoluble»**, se
arregla reescribiendo. Lo que queda es de contenido, no de forma:

1. **Defecto clínico de contenido — ID 29.** `Nunca suspenda medicación de la
   diabetes (pastillas o insulina)` contradice las reglas de días de enfermedad
   de ADA, que exigen pausar SGLT2i. **El sistema reproduce el documento
   fielmente; el documento es lo que está mal.** Requiere decisión clínica.
2. **Contradicción entre fuentes**, detectada por la sesión ciega: la GPC dice
   que en la primera defecación «no es habitual que se produzcan dolor ni
   hemorragia» y el documento de hemorroides clasifica el dolor al defecar como
   complicación **frecuente**.
3. **Hueco de cobertura en hemorroides**: 8 preguntas previsibles que **ningún**
   reescrito puede responder por falta de material — vuelta al trabajo, conducir,
   deporte, ingreso y días, puntos y cura de la herida, cómo hacer los baños de
   asiento, señales de alarma, analgésico en casa, laxantes, ir acompañado. **Es
   el postoperatorio domiciliario entero.**
4. **Estado del corpus**: hemorroides es 1,1 KB, **sin fuente propia y nunca
   destilado por nosotros** (§CORREGIDO (1)). No es material de producción.

### C2 — forma del borrador, con lo que el usuario quiere transmitir

Cuatro puntos, en este orden (de [[audit-response-plan]], más lo de 2026-07-23):

1. **Cómo debe hacerse la evaluación**: contra **el contenido del corpus**, no
   contra conocimiento clínico general.
2. **Los rechazos se evalúan contra el contenido Y contra las reglas del
   sistema**: hay que saber bajo qué reglas opera para distinguir un rechazo
   correcto de un fallo. Falta ese código en su rúbrica.
3. **El corpus era de desarrollo**, y hemorroides en particular no es material de
   producción.
4. **Con esos criterios su 9 % pasa a XX %** — y hay que decir el detalle.
5. **El documento de 1,1 KB es la causa de las respuestas telegráficas**, y está
   medido — no es una excusa. Ver §El punto 5 abajo.

✅ **XX YA ESTÁ CALCULADO (A2, 2026-07-23): 84 % de corrección y 78 %
presentable**, sobre las 134 y contra el corpus servido. Los números, por si hay
que elegir cuál se manda:

| | corrección | presentable |
| --- | ---: | ---: |
| **gemma-4, las 134** ← **el que se manda** | **84 %** | **78 %** |
| Ministral V13, las 134 (lo que ellos auditaron) | 63 % | 50 % |
| Ministral V13, sin hemorroides (103) | 62 % | 56 % |
| gemma, solo diabetes (55) | 71 % | 65 % |

El usuario recordaba «60 %»; el 63 % era **del modelo viejo**, que es el que
ellos auditaron. Hay que decir las dos cosas: **con lo que auditaron, 63 %; con
lo que se entrega, 84 %.**

### El punto 5 — hemorroides 1,1 KB es la causa de la telegrafía, medido

Ellos reportan «hemorroides 61,3 % nulas y **96,8 % ≤12 palabras**» y lo leen
como defecto del sistema. Es defecto **del documento**, y tenemos las dos mitades
de la prueba:

**(a) La telegrafía está concentrada en ese documento**, no repartida
(gemma, 1206 generaciones, umbral 80 caracteres):

| | respuestas telegráficas |
| --- | ---: |
| cirugia-abdominal (7,6 KB) | **2 %** |
| diabetes (13,0 KB) | 11 % |
| **hemorroides (1,1 KB)** | **56 %** |
| total 134 | 18 % |
| **total sin hemorroides** | **7 %** |

Hemorroides aporta **el 71 % de todas las respuestas telegráficas con el 23 % de
las preguntas**.

**(b) Y es causal, no correlación: reescribiendo el documento sin tocar un solo
hecho, la telegrafía cae de 56 % a 23 %** (job 7329, mismo modelo, mismo prompt,
mismas 31 preguntas). Ampliándolo desde fuentes, a 16 %.

Con eso el punto se sostiene sin pedir clemencia: **el sistema devuelve frases
cortas porque el documento son viñetas cortas**; cambiando la redacción del
documento, y solo eso, la mitad del defecto desaparece. Encadena con el punto 3
(el corpus era de desarrollo) y con §CORREGIDO (1) — ese documento **ni siquiera
lo destilamos nosotros**.

Y hay que conceder lo que es cierto: **sí hay preguntas que merecen revisión**, y
caen en dos grupos — (a) las que vienen de una **deficiencia de la documentación**
(→ guidelines, A1) y (b) las que son **mejorables de verdad** (→ v2, fase B).

**Contexto interno, NO va en el email:** el usuario cree que los auditores no
hicieron la evaluación a mano, sino que se la pasaron a un LLM sin apenas
contexto. Explicaría el ancla genérica (ADA/ERAS/ASA/ASCRS) y los 14 falsos
positivos. **No acusar** — sería contraproducente y no se puede probar. Pero sí
justifica **explicar el método con mucho detalle**: quien lo evaluó no conocía el
sistema. Y justifica el punto de que **evaluaron la v1.1** pese a que se les
ofreció una versión más reciente.

### Ideas del usuario que NO están en este repo

Las «ideas sueltas» para la respuesta se discutieron en **otra conversación** que
no está en el contexto de esta. Lo recogido arriba es lo que el usuario resumió
el 2026-07-23. **Antes de dar C2 por completo, recuperar esa conversación o
volver a discutirlas.**

## PLAN 2026-07-23 — acordado con el usuario. El frente es corpus y calidad

Contexto: con gemma la **decisión** está resuelta (91 %, volteo 1 %). Lo que no
sabemos es si gemma **responde** mejor — la capa de calidad solo está medida para
Ministral. Y dos diagnósticos de este documento resultaron falsos (abajo).

| # | tarea | coste | depende de |
| ---: | --- | --- | --- |
| 1 | **Números sin hemorroides** — recalcular decisión y correctitud sobre las 103 de diabetes+cirugía | minutos, local | — |
| 2 | **Curva D2/D4 de gemma** — su propia curva de escalado; da además el warm para dimensionar el `init` | job cluster | — |
| 3 | **Diabetes cualitativo sobre gemma** — mal respondidas, cuestionables, breves | lectura, local | — |
| 4 | **Reescritura de hemorroides**: `v4-forma` + `v4-completo` | escritura + job | guidelines |
| 5 | **`.v4` del 108** aislado, para no perder atribución | brazo del mismo job | — |
| 6 | **Guidelines de documentación** | escritura | 3, 4 |
| 7 | **Escalado clínico**: 108 (forma), 29 (SGLT2i), estado de hemorroides | escritura | 6 |
| 8 | Prompts sobre gemma — **solo si 3 destapa casos concretos** | — | 3 |
| 9 | Limpieza de repo, reorganización, empaquetado v1.2 con la etapa `init` | — | todo |
| 10 | **Solo al final**, la respuesta a los auditores | — | todo |

### RESULTADOS 2026-07-23 — tareas 1, 2 (D2) y 3 hechas

**Tarea 1 — sin hemorroides. Contra lo esperado, el titular NO mejora.**

| | 134 | 103 sin hemorroides |
| --- | ---: | ---: |
| decisión ministral | 78,1 % | 79,5 % |
| decisión gemma-4 | 91,0 % | 88,2 % |
| ventaja de gemma | +12,9 | **+8,7** |
| A · corrección (V13) | 63 % | **62 %** |
| B · presentable (V13) | 50 % | **56 %** |

Tres lecturas: (a) **el argumento «nuestro número real es mejor sin el corpus de
desarrollo» NO se sostiene para la corrección** (63→62); (b) donde sí cambia es
lo **presentable, 50→56 %**, porque hemorroides aporta **11 de las 17
telegráficas** con solo 31 preguntas — primera evidencia de que lo telegráfico
es **artefacto del corpus**, la hipótesis que cierra la tarea 4; (c) **la ventaja
de gemma se encoge** — hemorroides es donde más gana, porque allí el error de
Ministral era casi todo sobre-rechazo. Flag: `--exclude-procedure`.

**Tarea 2 — curva D2 de gemma (job 7324, cirugia-abdominal, `.venv-latest`).**
Salidas en `eval/model_scaling/d2/`.

| nT | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tok/s | 2,74 | 4,35 | 6,78 | 9,71 | 13,75 | 13,92 | **15,74** |
| speedup | 1,00× | 1,59× | 2,47× | 3,54× | 5,02× | 5,08× | 5,74× |
| eficiencia | 100 % | 79 % | 62 % | 44 % | 31 % | 16 % | 9 % |
| warm (s) | 45,0 | 45,0 | 45,0 | 44,9 | 24,2 | 14,8 | **7,7** |

**Nada de esto se parece a Ministral** (13,8× de tope, con **regresión** a 64).
gemma escala mucho peor (5,74× de tope) pero **no regresa a 64**: mete una
meseta 16→32 (**+1,2 %**) y luego sube +13 % a 64. Lectura: el cuello es el
**ancho de banda de memoria, no los cores** — a 64 hilos usa los dos sockets y
duplica el BW agregado, coherente con un MoE Q4 de 17 GB que lee ~2,5 GB por
token. Consecuencias:

- **La pregunta del usuario queda contestada por goleada**: de nT=16 a nT=32 se
  duplican los cores para ganar **1,2 %**. Dos réplicas de 16 baten a una de 32
  sin discusión. (D4, job 7325, lo confirma o no bajo contención, que es donde
  un MoE sensible al BW puede sufrir más.)
- **El `init` debe generar los snapshots con todos los cores**: 7,7 s por
  procedimiento a nT=64 frente a 45 s a nT=8. `load_state` ~585 ms, estable.

**Tarea 2 — curva D4 de gemma (job 7325, saturado, Σ(N×nT)=64).** Salidas en
`eval/model_scaling/d4/`.

| config | solo (D2) | saturado | drop | usuarios | veredicto |
| --- | ---: | ---: | ---: | ---: | --- |
| nT=4 × N=16 | 6,78 | **4,78** | −29 % | 16 | **bajo el suelo de 6** |
| **nT=8 × N=8** | 9,71 | **8,60** | −11 % | **8** | **el punto bueno** |
| nT=16 × N=4 | 13,75 | 12,52 | −9 % | 4 | pasa |
| nT=32 × N=2 | 13,92 | 14,95 | +7 % | 2 | pasa |

**El default `nT=8 N=8` queda confirmado, y no por casualidad**: es el último
punto donde la eficiencia paralela sigue en 44 % y la contención solo cuesta
−11 %. A nT=4 N=16 el modelo cae **por debajo del suelo** (4,78), con
`load_state` subiendo a ~750 ms y ttft a 1,4-2,1 s: ahí las réplicas ya compiten
por ancho de banda, que es el cuello de este MoE.

Y la pregunta del usuario queda contestada más fuerte de lo que se planteó: **no
solo dos réplicas de nT=16 baten a una de nT=32 — ocho de nT=8 baten a las dos.**
El +7 % de nT=32 saturado sobre su propio solo es casi seguro ruido (2 celdas × 2
preguntas); no sobreinterpretarlo. Y **no comparar los absolutos con el sweep de
Ministral de mayo**: distinto build y distinta metodología. Lo comparable entre
modelos es la *forma* de la curva.

**Tarea 3 — diabetes con gemma. El hallazgo está en
[[gemma-answer-shape]].** Resumen: A corrección **71 %** (vs 51 % de Ministral),
B presentable **65 %** (vs 44 %), decisión 46/55, **1 sola pregunta voltea**
(Ministral: 16). Los **9 fallos son sobre-rechazo, cero invención**.

**La hipótesis del «registro emocional» queda refutada por sus propios datos**:
gemma responde bien 41, 48 y 55, igual de emocionales. Lo que decide es si
**existe una frase del corpus que afirme o niegue la pregunta**: 41/48/55 la
tienen, 31/45/47 no (verificado: «se cura» **no aparece nunca**; «culpa» aparece
una vez enterrada en una viñeta sobre pedir ayuda). Y cuando no la hay pero algo
comparte léxico, engancha la frase equivocada: **«ceguera» aparece una sola vez
en todo el documento, en la lista de MITOS**, y por eso al «¿me voy a quedar
ciego?» contesta «la insulina inyectada reduce complicaciones, incluida la
ceguera» — que no responde y sugiere que necesita insulina.

Dos reglas nuevas para las guidelines (tarea 6), y una corrección de método:
**cada pregunta previsible necesita una frase que la responda** (no basta con que
sea deducible), y **el contenido clínico no puede vivir solo en una sección de
mitos/FAQ**. El guardarraíl automático de «telegráfica» (<80 car) es **mal
predictor**: marca 9, 40 y 51, que contestan perfectamente en una línea, y no ve
que el 42 es malo con 65 car.

### TAREA 4 — A/B de corpus de hemorroides, HECHO 2026-07-23, job 7329

gemma + V13 fijos, 31 preguntas × 9 seeds, 4 brazos. `vA`/`vB` los escribió una
**sesión ciega** desde `docs/corpus_rewrite_brief.md`, aislada en un directorio
sin acceso al set de evaluación; `mio` es mi versión, que conocía 5 preguntas y
sirve de **control de contaminación**. Salidas en `eval/corpus_ab/`.

| brazo | decisión | telegráficas | 108 | tamaño |
| --- | ---: | ---: | ---: | ---: |
| control (corpus servido) | **100 %** | **56 %** | 8/9 roto | 1,1 KB |
| mio | 90,3 % | 48 % | **0/9** | 2,0 KB |
| **vA** (ciego, mismos hechos) | 93,5 % | **23 %** | **0/9** | 2,6 KB |
| vB (ciego, ampliado) | 90,3 % | **16 %** | **0/9** | 8,1 KB |

**1. El 108 CAE, por fin, y respondiendo.** Es el G1 que sobrevivió a 6 variantes
de prompt, a los corpus v2 y v3, y al cambio de modelo. Leído:

    control  «Debes ajustar la medicación (anticoagulantes, etc.).»      <- el defecto
    los 3    «El equipo médico le ajusta la medicación ... incluidos
              los anticoagulantes.»                                      <- ARREGLADO

Y confirma **por qué** fallaron v2/v3: pusieron el acto en subordinada («es su
médico quien decide *cómo ajustar*») y el modelo rechazó. Estas tres ponen **el
acto en la principal con el actor de sujeto** y funciona. **Dos autores
independientes dieron con la misma estructura** aplicando R2+R5 del brief.

**2. La telegrafía, que era el objetivo real, se desploma: 56 % → 23 % → 16 %.**
Hemorroides aportaba 11 de las 17 telegráficas de las 134 con solo 31 preguntas.

**3. La autosuficiencia extrema gana, y mi versión es la peor de las tres.** vA
repite «de la operación de hemorroides» unas quince veces y se lee mal; la mía es
más natural y menos autosuficiente. **Gana vA por el doble.** Resuelve el choque
R1-vs-R7 del brief **a favor de R1**: un texto que se lee peor sirve mejor. Y el
control de contaminación sale limpio — conocer las preguntas no me dio ventaja.

### ⚠️ La decisión NO era medible aquí, y la verdad de terreno quedó obsoleta

Dos fallos de instrumento, y conviene no repetirlos:

- **El control ya estaba al 100 % de decisión en hemorroides.** No había margen:
  las reescrituras solo podían empeorar. Comprobado aritméticamente contra el
  91,0 % global (122/134 − 91/103 = 31/31). El experimento se midió con una
  métrica cuyo techo ya estaba tocado; **la métrica buena era la telegrafía**.
- **`MUST_REFUSE` se derivó del corpus original**, así que un corpus reescrito
  hace respondibles preguntas que antes no lo eran y el scorer las cuenta como
  fallo. Caso claro, el **109** («¿en qué se diferencia mi operación de tratarlo
  con láser?»): vA y vB responden **bien**, conectando dos hechos que el original
  tenía sueltos, y el scorer lo puntúa como regresión. **No es una regresión, es
  una mejora que la verdad de terreno no puede ver.** Cualquier A/B de corpus
  futuro necesita re-derivar la respondibilidad, o solo mirar probes y lectura.

Leídas las cuatro que se mueven: **109** mejora (arriba). **111** («¿por qué
fibra?») — `mio` responde «debe seguir dieta rica en fibra», que **no contesta el
porqué**: regresión real mía, y rechazar es lo correcto. **133** («¿secuela para
siempre?») — las tres rechazan y el control contesta listando complicaciones sin
decir si son permanentes; discutible, probablemente mejor rechazar. **134** —
vA acierta, `mio` y `vB` rechazan: regresión real de esos dos.

⚠️ **Y el probe del 105 canta falso por quinta vez esta semana**: marca 9/9 roto
en los tres brazos y las tres conservan la disyunción intacta («anestesia
regional o anestesia general»). La regex no casa la redacción nueva.

**Veredicto: gana `vA`.** Una regresión real discutible (133), una mejora que el
scorer no ve (109), el 108 arreglado y la telegrafía a menos de la mitad.

### ⚠️ CORREGIDO 2026-07-23 (1): hemorroides NO lo destilamos nosotros

Este documento afirma dos veces que «la destilación comprimió las reglas en
fragmentos telegráficos». **Es cierto para diabetes y cirugía, y falso para
hemorroides** — que es justo donde viven el 108 y el 105, o sea el G1 que
sobrevive a todo modelo y todo prompt. Verificado en el repo:

- `corpus/markdown/hemorroides.md` (1146 B) es byte-idéntico a
  `corpus/archive/hemorroides_v1_320.md`. **No hay historial de versiones**, al
  contrario de diabetes (`v3_3517`) y cirugía (`v2_2104` → `v3_1824`).
- **No existe fuente de hemorroides** en `corpus/sources/`. Ese 1,1 KB *es* el
  material que llegó.
- El clavo: su hermano `corpus/sources/resumen-fisura-anal.md`, material fuente
  sin tocar, ya trae las mismas patologías — `Se hace en quirófano, con anestesia
  regional o general.` (el 105) y **`- Ajustar medicación habitual.` (el 108,
  infinitivo sin sujeto, en el origen)**.

**Consecuencia**: las guidelines no son (solo) una nota interna sobre cómo
destilamos — son un **requisito sobre el material que se nos entrega**. El
proveedor manda documentos titulados «RESUMEN Cirugía de X» con seis secciones
de viñetas sin verbo, y ésa es la forma que rompe al modelo.

### ⚠️ CORREGIDO 2026-07-23 (2): la curva de escalado ya existía

No es cierto que «solo hemos probado 8 y 16 cores»: el sweep D2 barrió
nT = 1,2,4,8,16,32,64 ([[sweep-done-2026-05-21]]). Lo que falta **no** es medir
de cero, es **repetir D2/D4 con gemma** — la curva es propiedad del modelo, y
gemma es un MoE de 26B con ~4B activos. Herramientas ya hechas:
`tools/sweep/{run_sweep.py, run_d4.py, d2_native.sbatch, d4_native.sbatch}`.

Y queda contestada la pregunta del usuario sobre nT=32: **tiene razón**, dos
réplicas de nT=16 baten a una de nT=32. La eficiencia paralela lo dice sola
(diabetes): nT=8 **96 %**, nT=16 78 %, nT=32 **43 %**, nT=64 20 % — de 16 a 32
duplicas cores para ganar 11 %. D4 ya lo midió a 64 cores: `nT=16 N=4` da 28,6
tok/s agregados frente a 21,3 de `nT=32 N=2`. Dos matices: bajo saturación
nT=16 N=4 pierde **−21 %** (NUMA cruzada) y nT=32 N=2 solo −3 %; y el despliegue
actual **ya lleva esa lógica más lejos** (`nT=8 N=8`). Lo que se paga al trocear
es la velocidad con **un solo usuario** (19,5 vs 10,9 tok/s) — que es justamente
el hueco que tapa la idea de **instancias dinámicas según demanda** (a futuro).

### Qué son el 91 % y el 78 %, y el número que falta

**Son solo la decisión responder-vs-rechazar** (39 deben rechazarse, 95 deben
responderse), promediada sobre 9 seeds. **No juzgan si la respuesta es correcta.**
En preguntas: gemma ~122/134, Ministral ~105/134.

El número de «respondidas como se espera» existe **solo para V13+Ministral**, y
juzgado a mano una vez: **63 % (84/134) por correctitud** y **50 % (67/134) si
además debe ser presentable** (ver [[audit-response-plan]]). **Para gemma no
existe** — nadie ha leído sus 134 respuestas con criterio de calidad. Es el hueco
que cierran las tareas 3 y 1.

Nota que abarata ese trabajo: con Ministral una tirada no era representativa
(22 % de volteo); **con gemma sí lo es (1 %)**.

### Salvedades acordadas para las tareas 1 y 4

- **Tarea 1 no es «nuestro mejor número».** Quitar el peor documento y publicarlo
  es cherry-picking. La etiqueta que sí se sostiene es **«corpus de producción vs
  corpus de desarrollo»**, y se apoya en la corrección (1) de arriba: hemorroides
  no es corpus nuestro ni de producción.
- **Tarea 4 no inventa clínica.** El brazo `v4-forma` reescribe la forma **sin
  tocar un solo hecho** — ésa es la prueba limpia de las guidelines. El
  `v4-completo` solo puede ampliar desde fuentes que existen
  (`gpc_555_cma_iacs_compl-pacientes.md`, `via-clinica-cirugia-adulto-rica-2021-paciente.md`),
  con la procedencia anotada línea a línea. **Es material de prueba, no corpus
  desplegable**: no entra en producción sin validación clínica.
- **Por qué la tarea 4 es el experimento fuerte**: todas las ediciones de corpus
  anteriores fueron de 1-3 líneas, con efecto esperado ~2 puntos = el suelo de
  ruido (este documento ya lo reconoce en §A/B DE CORPUS v3). Hemorroides entero
  son **31 preguntas, 13 respondibles, 0/13 buenas hoy**: ahí la señal sí puede
  superar al ruido. Y decide la hipótesis abierta de [[audit-response-plan]] — si
  lo telegráfico es artefacto del corpus fino o propiedad del modelo.

## CRIBA 2 — decisión, HECHA 2026-07-22, job 7320. gemma-4-26B gana

`tools/hpc/model_ab.sbatch`, los dos modelos en `.venv-latest` (0.3.34), V13 fijo,
134 × 9 seeds. Salidas en `eval/model_ab/`. El brazo Ministral reproduce **78,1 %**
—el ancla del A/B de prompts al dígito—, así que la comparación es limpia y no
confunde modelo con build.

| métrica | ministral | gemma-4-26B |
| --- | ---: | ---: |
| acierto de decisión | 78,1 % | **91,0 %** |
| neto emparejado /1206 | — | **+155** |
| ratio gana:paga | — | **6,7×** (gana 182 SR, paga 27) |
| estabilidad (volteos) | 22 % | **1 %** |
| think-leaks / vacías | — | **0 / 0** en 1206 gen |

**gemma arregla las fronteras que no movió ni un prompt ni el corpus, leído no solo
por probe:** 105 («regional o general»), 67 (conserva la condición), 87 (fiel, no
inventa), 29 (no suelda fiebre>39 con paracetamol), 26 (sin prescripción
imposible). **Rechaza bien el 52** (la mortalidad nocturna inventada, el peor
defecto clínico) y responde el 103 (sobre-rechazo legítimo).

Salvedades honestas:
- **El 108 NO se arregla** (probe 8/9, leído: sigue «debes ajustar» al paciente).
  Es la familia G1 de sujeto → **escalar al cliente clínico pase lo que pase con
  el modelo**.
- El probe del 84 vuelve a cantar falso (9/9 «roto», se lee bien). Cosmético.
- Los 27 «rechazos correctos rotos» se reducen a **3 preguntas** en mayoría (76,
  99, 100) y **ninguna inventa** — responden del corpus, el «derivar» que los
  propios auditores validan. El downside es que nuestra verdad de terreno es más
  estricta que ellos, no que gemma alucine.

**El coste** (job 7319, diabetes peor caso): **7,34 tok/s por usuario a 8
concurrentes** (el usuario lo acepta, y también 4 usuarios a velocidad cómoda),
**594 ms de `load_state` por petición**, **1,4 GB de snapshot por procedimiento**,
**fichero de 17 GB — rompe la imagen portable de un solo fichero de v1.1**.

**Verificado de paso:** 0 think-leaks, 0 vacías, 0 deriva de idioma en las 1206
generaciones. El parche `_force_no_thinking` aguantó en producción-shape.

**Estado del entorno (cluster) para retomar:**
- `.venv-latest` (0.3.34) es el venv de la criba; se le instalaron a mano
  **`pydantic-settings` y `httpx`** (este último lo arrastra `audit_replay` vía
  `rag_client`, y sin él el job muere al minuto 1). Ya están; no reinstalar.
- Los 6 GGUF están en `~/Projects/cpu-rag/models/` del cluster (ver
  [[models-available]]). El nodo tiene 1 TB de RAM y ~406 GB de disco libre.
- Salidas de las tres cribas **versionadas** en el repo: `eval/model_screen/`
  (solo, saturated, saturated_diabetes) y `eval/model_ab/{ministral,gemma4}/`.
  El scorer: `python3 tools/audit_score.py --baseline ministral --run
  ministral=eval/model_ab/ministral --run gemma4=eval/model_ab/gemma4` (usar
  `python3` del sistema, NO `uv run` — intenta compilar llama-cpp y falla).
- NO hay pool vivo. La criba corre en proceso y no lo necesita.

## CRIBA DE MODELOS — criba 1 HECHA 2026-07-22, jobs 7314 (solo) y 7316 (8×8)

V13 sin tocar, cirugia-abdominal, `.venv-latest` (0.3.34). Herramientas nuevas:
`tools/bench_model.py`, `tools/hpc/model_screen.sbatch`,
`tools/model_screen_report.py`. Salidas versionadas en `eval/model_screen/`.

| modelo | arch | tok/s 8×8 | load_state | pickle | veredicto |
| --- | --- | ---: | ---: | ---: | --- |
| Ministral-3-3B | mistral3 | **15,38** | 231 ms | 536 MiB | el que hay |
| granite-4.1-3b | granite | **13,97** | **186 ms** | **431 MiB** | PASA, el barato |
| gemma-4-26B-A4B | gemma4 | **8,21** (6,85–9,04) | 468 ms | 1061 MiB | justo |
| Qwen3.5-4B | qwen35 | 12,55 solo | — | 617 MiB | **DESCARTADO** |
| Qwen3.5-35B-A3B | qwen35moe | 12,20 solo | — | 599 MiB | **DESCARTADO** |

**Los dos Qwen3.5 decodifican bien; caen por otra cosa.** Son híbridos SSM, y
llama.cpp **no sabe desalojar parcialmente una memoria recurrente**, así que el
acierto de prefijo de `Llama.generate` se rechaza —«partial kv removal not
supported, re-evaluating full prompt»— y **cada petición re-prefillea el fulldoc
entero**: ttft **27,8 s** (4B) y **49,2 s** (35B) frente a 0,46 s de Ministral.
Todo nuestro diseño es ese `load_state`.

⚠️ **Esto tumba la premisa de [[plan-b-model-benchmark]]**, que decía que los
híbridos Mamba eran el mejor encaje para los snapshots. El estado casi constante
es real y **da igual**: la reutilización se rechaza antes de que el tamaño
importe. Y está **probado, no supuesto** — calentando solo el prefijo común (una
extensión pura, sin nada que desalojar) sigue re-prefilleando, y reenviar el
prompt del warm **idéntico** también. No se arregla cambiando la forma del warm.

**El ancla es la columna del titular, no el suelo de 6 tok/s.** El 8×8 no le
cuesta nada a los densos de 2-3 GB en este nodo (15,2 → 15,4) y ~15 % al MoE de
26B (9,7 → 8,2), así que los 11,4 tok/s del pool son **coste del pool, no
contención de cores**. Escalando por ese factor, gemma-4 queda en ~6,2 tok/s por
el pool, con una celda en 5,1: **en** el suelo, no por encima.

**Y las respuestas** (un seed, seis preguntas: criba, no veredicto): los cuatro
candidatos **conservan la condición del 67** que no movieron ni seis variantes de
prompt ni dos ediciones de corpus, y **gemma-4 es el único que responde el 103**
(la mitad sobre-rechazada del par 91/103). granite sobre-rechaza el 86 y el 87,
que Ministral acierta. Es el primer indicio de que las fronteras rotas son
**capacidad del modelo**, no solo defecto de prompt o de corpus.

Dos trampas que costaría caro no saber:

- **El default de `enable_thinking` no es propiedad de la familia**: el
  Qwen3.5-4B lo trae apagado y el **35B-A3B encendido**, y
  `create_chat_completion` (0.3.34) **no acepta `**kwargs`**, así que no se puede
  pasar por llamada. Arreglado en `src/llm.py::_force_no_thinking`, y **afirmado
  byte-inerte para Ministral** — el caché de snapshots hashea el prompt.
- El `.venv-latest` del cluster ya lleva `pydantic-settings`, así que corre el
  harness entero (`audit_seed_sweep.py` incluido) sobre 0.3.34.

## RESULTADO DEL A/B — 2026-07-22, job 7290

`reports/audit_score.md` se regenera desde `eval/audit_ab/` (versionado):

    uv run python tools/audit_score.py --baseline v13 \
        --run v13=eval/audit_ab/v13 --run v14a=eval/audit_ab/v14a ... 

**Acierto de decisión, media sobre 9 seeds y las 134:** v13 78,1 % · v14a 77,5 %
· v14b **79,3 %** · v14c 77,6 % · v14d 78,4 % · v14e 77,8 %. La dispersión
*dentro* de cada brazo es ±2-3 puntos, así que **todas las diferencias caen
dentro del ruido**. El criterio de parada de §2.3 no lo cumple ninguna.

### Lo que sí quedó medido, y es el hallazgo de la ronda

**Sobre-rechazo e invención son un solo umbral, no dos comportamientos
separables.** El diff emparejado, partido por dirección:

| variante | gana sobre-rechazos | paga rechazos correctos | ratio |
| --- | ---: | ---: | ---: |
| v14a | 97 | 96 | **1,0×** |
| v14b | 69 | 33 | 2,1× |
| v14c | 53 | 43 | 1,2× |
| v14d | 44 | 28 | 1,6× |
| v14e | 72 | 45 | 1,6× |

V14a (aflojar el test de la REGLA) baja el rechazo de 44→29 % en diabetes y
39→21 % en cirugía, y **paga un rechazo correcto por cada sobre-rechazo que
gana**. No aprendió a distinguir «el documento lo resuelve con otras palabras»
de «el documento no dice nada»: movió el umbral. Sus regresiones son casi todas
FN (13, 19, 37, 46, 50, 53, 72, 73, 76, 79, 92, 94, 95, 99, 100, 102, 104, 109),
justo el grupo que se predijo. El mejor ratio es el de v14b (2,1×), pero su neto
son +14 celdas de 1206: dentro del azar.

### Fronteras: solo tres se mueven, y hay que leerlas

| id | v13 | mejor | ¿confirmado leyendo? |
| ---: | ---: | --- | --- |
| 105 | 9/9 | **v14a y v14d 0/9** | **sí** — «con anestesia regional o general» |
| 84 | 7/9 | **v14a 0/9** | **sí** — responde el género, no la laparoscopia |
| 87 | 9/9 | v14d 2/9 | parcial — la mayoritaria reproduce bien el documento |
| 108 | 9/9 | **ninguna** | — |
| 67 | 8/9 | **ninguna** | — |
| 29 | 7/9 | **ninguna** | — |
| 26 | 2/9 | **empeora**: v14a y v14c 7/9 | sí — la prescripción imposible pasa a mayoritaria |

⚠️ La primera pasada del scorer decía que v14b arreglaba el 108 (0/9) y que v14d
arreglaba el 67 (2/9). **Las dos eran falsas** y salieron al leer el texto: el
probe del 108 era `debes\s+ajustar` y el modelo escribe «debes **ajustar tu
medicación**», con la negrita entre medias; el del 67 solo buscaba «te
dará/darán» y no veía «puede darte». Arreglado en `b983a4c`: los probes casan
sobre texto sin marcado y `--self-check` afirma que poner en negrita cada
palabra no cambia ningún veredicto. **Moraleja para la próxima: ningún número
del apartado 3 vale sin leer la respuesta.**

### Por qué falló cada una — esto es lo que se lleva a V15

- **108 no es arreglable por prompt.** El fulldoc dice «Ajustar medicación
  (anticoagulantes, etc.)»: un infinitivo sin sujeto dentro de una lista de
  preparación. No hay sujeto que preservar, así que el modelo pone el del
  interlocutor porque la pregunta va en segunda persona. V14c no podía ganar.
  **Es la misma familia que el ID 29 y hay que escalarla al cliente clínico**:
  el estilo telegráfico del corpus borra el agente.
- **67 falló por desajuste estructural.** Su condición («cuando el grado de
  ansiedad y temor sea elevado») vive en **otra frase** que la promesa, y el
  ejemplo de V14d modelaba una condición *dentro* de la misma frase («Si nota
  picor, aplique la crema»). Preservarla exige unir dos frases, no copiar una.
  Es concreto y comprobable: V15 debe modelar la condición en frase vecina.
- **26 empeora justo con las variantes que empujan a responder más.** Más
  respuesta = más soldadura. Coherente con que fusión y sobre-rechazo compartan
  umbral.
- **V14e confirma la dirección pero no compra corrección.** Rechazo en
  hemorroides 54→42 %, y telegráficas 38→**49 %**. Con 1,1 KB de corpus,
  responder más solo produce más frases sueltas. Refuerza lo que ya decía este
  documento: hemorroides no tiene arreglo por prompt.

### Estabilidad (voltea la decisión entre los 9 seeds, sobre las 134)

v13 22 % · v14d **21 %** · v14c 22 % · v14a 25 % · v14b 26 % · v14e 28 %. Ojo:
esta cifra no es comparable con el 30 % del barrido de seeds, que era sobre las
54 discutidas. La línea base correcta aquí es el 22 % de v13.

## V15 — RECHAZADO 2026-07-22, job 7299

V15 = V13 + el ejemplo de V14d revisado para modelar también la condición que
encabeza una lista (la forma real del 67, verificada en §Premedicación). Salió
**peor que V14d en todo menos en un probe**:

| | acierto | ratio gana/paga | estabilidad | 105 | 67 | 87 | 26 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v13 | 78,1 % | — | 22 % | 9/9 | 8/9 | 9/9 | 2/9 |
| **v14d** | **78,4 %** | **1,6×** | **21 %** | **0/9** | 9/9 | **2/9** | 4/9 |
| v15 | 77,4 % | 1,0× | 25 % | 1/9 | 6/9 | 4/9 | 7/9 |

Y el 6/9 del 67 **no es un arreglo**: leídas, las mayoritarias siguen diciendo
«el equipo médico puede darte medicación» sin la condición. La hipótesis de V15
falla, y de paso degrada lo que V14d había ganado (105 y 87), empeora el 26 y
pierde estabilidad. Es dilución: 1105 t frente a 1042 t de V14d y 915 t de V13.

**Lección, coherente con `docs/prompt_versions.md`: un ejemplo, un
comportamiento.** Fundir dos mecanismos en un ejemplo diluyó el que funcionaba.

**Candidato a entregable: V14d**, no V15. Arregla el 105 (9/9→0/9, leído: «con
anestesia regional o general») y el 87 (9/9→2/9) sin coste medible de acierto ni
de estabilidad. No mueve la decisión — nada la mueve.

Comprobado de paso: el brazo v13 de este job es **byte a byte idéntico** al del
7290 (1206/1206 respuestas) pese a otro nodo y otro job. El harness es
determinista, así que las próximas rondas pueden reutilizar brazos ya medidos en
vez de re-correr el control.

## A/B DE CORPUS — HECHO 2026-07-22, job 7300. Refuta la hipótesis simple

Tres ediciones, una por documento, solo gramática y estructura, sin añadir ni
quitar hechos. Control: el brazo v13 ya versionado (el harness reproduce entre
jobs, comprobado). Los `.v2.md` **no están en git** — `corpus/markdown/` es
gitignored; viven en el portátil y en el cluster.

| id | edición | probe | ¿qué pasó de verdad? |
| ---: | --- | --- | --- |
| 29 | verbo y sujeto: «Si tiene fiebre, puede tomar paracetamol» | 7/9 → **0/9** | **no lo arregla: lo borra.** Paracetamol desaparece de las 9 respuestas |
| 108 | actor nombrado: «Su médico le indicará cómo ajustar…» | 9/9 → **0/9** | **no lo arregla: rechaza.** «No tengo información sobre cómo ajustar tu medicación con anticoagulantes» |
| 67 | lista plegada dentro de la frase con la condición | 8/9 → 8/9 | **sin cambio. Hipótesis refutada** |

Acierto de decisión **76,5 % frente al 78,1 %** de base, neto −19 emparejado, y
33 de las 40 regresiones son sobre-rechazo. Rechazo en hemorroides 54 → **59 %**,
justo el documento donde nombré al actor. Diabetes y cirugía, sin mover.

**Las dos «victorias» de los probes son el modelo fallando de otra manera.** El
108 pasa de mandar al paciente a rechazar — y encima el rechazo **nombra el
tema**, que viola la propia regla de V13. El 29 deja de soldar porque deja de
mencionar el paracetamol.

**Lo del 67 corrige un diagnóstico mío de esta misma sesión.** Dije que la
condición vivía «encabezando una lista». Falso: la condición y la promesa ya
estaban en la misma frase, y solo el *contenido* era lista. Plegarla no cambió
nada. El modelo simplemente suelta la subordinada al comprimir.

### La hipótesis que sí queda en pie, y cómo probarla

**El modelo cita los fragmentos telegráficos; en prosa, parafrasea, omite o
rechaza.** El fragmento sin verbo es material citable y por eso se suelda con el
de al lado; la frase completa deja de ser citable y el modelo la trata como algo
que debe reformular — y al reformular, la pierde.

Si es cierto, la guideline **no** es «escribe frases completas», que es lo que yo
asumí y lo que este experimento acaba de tumbar. Sería algo más exigente: **cada
frase tiene que ser autosuficiente** — llevar su condición, su actor y su alcance
dentro, de forma que reformularla no pueda perder nada.

Siguiente test barato con la maquinaria montada: un `.v3` donde las mismas tres
reglas se escriban autosuficientes (p. ej. «Si tiene fiebre, puede tomar
paracetamol; consulte igualmente si supera los 39 °C»), y comparar v2 contra v3.
Discrimina «gramática» de «autosuficiencia», que es lo que este A/B no separa.

Salvedad: un brazo, tres ediciones, una por documento. Como cada edición vive en
un procedimiento distinto, los tres resultados **sí** son atribuibles por
separado; lo que no está separado es edición vs. resto del documento, porque
cambiar una línea cambia el snapshot entero (de ahí que la 26 empeore, 2/9 → 5/9,
sin haberla tocado).

## A/B DE CORPUS v3 — HECHO 2026-07-22, job 7303. Autosuficiencia > gramática

`.v3` reescribe **las mismas tres líneas** que el v2, pero haciéndolas
**autosuficientes**: cada frase lleva dentro su condición, su actor y su alcance.
Un brazo, V13 sin tocar, 134 × 9 seeds. Salidas en `eval/audit_corpus_v3/v13/`.

⚠️ **NO leer esto como «el v3 sube el acierto».** Se escribió así primero y era
sobreventa. El **79,4 % NO es distinguible del 78,1 %**: los rangos se solapan y
**el v14b ya sacó 79,3 % con neto +14/1206 y este mismo documento lo declaró
«dentro del azar»** (§RESULTADO DEL A/B). Mismo tamaño, mismo veredicto. Lo que
sí destaca es la **estabilidad, 16 %**, fuera del rango de las seis variantes de
prompt (21-28 %) — y aun así es un solo brazo. **El resultado del v3 es el 67
leído, no el porcentaje.**

Y hay una razón de método por la que el porcentaje **no puede** ser el
instrumento aquí: la edición toca **3 líneas de 3 documentos**. Aunque
arreglaran las tres preguntas objetivo perfectamente, el efecto esperado sobre
134 preguntas es ~2 puntos — exactamente el suelo de ruido. El agregado sirve de
**guardarraíl** (¿ha roto algo global?), no de evidencia de mejora. La evidencia
son los probes dirigidos + la lectura.

Los números, con esa lectura:

| | acierto medio | min–max | neto emparejado | estabilidad |
| --- | ---: | --- | ---: | ---: |
| v13 | 78,1 % | 76,9–79,1 | — | 22 % |
| corpus_v2 | 76,5 % | 75,4–80,6 | −19 | 21 % |
| **corpus_v3** | **79,4 %** | **78,4–79,9** | **+15** | **16 %** |

El rango entero del v3 queda por encima de la media de v13, y el 16 % de volteo
es la mejor estabilidad medida hasta hoy. Rechazo **baja en los tres**
procedimientos (44→40, 39→37, 54→52). Telegráficas suben solo en hemorroides
(38→41 %), que es el documento de 1,1 KB de siempre.

### Pero de las tres fronteras solo cae una — y hay que decirlo así

| id | edición v3 | probe | leído |
| ---: | --- | --- | --- |
| **67** | «La medicación… **no se da a todos los pacientes**: solo se administra cuando…» | 6/9 → **0/9** | **ARREGLADO DE VERDAD.** Las 9 conservan contenido **y** condición. Y su par de control, el **86**, converge en la misma respuesta buena |
| 108 | actor + prohibición explícita de automedicarse | 5/9 → 0/9 | **NO.** Rechaza: «No tengo información sobre eso». Idéntico fallo que el v2 |
| 29 | «puede tomar paracetamol, **sea cual sea la temperatura**; que supere 39 °C es motivo de consulta, no umbral» | 7/9 → 0/9 | **NO.** El paracetamol **desaparece de las 9**. Idéntico fallo que el v2 |
| 26 | (no tocada) | 2/9 → 1/9 | parcial, y **con invención nueva**: un seed escribe «sesiones de al menos 15 minutos», que no está en el documento |

O sea: el 67 confirma la hipótesis de autosuficiencia con un cambio y una
variable —el v2 plegó esa misma lista y no movió nada (8/9 → 8/9)—, y el 108 y
el 29 **repiten el fallo del v2**.

### La hipótesis refinada, que es lo que se lleva a la próxima ronda

Lo que distingue al 67 de los otros dos es **qué quedó en la oración principal**:

    67   principal = la medicación y sus dos pastillas;  la restricción, dentro  -> se conserva
    108  principal = «es su equipo médico quien decide»; el ajuste, subordinado  -> rechaza
    29   principal = la consulta por >39 °C;             el paracetamol, aparte  -> lo suelta

**El modelo conserva la oración principal y suelta lo subordinado.** No basta con
que la frase sea autosuficiente: **el contenido accionable tiene que ser la
principal**, y la condición/alcance colgar de él, no competir con él. Es
concreto y comprobable — un `.v4` que reescriba 108 y 29 con el acto en la
principal («Puede tomar paracetamol si tiene fiebre, sea cual sea la
temperatura») discrimina esto de lo ya medido.

⚠️ Y por cuarta vez esta semana, un probe cantó un arreglo falso: el **87** pasó
a 0/9 sin que el v3 tocara su documento, y leído sigue diciendo «Sí, necesitarás
ayuda para caminar». Cambió la redacción lo justo para esquivar la regex.

## LLAMA-CPP — EL PIN CAE 2026-07-22. Medido, no heredado

`pyproject.toml` pineaba `==0.3.19` por una única observación sobre 0.3.23
(no emitía `vpdpbusd`/`tdpbssd`). **Re-testado con 0.3.34** en `computo02`, mismo
toolchain (`spack gcc@12.5.0`) y **los mismos `CMAKE_ARGS`** de
`build_venv_native.sh`, cambiando solo el número de versión:

| | 0.3.19 (ggml 0.9.8) | 0.3.34 (ggml 0.16.0) |
| --- | ---: | ---: |
| vpdpbusd / tdpbssd / tileloadd | 1757 / 94 / 96 | **1757 / 94 / 96** |
| decode @ nT=8 | 16,62 tok/s | 16,69 tok/s |
| prefill | 122,9 tok/s | **162,9 tok/s (+33 %)** |
| warm del prefijo | 15,20 s | 11,47 s |
| pickle del snapshot | 445,8 MiB | 445,8 MiB |
| `load_state` por petición | 207 ms | 205 ms |
| `_seed = state.seed` | `llama.py:2185` | `llama.py:2237` — **sigue** |

Comprobado que son builds distintos (md5 y tamaño distintos, ggml 0.9.8 vs
0.16.0); los contadores coinciden porque los kernels int8 salen de las mismas
plantillas. **La premisa del pin es falsa hoy.** Subido a `==0.3.34`.

**Dos moralejas de método.** La verificación original miró solo
`libggml-cpu.so`; el ggml moderno puede repartir variantes de ISA en librerías
hermanas, así que **hay que objdumpear todos los `.so`** del paquete (lo hace
`/tmp/…/build_venv_latest.sh`, copiado a `~/build_venv_latest.sh` en el cluster).
Y el +12-15 % de VNNI que justificaba el pin es a **nT=32**; a **nT=8**, que es
lo que desplegamos, era +3,5 %.

Lo que desbloquea, comprobado en el binario: 0.3.34 añade **`gemma4`** y las
plantillas `granite-4.0`/`granite-4.1`. **Qwen3.6 no aparece como arquitectura
propia** (hay `qwen35`, `qwen35moe`, `qwen3moe`, `qwen3next`) — probablemente se
declare como `qwen35moe`, pero **eso hay que probarlo, no darlo por hecho**.

Bajados ya a `models/` del cluster: `Qwen3.5-4B-Q4_K_M` (2,55 GB),
`Qwen3.5-35B-A3B-Q4_K_M` (20,5 GB), `granite-4.1-3b-Q4_K_M` (1,95 GB) y
`gemma-4-26B-A4B-it-UD-Q4_K_M`. El nodo tiene **1.027 GB de RAM** y 461 GB de
disco libre: la memoria **no** es la restricción, y para un MoE eso invalida el
«modelo más grande = menos usuarios» de [[plan-b-model-benchmark]] — en MoE la
velocidad la fijan los **activos** y llama.cpp mapea el GGUF con `mmap`, así que
las 8 réplicas comparten page cache.

## EL SIGUIENTE FRENTE ES EL CORPUS, NO EL PROMPT

Las tres fronteras que **ninguna** de las seis variantes tocó — 29, 67, 108 —
comparten una forma, y es del documento destilado, no del modelo:

    108  «Ajustar medicación (anticoagulantes, etc.)»   infinitivo sin sujeto
     67  «Cuando el grado de ansiedad... le darán:»     condición que encabeza una lista
     29  «- Si fiebre: paracetamol.»                    condicional sin verbo ni sujeto
         «Consulte si: ...; fiebre >39 °C; ...»         y su alarma en otra lista

La destilación comprimió las reglas en fragmentos telegráficos y repartió cada
regla entre viñetas. El modelo tiene que reensamblarlas y las reensambla mal:
pone al paciente de sujeto porque la pregunta va en segunda persona, suelta la
condición que encabeza la lista, y suelda dos viñetas de la misma sección. **Eso
no es arreglable desde el prompt y está medido: 6 variantes, 0 mejoras en las
tres.** Ver [[distillation-method]] — el objetivo de 2-4K tokens es lo que creó
la forma.

Propuesta de siguiente experimento, barato y con la maquinaria ya montada:
reescribir **solo esas tres líneas** del corpus devolviéndoles sujeto y verbo, y
correr el mismo harness con V13 sin tocar. Un cambio, una variable, ~30 min. Si
las tres caen, el frente de trabajo es la destilación y hay que decírselo al
cliente clínico junto con el ID 29.

### Recomendación para V15 — la que se escribió ANTES de medirlo

Un solo cambio sobrevive a la lectura y no cuesta ni acierto ni estabilidad:
**el ejemplo de V14d** (105 de 9/9 a 0/9, 87 a 2/9, estabilidad la mejor del
lote). V15 = V13 + ese ejemplo, **extendido para modelar también una condición
que vive en la frase vecina**, que es lo que le faltó al 67. Lo demás no entra:
V14a paga 1:1, V14b no supera el azar, V14c no podía ganar y V14e cambia
rechazo por telegrafía.

**Decisión explícita del usuario:** NO redactar todavía la respuesta a los auditores.
Primero cerrar nuestra propia evaluación. La respuesta será global.

## Los números que ya tenemos (no re-derivar)

Reproducción: **125/134 (93 %)** coinciden en responder-vs-rechazar. 0 fallos de red.
Donde ambos responden, el texto es a menudo idéntico palabra por palabra pese a
`temperature=0.1`. Las 9 divergencias NO son zona gris: incluyen sobre-rechazos
flagrantes (39, 40 — el fulldoc lo dice literal dos veces) e invenciones (111, 117).

**Evaluación propia, con la definición de zona gris ya adoptada** (ver abajo):

| Capa | Resultado |
| --- | --- |
| Decisión responder/rechazar | **106/134 = 79 %** |
| ├ rechazables acertadas | 35/39 = 90 % (fallamos en 8, 52, 111, 117) |
| └ respondibles respondidas | 71/95 = 75 % (24 sobre-rechazos reales) |
| Calidad de las 71 bien respondidas | **39 % buenas** (28/71), 32 % telegráficas, 14 % sin fundamento |

Ojo: las "buenas" bajaron de 45 % a 39 % tras el barrido de fronteras — cuatro que
dábamos por correctas (22, 67, 68, 84) resultaron defectuosas. Es una corrección a
la baja **nuestra**, no de ellos, y conviene decirlo así en la respuesta.

`audit_triage.py` ya emite **SR 24 / FN 35 / DEF 26 / OK 49**, que es lo que dice
esta tabla. Antes emitía 34/25 porque nunca se le aplicó la definición de zona gris;
sincronizado 2026-07-21. La cifra de "defectos genuinos" pasó de 60 a **50**.

| Especialidad | Decisión | Buenas / respondidas | Fulldoc |
| --- | ---: | ---: | ---: |
| cirugia-abdominal | **92 %** | 18/29 | 7,6 KB |
| diabetes | 73 % | 14/29 | 13,0 KB |
| hemorroides | 71 % | **0/13** | 1,1 KB |

Lectura: **la decisión es aceptable, la redacción no.** diabetes tiene la peor
decisión con el corpus más rico → 100 % prompt, es donde más rinde el trabajo.
hemorroides no tiene arreglo por prompt en expresión: 0 de 13.

Salvedad de método: el 79 % usa nuestro propio criterio de respondibilidad. Cada
veredicto cita la línea del corpus, pero conviene muestrear las 39 rechazables antes
de usar la cifra fuera.

## Definiciones fijadas — no re-litigar

**Zona gris = tema relacionado pero NO respondible desde el fulldoc; hay que
rechazar aunque sea tentador responder.** Es la definición que ya estaba en
`eval/datasets/*_grayzone.json` (todos `answerable=False`, categoría
`C_topic_related_no_answer`). Adoptada por el usuario 2026-07-21.

Consecuencia ya aplicada: las 10 de "cobertura parcial" (tema nombrado, no
desarrollado) son rechazables, no un tercer grupo. Eso movió sobre-rechazo 34→24 y
rechazos correctos 25→35. **Quedan dos grupos: 95 respondibles, 39 rechazables.**

**No proponer bajar a `temperature=0` ni cambiar el sampling.** Ya decidido: se
compara por equivalencia de comportamiento.

## El defecto de fondo (conclusión del triaje)

Un solo defecto con dos caras: **el sistema no distingue bien qué sostiene el
documento.** De prudente da sobre-rechazo (24); de servicial, inventa (14 falsos
positivos). Ambas caras se atacan en `app/prompt.py`, no en el corpus.

La evidencia decisiva son tres pares casi idénticos con resultado opuesto —
**112/128** (incontinencia), **91/103** (miedo antes de operarse), **30/6** (medirse
el azúcar). La variante rechazada es siempre la de carga emocional o de permanencia
("para siempre", "llorar", "desde ya"): el modelo trata el *registro* de la pregunta
como si fuera el *tema*. Cuantificado: **70 % de rechazo en las 20 preguntas
emocionales frente al 39 % del resto.**

Peores defectos por consecuencia clínica: **52** (inventa mortalidad por hipoglucemia
nocturna), **89** (inventa un "protocolo de ayuno estricto" y se contradice), **108**
(anticoagulantes en imperativo sin sujeto), **29** (ver abajo), **105/30** (exceso de
certeza). **24 y 26** filtran meta-comentario al paciente.

## Sobre su auditoría — verificado

Todas sus cifras cuadran exactamente (46,3 % rechazos, 73,1 % Alta+Crítica, 9 %
aceptable, hemorroides 61,3 % nulas y 96,8 % ≤12 palabras). El desacuerdo es de
diagnóstico, no de hechos. Tres cosas que NO hay que olvidar:

1. **Su ancla es externa, no nuestro corpus.** Prueba: hoja `Fonts clíniques`, con
   columna "Ús en l'auditoria" — ADA 2026, ERAS, ASA, ASCRS. Afecta también a cómo
   puntúan las respondidas: incluso a sus 9/10 les exigen contenido que no está en
   nuestro fulldoc (32, 81, 11).
2. **"Derivar" NO va contra nuestras reglas — en eso tienen razón.** Decir "eso
   depende de tu caso, consúltalo con tu equipo" no inventa nada ni requiere una
   palabra fuera del fulldoc. Su código `Sense resposta` penaliza rechazar algo que
   "es podria contestar **o derivar**". Es exactamente la tercera salida que
   proponemos. **No discutírselo: perderíamos.**
3. **Confunden tres tipos de hallazgo**: defecto del sistema, límite de alcance del
   corpus, y **defecto clínico del corpus**. El caso claro es el **ID 29**: su
   crítica más dura ("la resposta més perillosa del bloc") apunta a una frase que
   está literalmente en nuestro corpus — `Nunca suspenda medicación de la diabetes
   (pastillas o insulina); dosis habitual salvo indicación médica`. El sistema
   reprodujo fielmente el documento. **Eso hay que escalarlo al cliente clínico**,
   no arreglarlo en el prompt (las reglas de días de enfermedad de ADA sí exigen
   pausar SGLT2i). El 29 tiene *además* un defecto nuestro: funde "Si fiebre:
   paracetamol" con "Consulte si… fiebre >39 °C" en "si supera 39 °C usa
   paracetamol", convirtiendo un criterio de alarma en umbral de tratamiento.

Su rúbrica está bien diseñada (tiene códigos de fuga de metadatos, contradicción,
exceso de certeza, riesgo clínico). El problema es el ancla, no el instrumento. Lo
que le falta es un código para "rechazó correctamente dado este corpus".

## Barrido de reglas fundidas — HECHO 2026-07-21

Resultado: **11 de las 75 respuestas (15 %)** rompen la frontera entre dos reglas del
documento. Está en `audit_triage.py` como dict `RULE_BOUNDARY` y en el apartado
**4c** del informe. Cuatro mecanismos, y el nombre importa porque no todos son
"fusión":

| Mecanismo | N | Qué pasa |
| --- | ---: | --- |
| `fusión` | 6 | dos frases independientes soldadas en un condicional (26, 29, 22, 68, 84, 89) |
| `sujeto` | 3 | la regla conserva el contenido y cambia de actor (4, 87, 108) |
| `des-scope` | 1 | la regla pierde la condición que la acotaba (67) |
| `disyunción` | 1 | un "A o B" cerrado en A (105) |

**8 de los 11 son nuevos** — ni ellos ni nosotros los habíamos visto. Los tres que
hay que saber de memoria:

- **26** es el más limpio de explicar: "al menos 150 min semanales" + "caminar 30-45
  min diarios es muy beneficioso" → "150 minutos semanales repartidos en sesiones de
  al menos 30-45 minutos diarios", que **no puede ser cierto** (30 diarios ya son 210
  semanales). Dos frases correctas producen una prescripción imposible.
- **4** es el de peor consecuencia: "si usas insulina, ajustar la dosis según la
  glucemia postprandial" contradice de frente la regla de adherencia del fulldoc
  ("no modificar dosis ni suspender sin indicación"). No inventó un hecho: reasignó
  el sujeto de una regla.
- **67 vs 86** y **87 vs 88** son pares de control: mismo material, una conserva la
  frontera y la otra no.

Lectura que cambia el orden del trabajo: esos dos pares son al defecto de frontera
lo que 112/128 y 91/103 son al sobre-rechazo. **Las cuatro parejas dicen lo mismo** —
el sistema tiene el documento delante y decide distinto según cómo esté formulada la
pregunta. Sobre-rechazo y regla rota son la misma inestabilidad por los dos lados.
Esto refuerza que el pendiente nº 2 (inestabilidad) es la métrica de fondo.

Por qué cuesta verlo: **todos los ingredientes salen del documento**. No hay nada que
un detector de contenido inventado pueda marcar y la respuesta parece completa.

## Plan acordado 2026-07-21 — mapa de los 5 pasos (1, 4 y el método del 2, hechos)

Decisión de método: **medir la inestabilidad ANTES de tocar el prompt o el modelo.**
Motivo: los cinco pares de control demuestran que el sistema responde distinto al
mismo contenido según cómo se formule la pregunta. Eso es varianza, no una regla
que falte. Y **con varianza desconocida ningún A/B es interpretable**: "V14 arregla
5 preguntas" y "V14 no hizo nada y muestreamos dos veces un proceso ruidoso" se ven
igual. Por eso el paralelismo de variantes NO va primero: amplificaría el problema.

1. **Medir inestabilidad — HECHO 2026-07-21.** Varianza cero *a seed fijo* (y el
   seed está congelado dentro del pickle). Cuidado con leer eso como robustez:
   el barrido de seeds mide el suelo de ruido real, **30 % de volteo a t=0,1**.
   Los dos apartados siguientes, en ese orden.

2. **Iterar el prompt — ES LO SIGUIENTE. Plan completo en «PASO 2» abajo**, con
   el método ya fijado (§2.0, 2026-07-22). Corrección del usuario 2026-07-21:
   **se mide sobre las 134, no sobre los pares de control** — un cambio de
   prompt es global y puede arreglar 5 preguntas rompiendo 20. (Matizado
   2026-07-22 con la criba de §2.2: se *criba* por el objetivo de cada variante,
   se *valida* sobre las 134.) Si hay barrido de variantes: **3-4, no 8**, y
   **una variante por réplica (8 hilos), NO una por core** — una réplica de 1
   hilo va ~8× más lenta y mediría calidad en una config que no vamos a
   desplegar. La regla del nodo es `réplicas × N_THREADS ≤ 64`.

3. **Modelo más grande** solo para lo que sobreviva, y con esta cifra delante:
   hoy son ~11,4 tok/s a 8×8 con un 3B; un 7-8B Q4 es ~2,3× el cómputo → del orden
   de **5 tok/s, por debajo del suelo de 6**. Recuperarlo obliga a menos réplicas,
   o sea **menos usuarios concurrentes**. Es un intercambio de entregable, no un
   experimento gratis.

4. **Reordenar el triaje por consecuencia clínica — HECHO 2026-07-21.** Está en
   `audit_triage.py` como dict `CLINICAL` (id → gravedad + qué le pasa al
   paciente) y en el apartado **4d** del informe, cruzado con la estabilidad del
   barrido de seeds. Criterio: *qué le hace hacer al paciente*, no cuán
   equivocada está. Los sobre-rechazos **no** entran en la escala — su daño es el
   abandono, no una instrucción errónea. Reparto: **G1 3 · G2 5 · G3 12 · G4 6**.

   Los tres **G1 son 4, 29 y 108**, y comparten forma: convierten en instrucción
   al paciente algo que el documento dirige al equipo, o desactivan un criterio
   de alarma. Ninguno inventa un hecho — todos reasignan o funden reglas del
   corpus. **Confirma que §4c (fronteras) es la prioridad, no el contenido
   inventado.** Cruzado con estabilidad: 108 es 9/9, 29 es 8/9 — atacables y
   medibles; 4 salió `n/r` en el barrido (confundido con el build, no concluir
   que no exista).

   Aserciones nuevas: falla si un DEF se queda sin gravedad, si se gradúa algo
   que no es DEF, o si `BOUNDARY_PERSISTENCE` sale de `RULE_BOUNDARY`.

5. Solo entonces: redactar la respuesta global.

## Medición de inestabilidad — HECHA 2026-07-21. Resultado: varianza CERO

10 pasadas × 134 preguntas contra los pools vivos del job 7222
(`eurehpccomputo01`), sin tocar prompt, snapshot ni imagen.

| Procedimiento | Rechazos por pasada | sd | Oscilantes |
| --- | --- | ---: | ---: |
| diabetes | 24/55 las 10 veces | 0,0 | 0 |
| cirugia-abdominal | 19/48 las 10 veces | 0,0 | 0 |
| hemorroides | 16/31 las 10 veces | 0,0 | 0 |

Y no es solo la decisión: **las 1340 generaciones son idénticas byte a byte**, y
además idénticas al replay tomado 7 h antes contra ese mismo pool (134/134).

### Por qué: hay un seed congelado, metido sin querer por los snapshots

No es que a `temperature=0.1` el muestreo sea argmax. **Es un seed fijo**, y la
cadena es ésta:

1. `llama_cpp/llama.py` — `LlamaState` guarda un campo `seed`, y
   `load_state()` lo restaura: `self._seed = state.seed`. El pickle del snapshot
   **lleva un seed dentro**.
2. `app/routes/query.py:63` — hacemos `load_state()` **antes de cada
   generación**, dentro del lock, para devolver el KV al prefijo.
3. `llama_cpp/llama.py:1332` — al generar sin `seed` explícito:
   `set_seed(random.Random(self._seed).randint(0, 2**32))`.

El paso 1 resetea `_seed` a la constante del pickle antes de cada petición, así
que el paso 3 deriva **siempre el mismo número**. Nunca pasamos `seed` en
ninguna parte de nuestro código (`grep` en `app/`: solo aparece `temperature`).

Y el seed del pickle **tampoco es aleatorio**: sale de `LLAMA_DEFAULT_SEED`
(`0xffffffff`) encadenado por orden de warmup, en una única cadena determinista:

    4294967295 → 872737089 → 1894574933 → 1634406207 → 2999353390 → …

⚠️ **CORREGIDO 2026-07-22.** La versión anterior daba una sola columna de seeds y
eso indujo un error de método (ver §2.0). Hay que distinguir **dos** valores, y el
seed *almacenado* en la posición N es el *derivado* de la N−1. Leídos con
`pickle.load(...).seed` de los pickles que sirve el pool:

| procedimiento | pickle | pos | seed almacenado | **seed de generación real** |
| --- | --- | ---: | ---: | ---: |
| diabetes | `f8dbaf71…` | 1ª | 872737089 | **1894574933** |
| hemorroides | `ad0099cc…` | 2ª | 1894574933 | **1634406207** |
| cirugia-abdominal | `66e16bfb…` | 3ª | 1634406207 | **2999353390** |

O sea que **el seed depende solo de la posición del procedimiento en el orden de
warmup del proceso que construye el snapshot**, no de la máquina ni del momento.
Y como los pickles servidos son de mayo, de un warmup de los **tres** anterior a
los perfiles, **ningún camino de código actual los reconstruye**: hoy `aiciblock`
es `{hemorroides, cirugia}`, así que un build del perfil los pone en posiciones 1
y 2 y les toca otra cadena.

### Ellos corrieron con la misma configuración — comprobado, no supuesto

`dist/rag-deliverable-v1.1/` es lo que se les entregó. De su imagen
(`images/cpu-rag-api-1.2.0-portable.tar`, extraída sin docker):

- `app/routes/query.py:75` → `temperature=0.1`; `app/snapshot_builder.py:118`
  → `temperature=0.1`. Sin `seed`. **Idénticas a las nuestras de hoy.**
- `llama_cpp` **0.3.23** con el mismo mecanismo (`llama.py:1345` y `:2198`).
- `snapshots/` va **vacío** en el entregable y lleva los tres corpus → los
  generaron ellos al arrancar, y les tocaron esos mismos seeds.

Diferencias que sí quedan frente a nuestro cluster, y que explican parte de las
9 divergencias: **llama-cpp 0.3.23 (suyo) vs 0.3.19 native VNNI (nuestro)**, y
que v1.1 calienta los tres procedimientos en un proceso mientras nuestros
perfiles los reparten 1 + 2 → **distinta posición en la cadena de seeds**.

### Lo que cambia

1. **Los cinco pares de control no son ruido.** Son sensibilidad *determinista al
   enunciado*: mismo documento, trato opuesto, las 10 veces
   (30=0/10 respondida vs 6=10/10 rechazada; 91=0/10 vs 103=10/10; 112=0/10 vs
   128=10/10). Es un hallazgo **más fuerte** que la varianza y reproducible a
   voluntad — para la respuesta a los auditores es mejor argumento, no peor.
2. **Pero el suelo de ruido de 0 NO es transferible a un A/B.** Es 0 porque el
   seed está clavado, no porque el modelo esté seguro. Todo número que tenemos
   —su scorecard y el nuestro— es **una sola tirada**. Cuánto enmascara eso es lo
   que mide `tools/audit_seed_sweep.py` (job 7246).
3. ~~**Regla operativa para el A/B de prompts:** construir siempre uno por
   proceso.~~ ⚠️ **DESCARTADO 2026-07-22.** Iba en la dirección equivocada: uno
   por proceso da posición 1 a los tres, o sea seed de generación 1894574933 para
   todos, que coincide con producción **solo en diabetes** y cambia el seed de
   hemorroides y cirugía a la vez que el prompt. Con 42 % de volteo en
   hemorroides, ese confundido se come cualquier efecto del prompt. La regla
   buena está en §2.0: **el seed no se fija, se promedia.**
4. **El test NO decidió el reparto prompt-vs-modelo.** La hipótesis era que el
   sobre-rechazo saldría establemente rechazado y las fronteras rotas oscilarían.
   No oscila nada, así que la partición no separa.

Salidas versionadas en `eval/audit_stability/<procedimiento>.json`, por el mismo
motivo que los replays.

## Barrido de seeds — HECHO 2026-07-21 (job 7246). El suelo de ruido real

`tools/audit_seed_sweep.py`, 9 seeds × {t=0,1 · t=0,8} sobre las 54 preguntas que
el triaje discute (SR + DEF + los pares de control). Carga el modelo en proceso
sobre `.venv-native`; el seed no se puede inyectar por la API porque va dentro
del pickle.

**A t=0,1, la que desplegamos:**

| Procedimiento | Q | Rechazos por seed | Texto cambia | **Decisión voltea** |
| --- | ---: | --- | ---: | ---: |
| diabetes | 27 | 9–14 | 78 % | **33 %** (9) |
| cirugia-abdominal | 15 | 4–4 | 73 % | **13 %** (2) |
| hemorroides | 12 | 5–8 | 58 % | **42 %** (5) |
| **total** | **54** | | | **30 % (16)** |

A t=0,8 sube a 70 / 60 / 67 %. Ojo con cirugía: el **recuento** de rechazos es
constante (4 con los 9 seeds) mientras **la composición** cambia. Mirar solo la
cifra agregada habría dado sd=0 y escondido el trasiego.

### Lo que aguanta y lo que no

**Los tres pares de sobre-rechazo SOBREVIVEN al barrido** — es lo más sólido que
tenemos para la respuesta:

| Par | Miembro tratado | Miembro rechazado |
| --- | ---: | ---: |
| 30 / 6 | 0/9 rechazos | 8/9 |
| 91 / 103 | 0/9 | 8/9 |
| 112 / 128 | 0/9 | **9/9** |

La sensibilidad al enunciado **no es ruido de muestreo**. Pero el 30 % de las
demás preguntas discutidas sí lo es.

**Y la hipótesis sobre las fronteras rotas queda respaldada**: son mucho más
estables que la decisión. Verificado leyendo los textos, no solo por keyword:

| ID | Mecanismo | ¿Rompe la frontera? |
| ---: | --- | --- |
| 105 | disyunción | **9/9** — «Anestesia regional.» idéntico las 9 veces |
| 108 | sujeto | **9/9** — «Sí, debes ajustar tu medicación anticoagulante» |
| 67 | des-scope | **9/9** — «Sí, el equipo te dará medicación», sin la condición |
| 87 | sujeto | **9/9** — siempre traslada el «con ayuda» al caminar |
| 84 | fusión | **9/9** — siempre responde el género con la especie |
| 29 | fusión | **8/9** — «Si la fiebre supera los 39 °C, usa paracetamol» |
| 26 | fusión | **3/9** — la prescripción imposible es intermitente |
| 4, 22, 89 | — | **0/9** — no se reprodujeron en esta configuración |
| 68 | fusión | no concluyente con este probe |

Es decir: **6 de los 10 concluyentes se reproducen en ≥8 de 9 seeds**, frente al
~30 % de volteo de la decisión. Sobre-rechazo y frontera rota NO son la misma
inestabilidad, al contrario de lo que decía el apartado del barrido de fronteras.

### Consecuencias

1. **El seed NO es un hiperparámetro; la temperatura sí.** El seed elige *qué*
   tirada sale, no cambia la distribución: los seeds son intercambiables y el
   número esperado de rechazos es el mismo para todos. El 9–14 de diabetes es
   dispersión alrededor de ~12,4, no que un seed sea mejor — el seed 1 acierta en
   una pregunta y falla en otra donde el 21 hace lo contrario. Elegir el mínimo
   de 9 tiradas sobre 27 preguntas es seleccionar ruido, y regresaría a la media
   en la pregunta 135. Lo que sí mueve la dispersión es la temperatura: 33/13/42 %
   de volteo a t=0,1 frente a 70/60/67 % a t=0,8.

   Corolario incómodo: **nuestro determinismo es frágil, no robusto.** Se sostiene
   solo mientras no cambien el seed del pickle, el orden de warmup, la versión de
   llama-cpp y el número de hilos — y hemos visto moverse a los cuatro (su build
   0.3.23 coincide en 125/134; `.venv-native` a nT=32, en 29/54). A t=0 saldría la
   moda pase lo que pase con eso.

   **Decisión pendiente del usuario, no darla por tomada:** este apartado dice
   «no proponer bajar a `temperature=0`». Esa regla era para el **protocolo de
   reproducción** (no auditar una configuración distinta de la que corrieron
   ellos) y sigue siendo correcta para eso. Si t=0 es la configuración adecuada en
   **producción** es otra pregunta, que estos datos reabren.
2. **Los veredictos individuales de las preguntas que voltean son una sola
   tirada** — los nuestros y los suyos. Hay que decirlo en la respuesta. Lo que
   NO es una tirada: los pares de control y las fronteras estables.
3. **Prioridad de trabajo, ordenada por lo que ahora sabemos**: las fronteras
   rotas estables (105, 108, 67, 87, 84, 29) son el objetivo con mejor relación
   señal/ruido — reproducibles al 100 %, así que un cambio de prompt sobre ellas
   se mide limpio. El sobre-rechazo, con 30 % de volteo, necesita más cuidado
   estadístico del que suponíamos.

### Salvedades de método — importantes

- **El barrido reconstruyó el snapshot** (`BUILT`, no `HIT`), y por eso solo
  **29 de 54** respuestas del seed congelado coinciden con el replay. La
  **validez interna es buena** (las 18 celdas comparten build, hilos y snapshot;
  solo varía el seed), pero los absolutos no son los de producción. El «0/9» de
  4, 22 y 89 mezcla efecto de seed con efecto de build: no concluir de ahí que
  esos defectos no existan — el replay los documenta.

  ⚠️ **CORREGIDO 2026-07-22 — la culpa NO era del `nT=32`.** Esta nota lo daba
  por causa principal. Medido en dos jobs con el sbatch nuevo:

  | config | coincidencia con el replay |
  | --- | ---: |
  | nT=32, sin pinear, snapshot reconstruido (7246) | 29/54 |
  | **nT=8 pineado**, snapshot **reconstruido** (7274) | 4/10 |
  | **nT=8 pineado**, snapshot **HIT del pool** (7276) | **54/54** |

  O sea: bajar a 8 hilos por sí solo no arregla nada; lo que decide es **qué
  pickle se carga**. Con el pickle que sirve el pool sale byte a byte lo mismo
  (la única «diferencia» en 54 era un `\n` final que el transporte del pool
  recorta). Los seeds almacenados que leyó cada celda coinciden exactamente con
  la tabla de arriba, así que esa tabla queda confirmada por segunda vía.

  **Consecuencia para el A/B, que no es menor:** un prompt distinto es otra
  clave de caché, así que **ninguna variante puede dar HIT**. Todos los brazos
  reconstruyen — **V13 incluido**. Un brazo V13 servido desde el pickle de mayo
  no sería un control válido: estaría comparando prompt *y* procedencia a la
  vez, y acabamos de medir que la procedencia mueve ~la mitad de las
  respuestas. La comparación es interna a propósito; el ancla con producción es
  el replay final contra el pool.
- ⚠️ **CORREGIDO 2026-07-22 — la colisión nunca ocurrió.** Se decía que el
  barrido sobrescribió los `.pkl` que servía el pool 7222. No es cierto: el pool
  sirve `snapshots/<perfil>/` y el barrido escribió en la **raíz** `snapshots/`,
  porque importa `app.config`, cuyo default era `./snapshots` sin perfil. Mismos
  nombres (son content-addressed), directorio distinto, y los servidos siguen
  siendo los del 20 de mayo intactos. Comprobado por mtime y leyendo el `seed` de
  ambos juegos de pickles.
- Y por lo mismo, el `BUILT` no fue por un `load_state()` fallido sino por un
  MISS liso: en la raíz no había ningún pkl. La causa raíz está arreglada — ver
  «Layout de snapshots» abajo — pero el arreglo **arma** el peligro que la
  inconsistencia desarmaba por accidente: ahora una herramienta offline resuelve
  al directorio vivo. De ahí la guarda `--snapshots-root` en el barrido.
- El sbatch necesita `python -u` o no se ve progreso (ya corregido).

Alcance: probado **dentro de un despliegue fijo** (job 7222; los replays salieron
de ese mismo job). El reparto por réplicas sí varió entre pasadas (`--workers 4`
sobre 4 réplicas con `least_conn`) y aun así salió idéntico — coherente con que
el seed viaja dentro del pickle que comparten todas las réplicas.

### Coste de iterar el prompt — ya resuelto

`app/` va horneado en el `.sif`, así que cambiar el prompt costaba rebuild Docker en
el portátil + `scp` del tar + `apptainer build`. Añadido `APP_DIR` a
`launch_pool.sh`: monta el árbol de trabajo sobre `/app/app` y el ciclo pasa a ser
editar → regenerar snapshot → relanzar.

    APP_DIR=./app tools/hpc/launch_pool.sh     # opt-in, nunca por defecto

Es deliberadamente opt-in: lo que se entrega debe seguir siendo la imagen
autocontenida. **Los snapshots no hay que invalidarlos a mano**: la clave de caché
hashea el system prompt (`app/snapshot_cache.py:42`), así que un prompt distinto es
otra clave y las variantes conviven sin pisarse. Coste: ~90 s × procedimiento, y
**460–710 MB por (prompt × procedimiento)** — ese es el límite real del paralelismo,
no los cores.

## PASO 2 — Iterar el prompt. EMPEZAR AQUÍ EN LA PRÓXIMA SESIÓN

Antes no había plan para esto: el paso 2 eran tres líneas y una de sus premisas
la tumbó el usuario. Esto es el plan.

### 2.0 El método — fijado con el usuario 2026-07-22. NO re-litigar

La versión anterior ponía aquí una puerta bloqueante: reconstruir el snapshot de
V13, replay de las 134, exigir 134/134 byte a byte. **Descartada, y era además
insatisfacible** — con los seeds reales de la tabla de arriba, un rebuild uno por
proceso nunca reproduce hemorroides ni cirugía. Pero el fallo de fondo era otro:
estaba resolviendo el problema equivocado. El defecto no es que el seed se mueva,
es planificar comparaciones de **una sola tirada**.

**Decisión: el seed no se fija ni se ignora — es el eje sobre el que se promedia.**

- **No se tunea.** Los seeds son intercambiables en esperanza; elegir el mejor de
  9 es seleccionar ruido que regresa a la media en la pregunta 135.
- **Tampoco se ignora.** A t=0,1 el **30 % de las 54 preguntas discutidas voltea
  de decisión solo por el seed** (hemorroides 42 %). Una tirada por variante no
  puede separar el efecto del prompt de eso. No tunear una variable y no
  controlarla son movimientos opuestos.
- **Multi-seed por defecto en todas las preguntas**, no solo en las ruidosas. La
  estabilidad frente al seed es criterio de aceptación, no solo un problema de
  potencia: una pregunta que voltea es defectuosa aunque su respuesta media sea
  aceptable, y con una tirada no verías que V14 arregla 108 pero desestabiliza
  otra cosa. El scorer emite acierto medio ± dispersión **y** columna de
  estabilidad.
- **No se descarta por coste**: ~2 s/pregunta × 134 × 9 seeds × 5 variantes ≈
  3,3 h en serie, ~30 min repartido en los 8 procesos del nodo.
- **Consecuencia que simplifica mucho**: como los seeds se inyectan explícitos
  (`create_chat_completion(seed=…)`; `llama.py:1329` los respeta y salta la
  cadena del pickle), **el seed almacenado deja de importar y el harness no
  necesita persistir snapshots**. Calienta una vez en proceso y reutiliza para
  los 9 seeds. Ahorra ~8 GB de pickles por variante y elimina de raíz el riesgo
  de pisar los del pool.
- **No se toca `app/routes/query.py`.** Se consideró clavar un seed explícito por
  procedimiento en producción y se rechazó: arregla el A/B por el lado
  equivocado. Queda abierta, aparte, la pregunta de si producción debería ir a
  t=0 (ver §Consecuencias del barrido de seeds).

Fijar igual en todas las variantes: `N_THREADS`, build (`.venv-native`/imagen), y
el `--snapshots-root`, que debe ser **de scratch**.

**El A/B NO usa el pool.** Hay dos formas de correr el modelo y conviene tenerlas
separadas en la cabeza:

| | vía pool / API | en proceso |
| --- | --- | --- |
| qué es | nginx + N réplicas en contenedor; `audit_replay.py` habla HTTP | carga el modelo en un proceso sobre `.venv-native`, como `audit_seed_sweep.py` |
| el seed | **no se puede inyectar**: sale del pickle y la API no tiene parámetro | `create_chat_completion(seed=…)` |
| otro prompt | hay que relanzar el pool | es una variable del proceso |

El método necesita inyectar 9 seeds → **en proceso, obligatorio**. Y además son
5 variantes × 3 procedimientos = **15 prefijos**, cada uno con su warm: en
proceso cada worker se queda con su par y calienta una vez; por pool habría que
relanzarlo quince veces.

El conflicto con el pool es **de nodo, no de método**: `pool_dual.sbatch` pide
`--exclusive`, o sea los 64 cores, y el barrido quiere esos mismos 64. La salida
**no** es quitar el `--exclusive` (se midió, ver «Reserva del nodo»): es
`scancel` del pool mientras se barre, o —mejor— repartir los NUMA dentro de un
solo job exclusivo y darle 1-2 al servicio. El pool sigue haciendo falta **al
final**: cuando haya un prompt elegido, el replay contra el despliegue real es lo
que permite decir «esto es lo que responde el sistema entregado», no «lo que
responde mi harness».

Encuadre para la respuesta a los auditores: hoy producción es determinista byte a
byte **por accidente, no por robustez**. El valor es arbitrario y se mueve si
cambia el orden de warmup, la versión de llama-cpp o los hilos. Para el cliente
eso es peor que aleatorio: es un billete de lotería ya jugado que nadie sabe que
lo es.

### 2.0-bis El vehículo — HECHO 2026-07-22

`tools/hpc/numa_fanout.sh` (nuevo) despacha celdas, **una por nodo NUMA**, con
`numactl --cpunodebind=N --membind=N`. Si hay más celdas que slots las corre en
olas, así que las 15 del A/B caen en 2 tandas sin tocar nada. Cada celda tiene
su log y su código de salida.

`seed_sweep.sbatch` reescrito encima: los 3 procedimientos **en paralelo,
pineados, `N_THREADS=8`**, y con el `spack load numactl` que faltaba. Knobs por
entorno: `SUBSET`, `SEEDS`, `TEMPERATURES`, `OUT_DIR`, `SWEEP_SNAPSHOTS_ROOT`,
`NUMA_START`, `NUMA_COUNT`, `N_THREADS`.

`fanout_init` **se niega a arrancar** si el job no posee cores suficientes en
algún NUMA del rango (compara `Cpus_allowed_list` con el `cpulist` de cada
nodo). Eso convierte en fallo ruidoso el modo que este documento marca como «el
peligroso»: pinear a un NUMA del que posees *algunos* cores y sobresuscribir en
silencio.

Dos trampas que costaron un job cada una, y que valen para el sbatch del A/B:

- **No llamar `SNAPSHOTS_ROOT` al knob**: ese nombre es el setting de
  `app.config`, `sbatch` exporta el entorno de envío, y entonces el guard del
  tool ve que le apuntan a «la raíz de servicio» y aborta. Por eso se llama
  `SWEEP_SNAPSHOTS_ROOT`.
- **`OUT_DIR` en home, no en `/tmp`**: `/tmp` es local al nodo de cómputo y
  desde el login no se ve. Recuperarlo obliga a un `srun -w <nodo> cp`.

Coste real medido, que **corrige la estimación de §2.0**: ~5,7 s por pregunta a
nT=8, no ~2 s. El A/B son 15 celdas; la más larga es diabetes (55 preguntas × 9
seeds ≈ 47 min) y el resto va en paralelo, así que **~1,5-2 h de pared en dos
olas**, no 30 min. Cabe en el walltime de 4 h, pero no es gratis.

Son 15 celdas (5 variantes × 3 procedimientos) en 2 tandas de 8. Regla:
`Σ(procesos × hilos) ≤ 64` y ningún NUMA con dos inquilinos.

### 2.1 `tools/audit_score.py` — HECHO 2026-07-22

    uv run python tools/audit_score.py --self-check          # antes de fiarte
    uv run python tools/audit_score.py \
        --run v13=eval/audit_ab/v13 --run v14a=eval/audit_ab/v14a --baseline v13

Come un directorio de `<procedimiento>.json` por variante, y traga **las dos
formas** que ya producimos: la del barrido de seeds (muchas condiciones) y la
del replay (una sola). Emite los cinco apartados de abajo.

**`--self-check` es lo que impide que mienta en silencio**: puntúa el replay
servido y afirma que salen los números que este documento ya discute — 106/134,
los 7 probes viendo su rotura, y **ningún probe disparando en su par de
control**. Ese último pilló un fallo real: el probe del 67 («promete la
premedicación») disparaba también en el 86, que la promete *conservando* la
condición; sin esa aserción, un arreglo del 67 se habría contado como
«no cambió nada». Los probes llevan por eso un tercer campo de excepción.

Comprobado de paso al construirlo: los guardarraíles reproducen exactamente los
rechazos de la medición de estabilidad (44 % / 40 % / 52 % = 24/55, 19/48,
16/31).

Lo que emite:

1. **Acierto de decisión sobre las 134** — la métrica principal. La verdad de
   terreno es reutilizable y hay que codificarla:

       DEBEN RECHAZARSE = los 35 FN + {8, 52, 111, 117}  = 39
       DEBEN RESPONDERSE = las otras                      = 95

   (esos 4 son DEF que respondieron con material inventado donde el fulldoc no
   sostiene respuesta; por eso cuentan como rechazables). Línea base: **106/134
   = 79 %**.
   Con multi-seed la cifra es **media ± dispersión sobre los 9 seeds**, no un
   número.
2. **Diff por pregunta contra la línea base**, con **ganancias y regresiones por
   separado** y **emparejado por (pregunta, seed)** — comparar la media de V14
   contra la media de V13 sin emparejar tira la mitad de la potencia.
   Innegociable: una variante que gana 6 y pierde 5 no es progreso, y el agregado
   lo escondería — cirugía ya enseñó que el recuento de rechazos puede quedarse
   clavado mientras la composición cambia.
3. **Probes de frontera** sobre 105, 108, 67, 87, 84, 29, 26 (las regex de la
   sesión del 21 no estaban en el repo; reescritas y verificadas contra el
   texto real de cada respuesta). Son el objetivo prioritario según §4d.
4. **Guardarraíles**: tasa de rechazo y de respuesta telegráfica por
   procedimiento, para detectar que una variante «mejora» volviéndose charlatana.
5. **Columna de estabilidad**: por pregunta, en cuántos de los 9 seeds sale la
   misma decisión. Es objetivo, no diagnóstico — una variante que sube el acierto
   medio 2 puntos pero baja la estabilidad no vale. Línea base contra la que
   compararla: 30 % de volteo global (diabetes 33, cirugía 13, hemorroides 42).

Lo que **no** se puede automatizar es si una respuesta nueva es *buena*. Pero
solo hay que leer **el diff**, que es pequeño.

### 2.2 Las variantes — hipótesis con evidencia, un cambio cada una

Un solo cambio por variante, o el efecto no es atribuible. Luego se combinan las
que ganen.

| | Cambio | A qué defecto apunta | Evidencia |
| --- | --- | --- | --- |
| **V14a** | Tercera salida: «derivar» — si el tema es suyo pero el documento no lo resuelve, decir que depende de su caso y que lo consulte con su equipo | los 24 sobre-rechazos | su propia rúbrica puntúa «es podria contestar **o derivar**»; no inventa nada ni usa una palabra fuera del fulldoc |
| **V14b** | El **registro** de la pregunta no cambia si el documento la cubre | pares 112/128, 91/103, 30/6 | 70 % de rechazo en las 20 emocionales frente al 39 % del resto; los pares sobreviven al barrido de seeds |
| **V14c** | Preservar el **sujeto**: si el documento asigna una acción al equipo clínico, no convertirla en instrucción al paciente | los tres G1 (4, 108, 87) | §4d: los tres G1 comparten exactamente esta forma |
| **V14d** | Preservar **condiciones y alternativas** literales: no cerrar un «A o B», no soltar la condición que acota una regla | 67 (des-scope), 105 (disyunción) | ambos 9/9 estables en el barrido → medibles limpiamente |

Nota: el apartado del barrido de fronteras decía que no se puede escribir una
regla que diga «no sueldes dos reglas». Cierto en general, **pero 67 y 105 sí son
escribibles** — condición y disyunción son formas concretas. V14d prueba eso,
no el caso general.

**Criba operativa (usuario, 2026-07-22).** Cada variante se valida sobre las 134,
eso no se negocia. Pero antes se mira **solo su propio objetivo**, y si no mueve
nada ahí, se descarta sin pagar el resto. El barrido de seeds dice qué preguntas
sirven de criba: **las estables**. Una que voltea al 30 % puede «arreglarse»
sola y no discrimina.

| variante | criba | ¿vale con una tirada? |
| --- | --- | --- |
| V14c | 4, 108, 87 | sí — 9/9 |
| V14d | 67, 105 | sí — 9/9 |
| V14b | pares 30/6, 91/103, 112/128 | sí — sobreviven al barrido |
| V14a | los 24 sobre-rechazos | **no** — la criba tiene que ser multi-seed |

V14a es la excepción justamente porque apunta a la zona ruidosa: cribarla con una
tirada la mataría por azar.

Coste: el harness genera en proceso, así que **no hay que construir ni almacenar
snapshots por variante** (§2.0) — se caen los ~6 GB y los 90 s × build que decía
esta nota. Queda el cómputo: ~30 min repartido en los 8 procesos del nodo para
las 5 variantes × 134 × 9 seeds.

### 2.3 Consolidación — cómo se pasa de 4 variantes a 1 prompt

Faltaba en el plan. **Las letras son la fase de aislamiento, los números la de
consolidación.** No es un abanico por ronda: no habrá V15a-d, V16a-d…

- **V14a–d** — cuatro cambios de uno en uno. **No son candidatos a entregable**,
  son un experimento de atribución: sirven para saber qué hace cada cambio por
  separado. Ninguno se entrega tal cual.
- **V15** — **un solo prompt** con los que hayan ganado. Pueden ser los cuatro o
  pueden ser dos.
- **V16** — solo si V15 sale peor que la suma de sus partes; y entonces es una
  reescritura de la combinación, no un abanico nuevo.

Lo que hace que V15 no sea gratis: **los efectos no suman**. Dos cambios que
ayudan por separado pueden estorbarse, y aquí hay un choque previsible concreto —
**V14a** dice «si el tema es suyo pero el documento no lo resuelve, deriva» y
**V14d** dice «preserva la condición literal que acota la regla»; en una pregunta
acotada a medias, uno empuja a derivar y el otro a ceñirse. Por eso **V15 se mide
entera** sobre las 134 × 9 seeds, no sumando los deltas de a–d.

**Criterio de parada** (tampoco estaba escrito): se para cuando el acierto de
decisión sube desde el 79 % **sin regresión en los tres G1 (4, 29, 108) y sin
empeorar la estabilidad**. No «cuando ya no mejore» — eso no termina.

### 2.4 Riesgo a vigilar: dilución, no latencia

El prompt V13 son **915 tokens**. ⚠️ **CORREGIDO 2026-07-22:** este apartado decía
que «cada añadido cuesta prefill». **Es falso, y engaña en la dirección cómoda.**
El snapshot cachea `(system + fulldoc)` — `snapshot_builder.py:112` calienta con
el system, el fulldoc y un `PREGUNTA: hola` de relleno — y en cada petición
`load_state()` restaura eso y solo se prefillean los tokens de la pregunta real.
Un system prompt más largo **no cuesta latencia por consulta**: cuesta snapshot
más gordo y un warm más lento, una sola vez.

Lo cual quita la excusa fácil («no lo alargues, va lento») y deja el riesgo de
verdad, que es peor porque no se ve: **dilución**. 915 tokens con cuatro
instrucciones nuevas compitiendo por atención en un 3B. No se mide con un
cronómetro sino con el scorer, y por eso la métrica es el acierto sobre las 134 y
no solo las preguntas objetivo: **mirar siempre el total y las regresiones**.

## Reserva del nodo — RESUELTO 2026-07-22: exclusiva + partición autogestionada

Planteado por el usuario: `--exclusive` viene de la época de las pruebas de
velocidad y ya no hace falta; si hacen falta dos instancias, se piden dos mitades
de nodo. Se midió, y **la medición mandó a la propuesta al revés**: sin
exclusividad Slurm entrega cores fragmentados salvo que el nodo esté vacío. La
salida buena la dio el propio usuario a continuación — **quedarse la exclusiva y
repartir los NUMA nosotros**. Ver «DECISIÓN» abajo; la sección intermedia es la
evidencia que lo justifica y conviene no re-derivarla.

**Pero `--exclusive` sostiene una segunda cosa que sigue viva.** El pinning usa
**IDs absolutos de nodo NUMA**: `launch_pool.sh:188` hace `node=$((NUMA_START+i))`
y `pool_dual.sbatch` fija `NUMA_START=0` (glucowise) y `NUMA_START=4` (aiciblock).
Eso solo es cierto si el job posee los 8 nodos NUMA.

### Medido en el cluster 2026-07-22, no supuesto

Slurm: `SelectType=select/cons_tres`, `CR_CORE_MEMORY`,
`TaskPlugin=task/cgroup,task/affinity` → **confina por cgroup y soporta compartir
nodo**. Partición `cpu` = 4 nodos, 2 sockets × 32 cores, 8 NUMA de 8 (sub-socket).
Tres sondas con `~/numaprobe.sh` en el cluster:

| petición | nodo | `Cpus_allowed_list` | NUMA completos |
| --- | --- | --- | --- |
| 16 cores, no excl. | computo03 (`mixed`) | `36-39,44-55` | solo **[6]** |
| 32 cores `--sockets-per-node=1` | computo03 (`mixed`) | `0-9,30-31,36-39,44-59` | **[0,6]**, a caballo de los 2 sockets |
| 32 cores, no excl. | computo01 (**vacío**) | `0-31` | **[0,1,2,3]** — socket limpio |

**Conclusión: la propuesta funciona, pero solo si el nodo está vacío**, y eso
Slurm no lo garantiza. `--sockets-per-node=1` **no** fuerza alineación: en un nodo
fragmentado devolvió 32 cores repartidos por 7 de los 8 NUMA.

⚠️ **CORREGIDO respecto a la versión anterior de este apartado:** no «rompe la
localidad en silencio». Son **dos modos** y solo uno es silencioso:

- Pinear a un NUMA del que **no posees ningún core** → **falla ruidoso**:
  `numa_sched_setaffinity_v2_int() failed: Invalid argument`. Verificado.
- Pinear a un NUMA del que posees **algunos** cores → **funciona**, y la réplica
  arranca con `N_THREADS=8` sobre los 4 cores reales que hay. **Sobresuscripción
  dentro de la réplica, sin aviso.** Éste es el peligroso.

`Mems_allowed_list` sale `0-7` siempre — la memoria **no** se confina. Un
`--membind` ajeno «funcionaría» trayendo memoria remota; nos salva emparejar
siempre `--cpunodebind=N --membind=N` con N propio.

### DECISIÓN: `--exclusive` se queda, y gestionamos la partición nosotros

Propuesta del usuario 2026-07-22, y es la buena. En vez de pelearse con el
repartidor de Slurm, **se pide el nodo entero y se reparten los 8 nodos NUMA a
mano**. Con eso:

1. **El invariante del pinning se mantiene** — los IDs absolutos 0-7 son
   nuestros. **Se cae todo el trabajo de derivación** descrito arriba: no hace
   falta. `launch_pool.sh` ya está parametrizado (`NUMA_START`, `N_REPLICAS`),
   así que un pool de 2 réplicas en NUMA 6-7 es `NUMA_START=6 N_REPLICAS=2`.
   **Cero cambios de código en el lanzador.**
2. **Sin vecino ruidoso** → el ancho de banda de memoria es nuestro y los tok/s
   son predecibles.
3. **Y lo importante: mata la salvedad de método del barrido.** Hoy
   `seed_sweep.sbatch` pide `--exclusive` y **desperdicia el nodo**: corre los
   tres procedimientos **en serie**, a `N_THREADS=32`, **sin pinear**. Ese nT=32
   es justo lo que hacía que solo 29 de 54 respuestas coincidieran con el replay.
   Con **8 workers pineados a un NUMA cada uno y `N_THREADS=8`** se mide en **la
   misma configuración que sirve producción**, y de paso en paralelo.

El argumento de contención (había 4 jobs de otro usuario encolados a 16 CPUs
mientras el 7248 retenía 64) **no desaparece pero cambia de sentido**: era un
argumento contra retener el nodo *ocioso*. Si de verdad usamos los 64 cores, la
reserva es legítima. Lo que no se debe hacer es dejar el pool ocho horas
sujetando un nodo que no usa.

Regla del nodo, sin cambios: **`Σ(procesos × hilos) ≤ 64` y ningún NUMA con dos
inquilinos.** Reparto para el A/B: 15 celdas (5 variantes × 3 procedimientos) en
2 tandas de 8. Si se quiere el servicio vivo a la vez, se le dan 1-2 NUMA y el
barrido usa el resto.

**Lo que falta implementar** (solo esto): reescribir `seed_sweep.sbatch` —y el
futuro sbatch del A/B— para **abanicar 8 workers pineados con `numactl
--cpunodebind=N --membind=N` a `N_THREADS=8`** en vez de tres pasadas en serie a
nT=32. `pool_dual.sbatch` y `pool.sbatch` **se quedan como están**.

**Precisión sobre «dos mitades».** `pool_dual.sbatch` ya mete los dos perfiles en
un nodo dentro de **un solo job** — la densidad ya la tienes. Partirlo en dos jobs
compra **ciclos de vida independientes** (cancelar glucowise sin tumbar
aiciblock), y no garantiza que caigan en el mismo nodo. Con la partición
autogestionada eso se resuelve dentro del job.

La sonda queda en `~/numaprobe.sh` del cluster por si hace falta re-verificar.

## Layout de snapshots — ARREGLADO 2026-07-22

Causa raíz de la falsa «colisión». El scoping por perfil vivía **solo en los
lanzadores**: `launch_pool.sh` bindeaba `./snapshots/$PROFILE` sobre
`/app/snapshots`, mientras `app/config.py` tenía `snapshots_dir = ./snapshots`.
Dentro del contenedor cuadraba; desde el host, cualquier cosa que importara
`app.config` caía en la raíz — un directorio que no sirve nadie — y construía en
silencio un juego paralelo de pickles.

Arreglado así: el campo pasa a ser `snapshots_root` y **`snapshots_dir` es ahora
una propiedad derivada**, `<root>/<perfil>`, igual que `fulldoc_procedures`. Los
lanzadores bindean la **raíz** y el contenedor añade el perfil. Tocados:
`app/config.py`, `docker-compose.yml`, `tools/hpc/{launch_pool.sh, smoke.sh,
pool.sbatch, build_native_image.sh}`. Los cinco consumidores
(`main.py`, `snapshot_builder.py`) no cambian: siguen leyendo
`settings.snapshots_dir`, que ahora significa lo correcto.

Verificado dentro del SIF con los dos perfiles: resuelven a `snapshots/glucowise`
y `snapshots/aiciblock` y encuentran los mismos hashes que se servían.
**Migración no-op**: no hay que mover ni regenerar nada.

**Efecto secundario que hay que tener presente**: el arreglo *arma* el peligro
que la inconsistencia desarmaba. Una herramienta offline resuelve ahora al
directorio vivo del pool, y `query.py` re-lee el pkl en cada petición. Por eso
`audit_seed_sweep.py` lleva `--snapshots-root` (default
`/tmp/cpu-rag-offline-snapshots`) y **se niega a arrancar** si le apuntas a la
raíz de servicio. Copiar esa guarda en el scorer y en cualquier tool nueva.

Limpieza hecha: borrados los 3 `.pkl` + 3 `.meta.json` huérfanos (1,7 GB) que el
barrido dejó en la raíz del cluster.

## Herramientas (en `tools/`, ya commiteadas)

    uv run --with openpyxl python tools/audit_extract.py      # portátil -> JSON
    ./.venv-native/bin/python tools/audit_replay.py \         # en el cluster
        --procedure diabetes --api-url http://<nodo>:8080 \
        --out eval/audit_replay/diabetes.json --workers 4
    uv run python tools/audit_triage.py                       # -> reports/audit_triage.md
    ./.venv-native/bin/python tools/audit_stability.py \      # en el cluster
        --procedure diabetes --api-url http://<nodo>:8080 --runs 10
    PROFILE=glucowise ./.venv-native/bin/python \             # en un nodo ocioso
        tools/audit_seed_sweep.py --procedure diabetes --seeds 8
        # --snapshots-root default /tmp/cpu-rag-offline-snapshots; NUNCA ./snapshots

`audit_triage.py` guarda **los 134 veredictos en código** con la línea del corpus que
los justifica, más `FALSE_POSITIVE`, `NEIGHBOURING`, `RULE_BOUNDARY` y `CORPUS_SIZE`.
Dos aserciones: una falla si un veredicto de rechazo no cuadra con un rechazo real,
otra si un hallazgo de frontera cae en una pregunta no respondida o marcada OK. Es la
fuente de verdad del triaje: para cambiar una clasificación, se edita ahí y se
regenera.

Coste del replay: 55 preguntas en 72 s; las 134 en menos de 2 min.

## Trampas operativas

- **Los replay viven en `eval/audit_replay/<procedimiento>.json` y SÍ están
  versionados** (movidos ahí 2026-07-21). No son regenerables —`temperature=0.1` sin
  semilla y el pool que los produjo ya no existe—, así que no vuelvan a `reports/`.
- **`reports/` sigue en `.gitignore`**: `audit_triage.md` se regenera y
  `audit_questions.json` no. Este último sale del xlsx, que por decisión del usuario
  **no se commitea**: sin ese xlsx en local no se puede relanzar un replay, aunque el
  triaje sigue funcionando porque los replay ya llevan dentro sus preguntas y notas.
- **Toda herramienta offline que pueda construir un snapshot debe llevar
  `--snapshots-root` de scratch.** Desde el arreglo de layout, el default resuelve
  al directorio que sirve el pool. Ver «Layout de snapshots».
- Los clientes corren bajo `./.venv-native/bin/python` en el cluster; el `python3`
  del sistema no tiene `httpx`.
- El LB **no es alcanzable desde el portátil**. Todo por `ssh hpc`.
- `RAG_API_KEY` sale de `~/Projects/cpu-rag/.env` en el cluster.
- El repo local **no es** el del cluster; los cambios viajan por `rsync`/`scp`.
  **La deriva es en los dos sentidos: nada de `rsync` a lo bruto.** Comprobado
  2026-07-22 con `rsync -rcn`: `app/` solo difería en lo nuestro, pero en
  `tools/` divergían además `audit_extract.py`, `audit_triage.py`, `demo_rag.py`,
  `run_eval.py` y `hpc/seed_sweep.sbatch`, con cambios hechos **en el cluster**
  (ahí vive el `python -u` del sbatch). Diferenciar fichero a fichero y copiar
  solo lo tocado.
- **`APP_DIR` no existía en el `launch_pool.sh` del cluster** hasta 2026-07-22,
  pese a que este documento lo daba por disponible desde el 21: se añadió en el
  repo local y nunca se copió. Ya está. Moraleja: «añadido a X» sin copiar no
  vale de nada aquí.
- `docs/evaluation_framework.md` tiene WIP sin commitear del usuario — no pisarlo.

## Servicio

⚠️ **No hay pool vivo** (2026-07-22 ~20:30): el job 7248 agotó su walltime. Hacía
falta solo para el replay final contra el despliegue real; el A/B corre en
proceso y no lo necesita. Relanzar con las instrucciones de abajo cuando toque
anclar un prompt elegido contra producción. El 7222 del 2026-07-21, en
`computo01`, es el que produjo los replays versionados.

    curl -s http://eurehpccomputo04:8080/health   # glucowise -> diabetes
    curl -s http://eurehpccomputo04:8081/health   # aiciblock -> hemorroides, cirugia

**El layout nuevo está verificado contra la línea base: 134/134 byte-idénticos**
(replay completo contra este pool, diff contra `eval/audit_replay/`). Distinto
nodo, `config.py` refactorizado y `app/` montado por bind, y no se mueve un byte.
El refactor es inerte y el determinismo del seed congelado aguanta el cambio de
nodo. Nota de paso: eso es justo lo que exigía la vieja puerta §2.0, y pasa —
porque **no** se reconstruyeron los snapshots. Era la receta de «reconstruir uno
por proceso» lo que la hacía insatisfacible, no el criterio.

**Ojo al relanzar sin `APP_DIR`:** el `.sif` **hornea `app/`**, así que lleva el
`config.py` viejo mientras `launch_pool.sh` ya bindea la raíz. El contenedor
miraría en `/app/snapshots` (= la raíz, vacía) y `main.py` abortaría con «No
snapshots found» — falla ruidoso, no silencioso. Dos salidas: rehacer la imagen,
o `APP_DIR=./app`, que es además lo que hace falta para iterar el prompt.

Para relanzar:

    ssh hpc && cd ~/Projects/cpu-rag
    sbatch --export=ALL,APP_DIR=./app tools/hpc/pool_dual.sbatch   # ver el aviso de arriba
    squeue -u $USER -o "%.8i %.20j %.10T %.20N"   # el nodo puede cambiar; NO lo asumas
    curl -s http://<nodo>:8080/health              # glucowise  -> diabetes
    curl -s http://<nodo>:8081/health              # aiciblock  -> hemorroides, cirugia

`--workers 4` por pool (4 réplicas × 8 hilos por perfil). Los snapshots están y son
válidos para los 3 procedimientos: no hay que regenerar nada nunca.

**Los replay base están versionados en `eval/audit_replay/*.json`** — no hace
falta relanzar el servicio salvo que se quiera volver a medir. (En el cluster hay
además copias en `reports/`, que es gitignored; la fuente de verdad es `eval/`.)
