# Historial de desarrollo

Lo que se midió, lo que se decidió y por qué. Sustituye a `ARRANQUE.md`,
`ARRANQUE_v2.md`, `ARRANQUE_v4.md` y `V1.2_PLAN.md`, que eran briefs de sesión y
se borraron el 2026-07-28.

Solo va aquí lo que sigue siendo cierto y hace falta para sacar una versión
nueva. Los números que están en un informe no se repiten: se apunta al informe.

## Qué es el sistema

Un asistente que responde preguntas de pacientes **desde un único documento por
procedimiento**, sin recuperación: el documento entero entra en el contexto y se
cachea como snapshot. Tiene prohibido usar conocimiento propio y se abstiene
cuando el documento no cubre la pregunta. Esa abstención es una instrucción
explícita del prompt, no una laguna.

La parte de recuperación (chunking, embeddings, BM25, reranker, Qdrant) se separó
a `~/Projects/gpu-rag-deprecated/` el 2026-05-13 y no es de este repo. Cuidado con
el nombre: `~/Projects/gpu-rag/` es otra cosa desde el 2026-07-31 — ver abajo.

Este repo es **solo CPU y se cierra en la v2.2**, la versión entregada y auditada.
El port a GPU se hizo y se llevó a su propio repo, `~/Projects/gpu-rag/`, el
2026-07-31: allí la maquinaria de snapshots sobra (restaurar cuesta más que
reprefilar) y es otra arquitectura. Todo lo de GPU —el plan de migración, el
estado de la v3.0, los requisitos de hardware— vive allí, no aquí.

Desde el 2026-07-21 hay **un solo código y dos perfiles**, elegidos por la
variable `PROFILE`: `glucowise` (diabetes) y `aiciblock` (hemorroides y cirugía
abdominal). Cada perfil sirve sus procedimientos en su puerto.

## Las versiones

| | modelo | corpus | trato | leídas de 134 |
| --- | --- | --- | --- | ---: |
| v1.1 | Ministral-3-3B Q4_K_M | los tres originales | usted | 83 (62 %) |
| v2 | gemma-4-26B-A4B Q4_K_M | diabetes v4, hemorroides vA, cirugía original | usted | 115 (86 %) |
| v2.1 | igual que v2 | igual que v2 | usted | — |
| v2.2 | igual que v2 | los tres tuteados | tú | 119 (89 %) |

Las tres cifras salen de leer las 134 preguntas una a una contra el documento
que cada versión servía: `docs/auditoria_134_evaluacion.md` (v1.1),
`auditoria_134_v2.md` y `auditoria_134_v22.md`.

La v1.1 es la que audita el cliente y la que tiene desplegada. **La v2.2 es la
que se entrega.** La v2.1 solo cambió el literal de abstención y quedó absorbida
por la v2.2; no se envió.

Volver atrás entre v2, v2.1 y v2.2 es cambiar las rutas de `PROFILES` y
`prompt_variant` en `app/config.py`. Los documentos de cada versión están
archivados en `corpus/archive/`, porque `corpus/markdown/` está en `.gitignore`.

## Cómo se evalúa, y por qué así

**La corrección se lee, no se calcula.** Correcta = responde lo que se pregunta
apoyada en el documento, sin inventar y sin fundir ni des-acotar una regla; o se
abstiene donde el documento no da material.

Hubo un puntuador automático y se borró el 2026-07-28 con el resto de
`tools/audit_*.py`. Todos sus números acabaron corregidos leyendo: daba 60 % sobre
la v2 (era un suelo, no una medida), marcaba como rotas dos preguntas que eran un
artefacto de sus regex, y puntuaba la serie D1 a 70,9 % porque el literal de
abstención nuevo no casaba con su predicado y contaba 44 abstenciones como
respuestas. El medidor, no la variante.

De ahí salen dos reglas que conviene no volver a romper:

- Un número que salga del equipo viene de leer **las 134**. Leer una parte solo
  vale dentro de un experimento acotado donde se sigue un delta.
- Nunca compartir un veredicto entre dos ejecuciones porque las respuestas se
  parezcan. Un umbral de similitud decidiendo un veredicto es el mismo error con
  otro disfraz.

Lo que sí es mecánico y se puede medir solo: si la **decisión** cambia (responde
o se abstiene) entre semillas o entre variantes. Eso no es corrección.

Los tres informes se editan a mano y no se regeneran desde ningún sitio. Las 134
preguntas, con la respuesta que registró el cliente y su puntuación, están en
`reports/audit_questions.json`.

## Modelo

**gemma-4-26B-A4B sustituyó a Ministral-3-3B el 2026-07-22**, tras una criba en
el clúster. Decide 91 % frente al 78 % de Ministral con el prompt V13, arregla los
defectos de frontera y voltea ~1 % por semilla. Cuesta 7,3 tok/s y un modelo de
17 GB.

Qwen3.5 se descartó en la misma criba: es Mamba, y con arquitectura de estado no
hay reutilización de prefijo, que es justo lo que hace viable servir un documento
entero en CPU.

**gemma recupera frases, no sintetiza.** Responde bien si y solo si una frase del
corpus ya afirma lo que se pregunta. Esto es lo que hace que el corpus, y no el
prompt, sea el frente principal.

El modelo no se guarda en el repo: `fetch_model.sh` lo descarga de la
redistribución pública de Unsloth en Hugging Face. El único GGUF local es el de
Ministral, que sirve la v1.1.

## Corpus

Es el frente que más mueve el resultado, y está documentado aparte en
`docs/corpus_guidelines.md` (siete reglas). Lo esencial:

- **Autosuficiencia**: el contenido accionable tiene que ir en la oración
  principal. Leer la sección de alcance de las guías antes de aplicarlas: las
  reglas **arreglan documentos telegráficos y estropean los que ya son prosa**.
- La destilación v4 de diabetes ganó a la v1 (decisión 85,5 % contra 83,4 %,
  telegráficas 7 % contra 11 %). La v4 de cirugía se midió **peor** (88,7 % contra
  93,8 %, telegráficas por cinco) y no entró.
- La reescritura vA de hemorroides es solo de forma, con los mismos hechos y sin
  invención: arregla la frontera de los anticoagulantes y baja las telegráficas
  del 56 % al 23 %.
- **El trato lo fija el corpus, no el prompt.** Con los documentos en usted salían
  0 respuestas en tú de 89, por mucho que los ejemplos del prompt tutearan. Por
  eso el literal de abstención tiene que girar con el corpus.
- El tuteo **refuerza** la autosuficiencia: «toma» y «debe» son también tercera
  persona y obligan a poner «usted» explícito, mientras que «tomas» solo puede ser
  el paciente, así que el pronombre se cae sin perder claridad.
- La conversión a tú se hizo **a mano, leyendo**. Un primer intento con script
  metió errores que solo se ven leyendo, incluido un cambio de sentido. Para ~30 K
  de texto con juicio en cada frase, leer es más rápido y más seguro.

Los documentos de hemorroides y cirugía son material de desarrollo, no de
producción. El de hemorroides llegó como un resumen de unas 160 palabras y no
existe documento fuente. Solo el de diabetes está confirmado por el cliente.

## Prompt

Vive en `app/prompt.py`; `get_system_prompt(procedure, variant)` y el mecanismo de
variantes (`_replacing`, `_plus_example`) ya existen. El histórico de versiones
está en `docs/prompt_versions.md`.

- **V13** es el que sirvió la v1.1 y la v2: 915 tokens, agnóstico del
  procedimiento, temperatura 0,1.
- El A/B de prompts del 2026-07-22 no lo ganó ninguna variante. El hallazgo útil
  es que **la sobre-abstención y la invención son un mismo umbral**: apretar para
  que responda más hace que invente más.
- **d1c-tu** es el literal de abstención que sirve la v2.2. Alcanza a las 42
  abstenciones, 24 de ellas emocionales, sin bajar el acierto y con un 18 % menos
  de tokens de salida que el intento anterior.
- Un prompt más corto (G1) empata en decisión y ahorra 229 tokens, pero dispara
  las telegráficas de hemorroides del 23 % al 58 %. No desplegar.

`max_tokens` está en 320. Es suficiente salvo para una respuesta de las 134 (la
29, la lista de días de enfermedad), que se corta a media palabra. Si se toca,
hay que regenerar snapshots y volver a leer las 134.

## Snapshots y reproducibilidad

El documento entero se precalcula como estado KV y se guarda en un pickle bajo
`snapshots/<perfil>/`. Arranque en frío ~92 s; con snapshot, ~1 s.

- **La semilla va congelada dentro del pickle.** Por eso dos ejecuciones contra el
  mismo snapshot dan respuestas idénticas byte a byte, y cualquier cifra de la
  auditoría es una sola tirada.
- Mismo servidor y misma imagen producen un pickle **idéntico byte a byte**. Un
  snapshot reutilizado reproduce producción; uno reconstruido en otra caja, no.
  **Todo brazo de un A/B tiene que reconstruir el suyo, incluido el de control.**
- Una variante de prompt cambia la clave del snapshot, así que cada una calienta
  su propio pickle y nunca colisionan.
- Las herramientas offline tienen que escribir en un directorio de snapshots
  aparte: pueden pisar el pickle de un pool vivo.
- El overhead del pickle (~490 MB por `LlamaState.scores`) sigue sin atacar.

## Rendimiento y despliegue

- **Clúster (dos zócalos, 8 nodos NUMA):** el reparto bueno es nT8 N8, 11,62
  tok/s por usuario. Cruzar zócalos cuesta un 17-20 %. Mantener `--exclusive` y
  partir los nodos NUMA a mano.
- **EC2 single-socket (r7i.2xlarge, 8 vCPU, ~0,53 $/h):** con gemma lo mejor es
  **N=1 a todos los hilos**. Replicar no sube el rendimiento agregado (~4,8 tok/s,
  limitado por ancho de banda), solo cambia latencia por concurrencia.
- **La build nativa VNNI/AMX no aporta en gemma**: ~0 % en decode, −17 % en
  prefill, y cambia respuestas. El +15-30 % que se midió era con Ministral. Se
  sirve **solo la imagen portable**.
- La caché de pip reutiliza la rueda nativa entre variantes: la imagen portable
  dio SIGILL en CPUs sin AVX-512 hasta añadir `pip install --no-cache-dir` y
  apagar explícitamente las ISA altas. Probar la portable en el portátil antes de
  enviarla.
- Concurrencia == número de réplicas: cada réplica sirve una petición.
- Aceptable a partir de 6 tok/s en el peor caso; ~11 tok/s es bueno. Se optimiza
  el número de réplicas, no el pico de una sola.
- Actualizar una versión desplegada corta ~70 s, no 8 minutos: los snapshots de
  dos versiones conviven, así que se genera en caliente y solo el reinicio corta.
  El procedimiento está en `docs/actualizacion_version_desplegada.md`.

Trampas que costaron tiempo y no son evidentes:

- En Apptainer hay que pasar `--pwd /app`, o el contenedor ejecuta el `app/` del
  anfitrión e ignora todos los binds.
- En el clúster es `module load singularity`, no `spack load apptainer`: desde
  julio de 2026 es ambiguo y mata los trabajos bajo `set -e`.
- Cambios en `app/` necesitan `up -d --build`, no un restart.
- Con seis versiones acumuladas el disco llegó al 100 %, la generación se cortó a
  media escritura dejando un `.pkl.tmp`, y el servicio entró en bucle de reinicio
  con «No snapshots found», un error que no menciona el disco.

## Lo que queda abierto

- **Material psicológico para cirugía y hemorroides.** Es la mitad pendiente del
  problema de las preguntas emocionales, y es trabajo de corpus, no de prompt: las
  emocionales que hoy se responden bien son las de diabetes, el único documento
  con sección psicológica.
- **Derivar** cuando el documento no cubre pero se podría orientar. Se probó y no
  convence: la variante convierte 14 de 45 abstenciones pero solo 4 de las 21
  emocionales, siempre con la misma plantilla, y deriva 7 preguntas que el
  documento sí responde.
- Validar la calidad de las evaluaciones bajo la build nativa: VNNI cambia el
  orden de acumulación en int8.
- Replicar el barrido de rendimiento en EC2 cuando haga falta dimensionar.
