# Estado del proyecto

Qué sirve este repo hoy, con qué números y bajo qué restricciones. Solo estado
actual: las decisiones que ya no condicionan nada no están aquí, y los números
que viven en un informe se apuntan, no se repiten.

## El sistema

Un asistente que responde preguntas de pacientes **desde un único documento por
procedimiento**, sin recuperación: el documento entero entra en el contexto y su
estado KV se reutiliza entre peticiones. Tiene **prohibido usar conocimiento
propio** y se abstiene cuando el documento no cubre la pregunta. Esa abstención
es una instrucción explícita del prompt, no una laguna, y condiciona cómo se
evalúa.

Hay **un solo código y dos perfiles**, elegidos por la variable `PROFILE`:
`glucowise` (diabetes) y `aiciblock` (hemorroides y cirugía abdominal). Cada
perfil sirve sus procedimientos en su puerto.

Este repo es **solo CPU y se cierra en la v2.2**. La versión de GPU es otra
arquitectura y vive en `~/Projects/gpu-rag/`. Cuidado con el nombre:
`~/Projects/gpu-rag-deprecated/` es la app de recuperación vieja, sin relación con
ninguna de las dos.

## Lo que se sirve

| | v2.2 (la que se entrega) | v1.1 (la desplegada en cliente) |
| --- | --- | --- |
| modelo | gemma-4-26B-A4B Q4_K_M | Ministral-3-3B Q4_K_M |
| corpus | los tres documentos tuteados | los tres originales |
| trato | tú | usted |
| prompt | V13 + literal de abstención `d1c-tu` | V13 |
| correctas de 134 | **119 (89 %)** | 83 (62 %) |

Las dos cifras salen de leer las 134 preguntas una a una contra el documento que
cada versión sirve: `docs/auditoria_134_v22.md` y `docs/auditoria_134_evaluacion.md`.
Existe una lectura intermedia de la v2 en `docs/auditoria_134_v2.md` (115, 86 %);
la v2 y la v2.1 quedaron absorbidas por la v2.2 y no se enviaron.

Cambiar de versión servida es cambiar las rutas de `PROFILES` y `prompt_variant`
en `app/config.py`. Los documentos de cada entrega están en `corpus/archive/`,
porque `corpus/markdown/` está en `.gitignore`.

## Cómo se mide la calidad

**La corrección se lee, no se calcula.** Correcta = responde lo que se pregunta
apoyada en el documento, sin inventar y sin fundir ni des-acotar una regla; o se
abstiene donde el documento no da material.

No hay puntuador automático, y es deliberado: el que hubo dio 60 % sobre una v2
que leída daba 86 %, marcó como rotas dos preguntas que eran artefacto de sus
propias regex, y puntuó una serie a 70,9 % porque el literal de abstención nuevo
no casaba con su predicado y contaba 44 abstenciones como respuestas. Tampoco hay
herramientas de auditoría en `tools/`: pasar las 134 contra un endpoint son unas
treinta líneas y se escribe en el momento.

Tres reglas que no conviene romper:

- Un número que salga del equipo viene de leer **las 134**. Leer una parte solo
  vale dentro de un experimento acotado donde se sigue un delta.
- Nunca compartir un veredicto entre dos ejecuciones porque las respuestas se
  parezcan. Un umbral de similitud decidiendo un veredicto es el mismo error con
  otro disfraz.
- Lo único mecánico y fiable es si la **decisión** cambia (responde o se abstiene)
  entre semillas o entre variantes. Eso no es corrección.

Los tres informes de `docs/auditoria_134_*.md` se editan a mano y no se regeneran
desde ningún sitio. Las 134 preguntas, con la respuesta que registró el cliente y
su puntuación, están en `reports/audit_questions.json`; las respuestas de cada
ejecución, en `eval/<brazo>/<procedimiento>.json`.

## Modelo

**gemma-4-26B-A4B**, Q4_K_M. Decide 91 % frente al 78 % de Ministral-3-3B con el
mismo prompt, arregla los defectos de frontera y voltea ~1 % por semilla. Cuesta
7,3 tok/s y un modelo de 17 GB.

**gemma recupera frases, no sintetiza.** Responde bien si y solo si una frase del
corpus ya afirma lo que se pregunta. Por eso el corpus, y no el prompt, es el
frente principal.

No sirve cualquier arquitectura: con modelos de estado (Mamba, p. ej. Qwen3.5) no
hay reutilización de prefijo, que es justo lo que hace viable mandar un documento
entero en CPU.

El modelo no se guarda en el repo: `tools/fetch_model.sh` lo descarga de la
redistribución pública de Unsloth en Hugging Face. El único GGUF local es el de
Ministral, que sirve la v1.1.

## Corpus

Es el frente que más mueve el resultado. Las reglas de redacción están en
`docs/corpus_guidelines.md`. Lo que hay que saber antes de tocar nada:

- **Autosuficiencia**: el contenido accionable va en la oración principal. **Leer
  antes la sección de alcance de las guías**: las reglas arreglan documentos
  telegráficos y estropean los que ya son prosa.
- **El trato lo fija el corpus, no el prompt.** Con los documentos en usted salían
  0 respuestas en tú de 89, por mucho que los ejemplos del prompt tutearan. Por
  eso el literal de abstención tiene que girar con el corpus.
- El tuteo **refuerza** la autosuficiencia: «toma» y «debe» son también tercera
  persona y obligan a poner «usted» explícito, mientras que «tomas» solo puede ser
  el paciente, así que el pronombre se cae sin perder claridad.
- **Convertir el trato se hace a mano, leyendo.** Un script metió errores que solo
  se ven leyendo, incluido un cambio de sentido. Para ~30 K de texto con juicio en
  cada frase, leer es más rápido y más seguro.
- La reescritura solo de forma funciona: en hemorroides, con los mismos hechos y
  sin invención, arregló la frontera de los anticoagulantes y bajó las
  telegráficas del 56 % al 23 %.
- Destilar más no siempre gana. En diabetes sí (decisión 85,5 % contra 83,4 %);
  en cirugía la destilación salió **peor** (88,7 % contra 93,8 %, telegráficas por
  cinco) y no entró.

Los documentos de hemorroides y cirugía son **material de desarrollo, no de
producción**. El de hemorroides llegó como un resumen de unas 160 palabras y no
existe documento fuente. Solo el de diabetes está confirmado por el cliente. Esto
acota lo que significan las cifras de arriba.

## Prompt

Vive en `app/prompt.py`. Detalle en `docs/prompt_versions.md`.

Se sirve **V13 con el literal de abstención `d1c-tu`**: 915 tokens, agnóstico del
procedimiento, temperatura 0,1. El literal alcanza a las 42 abstenciones, 24 de
ellas emocionales, sin bajar el acierto y con un 18 % menos de tokens de salida
que el intento anterior.

Dos límites conocidos del prompt como palanca:

- **La sobre-abstención y la invención son un mismo umbral.** Apretar para que
  responda más hace que invente más. Ninguna variante del A/B lo esquivó.
- **Acortarlo no sale gratis.** Un prompt más corto empata en decisión y ahorra
  229 tokens, pero dispara las telegráficas de hemorroides del 23 % al 58 %.

`max_tokens` está en 320. Es suficiente salvo para una respuesta de las 134 (la
29, la lista de días de enfermedad), que se corta a media palabra. Si se toca, hay
que regenerar snapshots y volver a leer las 134.

## Prefijo KV y reproducibilidad

El documento entero se precalcula como estado KV. De dónde sale lo decide
`snapshot_mode` (`memory` por defecto, `disk` para el pool, `off` para
diagnosticar); la tabla comparativa está en el README. Arranque en frío ~92 s;
con el prefijo caliente, ~1 s.

- **En `disk` la semilla va congelada dentro del pickle.** Dos ejecuciones contra
  el mismo snapshot dan respuestas idénticas byte a byte, así que cualquier cifra
  de la auditoría es **una sola tirada**.
- Mismo servidor y misma imagen producen un pickle idéntico byte a byte. Un
  snapshot reutilizado reproduce producción; uno reconstruido en otra caja, no.
  **Todo brazo de un A/B tiene que reconstruir el suyo, incluido el de control.**
- Una variante de prompt cambia la clave del snapshot, así que cada una calienta
  su propio pickle y nunca colisionan.
- Las herramientas offline tienen que escribir en un directorio de snapshots
  aparte: pueden pisar el pickle de un pool vivo.
- El overhead del pickle (~490 MB por `LlamaState.scores`) sigue sin atacar.

## Rendimiento y despliegue

- **Clúster (dos zócalos, 8 nodos NUMA):** el reparto bueno es nT8 N8, 11,62 tok/s
  por usuario. Cruzar zócalos cuesta un 17-20 %. Mantener `--exclusive` y partir
  los nodos NUMA a mano.
- **EC2 single-socket (r7i.2xlarge, 8 vCPU, ~0,53 $/h):** con gemma lo mejor es
  **N=1 a todos los hilos**. Replicar no sube el rendimiento agregado (~4,8 tok/s,
  limitado por ancho de banda), solo cambia latencia por concurrencia.
- **Se sirve solo la imagen portable.** La build nativa VNNI/AMX no aporta en
  gemma: ~0 % en decode, −17 % en prefill, y cambia respuestas. El +15-30 % que se
  midió era con Ministral.
- Concurrencia == número de réplicas: cada réplica sirve una petición a la vez.
- Aceptable a partir de 6 tok/s en el peor caso; ~11 tok/s es bueno. Se optimiza
  el número de réplicas, no el pico de una sola.
- Actualizar una versión desplegada corta **~70 s**, no 8 minutos: los snapshots de
  dos versiones conviven, así que se genera en caliente y solo el reinicio corta.
  Procedimiento en `docs/actualizacion_version_desplegada.md`.

Trampas que cuestan tiempo y no son evidentes:

- La caché de pip reutiliza la rueda nativa entre variantes, y así la imagen
  portable dio SIGILL en CPUs sin AVX-512. Hace falta `pip install --no-cache-dir`
  y apagar explícitamente las ISA altas. Probar la portable antes de enviarla.
- En Apptainer hay que pasar `--pwd /app`, o el contenedor ejecuta el `app/` del
  anfitrión e ignora todos los binds.
- En el clúster es `module load singularity`, no `spack load apptainer`: desde
  julio de 2026 es ambiguo y mata los trabajos bajo `set -e`.
- Cambios en `app/` necesitan `up -d --build`, no un restart.
- Si el disco se llena, la generación se corta a media escritura dejando un
  `.pkl.tmp` y el servicio entra en bucle de reinicio con «No snapshots found»,
  un error que no menciona el disco.

## Lo que queda abierto

- **Material psicológico para cirugía y hemorroides.** Es la mitad pendiente del
  problema de las preguntas emocionales, y es trabajo de corpus, no de prompt: las
  emocionales que hoy se responden bien son las de diabetes, el único documento
  con sección psicológica.
- Validar la calidad de las evaluaciones bajo la build nativa: VNNI cambia el
  orden de acumulación en int8.
- Replicar el barrido de rendimiento en EC2 cuando haga falta dimensionar.

Descartado y no conviene volver a intentarlo sin material nuevo: **derivar** cuando
el documento no cubre pero se podría orientar. Convierte 14 de 45 abstenciones
pero solo 4 de las 21 emocionales, siempre con la misma plantilla, y deriva 7
preguntas que el documento sí responde.
