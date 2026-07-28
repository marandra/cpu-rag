# cpu-rag

Servicio de preguntas y respuestas para pacientes, solo CPU. Cada procedimiento
tiene **un único documento markdown** que se manda entero como contexto a un
modelo local de llama.cpp. No hay recuperación, ni embeddings, ni base vectorial.

La variante con recuperación (chunking, embeddings, Qdrant, reranker) vive en el
repo hermano [`gpu-rag`](../gpu-rag/) y no es de aquí.

El sistema tiene **prohibido usar conocimiento propio** y se abstiene cuando el
documento no cubre la pregunta. Eso es una instrucción explícita del prompt, no
una laguna, y condiciona cómo se evalúa: ver «Evaluación» más abajo.

## Cómo funciona

```
corpus/markdown/<documento>.md ──► se precalcula el estado KV (snapshot .pkl)
                                            │
                                            v
        POST /query  ──►  reutiliza el prefijo + PREGUNTA: <q>  ──►  SSE
```

El prefijo (system prompt + documento) se paga una vez y se guarda en
`snapshots/<perfil>/`. Arranque en frío ~92 s; con snapshot, ~1 s. Solo la
pregunta y la respuesta cuestan generación.

**La semilla va congelada dentro del pickle**, así que dos ejecuciones contra el
mismo snapshot dan respuestas idénticas byte a byte.

## Perfiles

Un solo código, dos despliegues, elegidos por el fichero de `profiles/`:

| Perfil | Procedimientos | Puerto |
| --- | --- | --- |
| `glucowise` | diabetes | ver `profiles/glucowise.env` |
| `aiciblock` | hemorroides, cirugia-abdominal | ver `profiles/aiciblock.env` |

Cada perfil tiene su directorio de snapshots, sus contenedores y su puerto de
balanceador, así que los dos pueden correr a la vez.

```bash
cp env.example .env          # y poner RAG_API_KEY
./run.sh glucowise generate  # construye los snapshots (trabajo de una vez)
./run.sh glucowise up -d
./run.sh aiciblock up -d
./run.sh glucowise logs -f
```

Todo lo que va detrás del perfil se pasa tal cual a `docker compose`. Un cambio
en `app/` necesita `up -d --build`, no un restart.

Para desarrollo sin contenedor: `uv sync && uv run uvicorn app.main:app --reload`.
Usar siempre `uv run`.

## Endpoints

| Método | Ruta | Auth | Descripción |
| --- | --- | :--: | --- |
| `GET` | `/health` | no | Estado, modelo, procedimientos cargados |
| `POST` | `/query` | sí | SSE (`chunk` / `done` / `error`). Cuerpo: `{question, procedure}` |

Auth por cabecera `X-API-Key`, valor en `RAG_API_KEY`.

## Configuración

Todo en `app/config.py`, sobreescribible por entorno o `.env`. Lo que importa:

| Ajuste | Notas |
| --- | --- |
| `PROFILES` | perfil → procedimiento → ruta del documento. **Cambiar de versión servida es cambiar estas rutas** |
| `prompt_variant` | variante del system prompt; cambia la clave del snapshot |
| `model_path` | GGUF |
| `n_ctx` | 32768; tiene que cubrir system + documento + pregunta + respuesta |
| `max_tokens` | 320 |

La temperatura está fijada a 0,1 en `app/routes/query.py`.

## Documentos y versiones

Los documentos servidos viven en `corpus/markdown/` (que está en `.gitignore`) y
se archivan en `corpus/archive/`, que sí se versiona. Ahí está la copia buena.

| Versión | Modelo | Documentos |
| --- | --- | --- |
| v1.1 | Ministral-3-3B | `diabetes.md`, `hemorroides.md`, `cirugia-abdominal.md` |
| v2.2 | gemma-4-26B-A4B | `diabetes.v5-tu.md`, `hemorroides.v2-tu.md`, `cirugia-abdominal.v2-tu.md` |

La **v1.1** es la que tiene desplegada el cliente. La **v2.2** es la que se
entrega. Los dos bundles están en `dist/`; volver de una a otra es cambiar las
rutas de `PROFILES` y `prompt_variant`.

Solo el GGUF de Ministral está en `models/`. El de gemma (17 GB) no se guarda:
`tools/fetch_model.sh` lo baja de Hugging Face.

## Añadir o cambiar un documento

1. Escribirlo siguiendo `docs/corpus_guidelines.md`. **Leer antes la sección de
   alcance**: las reglas arreglan documentos telegráficos y estropean los que ya
   son prosa.
2. Dejarlo en `corpus/markdown/` y archivar una copia en `corpus/archive/`.
3. Añadir o cambiar la ruta en `PROFILES` (`app/config.py`).
4. `./run.sh <perfil> generate` y levantar de nuevo.
5. Leer las 134 preguntas contra el documento nuevo antes de dar ninguna cifra.

## Evaluación

**La corrección se lee, no se calcula.** Correcta = responde lo que se pregunta
apoyada en el documento, sin inventar y sin fundir ni des-acotar una regla; o se
abstiene donde el documento no da material.

Hubo un puntuador automático y se borró: todos sus números acabaron corregidos
leyendo. Lo único mecánico y fiable es si la **decisión** cambia (responde o se
abstiene) entre semillas o variantes; eso no es corrección.

- Las 134 preguntas del cliente, con su respuesta y su puntuación:
  `reports/audit_questions.json`.
- Las respuestas de cada ejecución: `eval/<brazo>/<procedimiento>.json`, en
  `rows[].our_answer`.
- Los resultados leídos: `docs/auditoria_134_*.md`. **Son el informe**, se editan
  a mano y no se regeneran desde ningún sitio.

No hay herramientas de auditoría en `tools/`: se borraron a propósito. Pasar las
134 contra un endpoint son unas treinta líneas y se escribe en el momento.

## Estructura

```
app/
  main.py             FastAPI + lifespan (carga del modelo, snapshots)
  config.py           Settings y PROFILES
  prompt.py           plantilla del system prompt y sus variantes
  generate.py         CLI de generación de snapshots
  snapshot_builder.py construcción y cacheo del estado KV
  routes/query.py     streaming SSE
src/llm.py            envoltorio de llama-cpp
corpus/markdown/      documentos servidos (gitignored)
corpus/archive/       copias versionadas de los documentos de cada entrega
models/               GGUF (gitignored)
snapshots/<perfil>/   estado KV precalculado (gitignored, regenerable)
dist/                 bundles entregables (gitignored)
profiles/             un .env por perfil de despliegue
infra/ec2-test/       módulo OpenTofu de la caja de pruebas
tools/hpc/            lanzadores Apptainer para el clúster
tools/sweep/          barridos de rendimiento
eval/                 ejecuciones de las 134 y datasets de evaluación
docs/                 documentación (empezar por historial.md)
```

## Documentación

- `docs/historial.md` — **empezar por aquí**: qué se midió, qué se decidió y por
  qué; el estado de cada versión y lo que queda abierto.
- `docs/corpus_guidelines.md` — las siete reglas de redacción del corpus.
- `docs/auditoria_134_*.md` — las tres lecturas de las 134 preguntas.
- `docs/actualizacion_version_desplegada.md` — cómo actualizar una versión ya
  desplegada (corte real ~70 s).
- `docs/ec2_test_env.md` — la caja de pruebas y cómo reproducir allí.
- `docs/evaluation_framework.md`, `docs/prompt_versions.md`,
  `docs/gpu_migration_plan.md`.
