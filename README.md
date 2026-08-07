# cpu-rag

Servicio de preguntas y respuestas para pacientes, solo CPU. Cada procedimiento
tiene **un único documento markdown** que se manda entero como contexto a un
modelo local de llama.cpp. No hay recuperación, ni embeddings, ni base vectorial.

Este repo es la **v2.2**, la entregada y auditada. Para GPU, otra arquitectura y
otro repo: [`gpu-rag`](../gpu-rag/).

El sistema tiene **prohibido usar conocimiento propio** y se abstiene cuando el
documento no cubre la pregunta. Eso es una instrucción explícita del prompt, no
una laguna, y condiciona cómo se evalúa: ver «Evaluación» más abajo.

## Cómo funciona

```
corpus/markdown/<documento>.md ──► se precalcula el estado KV (el prefijo)
                                            │
                                            v
        POST /query  ──►  reutiliza el prefijo + PREGUNTA: <q>  ──►  SSE
```

El prefijo (system prompt + documento) se paga una vez. Arranque en frío ~92 s;
con el prefijo ya calentado, ~1 s. Solo la pregunta y la respuesta cuestan
generación.

De dónde sale ese prefijo lo decide `snapshot_mode`:

| modo | por petición | arranque | cuándo |
| --- | --- | --- | --- |
| `memory` (por defecto) | ~0,4-0,6 s | +60-80 s por procedimiento | una instancia; no hay pickles, ni paso `generate`, ni copia de staging |
| `disk` | ~0,4-0,6 s | instantáneo | el pool: N réplicas calentando a la vez es mucho peor que leer un pickle |
| `off` | ~70 s al cambiar de procedimiento | instantáneo | solo para diagnosticar |

`memory` es el defecto **por simplicidad, no por velocidad** —entre las dos
primeras filas no hay diferencia medible—: quita el paso `generate`, los pickles
de ~0,5 GB por procedimiento y la copia de staging, y convierte cambiar un
documento en un reinicio.

**En `disk` la semilla va congelada dentro del pickle**, así que dos ejecuciones
contra el mismo snapshot dan respuestas idénticas byte a byte. En `memory` el
prefijo se calienta en vivo: determinista dentro del proceso, no entre
arranques.

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
./run.sh glucowise generate  # construye los snapshots; el pool va en modo disk
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
| `n_ctx` | 8192; tiene que cubrir system + documento + pregunta + respuesta. El documento más largo (diabetes) deja poco margen |
| `max_tokens` | 320 |
| `snapshot_mode` | de dónde sale el prefijo KV: `memory` (en RAM, por defecto), `disk` (pickles; lo que fuerza el pool de `docker-compose.yml`) u `off` (sin estado, solo diagnóstico) |

La temperatura está fijada a 0,1 en `app/routes/query.py`.

## Documentos y versiones

Los documentos servidos viven en `corpus/markdown/` (que está en `.gitignore`) y
se archivan en `corpus/archive/`, que sí se versiona. Ahí está la copia buena.

| Versión | Modelo | Documentos |
| --- | --- | --- |
| **v2.2** — la que se entrega | gemma-4-26B-A4B | `diabetes.v5-tu.md`, `hemorroides.v2-tu.md`, `cirugia-abdominal.v2-tu.md` |
| v1.1 — deprecada, aún desplegada en cliente | Ministral-3-3B | `diabetes.md`, `hemorroides.md`, `cirugia-abdominal.md` |

Los dos bundles están en `dist/`; cambiar de una a otra es cambiar las rutas de
`PROFILES` y `prompt_variant`.

El GGUF de gemma (17 GB) no se guarda en el repo: `tools/fetch_model.sh` lo baja
de Hugging Face. En `models/` solo está el de Ministral.

## Añadir o cambiar un documento

1. Escribirlo siguiendo `docs/corpus_guidelines.md`. **Leer antes la sección de
   alcance**: las reglas arreglan documentos telegráficos y estropean los que ya
   son prosa.
2. Dejarlo en `corpus/markdown/` y archivar una copia en `corpus/archive/`.
3. Añadir o cambiar la ruta en `PROFILES` (`app/config.py`).
4. `./run.sh <perfil> generate` y levantar de nuevo. (En `memory` u `off` basta
   con reiniciar: no hay nada que pregenerar.)
5. Leer las 134 preguntas contra el documento nuevo antes de dar ninguna cifra.

## Evaluación

**La corrección se lee, no se calcula.** Correcta = responde lo que se pregunta
apoyada en el documento, sin inventar y sin fundir ni des-acotar una regla; o se
abstiene donde el documento no da material.

No hay puntuador automático, y es deliberado: el que hubo acabó con todos sus
números corregidos a mano. Lo único mecánico y fiable es si la **decisión** cambia
(responde o se abstiene) entre semillas o variantes; eso no es corrección.

- Las 134 preguntas del cliente, con su respuesta y su puntuación:
  `reports/audit_questions.json`.
- Las respuestas de cada ejecución: `eval/<brazo>/<procedimiento>.json`, en
  `rows[].our_answer`.
- Los resultados leídos: `docs/auditoria_134_*.md`. **Son el informe**, se editan
  a mano y no se regeneran desde ningún sitio.

Tampoco hay herramientas de auditoría en `tools/`, y también es deliberado: pasar
las 134 contra un endpoint son unas treinta líneas y se escribe en el momento.

## Estructura

```
app/
  main.py             FastAPI + lifespan (carga del modelo, snapshots)
  config.py           Settings y PROFILES
  prompt.py           plantilla del system prompt y sus variantes
  generate.py         CLI de generación de snapshots (solo en modo `disk`)
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
docs/                 documentación (empezar por estado.md)
```

## Documentación

- `docs/estado.md` — **empezar por aquí**: qué se sirve, con qué números, bajo qué
  restricciones y qué queda abierto.
- `docs/corpus_guidelines.md` — las siete reglas de redacción del corpus.
- `docs/auditoria_134_*.md` — las lecturas de las 134 preguntas, una por versión.
- `docs/actualizacion_version_desplegada.md` — cómo actualizar una versión ya
  desplegada (corte real ~70 s).
- `docs/ec2_test_env.md` — la caja de pruebas y cómo reproducir allí.
- `docs/evaluation_framework.md`, `docs/prompt_versions.md`.
