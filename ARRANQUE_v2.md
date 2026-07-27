# ARRANQUE v2 — construir el entregable de gemma (Fase B)

> Brief de arranque para una **conversación nueva**. El usuario dirá «lee
> ARRANQUE_v2.md y empezamos». Este documento es autosuficiente: no hace falta
> releer todo `ARRANQUE.md` (ése es el historial de la fase de auditoría, del
> que esto es la continuación). Bórralo cuando la Fase B acabe.

## Qué es esto

La fase de auditoría cerró (A1-A4, A3). La decisión de modelo está tomada:
**se adopta gemma-4-26B** en lugar de Ministral V13. Falta **empaquetar un
entregable v2** para el cliente. Es un **v2, no un v1.2**: cambia el modelo, el
empaquetado, el corpus y la config de servicio. Llamarlo v1.2 vende corto lo que
es.

Todo lo de abajo está **medido**, no supuesto. Las referencias `[[...]]` y los
números detallados viven en `ARRANQUE.md` y en las memorias del proyecto.

## Lo que ya sabemos y no hay que volver a medir

**gemma vs Ministral (jobs 7320 / A2), las 134 preguntas:**

| | Ministral V13 | gemma-4 |
| --- | ---: | ---: |
| decisión responder/rechazar | 78,1 % | **91,0 %** |
| corrección (leído a mano) | 63 % | **84 %** |
| presentable | 50 % | **78 %** |
| volteo por seed | 22 % | **1 %** |
| invenciones / fusiones en 1206 gen | — | **0 / 0** |

**Velocidad — gemma es un MoE 26B/~4B activos, limitado por ancho de banda, no
por cores** (jobs 7324/7325). Curva D4 (saturado, Σ(N×nT)=64):

| config | tok/s por usuario | usuarios | veredicto |
| --- | ---: | ---: | --- |
| **nT=8 × N=8** ← **default** | **8,60** | **8** | el punto bueno |
| nT=16 × N=4 | 12,52 | 4 | pasa |
| nT=32 × N=2 | 14,95 | 2 | pasa |
| nT=4 × N=16 | 4,78 | 16 | **bajo el suelo de 6** |

Menos réplicas → más hilos cada una → más tok/s por usuario (el cuello es el BW,
con más hilos usas los dos sockets). Todos los puntos útiles **por encima del
suelo de 6**. `init` a nT=64: warm **~8-20 s por procedimiento**.

**Prompt: V13 se queda** (A3, job 7363). Ningún variante lo bate en decisión.
El único movimiento aprovechable es **encoger** (g1 = V13 sin el andamiaje
anti-invención, −255 t/petición, misma decisión), pero es **latencia, no
calidad**, y su capa de calidad no está releída. **No entra en el v2** salvo que
alguien relea su calidad; se anota como candidato de la fase de investigación
posterior.

## Decisiones de corpus para el v2

| procedimiento | qué se sirve en v2 | por qué |
| --- | --- | --- |
| **diabetes** | **v4** (22,4 KB) | **decidido por el usuario.** Mejor decisión (85,5 vs 83,4 %) y telegrafía (7 vs 11 %). Coste de velocidad **−5 %** en decode a nT=8, no bloquea (job 7389). |
| **cirugía-abdominal** | **served** (7,6 KB, el original) | v4 **medido peor** (88,7 vs 93,8 %, telegrafía ×5). **No entra.** |
| **hemorroides** | **vA (2,6 KB) — DECIDIDO 2026-07-24** | El served (1,1 KB) es la causa de la telegrafía y arrastra el 108. `hemorroides.vA` (2,6 KB, sesión ciega, **mismos hechos, sin invención**) ganó el A/B (job 7329): arregla el 108 y baja telegrafía 56→23 %. `MUST_REFUSE` **re-derivado en B1 → sin cambios** (las 11 rechazables de hemorroides no ganan terreno en vA; vA es superset solo-forma). |

### ⚠️ No perder el corpus anterior de diabetes

Al hacer que diabetes sirva la v4, **conservar el original (13 KB) por si hay
que volver**. Estado actual de los ficheros:

- `corpus/markdown/diabetes.md` (13 KB) — el **served actual**. `corpus/markdown/`
  está en `.gitignore`.
- Copia versionada del original: existe en el historial y en las fuentes; el v4
  versionado está en `eval/corpus_ab_v4/corpora/diabetes.v4.md` (y `-mio.md`, la
  versión contaminada de control — **ésa no se sirve nunca**).
- **Al aplicar v4**: preservar el original como `corpus/markdown/diabetes.v1.md`
  (o equivalente) **antes** de sobrescribir, y que la copia versionada del
  original quede clara en el repo. La reversión debe ser un cambio de una línea
  en `app/config.py` (`PROFILES["glucowise"]["diabetes"]`), no una arqueología.

### ⚠️ Al cambiar cualquier corpus, re-derivar la respondibilidad

`MUST_REFUSE` se deriva del corpus. Si diabetes pasa a v4, la verdad de terreno
del scorer puede quedar obsoleta y contar mejoras como regresiones (pasó con el
109 y el 111). Para diabetes v4 **ya se re-derivó y salió sin cambios**
(ARRANQUE.md §Re-derivación), así que este punto está cubierto para diabetes;
vale para hemorroides si se sirve vA.

## Las tareas (Fase B)

| # | tarea | notas |
| --- | --- | --- |
| ~~**B1**~~ ✅ | **Versión y alcance — HECHO 2026-07-24.** «v2» fijado; corpora: diabetes v4 ✔, cirugía served ✔, hemorroides vA ✔. `app/config.py` editado: `PROFILES` apunta a `diabetes.v4.md` + `hemorroides.vA.md` (reversión = 1 línea, originales intactos), `model_path` default → `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`. `MUST_REFUSE` re-derivado para hemorroides vA → sin cambios. Config estática ya parametrizable vía env del launcher (ver §Sobre lo dinámico). | HECHO |
| **B2** | **Etapa `init`**: tras instalar el contenedor, **descarga el modelo** (17 GB) **y genera los snapshots** a todos los cores. | ✅ **fuente decidida (2026-07-24): HF público, sin token.** Repo `unsloth/gemma-4-26B-A4B-it-GGUF`, fichero `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (~16,9 GB). HEAD anónimo → `302` a CDN (`user_id=public`) → `200`; **NO gated** (es la redistribución de Unsloth, no el repo oficial de Google). URL directa: `https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`. Medido: usar nT=máx (warm 7,7 s/proc a nT=64 vs 45 s a nT=8). Snapshots a los 3 procedimientos. |
| **B3** | **Empaquetado.** El modelo de 17 GB **rompe la imagen portable de un fichero de v1.1**. Imagen pequeña (~300 MB) + init que baja modelo y construye snapshots. | La imagen NO hornea el modelo, por eso el init lo baja. **Imagen NATIVA validada en cluster 2026-07-24 (job 7397):** `cpu-rag-api:2.0.0-spr-native` (tar 316 MB), `docker build` aquí → `apptainer build` allí. Smoke OK ambos perfiles: VNNI/AMX activos, snapshots v4/vA generan a nT=64, `/query` streamea, diabetes 7.62 tok/s single-replica nT8. `.dockerignore` añadido (excluye `-mio` y adelgaza contexto), `LABEL version=2.0.0`. **Bundle cliente HECHO 2026-07-24:** `dist/rag-deliverable-v2/` (imagen PORTABLE, `load_and_run.sh <profile>` fetch→load→generate→up, `profiles/*.env` glucowise→:8001 / aiciblock→:8002, `.env.example`, `fetch_model.sh`, README, 3 corpus servidos). Una imagen parametrizada por PROFILE (misma para los dos proyectos), corre una vez por proyecto; los dos coexisten en una caja para pruebas. **Bug SIGILL de la portable arreglado** (la cache de pip reusaba la wheel nativa; fix = recompilación fresca `--no-cache-dir` + high-ISA OFF explícito; validado en el laptop AVX2-only). EC2 de prueba nuestra: `docs/ec2_test_env.md` (r7i.2xlarge, ~$0.53/h). |
| ~~**B4**~~ ✅ | **Config de servicio estática parametrizable — HECHO 2026-07-24.** Bundle cliente: hilos de servicio expuestos como knob visible (`N_THREADS` en `.env`; `load_and_run.sh` pone **todos los cores** por defecto en vez del cap silencioso de 9 del app — gemma es BW-bound; guía para partir a la mitad si corren los dos profiles en una caja). Generate usa `RAG_GEN_THREADS`=nproc. **Bundle es pool-capable** (servicio `rag` escalable + nginx LB dinámico, `--scale rag=$N_REPLICAS`, default 1) — misma topología demo=producción, se escala por config; forma N×nT expuesta (N_REPLICAS + N_THREADS, producto ≤ cores). El pool del HPC (`launch_pool.sh`) ya parametrizado (default D4 nT8 N8). **Sin rebalanceo en caliente.** | HECHO |

**Coexistencia aiciblock/glucowise**: se queda por `PROFILE` (el request ya
enruta a proyecto+especialidad). Separar en duro por proyecto es futuro — es
borrar una entrada del dict de `app/config.py`. No tocar ahora.

## Sobre lo dinámico — decidido: estático parametrizable, NADA de rebalanceo vivo

El usuario preguntó por instancias dinámicas (bajar una de 16 y crear dos de 8
según demanda). Conclusión repasada con él:

- **Lo barato ya está hecho:** `tools/hpc/launch_pool.sh` **ya es
  parametrizable** — `N_REPLICAS`, `N_THREADS`, `BASE_PORT`, `LB_PORT`,
  `NUMA_START` son env vars; el «8» es un `${N_THREADS:-8}`, **un default, no un
  hardcode**. Arrancar un nodo como `N_THREADS=16 N_REPLICAS=4` o `nT=8 N=8` ya
  se elige al lanzar. **Esto cubre el ~80 % de lo que se pedía.** B4 es exponerlo
  limpio (mín/máx/forma), no inventarlo.
- **El rebalanceo en caliente es su propia línea y se DEFIERE.** En llama-cpp los
  hilos y el pinning NUMA (`numactl --cpunodebind/--membind`) se fijan **al
  arrancar el proceso**. «Partir una de 16 en dos de 8» no es ajustar hilos: es
  **matar el proceso y arrancar dos**, cada uno recargando 17 GB + restaurando
  snapshots. Es **autoescalado con arranque en frío**, no un dial. Y el diseño
  actual tiene **concurrencia == nº de réplicas** (cada réplica serializa la
  generación); la elasticidad «de verdad» apunta a **continuous batching** (un
  proceso, N slots), que es otro modelo de servicio.
- **Herramientas, para no reinventar la rueda** (ninguna es plug-in):
  - **`llama-server` del propio llama.cpp** (`--parallel`, slots, continuous
    batching) es lo más alineado, **pero rearquitectura fuera del diseño de
    snapshot-pickle** (nuestro prefijo KV congelado por procedimiento con seed
    determinista). Es un **spike de investigación**, no una tarea de entregable.
  - **Ray Serve / KServe / K8s HPA**: orquestadores de autoescalado NUMA-aware.
    Dependencia pesada, desproporcionada para un entregable a cliente.
- **Regla para el v2:** entregar con config **estática parametrizable**. Si algún
  día se hace elasticidad viva, la vía industrial es `llama-server` con slots o
  un orquestador, **no un lanzador casero que rebalancee**.

## Después del v2 (no ahora)

Usar el entregable como **plataforma para exprimir más el prompt**. A3 ya midió
que el prompt está cerca de su techo con gemma (lo único aprovechable es encoger
V13 → latencia), así que el retorno es pequeño y **no bloquea nada**. Va detrás.

## Punteros

- Historial completo de la auditoría: `ARRANQUE.md` (no hace falta para B, pero
  está ahí).
- Lanzador y su documentación: `tools/hpc/launch_pool.sh` (cabecera larga),
  `pool_dual.sbatch`, `stop_pool.sh`.
- init de snapshots: `python -m app.generate`, `app/snapshot_builder.py`.
- Config: `app/config.py` (`PROFILES`, `model_path`, `n_threads`).
- Bench de velocidad con override de corpus: `tools/bench_model.py --fulldoc`,
  `tools/hpc/model_scaling.sbatch` (`FULLDOC=`).
- HPC: ssh alias `hpc`, `module load singularity/1.4.1` + `spack load numactl
  target=x86_64_v3`, `.venv-latest` (llama-cpp 0.3.34, tiene arquitectura
  `gemma4`). El repo local **no es** el del cluster; se sincroniza fichero a
  fichero con `scp`, nunca `rsync` a lo bruto (deriva en los dos sentidos).
- Modelo servido: `models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (17 GB).
