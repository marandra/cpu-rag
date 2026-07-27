# ARRANQUE v3 — cerrar la puntuación y redactar la respuesta a la auditoría

> Brief de arranque para una **conversación nueva**. El usuario dirá «lee
> ARRANQUE_v3.md y seguimos». Autosuficiente. Continúa `ARRANQUE_v2.md` (fase de
> empaquetado, YA CERRADA) y `ARRANQUE.md` (historial de auditoría). Bórralo
> cuando la respuesta a la auditoría esté enviada.

## Qué está hecho (no rehacer)

**El entregable v2 está construido, validado en EC2 y empaquetado.**
- Bundle cliente listo: **`dist/rag-deliverable-v2.tar.gz`** (105 MB,
  `sha256 a7f02961857363aeca5800591924753f82f6febb1cea9a17956b562e7b970662`).
  Autocontenido, imagen portable, `load_and_run.sh` con el fix del `chmod`,
  README sin nativa ni changelog. **Aún NO enviado al cliente.**
- Desplegado y validado end-to-end en la EC2 de prueba (r7i.2xlarge, single-socket
  SPR). Bug del `chmod` (abortaba el 2º perfil bajo `set -e`) encontrado y
  arreglado (commit `2abe087`).
- **Nativa descartada** para gemma: ~0 % decode (BW-bound), solo −17 % prefill, y
  cambia respuestas → se entrega solo portable. [[gemma-native-no-decode-gain]]
- **Curva N_REPLICAS**: en single-socket replicar NO sube throughput (plano ~4.8
  tok/s); N=1 a full cores es lo mejor. [[gemma-single-socket-bw-ceiling]]
- Docs corregidos (commits `c0d9486`, `f5d8711`).

**La EC2 está PARADA** (`i-09ce0b4c77acc1fd6`, coste cómputo $0). Re-arrancar:
`./infra/ec2-test/rag-ec2.sh start` (IP elástica estable, `ssh rag` sigue). El
modelo y los snapshots persisten en EBS — no se regeneran. La API key de la caja
vive en `/opt/rag/.env` (distinta de la local; para el replay se exporta la de la
caja).

## El resultado medido: antes vs después (las 134)

Repro hecho contra el sistema entregado. Datos en el repo:
- **`eval/ec2/`** = gemma v2 entregado (diabetes/hemorroides/cirugia .json).
- **`eval/audit_replay/`** = Ministral V13 servido = **el sistema ORIGINAL que
  auditaron** (`BASELINE_CORRECT=106=79%`).

Reproducir la comparación (local, sin EC2 — solo lee los JSON):
```
python3 tools/audit_score.py --run "servido-ministral=eval/audit_replay" \
                             --run "ec2-gemma=eval/ec2" --baseline "servido-ministral"
```

| criterio | Ministral (orig.) | **gemma v2** | cliente |
| --- | ---: | ---: | ---: |
| **Decisión** responder/rechazar — *automático, sobre el run EC2* | 79.1% (106/134) | **91.0% (122/134)** | — |
| Corrección — *automático `scorecard()`, sobre el run EC2, CONSERVADOR* | 63% (84/134) | 60% (81/134) | **9%** |
| Presentable — *automático, sobre el run EC2, conservador* | 50% (67/134) | 55% (74/134) | — |
| Corrección / presentable — *lectura a MANO de un run gemma ANTERIOR (A2), NO del EC2* | — | **~84% / ~78%** | — |

⚠️ **El ~84%/78% NO está medido sobre el run EC2.** Es lectura a mano de un run
A2 previo, trasladado como estimación (gemma tiene 1% de volteo por seed y la
decisión reprodujo clavada el 91%). Lo único certificado sobre las respuestas del
EC2 (`eval/ec2/`) es el **91% de decisión** y el **60/55 automático conservador**.
Certificar el ~84/78 sobre este run es exactamente la tarea **C1**.

Diff emparejado: **gana 23, rompe 7, neto +16**; ratio 6.3× (discrimina mejor, no
mueve umbral). Fronteras: **5 de 7 defectos arreglados** (108 anticoagulante ✅,
67, 87, 29, 26); siguen rotos **105** (disyunción regional/general) y **84**
(género/especie). Telegrafía cae en todo (diabetes 13→7%, cirugía 8→2%,
hemorroides 42→23%). Cada lado es **una tirada** (gemma 1% volteo → estable).

⚠️ **Por qué el `scorecard()` automático (60/55) ≠ el 84/78 de mano:** su columna
`ok` (correctas) es un set FIJO de veredictos leídos a mano en
`tools/audit_triage.py:TRIAGE`; **no acredita** a gemma por responder bien lo que
Ministral fallaba. Por eso corrección/presentable de gemma **exige lectura a
mano** de las respuestas nuevas, no se computa.

## Tareas (Fase C)

| # | tarea | notas |
| --- | --- | --- |
| **C1** | **Leer a mano las que se mueven** | Las **7 roturas** `[3, 5, 21, 36, 76, 100, 109]` (4 responden de menos, 3 rechazan de menos) + muestra de las **23 ganancias** `[6, 8, 16, 25, 39, 40, 41, 47, 51, 52, 55, 61, 70, 83, 103, 110, 111, 114, 117, 121, 123, 127, 128]`. Objetivo: cerrar el antes/después de **corrección/presentable** (no solo decisión) y confirmar el ~84/78. Las respuestas lado a lado salen de `eval/ec2/*.json` (`our_answer`) vs `eval/audit_replay/*.json`. |
| **C2** | **Redactar la respuesta a la auditoría** | La regla de memoria («no redactar hasta cerrar el corpus») ya se cumple. Forma de 5 puntos + número titular en [[audit-response-plan]]. Titular: **gemma correcta/presentable en ~78–84% de las 134 (~105–112) vs el 9% del cliente**; reconocer «derivar» como legítimo; escalar defectos de corpus (ID 29 ya arreglado; 105/84 siguen); honestidad sobre corpus dev-grade [[corpus-is-dev-grade]]. |
| **C3** | **Enviar el `tar.gz` al cliente** | `dist/rag-deliverable-v2.tar.gz`. Si prueban en nuestra EC2: meter su IP en el security group para :8001/:8002 (ahora solo la nuestra). |

## Después (investigación, no bloquea)

Probar más prompts: A3 midió que el prompt está cerca del techo con gemma (solo
encoger V13 = latencia). Va detrás. [[project_prompt_iteration_plan]]

## Punteros

- Scorer: `tools/audit_score.py` (`--run LABEL=DIR`, `--scorecard`, `--self-check`,
  `--exclude-procedure`). Verdad de terreno + veredictos: `tools/audit_triage.py`.
- Replay: `tools/audit_replay.py` (necesita `reports/audit_questions.json` = las
  134; usa `httpx`; exportar `RAG_API_KEY` de la caja).
- Entregable y su doc: `dist/rag-deliverable-v2/README.md`.
- EC2: `infra/ec2-test/rag-ec2.sh {start|stop|status|ssh}`, `docs/ec2_test_env.md`.
- Historial: `ARRANQUE_v2.md` (empaquetado, cerrado — se puede borrar), `ARRANQUE.md`.
