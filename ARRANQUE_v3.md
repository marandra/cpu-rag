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
| **Corrección** — *lectura a MANO sobre el run EC2* (C1, 2026-07-27) | 63% (84/134) | **85–90% (114–120/134)** | **9%** |
| **Presentable** — *lectura a MANO sobre el run EC2* (C1) | 50% (67/134) | **80–84% (107–112/134)** | — |

✅ **C1 HECHO: el 84/78 estimado se queda corto — lo medido es 85–90 % / 80–84 %,
sobre las respuestas del EC2.** 97 de las 134 leídas una a una; las otras 37 se
transfieren exactas (ambos runs rechazan lo mismo). Veredictos en código, uno por
pregunta con su motivo: **`tools/audit_hand.py`** (`--show 29,84` para el porqué).
La banda son 6 marginales `[2, 26, 56, 89, 91, 110]` — apoyadas y sin invención,
pero sin responder del todo lo preguntado; el extremo bajo las descuenta.

Diff emparejado: **gana 23, rompe 7, neto +16**; ratio 6.3× (discrimina mejor, no
mueve umbral). Fronteras: **7 de 7 defectos arreglados** (108, 67, 87, 29, 26,
105, 84). ⚠️ El «siguen rotos 105 y 84» era **artefacto de las sondas**, no un
defecto: la de 105 exigía `regional o general` adyacente y gemma escribe
«regional o *anestesia* general»; la de 84 no tenía `unless` y disparaba con
cualquier mención de laparoscopia. Ambas corregidas; `--self-check` sigue verde
(baseline 106/134, las 7 siguen dando HIT en la rotura de Ministral).
Telegrafía cae en todo (diabetes 13→7%, cirugía 8→2%, hemorroides 42→23%).
Cada lado es **una tirada** (gemma 1% volteo → estable).

**Lo que la decisión automática no ve, y que la lectura sí:**
- Las **7 «roturas» cuestan 0 en corrección**. Tres son ganancias de contenido
  mal etiquetadas (36 rechaza bien — el fulldoc no tiene «qué llevar a la cita»
  y Ministral servía la lista de vacaciones; 100 y 109 eran «FN parcial» y gemma
  responde apoyada en el doc). Sólo 21 y 76 pierden algo real.
- El grueso de la ganancia está donde **la decisión no cambia**: de las 18 `DEF`
  (Ministral falla las 18 por construcción) **gemma arregla 15**, y el scorer
  automático no le da crédito por ninguna. Ahí están el 29 (G1: paracetamol y el
  >39 °C vuelven a ser cosas distintas), 60, 67, 68, 69, 87, 105, 108.
- Riesgo asumido: de las 49 `OK` gemma **sólo pierde 2** (31 responde
  «vacaciones» a «vida normal»; 42 responde con la insulina a «¿me quedaré
  ciego?»). Incorrectas restantes: 7, 30, 63 (63 es idéntica a la de Ministral).

⚠️ **Por qué el `scorecard()` automático (60/55) ≠ el 90/84 de mano:** su columna
`ok` (correctas) es un set FIJO de veredictos leídos a mano en
`tools/audit_triage.py:TRIAGE`; **no acredita** a gemma por responder bien lo que
Ministral fallaba. El 60/55 es un **suelo**, no una medida. Por eso
corrección/presentable de gemma exigía lectura a mano — ya hecha en
`tools/audit_hand.py`.

## Tareas (Fase C)

| # | tarea | notas |
| --- | --- | --- |
| ~~C1~~ | ~~Leer a mano las que se mueven~~ | **HECHO 2026-07-27.** Se leyeron las 97 que lo necesitaban, no sólo los 30 movers. Resultado arriba. Herramientas nuevas: `tools/audit_movers.py` (volcado lado a lado con pregunta, veredicto de terreno y crítica del auditor) y `tools/audit_hand.py` (los veredictos + el scorecard a mano). |
| **C2** | **Redactar la respuesta a la auditoría** | **AHORA.** La regla de memoria («no redactar hasta cerrar el corpus») ya se cumple y el número ya está medido. Forma de 5 puntos en [[audit-response-plan]]. Titular: **gemma correcta en 85–90 % de las 134 (114–120) y presentable en 80–84 % (107–112), vs el 9 % del cliente**; el sistema que auditaron daba 63 %/50 %. Reconocer «derivar» como legítimo; escalar defectos de corpus (29 arreglado; los que quedan son 7, 30, 31, 42, 63 + el 21/76); honestidad sobre corpus dev-grade [[corpus-is-dev-grade]] y sobre que es **una tirada**. |
| **C3** | **Enviar el `tar.gz` al cliente** | `dist/rag-deliverable-v2.tar.gz`. Si prueban en nuestra EC2: meter su IP en el security group para :8001/:8002 (ahora solo la nuestra). |

## Después (investigación, no bloquea)

Probar más prompts: A3 midió que el prompt está cerca del techo con gemma (solo
encoger V13 = latencia). Va detrás. [[project_prompt_iteration_plan]]

## Punteros

- Scorer: `tools/audit_score.py` (`--run LABEL=DIR`, `--scorecard`, `--self-check`,
  `--exclude-procedure`). Verdad de terreno + veredictos: `tools/audit_triage.py`.
- Lectura a mano del run EC2: `tools/audit_hand.py` (`--show IDS` da el motivo de
  cada veredicto). Volcado lado a lado para releer: `tools/audit_movers.py`
  (sin `--ids` saca justo los movers; con `--ids` lo que le pidas).
- Replay: `tools/audit_replay.py` (necesita `reports/audit_questions.json` = las
  134; usa `httpx`; exportar `RAG_API_KEY` de la caja).
- Entregable y su doc: `dist/rag-deliverable-v2/README.md`.
- EC2: `infra/ec2-test/rag-ec2.sh {start|stop|status|ssh}`, `docs/ec2_test_env.md`.
- Historial: `ARRANQUE_v2.md` (empaquetado, cerrado — se puede borrar), `ARRANQUE.md`.
