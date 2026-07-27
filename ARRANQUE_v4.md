# ARRANQUE v4 — pruebas de prompt en EC2, y de paso validar despliegue y actualización

> Brief para una **conversación nueva**. El usuario dirá «lee ARRANQUE_v4.md y
> seguimos». Autosuficiente. Sustituye a `ARRANQUE_v3.md`, que ya se puede borrar.

## Punto fijo: la v2 se puede enviar hoy

**Esto es lo que protege el trabajo. No tocarlo sin decirlo.**

- `dist/rag-deliverable-v2.tar.gz` — 105 MB, `chmod 444`, sha256 registrado al lado
  en `.sha256`. Verificar con `sha256sum -c dist/rag-deliverable-v2.tar.gz.sha256`.
- La configuración que define la v2 está **commiteada** (`bca2a32`): `app/config.py`
  (diabetes v4, hemorroides vA, cirugía el original, gemma por defecto), el
  `Dockerfile`, `infra/ec2-test/` y los tres documentos servidos archivados en
  `corpus/archive/*_v2servido_2026-07-27.md`. Ojo: `corpus/markdown/` está en
  `.gitignore`, por eso el archivado.
- La respuesta a la auditoría está redactada y con las cifras verificadas:
  `docs/respuesta_auditoria_borrador.md` (texto plano, listo para que el usuario le
  dé su estilo) + `docs/auditoria_134_evaluacion.md` y `docs/auditoria_134_v2.md`.

**Si un experimento no sale, se envía la v2 tal cual y no se ha perdido nada.** Ese
era el objetivo explícito de cerrar todo esto antes de empezar.

## Qué medimos, y cuál es la vara

La v2, leída a mano sobre `eval/ec2/`: **120 de 134 correctas (90 %)**; decisión
responder-vs-abstenerse **122/134 (91 %)**. La v1.1 auditada: 83/134 (62 %). Detalle
por pregunta con su motivo en `tools/audit_hand.py` (`--show 29,84`).

**Cualquier variante tiene que batir eso leyendo las respuestas, no un número.** La
puntuación automática (`tools/audit_score.py --scorecard`) sirve para cribar, no para
concluir: su columna de correctas es un set fijo leído sobre las respuestas de
Ministral y da 60 % sobre la v2, que es un suelo y no una medida. Ver
[[eval-by-reading]].

## Las dos tareas, en orden

### D1 — El texto de la abstención (comprometido en el correo)

Hoy el sistema responde `No tengo información sobre eso.` a **45** preguntas, de las
cuales **37 son correctas por diseño** y **21 de esas 37 son emocionales**: «me da
vergüenza pincharme delante de otras personas», «tengo miedo a las agujas», «¿me puedo
morir en esta operación?». Tiene razón en no tener nada que decir y lo dice de la peor
manera posible.

El cambio es **reescribir ese texto** para que reconozca la pregunta y remita al
equipo. Por qué esto y no la derivación: alcanza a las 37 (la derivación cubría 14) y
**no puede desplazar una respuesta correcta**, porque solo reescribe lo que ya es una
abstención.

Ya está medido lo que NO hay que hacer: el brazo G2 (`eval/audit_ab_g/g2`, gemma, 9
semillas) convierte 14 de 45 abstenciones pero solo 4 de las 21 emocionales, siempre
con la misma frase de plantilla, y **deriva 7 preguntas que el documento sí responde**
(3, 12, 20, 23, 27, 108, 128), perdiendo hasta 3 puntos de decisión. No repetirlo.

Criterio de aceptación: la decisión **no baja de 122/134**, y las 37 abstenciones leídas
a mano resultan mejores para un paciente. Si toca una sola respuesta correcta, fuera.

### D2 — Validar despliegue y actualización de versión en EC2

Es el motivo real de usar la caja: **verificar que sabemos actualizar una versión
desplegada**, no solo instalar de cero. Al desplegar cada variante hay que anotar el
procedimiento como si fuera el del cliente.

## Cómo se opera la EC2

Está **parada** (`i-09ce0b4c77acc1fd6`, coste cómputo 0). IP elástica estable, `ssh rag`
sigue funcionando. El modelo (17 GB) y los snapshots persisten en EBS, no se regeneran.

```
./infra/ec2-test/rag-ec2.sh start|stop|status|ssh
```

- La API key de la caja vive en `/opt/rag/.env` y **es distinta de la local**: para el
  replay hay que exportar la de la caja.
- Detalle del entorno: `docs/ec2_test_env.md`. Corrección importante ya aplicada allí:
  en single-socket **N=1 a full cores es lo mejor** (replicar no sube throughput,
  ~4.8 tok/s plano, BW-bound) y la build nativa **no aporta** en gemma (~0 % decode,
  −17 % prefill, y cambia respuestas) — se sirve solo la portable.
- **Acordarse de `stop` al terminar.** Cuesta ~$0.53/h.

## Herramientas

- `tools/audit_replay.py` — pasa las 134 contra un endpoint. Necesita
  `reports/audit_questions.json` (está) y `RAG_API_KEY` exportada (la de la caja).
- `tools/audit_movers.py` — volcado lado a lado para **leer**: pregunta, veredicto de
  terreno, crítica del auditor y las dos respuestas. Sin `--ids` saca los movers.
- `tools/audit_hand.py` — los veredictos a mano de la v2 + el scorecard.
- `tools/audit_annex.py` — regenera los dos anexos del cliente.
- `tools/audit_score.py` — decisión, diff emparejado, `--self-check`. Para cribar.
- `app/prompt.py` — `get_system_prompt(procedure, variant)`; el mecanismo de variantes
  ya existe (`_replacing`, `_plus_example`).

## Si la variante gana

Entonces, y solo entonces: reconstruir el bundle, **con número de versión nuevo**
(v2.1), sin pisar `dist/rag-deliverable-v2.tar.gz`, y actualizar en el correo la frase
del punto 5 («lo que sí vamos a hacer») por lo que se haya hecho. El correo está
escrito para que ese párrafo se pueda cambiar sin tocar el resto.

## Punteros

- Historial: `ARRANQUE_v3.md` (fase C, cerrada — se puede borrar), `ARRANQUE_v2.md`,
  `ARRANQUE.md`.
- Guías de redacción del corpus: `docs/corpus_guidelines.md` (siete reglas). La otra
  mitad de D1 (material psicológico para cirugía y hemorroides) es trabajo de corpus,
  no de prompt, y va por ahí.
