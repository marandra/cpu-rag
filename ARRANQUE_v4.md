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

Las tres versiones, leídas a mano pregunta a pregunta (2026-07-28): la **v1.1**
auditada **83/134 (62 %)**, la **v2** **115/134 (86 %)** y la **v2.2** —la que se
entrega— **119/134 (89 %)**. El detalle está en los tres informes de `docs/`, con
el motivo y la sección del documento en la que se apoya cada veredicto.

**Cualquier variante tiene que batir eso leyendo las 134, y el informe es el
resultado.** No hay puntuación automática: se borró, junto con el resto de los
scripts de auditoría, porque todos sus números acabaron corregidos leyendo. Si
hace falta una herramienta (pasar las 134 contra un endpoint, volcar dos runs
lado a lado), se escribe en el momento. Ver [[eval-by-reading]].

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

Los scripts de `tools/audit_*.py` **ya no existen**: se borraron el 2026-07-28. Se
escriben en el momento cuando hagan falta, que es más barato que mantenerlos y
evita volver a confundir una lectura con un cálculo. Lo que se necesita está todo
como datos:

- `reports/audit_questions.json` — las 134 preguntas, con la respuesta que
  registró el cliente y su puntuación.
- `eval/<brazo>/<procedimiento>.json` — las respuestas de cada ejecución, en
  `rows[].our_answer`. Pasar las 134 contra un endpoint son ~30 líneas: leer el
  JSON, POST a `/query` con `RAG_API_KEY`, guardar.
- `docs/auditoria_134_*.md` — los tres informes. **Son el resultado**, se editan a
  mano, y no se regeneran desde ningún sitio.
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
