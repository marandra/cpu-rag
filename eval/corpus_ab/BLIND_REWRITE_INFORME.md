# Informe de reescritura — cirugía de hemorroides (vA / vB)

Se han producido dos versiones a partir del «RESUMEN Cirugía de hemorroides»
incluido en `docs/corpus_rewrite_brief.md`:

- `hemorroides.vA.md` — mismos hechos, solo forma.
- `hemorroides.vB.md` — misma forma, más contenido tomado exclusivamente de
  `corpus/sources/`.

No se ha buscado ni consultado ningún conjunto de preguntas de evaluación. Las
preguntas de la sección 1 se han derivado del propio procedimiento.

---

## 1. Preguntas de paciente previstas (R3) y dónde se responden

«Sí» significa que existe una frase que afirma o niega la respuesta
directamente, no que sea deducible.

| # | Pregunta previsible | vA | vB |
|---|---|---|---|
| 1 | ¿Qué es una hemorroidectomía / qué me van a hacer? | Sí — «En qué consiste» | Sí |
| 2 | ¿Por qué me tienen que operar? ¿Cuándo está indicada? | Sí — frase de indicación | Sí |
| 3 | ¿Dónde me operan? | Sí — «en el quirófano» | Sí |
| 4 | ¿Me van a dormir? ¿Qué anestesia me ponen? | Sí — regional o general | Sí |
| 5 | ¿Cuánto dura la operación? | Sí — 30–60 min | Sí |
| 6 | ¿Cuántas horas tengo que estar en ayunas? | Sí — 6–8 h | Sí, + detalle sólidos/líquidos |
| 7 | ¿Puedo beber agua antes de entrar a quirófano? | **No** | Sí — líquidos claros hasta 2 h antes |
| 8 | ¿Qué hago con mis pastillas? ¿Y con el anticoagulante? | Sí — lo ajusta el equipo médico | Sí |
| 9 | ¿Me va a doler después? | Sí — dolor moderado | Sí, + curva de las primeras 24 h |
| 10 | ¿Qué hago si me duele mucho? | **No** | Sí — pedir calmante de rescate |
| 11 | ¿Cuánto tardaré en recuperarme? | Sí — 2–4 semanas | Sí |
| 12 | ¿Qué cuidados tengo que hacer en casa? | Sí — baños de asiento | Sí |
| 13 | ¿Qué puedo comer después? | Sí — fibra y líquidos | Sí, + reinicio progresivo |
| 14 | ¿Es normal sangrar al defecar después? | Sí — complicación frecuente | Sí |
| 15 | ¿Qué riesgos tiene la operación? | Sí — tres niveles de frecuencia | Sí |
| 16 | ¿Me puedo quedar incontinente? | Sí — muy raro | Sí |
| 17 | ¿Me volverán a salir las hemorroides? | Sí — «reduce las recaídas» | Sí |
| 18 | ¿Hay alguna alternativa a operarme? | Sí — dos frases | Sí |
| 19 | ¿Qué pasa si no me opero? | Sí | Sí |
| 20 | ¿Me puedo echar atrás después de firmar? | Sí | Sí |
| 21 | ¿Quién me informa y cuándo? ¿Qué firmo? | **No** | Sí — consultas de cirugía y anestesia |
| 22 | ¿Puedo fumar o beber alcohol antes de la operación? | **No** | Sí |
| 23 | ¿Tengo que preparar algo con la alimentación antes? | **No** | Sí — proteínas 7–10 días antes |
| 24 | ¿Puedo hacer ejercicio antes de la operación? | **No** | Sí |
| 25 | Estoy muy nervioso, ¿me pueden dar algo? | **No** | Sí — premedicación |
| 26 | ¿Cuándo me puedo levantar y andar? | **No** | Sí |
| 27 | ¿Cuándo me dan el alta? | **No** | Sí — tres criterios |
| 28 | ¿Qué puede hacer mi familia? | **No** | Sí |

Preguntas previsibles que **ninguna** de las dos versiones responde, porque no
hay material para ello ni en el original ni en las fuentes permitidas (se dejan
deliberadamente sin cubrir, para que el sistema pueda decir «no tengo
información»):

- ¿Cuándo puedo volver al trabajo, conducir, hacer deporte o tener relaciones?
- ¿Es una operación con ingreso o ambulatoria? ¿Cuántos días estaré ingresado?
- ¿Me ponen puntos? ¿Hay que quitarlos? ¿Cómo curo la herida?
- ¿Cómo hago los baños de asiento (con qué agua, cuántas veces, cuánto tiempo)?
- ¿Qué señales de alarma deben hacerme acudir a urgencias?
- ¿Qué analgésico puedo tomar en casa y cada cuánto?
- ¿Necesito laxantes o algún suplemento de fibra concreto?
- ¿Tengo que ir acompañado el día de la cirugía?

Los cinco últimos son, a mi juicio, el hueco más caro del corpus: son las
preguntas típicas del postoperatorio domiciliario de una cirugía perianal y el
material disponible no las cubre en absoluto.

---

## 2. Qué se dejó fuera de la versión A por dudar

Todo lo que sigue parecía clínicamente razonable, pero no está en el original,
así que se excluyó de vA (y, salvo indicación, también de vB):

1. **«No suspenda ni modifique su medicación por su cuenta».** Excluido
   explícitamente por el brief; se dejó fuera aunque es el corolario natural de
   la línea «Ajustar medicación».
2. **Quién elige el tipo de anestesia.** El original dice «anestesia regional o
   general» pero no dice quién decide cuál. No se atribuyó la decisión a nadie.
3. **Motivo del ayuno** (riesgo de aspiración) y **qué pasa si el paciente no
   cumple el ayuno** (suspensión de la cirugía). No están en el original.
4. **«La recuperación completa».** El original dice «recuperación en 2–4
   semanas»; se escribió «se recupera en un plazo de 2 a 4 semanas» sin añadir
   «completa» ni «vuelta a la actividad normal», que son lecturas distintas.
5. **Comparación implícita de «reduce recaídas».** El original no dice respecto
   a qué se reducen (¿a no operarse? ¿a las alternativas?). Se dejó la frase sin
   término de comparación.
6. **«Infección de la herida».** El original dice solo «infección»; se mantuvo
   «una infección», sin localizarla.
7. **Que las alternativas sean menos eficaces o que la cirugía sea el
   tratamiento definitivo.** El original solo las lista. (El documento de fisura
   anal sí lo afirma para *su* procedimiento; no se transfirió, ver §3.)
8. **Que el sangrado abundante o la incontinencia requieran atención urgente.**
   El original los clasifica como muy raros y no dice nada más.
9. **Frecuencia, duración o composición de los baños de asiento.** El original
   solo los nombra.

Dos decisiones fronterizas que sí se tomaron, por considerarlas explicitación de
un sujeto elidido en el sentido que autoriza el brief, y no adición:

- **«El equipo médico le ajusta la medicación… incluidos los anticoagulantes».**
  Es la decisión más discutible de vA. «Ajustar medicación» es un infinitivo sin
  sujeto y R5 obliga a ponerle uno; dejarlo al paciente sería justamente el
  defecto que R5 describe (convertir en instrucción al paciente algo dirigido al
  equipo clínico). Se atribuyó al equipo médico y no se añadió ninguna
  instrucción al paciente. Si se considera que la atribución es un hecho nuevo,
  ésta es la línea a revisar.
- **«El equipo médico le indica… / el equipo quirúrgico le opera en el
  quirófano».** El original usa impersonales («se realiza en quirófano», «es la
  cirugía para extirpar»); R5 exige sujeto y el único sujeto posible es el
  equipo.

---

## 3. Procedencia de cada bloque añadido en la versión B

Se marcan como `[RICA]` = `via-clinica-cirugia-adulto-rica-2021-paciente.md`,
`[GPC]` = `gpc_555_cma_iacs_compl-pacientes.md`,
`[FIS]` = `resumen-fisura-anal.md`.

### Sección «Información y consentimiento antes de la operación» (nueva)
| Frase | Fuente |
|---|---|
| El cirujano le explica en la primera consulta propósito, características, riesgos y resultados esperados | [GPC] §¿Quién le informará? ¿Cuándo? |
| El personal de enfermería le entrega la información por escrito en un folleto | [GPC] misma sección |
| El anestesiólogo le informa del plan anestésico unos días antes | [GPC] misma sección |
| Usted decide tras recibir la información y solo después firma el consentimiento | [GPC] misma sección |
| Puede consultar sus dudas en cualquier momento | [GPC] misma sección |
| Debe resolver con el cirujano las dudas sobre el resultado previsible | [GPC] misma sección |

### Sección «Preparación antes de la operación» (ampliada)
| Frase | Fuente |
|---|---|
| Acude a las consultas de cirugía, anestesia y enfermería antes del ingreso | [RICA] §Preparación previa al ingreso |
| Sólidos hasta 6 h antes, líquidos claros (manzanilla, zumo, solución azucarada) hasta 2 h antes | [RICA] §Preparación previa al ingreso |
| Nada en las 2 horas anteriores | [RICA] §Preparación previa al ingreso |
| Evitar alcohol y tabaco desde que se decide operar; menos complicaciones respiratorias | [RICA] §Preparación previa al ingreso |
| No tomar bebidas alcohólicas | [RICA] §Preparación previa al ingreso |
| Dieta rica en proteínas e hidratación 7–10 días antes; favorece cicatrización y defensa frente a infecciones | [RICA] §Nutrición preoperatoria |
| Ejercicio físico moderado antes del ingreso; la enfermera asesora | [RICA] §Ejercicio previo a la cirugía |
| Premedicación por ansiedad: pastilla la noche antes y 1–2 h antes | [GPC] §¿Qué es la premedicación? |
| Somnolencia y lagunas de memoria por esa medicación | [GPC] §¿Qué es la premedicación? |

### Sección «El dolor después de la operación» (nueva; el «dolor moderado» es del original)
| Frase | Fuente |
|---|---|
| El dolor es máximo las primeras 24 h y luego disminuye | [GPC] §¿Por qué no tengo que tener dolor? |
| Pedir calmante en lugar de aguantarse; aguantarse provoca complicaciones y retrasa la recuperación | [GPC] misma sección |
| Calmantes a ritmo fijo cada 6–8 h más un calmante de rescate a demanda | [GPC] §¿Cómo se trata el dolor postoperatorio? |
| Enfermería controla el nivel de dolor y ajusta el tratamiento | [GPC] misma sección |
| Personal sanitario disponible 24 h para el dolor | [GPC] misma sección |
| Con menos dolor se camina y se recupera fuerza antes | [GPC] §¿Por qué no tengo que tener dolor? |

### Sección «Cuidados después de la operación» (ampliada)
| Frase | Fuente |
|---|---|
| Empezar a beber lo antes posible el mismo día, pequeñas cantidades, luego alimentos fáciles de digerir | [RICA] §Alimentación oral temprana (y [GPC] §¿Cuándo puedo reanudar la alimentación?) |
| No tomar bebidas con gas | [RICA] §Alimentación oral temprana |
| Sentarse en el sillón el mismo día con ayuda; al día siguiente levantarse y dar paseos cortos | [GPC] §¿Cuándo puedo levantarme de la cama? |
| Caminar cuanto antes; la inmovilidad da coágulos, debilidad muscular y neumonía | [GPC] misma sección |

### Sección «El alta hospitalaria» (nueva)
| Frase | Fuente |
|---|---|
| El médico comunica con antelación la fecha probable de alta | [RICA] §Recomendaciones al alta |
| Tres criterios de alta: dolor controlado con analgésicos orales, tolerancia oral sin náuseas ni vómitos, autonomía en la movilidad | [RICA] §Recomendaciones al alta |
| Consultar al personal sanitario cualquier duda de manejo | [RICA] §Recomendaciones al alta |

### Sección «Cómo pueden ayudarle sus familiares y cuidadores» (nueva)
| Frase | Fuente |
|---|---|
| Los profesionales informan a familiares y cuidadores de riesgos y beneficios para que participen | [GPC] §¿Cómo pueden colaborar los familiares? |
| Ayudan a tomar decisiones y apoyan en el reinicio de la alimentación y el movimiento | [GPC] misma sección |

### Material de las fuentes que se decidió NO usar, y por qué

- **[FIS] `resumen-fisura-anal.md`: no se ha tomado nada.** Todo lo realmente
  común (quirófano, anestesia regional o general, ayuno 6–8 h, ajuste de
  medicación, baños de asiento, dieta con fibra y líquidos, mismos tres niveles
  de riesgo, retirada del consentimiento) ya estaba en el documento de
  hemorroides con cifras propias. Lo que quedaba —«la cirugía es el tratamiento
  más eficaz si fallan las pomadas», toxina botulínica, esfinterotomía,
  recuperación en 2–3 semanas, incontinencia limitada a gases y heces
  líquidas— es específico de la fisura y contradiría o desplazaría las cifras
  propias de la hemorroidectomía. Añadirlo habría sido conocimiento clínico
  transferido, no material de fuente.
- **[GPC] Cribado nutricional preoperatorio.** La fuente lo restringe
  explícitamente a «todos los pacientes que van a ser intervenidos de cirugía
  mayor abdominal». Fuera de alcance.
- **[GPC] Cirugía mínimamente invasiva / laparoscopia / gas abdominal.** No
  aplica a la vía perianal.
- **[GPC] Bebida con 200–400 ml de carbohidratos y control de glucemia en
  diabéticos.** Es una medida del protocolo de cirugía mayor abdominal y
  arrastra un subgrupo de pacientes; se dejó fuera para no inventar su
  aplicabilidad aquí.
- **[GPC] Analgesia epidural, bomba de ACP, vía intravenosa 24–48 h, Unidad de
  Recuperación Postanestésica.** Son dispositivos de cirugía mayor; incluirlos
  haría que el sistema respondiese con ellos a preguntas sobre el dolor de una
  hemorroidectomía.
- **[GPC] «La primera defecación suele tener lugar 2 o 3 días después de
  reiniciar la alimentación; no es habitual que se produzcan dolor ni
  hemorragia.»** **Contradice frontalmente** el documento de hemorroides, que
  clasifica el dolor al defecar y el sangrado leve como complicaciones
  *frecuentes*. Excluido por conflicto directo.
- **[GPC]/[RICA] Íleo postoperatorio / parálisis intestinal.** Específico de la
  manipulación del intestino en cirugía abdominal.
- **[RICA] Espirómetro incentivador y fisioterapia respiratoria.** La fuente lo
  presenta como propio de «toda cirugía», así que su inclusión sería
  formalmente admisible, pero describe un régimen de ingreso prolongado
  (cada 2 horas, 10 minutos) que no encaja con una intervención de 30–60
  minutos. Se excluyó como decisión de curación, no por falta de fuente; es
  reversible si se prefiere maximizar cobertura.
- **[RICA] Cifras concretas de movilización (sentado hasta 2 h el primer día,
  hasta 6 h y cuatro series de 60 metros el segundo) e ingesta de 1,5 litros al
  día siguiente.** Se conservó la pauta cualitativa y se descartaron las cifras,
  por la misma razón que el punto anterior.

---

## 4. Dónde chocaron las reglas entre sí

1. **R1 (autosuficiencia) vs R7 (no compenses con longitud).** Es el choque
   principal y afecta a casi todas las frases. Hacer cada frase autosuficiente
   obliga a repetir el anclaje «después de la operación de hemorroides» y el
   sujeto «usted» en frases contiguas, lo que a ojo humano es redundante.
   *Resolución:* se dio prioridad a R1 —es la regla medida— pero se alternó el
   sintagma («la hemorroidectomía», «la operación de hemorroides», «esta
   cirugía», «durante la recuperación») para no repetir literalmente, y no se
   añadió ninguna frase que no aporte un hecho. R7 se aplicó como prohibición de
   *frases* de relleno, no de *palabras* repetidas.

2. **R5 (sujeto explícito) vs versión A (ni un hecho más).** «Ajustar
   medicación» no tiene sujeto en el original y cualquier sujeto que se le
   ponga es, estrictamente, información que el original no da. No hay salida
   neutra: dejarlo en infinitivo incumple R5 y produce justo el defecto grave
   que R5 describe; ponerle sujeto roza la adición. *Resolución:* se atribuyó al
   equipo médico, por ser la lectura que R5 identifica como correcta, y se anotó
   arriba como la línea más discutible de vA. Lo mismo, con menos riesgo, en
   «se realiza en quirófano» → «el equipo quirúrgico le opera».

3. **R2 (lo accionable en la principal) vs fidelidad al énfasis del original.**
   El original enmarca varias cosas como categorías («Frecuentes: …»,
   «Alternativas: …»). Poner el hecho accionable en la principal obliga a
   convertir la categoría en un adverbio subordinado («usted puede presentar
   *con frecuencia* …», «usted puede tratar sus hemorroides *en lugar de
   operarse* …»). *Resolución:* se hizo así; la etiqueta de frecuencia y la
   condición de alternativa sobreviven dentro de la misma frase, que es lo que
   pide R1.

4. **R3 (una frase por pregunta previsible) vs la prohibición de añadir.** En
   vA hay ocho preguntas previsibles sin respuesta posible y en vB quedan otras
   ocho. R3 no puede satisfacerse sin material. *Resolución:* no se rellenó
   ninguna; se listaron en §1 como carencia del corpus. Ésta es exactamente la
   señal que la comparación A/B debería medir.

5. **R4 (el contenido en su sección temática) vs contenido que pertenece a dos
   secciones.** «Dolor al defecar» es a la vez complicación frecuente y respuesta
   a «¿me va a doler?»; «dieta rica en fibra» es cuidado postoperatorio y también
   alternativa terapéutica. *Resolución:* cada hecho se enunció una sola vez, en
   la sección donde el original lo sitúa, y la sección vecina no lo repite ni lo
   referencia. No se creó ninguna sección de mitos, FAQ ni anexo.

6. **Fuentes en conflicto entre sí (solo vB).** El original manda ayuno de 6–8
   horas; [RICA] permite sólidos hasta 6 horas y líquidos claros hasta 2 horas.
   No son idénticos: leídos juntos, un paciente puede entender que puede beber a
   las 3 horas de la cirugía pese a la regla de «6 a 8 horas de ayuno».
   *Resolución:* se conservó la cifra del documento propio como regla del
   procedimiento y se añadió el detalle de [RICA] a continuación, sin borrar
   ninguna de las dos. **Es una inconsistencia residual conocida de vB** y el
   punto que más merece validación clínica antes de cualquier uso real.
