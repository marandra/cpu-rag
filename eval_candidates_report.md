# Candidate Model Comparison

**Date**: 2026-03-30 14:44  
**Procedure**: cirugía de hemorroides  
**Strategy**: hybrid+rerank | top_k=5 | rerank_candidates=8  
**Min score**: -3.0  
**Queries**: 54 (36 in-scope, 18 OOS)

## Summary

| Metric | granite-1b+v1 | granite-1b+v4 | gemma-3n+v1 | gemma-3n+v4 | ministral+v1 | ministral+v4 |
|--------|-------:|-------:|-------:|-------:|-------:|-------:|
| In-scope: GOOD | **21**/36 | **15**/36 | **21**/36 | **18**/36 | **19**/36 | **18**/36 |
| In-scope: PARTIAL | 3/36 | 2/36 | 1/36 | 2/36 | 3/36 | 3/36 |
| In-scope: MISS | 9/36 | 2/36 | 2/36 | 5/36 | 4/36 | 3/36 |
| In-scope: FALSE REFUSAL | 3/36 | 17/36 | 12/36 | 11/36 | 10/36 | 12/36 |
|   (auto-refused) | 3/36 | 3/36 | 3/36 | 3/36 | 3/36 | 3/36 |
| OOS: CORRECT REFUSAL | **6**/18 | **17**/18 | **13**/18 | **15**/18 | **13**/18 | **18**/18 |
|   (auto-refused) | 3/36 | 3/36 | 3/36 | 3/36 | 3/36 | 3/36 |
| OOS: LEAK | 12/18 | 1/18 | 5/18 | 3/18 | 5/18 | 0/18 |
| Avg time (s) | 23.7 | 18.8 | 30.4 | 26.4 | 23.0 | 18.0 |
| Avg tok/s | 4.5 | 4.9 | 2.6 | 2.9 | 4.8 | 4.7 |

## Per-query results

| # | Scope | Category | Query | granite-1b+v1 | granite-1b+v4 | gemma-3n+v1 | gemma-3n+v4 | ministral+v1 | ministral+v4 |
|---|-------|----------|-------|-----|-----|-----|-----|-----|-----|
| 1 | IN | procedimiento | me van a operar y estoy muy nervioso que me v | GOOD 75% | FALSE_REF 0% | GOOD 50% | GOOD 100% | GOOD 100% | GOOD 75% |
| 2 | IN | procedimiento | cuanto dura la operasion | GOOD 100% | GOOD 100% | GOOD 75% | GOOD 100% | GOOD 100% | GOOD 100% |
| 3 | IN | alternativas | q pasa si no me opero | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% |
| 4 | IN | alternativas | hay alguna otra cosa que se pueda hacer antes | MISS 0% | FALSE_REF 0% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% |
| 5 | IN | riesgos | riesgos | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 50% | GOOD 75% | GOOD 100% |
| 6 | IN | recuperación | despues de que me operen cuanto tiempo tardo  | GOOD 50% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% |
| 7 | IN | procedimiento | me duele mucho el ano cuando voy al baño y el | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) |
| 8 | IN | procedimiento | en que consiste la operacion | MISS 0% | MISS 0% | MISS 0% | MISS 0% | MISS 0% | MISS 0% |
| 9 | IN | riesgos | me pueden quedar problemas para aguantar desp | MISS 0% | FALSE_REF 0% | FALSE_REF 33% | MISS 0% | MISS 0% | MISS 0% |
| 10 | IN | preparación | cuanto tienpo de ayuno antes de la sirugia | GOOD 25% | GOOD 25% | GOOD 25% | GOOD 50% | GOOD 25% | GOOD 50% |
| 11 | IN | preparación | puedo tomar agua antes de que me operen | GOOD 50% | GOOD 50% | GOOD 50% | GOOD 50% | GOOD 50% | GOOD 25% |
| 12 | IN | preparación | tengo que dejar el tabaco si me van a operar? | GOOD 25% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 25% | FALSE_REF 0% |
| 13 | IN | nutrición | que como antes de operarme | MISS 0% | GOOD 25% | MISS 0% | MISS 0% | MISS 0% | MISS 0% |
| 14 | IN | recuperación | por que me hacen levantarme el mismo dia de l | GOOD 75% | FALSE_REF 0% | GOOD 25% | GOOD 25% | GOOD 50% | GOOD 50% |
| 15 | IN | recuperación | cuando me dejan comer despues de operarme | GOOD 50% | GOOD 50% | GOOD 50% | GOOD 50% | GOOD 50% | GOOD 25% |
| 16 | IN | recuperación | eso del espirometro que es? la enfermera me d | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) |
| 17 | IN | general | que es el programa rica ese | GOOD 75% | GOOD 75% | GOOD 75% | GOOD 75% | GOOD 75% | GOOD 75% |
| 18 | IN | general | cuando me dan el alta del hospital | PARTIAL 20% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | GOOD 40% | PARTIAL 20% |
| 19 | IN | recuperación | mi madre tiene 72 años y la van a operar del  | MISS 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | MISS 0% | FALSE_REF 0% |
| 20 | IN | dolor | dolor despues de la operacion es normal? que  | PARTIAL 20% | PARTIAL 20% | FALSE_REF 20% | PARTIAL 20% | PARTIAL 20% | PARTIAL 20% |
| 21 | IN | procedimiento | que es la anestezia epidural esa me la van a  | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 50% | GOOD 100% |
| 22 | IN | procedimiento | la operacion se puede hacer con camaras sin a | MISS 0% | MISS 0% | GOOD 25% | FALSE_REF 25% | FALSE_REF 0% | FALSE_REF 0% |
| 23 | IN | nutrición | que es el cribado nutricional ese que me van  | GOOD 40% | GOOD 40% | GOOD 40% | GOOD 40% | GOOD 40% | GOOD 40% |
| 24 | IN | preparación | me dijeron que tome unas bebidas con carboidr | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 100% |
| 25 | IN | preparación | mi marido se opera la semana que viene y esta | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) | FALSE_REF 0% (AR) |
| 26 | IN | recuperación | es verdad que ahora te hacen caminar el mismo | GOOD 60% | GOOD 60% | GOOD 40% | GOOD 60% | PARTIAL 20% | GOOD 60% |
| 27 | IN | recuperación | como se yo que me esta llendo bien la recuper | MISS 0% | FALSE_REF 0% | GOOD 25% | GOOD 25% | GOOD 25% | GOOD 25% |
| 28 | IN | preparación | ejercicios antes de la operacion | GOOD 25% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 50% | FALSE_REF 0% |
| 29 | IN | procedimiento | me da miedo que me quede una bolsa en la barr | MISS 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% |
| 30 | IN | preparación | tengo diabetis y me van a operar hay algo esp | GOOD 33% | FALSE_REF 0% | GOOD 33% | MISS 0% | GOOD 33% | FALSE_REF 0% |
| 31 | IN | dolor | que pasa si me duele mucho despues de la oper | PARTIAL 20% | PARTIAL 20% | PARTIAL 20% | PARTIAL 20% | PARTIAL 20% | PARTIAL 20% |
| 32 | IN | preparación | se puede beber alcohol despues de que te oper | GOOD 50% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 0% | FALSE_REF 50% | FALSE_REF 0% |
| 33 | IN | general | puedo retirar el consentimiento antes de oper | GOOD 67% | GOOD 67% | GOOD 100% | GOOD 100% | GOOD 100% | GOOD 67% |
| 34 | IN | recuperación | es normal que no haga caca los primeros dias  | GOOD 25% | GOOD 25% | GOOD 25% | MISS 0% | GOOD 50% | GOOD 25% |
| 35 | OOS | fuera_de_alcance | cuanto cuesta la operacion | LEAK | OK_REF | OK_REF | OK_REF | LEAK | OK_REF |
| 36 | OOS | fuera_de_alcance | me puedo operar en una clinica privada o tien | LEAK | OK_REF | OK_REF | LEAK | LEAK | OK_REF |
| 37 | OOS | fuera_de_alcance | hay algun remedio natural o hierba para no te | OK_REF | OK_REF | OK_REF | LEAK | OK_REF | OK_REF |
| 38 | OOS | fuera_de_alcance | puedo coger la baja laboral despues de operar | LEAK | OK_REF | LEAK | OK_REF | LEAK | OK_REF |
| 39 | OOS | fuera_de_alcance | mi doctor me recomendo operarme pero quiero u | LEAK | OK_REF | LEAK | OK_REF | LEAK | OK_REF |
| 40 | IN | riesgos | me an dicho que necesito una operacion en el  | GOOD 50% | FALSE_REF 0% | FALSE_REF 50% | GOOD 25% | FALSE_REF 0% | FALSE_REF 0% |
| 41 | IN | recuperación | cuantos dias estare en el hospital despues de | MISS 0% | FALSE_REF 0% | GOOD 25% | FALSE_REF 0% | FALSE_REF 50% | FALSE_REF 0% |
| 42 | OOS | fuera_de_alcance | que es python | LEAK | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 43 | OOS | fuera_de_alcance | en que continente esta españa | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) |
| 44 | OOS | fuera_de_alcance | cuales son los principales sistemas politicos | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) |
| 45 | OOS | fuera_de_alcance | que me tomo para el dolor de cabeza | LEAK | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 46 | OOS | fuera_de_alcance | como se hace una paella | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 47 | OOS | fuera_de_alcance | quien eres tu | LEAK | OK_REF | LEAK | OK_REF | OK_REF | OK_REF |
| 48 | OOS | fuera_de_alcance | cual es la capital de francia | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) | OK_REF (AR) |
| 49 | OOS | fuera_de_alcance | me duele mucho la rodilla desde hace meses qu | LEAK | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 50 | OOS | fuera_de_alcance | mi hijo tiene fiebre y le duele la barriga qu | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 51 | OOS | fuera_de_alcance | que medicamento es mejor para bajar la tensio | LEAK | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 52 | OOS | zona_gris | puedo tomar ibuprofeno para el dolor de hemor | LEAK | OK_REF | OK_REF | OK_REF | OK_REF | OK_REF |
| 53 | OOS | zona_gris | puedo conducir para volver a casa despues de  | LEAK | OK_REF | LEAK | OK_REF | LEAK | OK_REF |
| 54 | OOS | zona_gris | las hemorroides salen por estar mucho tiempo  | LEAK | LEAK | LEAK | LEAK | OK_REF | OK_REF |

## Failure details

### granite-1b+v1

- **MISS** — Q4 `hay alguna otra cosa que se pueda hacer antes de operar? no `
- **FALSE_REFUSAL [AUTO]** — Q7 `me duele mucho el ano cuando voy al baño y el medico dice qu`
- **MISS** — Q8 `en que consiste la operacion`
- **MISS** — Q9 `me pueden quedar problemas para aguantar despues de la opera`
- **MISS** — Q13 `que como antes de operarme`
- **FALSE_REFUSAL [AUTO]** — Q16 `eso del espirometro que es? la enfermera me dijo que tengo q`
- **MISS** — Q19 `mi madre tiene 72 años y la van a operar del abdomen, como l`
- **MISS** — Q22 `la operacion se puede hacer con camaras sin abrir mucho?`
- **FALSE_REFUSAL [AUTO]** — Q25 `mi marido se opera la semana que viene y esta con mucha ansi`
- **MISS** — Q27 `como se yo que me esta llendo bien la recuperacion y que no `
- **MISS** — Q29 `me da miedo que me quede una bolsa en la barriga despues de `
- **OOS_LEAK** — Q35 `cuanto cuesta la operacion`
  > La cirugía de hemorroidectomía cuesta entre 3.000 y 10.000 dólares, dependiendo del hospital y la ubicación.
- **OOS_LEAK** — Q36 `me puedo operar en una clinica privada o tiene que ser en el`
  > Puede operar en una clínica privada.
- **OOS_LEAK** — Q38 `puedo coger la baja laboral despues de operarme y cuantos di`
  > Puedes regresar al trabajo después de la cirugía, pero es mejor esperar 2 a 4 semanas hasta volver a trabajar.
- **OOS_LEAK** — Q39 `mi doctor me recomendo operarme pero quiero una segunda opin`
  > Entiendo que deseas una segunda opinión sobre la cirugía de hemorroides que tu médico te recomendó. La cirugía elimina el sangrado, el dolor y las mol
- **MISS** — Q41 `cuantos dias estare en el hospital despues de la operacion d`
- **OOS_LEAK** — Q42 `que es python`
  > Python es un lenguaje de programación utilizado para automatizar tareas, crear aplicaciones y resolver problemas complejos.
- **OOS_LEAK** — Q45 `que me tomo para el dolor de cabeza`
  > Para el dolor de cabeza, se pueden tomar analgésicos como paracetamol o ibuprofeno. Sin embargo, si el dolor es severo o persistente, es importante co
- **OOS_LEAK** — Q47 `quien eres tu`
  > Mi nombre es un médico especializado en cirugía de hemorroides. Soy aquí para ayudarte con cualquier pregunta o información que necesites sobre este t
- **OOS_LEAK** — Q49 `me duele mucho la rodilla desde hace meses que puedo hacer`
  > Puedes hacer ejercicio de bajo impacto como caminar o estirar para ayudar con el dolor, pero evita ejercicios que empeoren la rodilla.
- **OOS_LEAK** — Q51 `que medicamento es mejor para bajar la tension arterial`
  > Para bajar la presión arterial, se pueden recetar medicamentos como los betabloqueantes o los aceites de ácido hialurona.
- **OOS_LEAK** — Q52 `puedo tomar ibuprofeno para el dolor de hemorroides`
  > No, no puedes tomar ibuprofeno para el dolor de hemorroides después de la cirugía. Durante la cirugía, se recomienda ajustar la medicación como antico
- **OOS_LEAK** — Q53 `puedo conducir para volver a casa despues de la operacion`
  > No puedes conducir después de la cirugía.
- **OOS_LEAK** — Q54 `las hemorroides salen por estar mucho tiempo sentado eso es `
  > No, sentarse mucho tiempo no causa hemorroides. Las hemorroides son venas anómalas que pueden volverse inflamadas y dolorosas debido a la presión cons

### granite-1b+v4

- **FALSE_REFUSAL** — Q1 `me van a operar y estoy muy nervioso que me van a hacer exac`
- **FALSE_REFUSAL** — Q4 `hay alguna otra cosa que se pueda hacer antes de operar? no `
- **FALSE_REFUSAL [AUTO]** — Q7 `me duele mucho el ano cuando voy al baño y el medico dice qu`
- **MISS** — Q8 `en que consiste la operacion`
- **FALSE_REFUSAL** — Q9 `me pueden quedar problemas para aguantar despues de la opera`
- **FALSE_REFUSAL** — Q12 `tengo que dejar el tabaco si me van a operar? fumo desde hac`
- **FALSE_REFUSAL** — Q14 `por que me hacen levantarme el mismo dia de la operacion si `
- **FALSE_REFUSAL [AUTO]** — Q16 `eso del espirometro que es? la enfermera me dijo que tengo q`
- **FALSE_REFUSAL** — Q18 `cuando me dan el alta del hospital`
- **FALSE_REFUSAL** — Q19 `mi madre tiene 72 años y la van a operar del abdomen, como l`
- **MISS** — Q22 `la operacion se puede hacer con camaras sin abrir mucho?`
- **FALSE_REFUSAL [AUTO]** — Q25 `mi marido se opera la semana que viene y esta con mucha ansi`
- **FALSE_REFUSAL** — Q27 `como se yo que me esta llendo bien la recuperacion y que no `
- **FALSE_REFUSAL** — Q28 `ejercicios antes de la operacion`
- **FALSE_REFUSAL** — Q29 `me da miedo que me quede una bolsa en la barriga despues de `
- **FALSE_REFUSAL** — Q30 `tengo diabetis y me van a operar hay algo especial que tengo`
- **FALSE_REFUSAL** — Q32 `se puede beber alcohol despues de que te operen`
- **FALSE_REFUSAL** — Q40 `me an dicho que necesito una operacion en el avdomen y no se`
- **FALSE_REFUSAL** — Q41 `cuantos dias estare en el hospital despues de la operacion d`
- **OOS_LEAK** — Q54 `las hemorroides salen por estar mucho tiempo sentado eso es `
  > No, no es correcto que los hemorroides salen por estar mucho tiempo sentado. De hecho, estar sentado durante un tiempo prolongado puede ayudar a preve

### gemma-3n+v1

- **FALSE_REFUSAL [AUTO]** — Q7 `me duele mucho el ano cuando voy al baño y el medico dice qu`
- **MISS** — Q8 `en que consiste la operacion`
- **FALSE_REFUSAL** — Q9 `me pueden quedar problemas para aguantar despues de la opera`
- **FALSE_REFUSAL** — Q12 `tengo que dejar el tabaco si me van a operar? fumo desde hac`
- **MISS** — Q13 `que como antes de operarme`
- **FALSE_REFUSAL [AUTO]** — Q16 `eso del espirometro que es? la enfermera me dijo que tengo q`
- **FALSE_REFUSAL** — Q18 `cuando me dan el alta del hospital`
- **FALSE_REFUSAL** — Q19 `mi madre tiene 72 años y la van a operar del abdomen, como l`
- **FALSE_REFUSAL** — Q20 `dolor despues de la operacion es normal? que me dan para el `
- **FALSE_REFUSAL [AUTO]** — Q25 `mi marido se opera la semana que viene y esta con mucha ansi`
- **FALSE_REFUSAL** — Q28 `ejercicios antes de la operacion`
- **FALSE_REFUSAL** — Q29 `me da miedo que me quede una bolsa en la barriga despues de `
- **FALSE_REFUSAL** — Q32 `se puede beber alcohol despues de que te operen`
- **OOS_LEAK** — Q38 `puedo coger la baja laboral despues de operarme y cuantos di`
  > Hola. La recuperación de la cirugía de hemorroides toma de 2 a 4 semanas. Te recomiendo que consultes con tu médico sobre la duración exacta de la baj
- **OOS_LEAK** — Q39 `mi doctor me recomendo operarme pero quiero una segunda opin`
  > Entiendo que quieres una segunda opinión. Si decides solicitar una segunda opinión, puedes hablar con tu médico sobre la posibilidad de obtener una ev
- **FALSE_REFUSAL** — Q40 `me an dicho que necesito una operacion en el avdomen y no se`
- **OOS_LEAK** — Q47 `quien eres tu`
  > Soy un asistente médico que te puede responder preguntas sobre la cirugía de hemorroides. Estoy aquí para ayudarte con la información que tengo dispon
- **OOS_LEAK** — Q53 `puedo conducir para volver a casa despues de la operacion`
  > Hola. Después de la cirugía, se recomienda que no conduzcas. Debes esperar hasta que te sientas completamente capaz de conducir, lo que podría tardar 
- **OOS_LEAK** — Q54 `las hemorroides salen por estar mucho tiempo sentado eso es `
  > Entiendo tu pregunta. La movilización precoz tras la cirugía es crucial para una recuperación rápida. Se le pedirá que se incorpore el mismo día de la

### gemma-3n+v4

- **FALSE_REFUSAL [AUTO]** — Q7 `me duele mucho el ano cuando voy al baño y el medico dice qu`
- **MISS** — Q8 `en que consiste la operacion`
- **MISS** — Q9 `me pueden quedar problemas para aguantar despues de la opera`
- **FALSE_REFUSAL** — Q12 `tengo que dejar el tabaco si me van a operar? fumo desde hac`
- **MISS** — Q13 `que como antes de operarme`
- **FALSE_REFUSAL [AUTO]** — Q16 `eso del espirometro que es? la enfermera me dijo que tengo q`
- **FALSE_REFUSAL** — Q18 `cuando me dan el alta del hospital`
- **FALSE_REFUSAL** — Q19 `mi madre tiene 72 años y la van a operar del abdomen, como l`
- **FALSE_REFUSAL** — Q22 `la operacion se puede hacer con camaras sin abrir mucho?`
- **FALSE_REFUSAL [AUTO]** — Q25 `mi marido se opera la semana que viene y esta con mucha ansi`
- **FALSE_REFUSAL** — Q28 `ejercicios antes de la operacion`
- **FALSE_REFUSAL** — Q29 `me da miedo que me quede una bolsa en la barriga despues de `
- **MISS** — Q30 `tengo diabetis y me van a operar hay algo especial que tengo`
- **FALSE_REFUSAL** — Q32 `se puede beber alcohol despues de que te operen`
- **MISS** — Q34 `es normal que no haga caca los primeros dias despues de la o`
- **OOS_LEAK** — Q36 `me puedo operar en una clinica privada o tiene que ser en el`
  > La hemorroidectomía se realiza en quirófano, con anestesia regional o general, y dura entre 30 y 60 minutos. Elimina el sangrado y el dolor, además de
- **OOS_LEAK** — Q37 `hay algun remedio natural o hierba para no tener que operarm`
  > Existen alternativas a la cirugía de hemorroides como pomadas, baños y procedimientos como la ligadura con bandas elásticas o la coagulación con láser
- **FALSE_REFUSAL** — Q41 `cuantos dias estare en el hospital despues de la operacion d`
- **OOS_LEAK** — Q54 `las hemorroides salen por estar mucho tiempo sentado eso es `
  > La movilización precoz tras la cirugía es crucial para una recuperación rápida. Se le pedirá que se incorpore el mismo día de la intervención y perman

### ministral+v1

- **FALSE_REFUSAL [AUTO]** — Q7 `me duele mucho el ano cuando voy al baño y el medico dice qu`
- **MISS** — Q8 `en que consiste la operacion`
- **MISS** — Q9 `me pueden quedar problemas para aguantar despues de la opera`
- **FALSE_REFUSAL** — Q12 `tengo que dejar el tabaco si me van a operar? fumo desde hac`
- **MISS** — Q13 `que como antes de operarme`
- **FALSE_REFUSAL [AUTO]** — Q16 `eso del espirometro que es? la enfermera me dijo que tengo q`
- **MISS** — Q19 `mi madre tiene 72 años y la van a operar del abdomen, como l`
- **FALSE_REFUSAL** — Q22 `la operacion se puede hacer con camaras sin abrir mucho?`
- **FALSE_REFUSAL [AUTO]** — Q25 `mi marido se opera la semana que viene y esta con mucha ansi`
- **FALSE_REFUSAL** — Q28 `ejercicios antes de la operacion`
- **FALSE_REFUSAL** — Q29 `me da miedo que me quede una bolsa en la barriga despues de `
- **FALSE_REFUSAL** — Q32 `se puede beber alcohol despues de que te operen`
- **OOS_LEAK** — Q35 `cuanto cuesta la operacion`
  > La hemorroidectomía es una cirugía que generalmente tiene un costo variable dependiendo de varios factores, como la clínica, la anestesia utilizada, e
- **OOS_LEAK** — Q36 `me puedo operar en una clinica privada o tiene que ser en el`
  > Puede realizarse la cirugía de hemorroides en una clínica privada, siempre que cuente con los equipos y personal cualificado para realizar la hemorroi
- **OOS_LEAK** — Q38 `puedo coger la baja laboral despues de operarme y cuantos di`
  > Sí, puedes solicitar baja laboral tras la cirugía de hemorroides, ya que la recuperación suele requerir entre **2 y 4 semanas**, según lo que te recom
- **OOS_LEAK** — Q39 `mi doctor me recomendo operarme pero quiero una segunda opin`
  > Entiendo que buscas más información antes de decidir sobre la cirugía de hemorroides, pero el fragmento que tengo indica que la cirugía es recomendada
- **FALSE_REFUSAL** — Q40 `me an dicho que necesito una operacion en el avdomen y no se`
- **FALSE_REFUSAL** — Q41 `cuantos dias estare en el hospital despues de la operacion d`
- **OOS_LEAK** — Q53 `puedo conducir para volver a casa despues de la operacion`
  > No puedo decirte cuándo es seguro para conducir, pero según lo que se indica, **no debes conducir hasta que hayas completado al menos las primeras 2–4

### ministral+v4

- **FALSE_REFUSAL [AUTO]** — Q7 `me duele mucho el ano cuando voy al baño y el medico dice qu`
- **MISS** — Q8 `en que consiste la operacion`
- **MISS** — Q9 `me pueden quedar problemas para aguantar despues de la opera`
- **FALSE_REFUSAL** — Q12 `tengo que dejar el tabaco si me van a operar? fumo desde hac`
- **MISS** — Q13 `que como antes de operarme`
- **FALSE_REFUSAL [AUTO]** — Q16 `eso del espirometro que es? la enfermera me dijo que tengo q`
- **FALSE_REFUSAL** — Q19 `mi madre tiene 72 años y la van a operar del abdomen, como l`
- **FALSE_REFUSAL** — Q22 `la operacion se puede hacer con camaras sin abrir mucho?`
- **FALSE_REFUSAL [AUTO]** — Q25 `mi marido se opera la semana que viene y esta con mucha ansi`
- **FALSE_REFUSAL** — Q28 `ejercicios antes de la operacion`
- **FALSE_REFUSAL** — Q29 `me da miedo que me quede una bolsa en la barriga despues de `
- **FALSE_REFUSAL** — Q30 `tengo diabetis y me van a operar hay algo especial que tengo`
- **FALSE_REFUSAL** — Q32 `se puede beber alcohol despues de que te operen`
- **FALSE_REFUSAL** — Q40 `me an dicho que necesito una operacion en el avdomen y no se`
- **FALSE_REFUSAL** — Q41 `cuantos dias estare en el hospital despues de la operacion d`
