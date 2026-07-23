# Informe ciego — reescritura de `cirugia-abdominal.md` → `cirugia-abdominal.v4.md`

Sesión independiente. No he abierto `eval/` (salvo para escribir este fichero),
`reports/`, `ARRANQUE.md`, `docs/corpus_guidelines.md`, `tools/audit_*.py` ni
ningún `.xlsx`. Los únicos ficheros leídos son
`docs/corpus_rewrite_brief_cirugia.md` y `corpus/markdown/cirugia-abdominal.md`.

---

## 1. Transformaciones aplicadas

**Inventario de hechos.** Antes de reescribir extraje 84 hechos atómicos del
original y comprobé uno a uno que estuvieran todos en la v4 y que la v4 no
tuviera ninguno más. Cifras, plazos y nombres de técnica se copiaron literales
(2 horas, 200–400 ml, 6 u 8 horas, 24 o 48 horas, 1 ó 2 horas, 3 a 5 días, 2 ó 3
días, 1 o 2 días, «una semana antes», ACP, catéter epidural, laparoscopio,
albúmina, Unidad de Recuperación Postanestésica).

**R6 / longitud de párrafo.** El original tenía 8 párrafos de 4–7 hechos
encadenados (el peor: el de calmantes/ACP, con 7). La v4 no tiene ningún párrafo
de más de un hecho principal: cada frase es un párrafo. Es el cambio de forma más
visible y el que ataca el defecto de «recortar la primera frase y soltar el
resto».

**R5 / sujeto explícito.** Eliminé todos los impersonales y pasivas del original:
«le explican» → *el cirujano le explica*; «le informan» → *el anestesiólogo le
informa*; «se le pedirá consentimiento» → *el equipo le pedirá su consentimiento*;
«le darán» (bebida y pastillas) → *el equipo le dará / le da*; «será trasladado»
→ *el equipo le traslada*; «se introduce un laparoscopio / se insufla gas» → *el
cirujano le introduce / le insufla*; «se coloca una bomba de ACP» → *el equipo le
coloca*; «el dolor se trata con calmantes» → *el equipo le trata el dolor con*;
«se recomienda beber y comer» → *usted debe beber y comer*; «se hace de forma
progresiva» → *usted reanuda la alimentación de forma progresiva*.

Criterio para elegir actor: solo actores que el propio documento nombra
(cirujano, anestesiólogo, personal de enfermería, médico, médico de cabecera,
«el equipo», «los profesionales», familiares/cuidadores, usted). Donde el
original no dice quién, usé **«el equipo»** o **«los profesionales»**, que son
los genéricos que el propio original ya emplea («el equipo programa», «Comunique
al equipo», «Los profesionales aseguran»). No inventé ningún rol nuevo.

**R2 / acto en la oración principal.** Reescribí los casos donde el acto colgaba
de una subordinada o de un marco: «Es usted quien decide» → *usted decide sobre
su tratamiento y usted firma el consentimiento*; «Hoy se recomienda beber y
comer» → *usted debe beber y comer* (evité «el equipo le recomienda que beba»,
que dejaría «recomienda» en la principal y perdería el acto); «Controlar el dolor
es importante» conserva forma nominal porque el hecho *es* la importancia, pero
la instrucción accionable («no debe aguantarse el dolor») va en principal
independiente. También «no hay peligro de sobredosis» pasa a principal con sujeto
(*usted no corre peligro de sobredosis al darse dosis adicionales*).

**R1 / autosuficiencia.** Cada frase repite su ancla: «después de la operación»,
«antes de la operación», «durante la cirugía laparoscópica», «si usted tiene
diabetes», «cuando su grado de ansiedad y de temor es elevado». El caso crítico
es la **lista de premedicación** del original: la condición («cuando el grado de
ansiedad y temor sea elevado») encabezaba y los dos puntos colgaban sin verbo ni
condición. En la v4 la lista desaparece y cada pastilla es una frase completa que
**repite la condición entera**. Es feo y es deliberado (R1 > R7).

Eliminé toda estructura de lista del documento: era el único sitio donde el
contenido dependía de una cabecera. La enumeración de líquidos permitidos queda
dentro de la frase, con la condición («hasta 2 horas antes de la operación»)
repetida dentro de la propia frase enumerativa.

**R3 / comparaciones implícitas y categorías generales.** Ver §2 y §4.

**R7.** No añadí fórmulas de cortesía, ni encuadres, ni cierres. Sí hay
repetición de anclas y de sujetos, que R7 no prohíbe.

---

## 2. Preguntas de paciente previstas (R3) y dónde las responde el texto

Derivadas del procedimiento, no de ningún conjunto de evaluación.

| # | Pregunta prevista | Sección de la v4 que la responde con una frase |
|---|---|---|
| 1 | ¿Qué es una cirugía mayor? ¿es arriesgada? | Qué es la cirugía mayor abdominal |
| 2 | ¿Qué anestesia me van a poner? | Qué es la cirugía mayor abdominal («suele requerir anestesia general») |
| 3 | ¿Cuánto tardaré en recuperarme? | Qué es la cirugía mayor abdominal («varios días o varias semanas») |
| 4 | ¿Qué órganos se operan en cirugía abdominal? | Qué es la cirugía mayor abdominal |
| 5 | ¿Qué son los cuidados perioperatorios y para qué sirven? | Qué son los cuidados perioperatorios |
| 6 | ¿Cuándo empiezan y cuándo terminan esos cuidados? | Qué son los cuidados perioperatorios (hasta el alta) |
| 7 | ¿Quién me explica los riesgos de la operación y cuándo? | Quién le informa y cuándo |
| 8 | ¿Quién decide si me opero? ¿tengo que firmar algo? | Quién le informa y cuándo |
| 9 | ¿Cuándo veo al anestesista y qué me cuenta? | Quién le informa y cuándo |
| 10 | ¿Me dan información por escrito? ¿de qué? | Quién le informa y cuándo (folleto) |
| 11 | ¿Con quién resuelvo dudas que me surjan después? | Quién le informa y cuándo |
| 12 | ¿Cuándo es la evaluación preoperatoria? | La evaluación preoperatoria («una semana antes») |
| 13 | ¿Me hacen las mismas pruebas que a todo el mundo? | La evaluación preoperatoria |
| 14 | ¿Tengo que firmar/consentir las pruebas? | La evaluación preoperatoria |
| 15 | ¿Qué es el cribado nutricional? ¿me lo harán a mí? | El cribado nutricional (a todos) |
| 16 | ¿Quién me hace el cribado nutricional? | El cribado nutricional (enfermería) |
| 17 | ¿Qué miran en el cribado nutricional? | El cribado nutricional |
| 18 | ¿Por qué importa estar bien nutrido? | El cribado nutricional |
| 19 | ¿Qué pasa si salgo desnutrido? | El cribado nutricional (valoración completa + tratamiento) |
| 20 | ¿Hasta cuándo puedo beber antes de la operación? | Qué puede beber antes de la operación (2 horas) |
| 21 | ¿Qué líquidos puedo beber? | Qué puede beber antes de la operación |
| 22 | ¿Me darán alguna bebida especial antes? ¿cuánta? | Qué puede beber antes de la operación (200–400 ml) |
| 23 | ¿Para qué sirve esa bebida con carbohidratos? | Qué puede beber antes de la operación |
| 24 | Tengo diabetes, ¿qué hago con las bebidas? | Si usted tiene diabetes |
| 25 | ¿Tengo que controlar la glucosa antes? ¿quién me ayuda? | Si usted tiene diabetes |
| 26 | ¿Qué es la cirugía mínimamente invasiva? | Cirugía mínimamente invasiva y cirugía convencional |
| 27 | ¿Duele menos la laparoscopia que la cirugía abierta? | Cirugía mínimamente invasiva y cirugía convencional (frase explícita nueva en forma) |
| 28 | ¿Qué es un laparoscopio? | La cirugía laparoscópica |
| 29 | ¿Me pueden acabar haciendo una incisión más grande? | La cirugía laparoscópica |
| 30 | ¿Por qué me duele el abdomen después de la laparoscopia? ¿cuánto dura? | La cirugía laparoscópica (gas, 1–2 días) |
| 31 | Estoy muy nervioso, ¿me darán algo? | Medicación antes de la operación (premedicación) |
| 32 | ¿Cuándo tomo esas pastillas? | Premedicación (noche antes; 1 ó 2 h antes) |
| 33 | ¿Es normal estar tan dormido y no acordarme de nada? | Premedicación |
| 34 | ¿A quién le digo que estoy incómodo o que necesito algo? | Premedicación («comunicar al equipo») |
| 35 | ¿Adónde me llevan al salir del quirófano? | Adónde le llevan a usted al salir del quirófano |
| 36 | ¿Tengo que aguantarme el dolor? | Control del dolor |
| 37 | ¿Quién decide mi tratamiento del dolor y dónde? | Control del dolor (anestesiólogo, URPA) |
| 38 | ¿Cuándo dolerá más? | Control del dolor (primeras 24 h) |
| 39 | ¿Cada cuánto me darán calmantes? ¿y si me duele entre dosis? | Control del dolor (6–8 h + rescate a demanda) |
| 40 | ¿Me darán los calmantes en vena o en pastillas? | Control del dolor (IV 24–48 h, luego oral) |
| 41 | ¿Qué es la bomba de ACP? ¿me puedo pasar de dosis? | Control del dolor |
| 42 | ¿Quién vigila mi dolor durante el ingreso? | Control del dolor (enfermería, 24 h) |
| 43 | ¿Los calmantes tienen efectos secundarios? | Control del dolor (frase general aislada) |
| 44 | ¿Qué es la epidural y cuándo se usa? | La analgesia epidural |
| 45 | ¿Dónde y quién me pone el catéter epidural? | La analgesia epidural (quirófano, anestesiólogo) |
| 46 | ¿Qué efectos puede darme la epidural? ¿hay que tratarlos? | La analgesia epidural |
| 47 | ¿Por qué no me funciona el intestino tras la operación? | Reanudar la alimentación |
| 48 | ¿Cuánto tarda en volverme el tránsito intestinal? | Reanudar la alimentación (horas; 3–5 días) |
| 49 | ¿Cuándo puedo volver a beber y a comer? | Reanudar la alimentación (primeras 24 h) |
| 50 | ¿Qué tomo primero y en qué postura? | Reanudar la alimentación (sorbos de agua, semisentado) |
| 51 | ¿Y si me dan náuseas o vomito? | Reanudar la alimentación |
| 52 | ¿Se me puede abrir la herida por comer pronto? | Reanudar la alimentación |
| 53 | ¿Cuándo haré de vientre por primera vez? ¿me dolerá? ¿sangraré? | Reanudar la alimentación (2–3 días) |
| 54 | ¿Por qué tengo que levantarme si me encuentro mal? | Levantarse de la cama (riesgos de la inmovilización) |
| 55 | ¿Cuándo me puedo sentar? ¿cuándo tengo que caminar? | Levantarse de la cama (mismo día / día siguiente) |
| 56 | ¿Y si me duele al levantarme? ¿y las sondas y drenajes? | Levantarse de la cama |
| 57 | ¿Qué papel tiene mi familia? ¿por qué les informan a ellos? | Colaboración de sus familiares y cuidadores |

### Preguntas previsibles SIN material en el documento (no rellenadas)

No inventé ninguna respuesta para éstas. Son huecos del corpus, no de la
redacción:

1. ¿Cuánto dura la operación?
2. ¿Cuántos días estaré ingresado? (solo hay «la recuperación puede llevar días
   o semanas», que no es lo mismo que la estancia)
3. ¿Hasta cuándo puedo comer **sólidos** antes de la operación? El documento
   solo regula los **líquidos** (2 horas). Es probablemente el hueco más
   preguntado del documento.
4. ¿Qué hago con mi medicación habitual (anticoagulantes, antihipertensivos…)?
5. ¿Tengo que dejar de fumar o de beber alcohol antes?
6. ¿Tengo que ducharme, depilarme o rasurarme la zona antes?
7. ¿Cómo cuido la herida? ¿cuándo puedo ducharme? ¿cuándo me quitan puntos o
   grapas?
8. ¿Cuándo me quitan las sondas y los drenajes? (solo se dice que los
   profesionales reducen su uso)
9. ¿Iré a la UCI? ¿cuánto tiempo estaré en la Unidad de Recuperación
   Postanestésica?
10. ¿Qué signos de alarma debo vigilar y a quién llamo al llegar a casa?
11. ¿Cuándo puedo volver al trabajo, conducir, hacer ejercicio o coger peso?
12. ¿Puede acompañarme un familiar? ¿puede quedarse a dormir? ¿horarios de
    visita?
13. ¿Qué pruebas concretas incluye la evaluación preoperatoria además del
    cribado nutricional? (el documento anuncia la categoría y solo instancia una)
14. ¿Para **qué** cirugías se usa la bomba de ACP y para cuáles la epidural? (el
    documento dice «algunas» y «determinadas» sin nombrarlas)
15. ¿Puedo cambiar de opinión después de firmar el consentimiento?
16. ¿Necesitaré transfusión? ¿me pondrán sonda urinaria?

---

## 3. Contenido movido de sección por R4

1. **Diabetes y control de glucosa** — de *«Bebidas con carbohidratos antes de la
   operación»* (segundo párrafo) a una **sección propia, «Si usted tiene
   diabetes»**. Motivo: era contenido de una condición del paciente enterrado en
   una sección cuyo título no contiene la palabra «diabetes»; el vocabulario que
   el paciente usaría no aparecía en ninguna cabecera.
2. **Traslado a la Unidad de Recuperación Postanestésica** — del primer párrafo
   de *«Control del dolor»* a una **sección propia, «Adónde le llevan a usted al
   salir del quirófano»**. Motivo: es un hecho del itinerario postoperatorio, no
   del dolor. En la sección de dolor se conserva la **prescripción del
   anestesiólogo**, repitiendo el nombre de la unidad como ancla (R1) pero sin
   repetir el hecho del traslado (R4: cada hecho una sola vez).
3. **«Todos los fármacos para el dolor pueden producir efectos no deseados»** —
   estaba pegado al final del párrafo de la epidural, lo que lo hacía leer como
   un hecho de la epidural. Pasa a **frase independiente al final de «Control del
   dolor»**, antes de la sección de epidural, como enunciado general; los efectos
   concretos de la epidural quedan en la sección de epidural.
4. **La analgesia epidural** — de párrafo final de *«Control del dolor»* a
   **sección propia**. No cambia de tema, pero la separa del bloque de
   calmantes/ACP y evita que el párrafo largo del dolor absorba su vocabulario.
5. **La evaluación preoperatoria** vs. **el cribado nutricional** — el original
   los mezclaba en una sola sección titulada «Cribado nutricional
   preoperatorio», donde el primer párrafo hablaba de la evaluación preoperatoria
   en general (fecha, consentimiento, tipos de prueba) y solo el segundo del
   cribado. Los separé en **dos secciones**, y añadí en forma —no en contenido—
   el enlace explícito entre la categoría («pruebas que se hacen a todos») y su
   única instancia (el cribado nutricional), que en el original solo era
   deducible.
6. **La cirugía laparoscópica** — de segundo párrafo de *«Cirugía mínimamente
   invasiva»* a **sección propia**, para que el bloque comparativo (invasiva vs.
   abierta) y el bloque descriptivo (laparoscopio, gas, molestia) no compartan
   párrafo ni compitan por la misma cabecera.
7. **«Qué son los cuidados perioperatorios»** — segundo párrafo de la sección
   inicial, ahora **sección propia**, porque define un término distinto del que
   titula la sección.

**No moví**: «los profesionales aseguran una buena analgesia» sigue en
*«Levantarse de la cama»* pese a ser vocabulario de analgesia, porque su
finalidad declarada en el original es la deambulación. Ver §5.

---

## 4. Dónde chocaron las reglas y cómo lo resolví

**R1 vs. R7 (tres veces).** La premedicación repite «Cuando su grado de ansiedad
y de temor es elevado» en tres frases consecutivas; la sección de diabetes repite
«Si usted tiene diabetes» en dos; la de líquidos repite «hasta 2 horas antes de la
operación» en dos frases seguidas. Se lee mal. Aplicado el criterio del brief:
**gana R1**.

**R2 vs. fidelidad de modalidad.** «Hoy se recomienda beber y comer lo antes
posible» → «Usted debe beber y comer lo antes posible». Cualquier forma que
conserve «se recomienda» con sujeto explícito («el equipo le recomienda que
beba…») pone *recomendar* en la principal y manda el acto a la subordinada, que
es exactamente el fallo que R2 describe. Elegí el imperativo con sujeto,
asumiendo un ligero refuerzo de modalidad (recomendación → deber). El original ya
usa «debe» para la instrucción paralela de deambulación («Debe comenzar a
caminar lo antes posible»), así que el registro es el mismo. **Lo anoto como la
única desviación de modalidad de toda la reescritura.**

**R4 vs. R1 (Unidad de Recuperación Postanestésica).** R4 pide enunciar cada
hecho una vez y en su sitio; R1 pide que la frase del anestesiólogo lleve dentro
su alcance («¿dónde me prescribe el tratamiento?»). Resolución: el **hecho** del
traslado vive solo en su sección; el **nombre** de la unidad se repite como ancla
en la sección de dolor. Repetir un ancla no es repetir un hecho.

**R4 vs. R5 (analgesia en la sección de deambulación).** «Los profesionales
aseguran una buena analgesia y reducen el uso de sondas y drenajes» es una
instrucción **al equipo**, y es justo el patrón que R5 marca como el defecto más
grave (que el sistema se lo reasigne al paciente). No la moví a la sección de
dolor porque su finalidad declarada es la deambulación, pero la partí en dos
frases, ambas con **«los profesionales que le atienden»** como sujeto explícito y
en posición inicial, y ambas con la finalidad («para favorecer su deambulación»)
como subordinada final. El sujeto no puede desprenderse sin romper la frase.

**R3 vs. «no añadir» (comparación implícita).** El original describe la cirugía
abierta («incisiones mayores, que pueden producir más dolor y alargar la
recuperación») y deja que el lector deduzca la propiedad de la mínimamente
invasiva. Añadí **una frase** que afirma directamente el término contrario:
«La cirugía mínimamente invasiva… puede producirle menos dolor y una recuperación
más corta que la cirugía convencional abierta». No es un hecho nuevo: es el mismo
hecho comparativo enunciado desde el otro término. El brief marca este patrón
como uno de los defectos a corregir.

---

## 5. Lo que dejé fuera por dudar entre reformulación y añadido

1. **«La cirugía convencional (abierta) sí abre las cavidades del organismo.»**
   El original solo dice que la mínimamente invasiva «evita abrir las cavidades»
   y que la abierta «usa incisiones mayores». Afirmar que la abierta abre las
   cavidades es la complementaria obvia, y probablemente cierta, pero el
   documento no la enuncia y no es un simple cambio de término comparativo (como
   sí lo es el del dolor). **Fuera.**
2. **«Solo» en la premedicación.** «Cuando el grado de ansiedad y temor sea
   elevado, le darán medicación» no equivale a «solo cuando». Escribir «el equipo
   le da esta medicación **solo** cuando su ansiedad es elevada» habría hecho la
   condición inequívoca (bueno para R1) pero habría añadido exclusividad. Repetí
   la condición en cada frase sin «solo». **Fuera.**
3. **Quién no puede beber hasta 2 horas antes.** «La **mayoría** de las personas
   pueden beber…» implica que hay personas que no. El documento nunca dice
   quiénes, y la yuxtaposición con el párrafo de diabetes invita a concluir que
   son los diabéticos. No lo escribí ni lo insinué: en la v4 la diabetes está en
   sección aparte precisamente para no sugerir esa inferencia. **Fuera.**
4. **A quién debe consultar sus dudas.** «Puede consultar sus dudas en cualquier
   momento» viene en el párrafo de enfermería, pero el original no nombra
   destinatario. Escribí «al equipo que le atiende» —el genérico que el propio
   documento usa— en lugar de «al personal de enfermería», que habría estrechado
   el hecho. Lo anoto porque es una explicitación de destinatario, no una
   copia literal.
5. **Alcance de «cómo proceder» en diabetes.** Al sacar la diabetes a sección
   propia, «el cirujano le indicará cómo proceder» se quedaba sin alcance.
   Restituí el alcance que le daba su sección de origen: «cómo debe proceder **con
   las bebidas** antes de la operación». No escribí «con su insulina», «con su
   medicación antidiabética» ni «con el ayuno», que serían añadidos clínicos.
6. **«Menos complicaciones que un paciente desnutrido».** Añadí el término de
   comparación, que en el original quedaba elíptico («un paciente bien nutrido
   tiene menos complicaciones»), tomándolo de la propia frase («detecta
   desnutrición»). Es la explicitación más discutible que sí hice; la marco aquí
   por si se prefiere revertirla.
7. **Nada de conocimiento clínico externo.** No añadí dolor en el hombro tras
   laparoscopia, ni profilaxis antitrombótica, ni medias de compresión, ni
   fisioterapia respiratoria, ni ninguna cifra, plazo o técnica ausente del
   original.

---

## 6. Dudas clínicas detectadas y NO tocadas

Las dejo constar sin modificar el texto; son cuestiones para el clínico
propietario del corpus, no de redacción.

1. **Posible incoherencia horaria en el preoperatorio.** «Pueden beber líquidos
   hasta **2 horas** antes de la operación» y «**unas horas** antes de la cirugía
   le darán 200–400 ml de bebida con hidratos de carbono». Si «unas horas» fueran
   menos de dos, las dos frases se contradicen. Conservé ambas expresiones tal
   cual.
2. **Quién administra la premedicación.** El original no lo dice en ningún punto
   («le darán»). Usé «el equipo». Si en el circuito real es enfermería de planta,
   convendría decirlo en el corpus.
3. **La sección de dolor mezcla dos escenarios de bomba de ACP**: la ACP con
   calmantes («para algunas cirugías») y la ACP conectada al catéter epidural
   («para determinadas cirugías»). Son dos usos distintos del mismo dispositivo y
   el documento no aclara si son excluyentes. Los mantuve separados en dos
   secciones sin afirmar relación entre ellos.
4. **«No hay peligro de sobredosis» es una afirmación absoluta** en un documento
   para pacientes. Está en el original y no la matizo, pero conviene que un
   clínico confirme que quiere esa formulación sin matices.
5. **Efectos no deseados sin instanciar.** El documento afirma que *todos* los
   fármacos para el dolor pueden producir efectos no deseados, pero solo nombra
   los de la epidural. Los efectos de los calmantes pautados y de la ACP quedan
   sin enunciar (ver §2, hueco 14).
6. **«El médico» del cribado nutricional** no se identifica (¿cirujano?
   ¿nutricionista? ¿médico de cabecera?). Mantuve «el médico», literal.
7. **Ayuno de sólidos ausente.** Un documento perioperatorio para pacientes que
   regula los líquidos pero no los sólidos deja abierta la pregunta más frecuente
   del preoperatorio. Es un hueco de contenido, no de redacción.
