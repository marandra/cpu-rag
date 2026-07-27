# Respuesta a la auditoría — BORRADOR VIVO. No enviar

---

## 0. El correo al que respondemos

> Os comparto una síntesis de la revisión realizada a partir de las 134 preguntas
> preparadas por Ramon y de la evaluación clínica y técnica posterior.
>
> En primer lugar, el ejercicio de Ramon ha sido un buen punto de partida: separar
> preguntas de usuarios con algún conocimiento, pacientes sin conocimientos previos y
> preguntas emocionales o personales ha permitido observar comportamientos muy distintos
> del sistema.
>
> Los principales resultados que hemos visto:
>
> - El 46,3 % responde «No tengo información», incluso ante preguntas frecuentes que
>   podrían recibir una orientación general segura.
> - El 73,1 % requiere una corrección crítica o alta.
> - Solo el 9 % alcanza un nivel aceptable.
> - En hemorroides, el 61,3 % de las respuestas son nulas y el 96,8 % tiene doce
>   palabras o menos, lo que apunta a un problema específico de cobertura, indexación o
>   vocabulario.
> - Se han identificado respuestas potencialmente inseguras sobre medicación durante
>   enfermedades agudas, ayuno y bebidas preoperatorias, anticoagulantes, glucómetros e
>   hipoglucemia.
> - Las preguntas relacionadas con miedo, culpa, ansiedad o vergüenza suelen terminar en
>   una abstención, sin validación ni orientación.
> - También aparecen respuestas como «HbA1c ≥6,5 %», «Anestesia regional» o «Duración:
>   30–60 minutos». Son datos relacionados con la consulta, pero no respuestas
>   completas: no explican su significado, de qué dependen ni qué debe hacer el
>   paciente. En otros casos, una recomendación condicionada se presenta como una regla
>   universal.

---

## Respuesta

Hola Xavi,

Gracias por el trabajo, y gracias a Ramon por el set de preguntas. Lo hemos adoptado
como banco de pruebas propio: no descartamos ninguna de las 134, las hemos reproducido
contra el sistema y las hemos vuelto a evaluar una a una. Vuestras cifras cuadran, no
hay discusión sobre los hechos. Sí sobre qué miden, y de eso va este correo.

### 1. El criterio de evaluación tiene que ser el de diseño

Si no, le estamos pidiendo al sistema funciones que no implementa —y que
deliberadamente no implementa—.

- **Las respuestas deben compararse con los documentos del corpus, no con conocimiento
  general.** En vuestra hoja `Fonts clíniques`, la columna «Ús en l'auditoria» lista
  ADA 2026, ERAS, ASA y ASCRS. Con esa ancla, una respuesta puede ser impecable —fiel al
  documento, sin inventar nada— y puntuar como deficiente porque le falta contenido que
  **no está en el documento que el sistema tenía delante**. Eso no mide el sistema, mide
  el documento.

- **El «No tengo información» es una función, no un fallo, y hay que leerlo junto al
  system prompt.** El sistema no es un asistente clínico: responde **exclusivamente**
  desde un documento por procedimiento, con prohibición explícita de usar conocimiento
  propio. Cuando el documento no resuelve la pregunta, abstenerse **es** la respuesta
  correcta. Esa restricción es el requisito de seguridad del que parte el producto: un
  sistema que rellena huecos con conocimiento general es exactamente el que no queremos
  desplegar en clínica. El prompt además fija casos en los que debe declinar **aunque
  pudiera** responder. Sin ese dato, vuestro 46,3 % de abstenciones es un bloque
  indistinguible; con él se parte en dos, y **35 de esas abstenciones son correctas por
  diseño**. Falta un código para ellas en la rúbrica.

- **Donde sí tenéis razón, y no lo discutimos:** vuestro código `Sense resposta` penaliza
  abstenerse cuando se podía **derivar**. Decir «esto depende de tu caso, coméntalo con
  tu equipo» no inventa nada y no requiere una palabra fuera del documento. Es una
  tercera salida que hoy no está bien resuelta, y nos la llevamos como mejora.

- **Una precisión medida, porque a nosotros nos costó verla:** sobre-rechazo e invención
  **no son dos defectos independientes, son un único umbral**. Lo probamos con seis
  variantes de prompt diseñadas para reducir las abstenciones: la que más las reduce paga
  un rechazo correcto por cada sobre-rechazo que gana. Pedirle al sistema que sea más
  servicial es pedirle que invente más. Por eso la solución no estaba en el prompt, y por
  eso este correo habla sobre todo de los documentos.

### 2. El corpus auditado es material de desarrollo, y se ha evaluado como si fuera definitivo

Esta distinción no quedó clara en la entrega, y eso es cosa nuestra.

- El corpus de **Aiciblock son textos generados para poder desarrollar**, no material de
  producción. El único documento confirmado por Joima es el de **diabetes de Glucowise**.

- El caso extremo es **hemorroides: 28 líneas, 164 palabras, 1,1 KB**, viñetas sin verbo
  ni sujeto. Y no hay documento fuente: ese resumen *es* todo el material que existe.
  Las respuestas telegráficas que citáis salen casi todas de ahí, porque el documento
  está escrito así. Vuestros propios ejemplos —«HbA1c ≥6,5 %», «Anestesia regional»,
  «Duración: 30–60 minutos»— son **literalmente líneas del documento**. Cuando el dato
  está aislado en una viñeta, el sistema no puede añadir lo que no está; distinto es
  cuando la información aparece repartida y sí la sintetiza.

- **Las respuestas «potencialmente inseguras» y las recomendaciones condicionadas
  presentadas como universales tienen el mismo origen: la redacción del corpus.** En la
  mayoría de los casos la condición, el actor o el alcance faltaban ya en el documento, o
  estaban en una viñeta separada de la regla que acotaban. Está detallado pregunta a
  pregunta en el anexo.

- Y un caso que os devolvemos porque **no lo podemos resolver nosotros**: vuestra crítica
  más dura —«la resposta més perillosa del bloc»— apunta a la medicación durante
  enfermedad aguda. La frase está literalmente en nuestro documento de diabetes
  (`Nunca suspenda medicación de la diabetes…; dosis habitual salvo indicación médica`).
  El sistema reprodujo el documento con fidelidad, y tenéis razón en el fondo: las reglas
  de días de enfermedad exigen pausar los SGLT2i. **Es el documento el que hay que
  corregir, y esa decisión es clínica, no técnica.**

### 3. De aquí hemos sacado unas guidelines de redacción, que es el entregable que más vale

Hemos derivado **guías de redacción del corpus** para las especialidades, con siete
reglas, cada una respaldada por un experimento. No son estilo. Por ejemplo: «escribe
frases completas» **por sí solo empeora el resultado** —lo medimos y bajó—; lo que
funciona es que el contenido accionable esté en la **oración principal** y que cada frase
lleve dentro su **condición, su actor y su alcance**.

Están validadas: reescribimos hemorroides **sin cambiar un solo hecho** y las respuestas
telegráficas cayeron del 56 % al 23 %; de paso se arregló el defecto de los
anticoagulantes, que había sobrevivido a seis variantes de prompt y a un cambio de
modelo.

Con un límite que preferimos deciros nosotros: **estas reglas reparan material
telegráfico, no mejoran cualquier texto.** Aplicadas al documento de cirugía abdominal,
que ya estaba en prosa, **empeoraron el resultado**. Así que el requisito aplica al
material que llega en forma de resumen esquemático, no a todo.

Lo que queda es generar la documentación de producción con estas guías y validar el
procedimiento sobre ella.

### 4. Sobre los perfiles de paciente

Aquí os apuntamos algo a favor: **la partición de Ramon —usuario informado, paciente sin
conocimientos previos, pregunta emocional— resultó la variable con más poder explicativo
de todo el análisis.**

Nuestros propios sets de evaluación (147 preguntas, siete ficheros) **no modelan al
paciente**: varían otro eje, el de qué debe hacer el sistema —respondible o no (76 de las
147 son irrespondibles a propósito), administrativa, fuera de dominio, meta,
conversacional—. Estaban construidos para castigar la invención, y en eso hicieron su
trabajo. Pero no medían registro ni acompañamiento, y por eso pasábamos nuestras pruebas
y suspendíamos las vuestras.

Vuestro sexto punto lo confirmamos y lo cuantificamos: en la versión que auditasteis, las
preguntas emocionales se abstenían el **81 %** de las veces. El material estaba en el
corpus todo el tiempo (§Aspectos psicológicos, §Premedicación). **En la versión actual
esa lista pasa a ser la más fuerte de las tres, con un 92 % de aciertos.** Incorporamos
el eje de paciente a nuestros sets.

### 5. Las 134, comentadas una a una — anexo adjunto

Adjuntamos **`auditoria_134_comentadas.md`**: para cada una de las 134, vuestra pregunta
y vuestra puntuación, la respuesta de la versión que auditasteis, la respuesta de la
versión actual, y **nuestro veredicto con el motivo y la línea del documento que lo
decide**. No es una recalificación en bloque: se puede discutir pregunta a pregunta.

Nuestros dos criterios, a propósito separados:

- **corrección** — ¿hace lo que debe, **dado el documento**?
- **presentable** — ¿además se la enseñarías a un paciente tal cual?

La segunda es más dura y la incluimos **porque nos incluye a nosotros**: hay respuestas
correctas que no son presentables, y eso es un problema aunque no sea un error.

### 6. El sistema que auditasteis ya no es el que vale

Esto es importante para leer las cifras: **auditasteis la v1.1**, que es la que estaba
entregada. Desde entonces cambiaron el modelo y dos de los tres documentos. Hemos pasado
vuestras 134 preguntas por la versión nueva y las hemos releído con los criterios de
arriba:

| sobre las 134                      |     corrección | presentable |
| ---------------------------------- | -------------: | ----------: |
| **versión actual** (aún no enviada) | **85 – 90 %** | **80 – 84 %** |
| versión que auditasteis (v1.1)      |         63 %  |      50 %   |
| vuestra evaluación                  |          —    |       9 %   |

El acierto en la decisión de responder-vs-abstenerse pasa del 79 % al 91 %. La horquilla
son seis respuestas en el filo: apoyadas en el documento y sin inventar nada, pero sin
responder del todo lo que se preguntaba; el extremo bajo las descuenta.

Dos honestidades que van pegadas al número: la mejora **no es solo del modelo** —dos de
los tres documentos están reescritos con las guías del punto 3, y esa es justamente la
prueba de que las guías funcionan—; y es **una sola ejecución**, como lo era la vuestra.

### 7. Lo que os ofrecemos ahora

- **Tenemos el sistema desplegado en un servidor a vuestra disposición** para que lo
  probéis vosotros mismos con las 134 o con lo que queráis. Decidnos y os damos acceso.

- **Tenemos preparado el instalador nuevo**, con el modelo, el corpus y el empaquetado
  actualizados. Quedamos a la espera de que nos dijerais **dónde lo ibais a instalar**
  para adaptarlo a ese entorno, y no hemos tenido respuesta. Para no seguir bloqueados os
  mandamos directamente una **versión general**, que funciona tal cual; si al instalarlo
  surge cualquier duda del entorno concreto, preguntadnos y lo ajustamos.

- Y una petición para la próxima ronda: **evaluad con los dos veredictos separados**
  —correcta dado el documento / suficiente para el paciente—. Es el cambio que hace que
  la evaluación mida el sistema y el material por separado, que es lo que necesitamos los
  dos para saber dónde trabajar.

Un abrazo,
Marcelo

## 1. Lo primero: lo que damos por bueno

Esto va delante de todo lo demás, y va sin matices.

**El conjunto de preguntas es válido y lo hemos adoptado como nuestro.** No hemos
descartado ninguna de las 134. Las hemos reproducido contra el sistema real, las hemos
vuelto a evaluar con nuestros propios criterios, y desde entonces son el banco de
pruebas con el que medimos cada cambio que hacemos: seis variantes de _prompt_, tres
ediciones de corpus, seis modelos y cuatro configuraciones de servicio se han decidido
contra estas 134 preguntas. La partición de Ramon —usuario informado, paciente sin
conocimientos previos, pregunta emocional— resultó **la variable con más poder
explicativo de todo el análisis**, y volvemos sobre ella en el punto 5.

**Vuestras cifras cuadran.** Las hemos recalculado una a una: 46,3 % de abstenciones,
73,1 % de corrección crítica o alta, 9 % aceptable, hemorroides 61,3 % nulas y 96,8 % de
doce palabras o menos. No hay discusión sobre los hechos. La hay sobre el diagnóstico, y
de eso va el resto de este correo.

**Y hay defectos reales, encontrados por vosotros y por nosotros.** De hecho nuestra
propia revisión encontró **más** defectos que la vuestra en la capa que vosotros no
podíais ver —11 respuestas que funden o des-acotan dos reglas del documento, 8 de ellas
nuevas—, y nos obligó a **bajar** nuestra propia nota: cuatro respuestas que dábamos por
correctas resultaron defectuosas al releerlas. Lo decimos porque marca el tono: no
venimos a defender el 9 %, venimos a explicar qué mide y qué no.

**Dos precisiones sobre qué evaluasteis**, que no cambian nada de lo anterior pero sí
cómo hay que leer las cifras: la versión auditada es la **v1.1**, que es la que estaba
entregada, y desde entonces el sistema ha cambiado de modelo; y en un punto concreto —lo
comentamos en el punto 6— el sistema reprodujo fielmente un documento clínico que es el
que está equivocado.

---

## 2. Punto 1 — la evaluación tiene que hacerse contra el corpus

Éste es el desacuerdo de fondo, y explica la mayor parte de la distancia entre vuestro 9
% y nuestro número.

El sistema **no es un asistente clínico**: es un sistema que responde **exclusivamente**
desde un documento por procedimiento, con prohibición explícita de usar conocimiento
propio. Si el documento no resuelve la pregunta, la respuesta correcta —la diseñada— es
«No tengo información sobre eso». Esa restricción no es una limitación técnica que
estemos disculpando: es **el requisito de seguridad del que parte el producto**. Un
sistema que completa con conocimiento general es exactamente el que no queremos
desplegar en un entorno clínico.

En vuestra propia hoja `Fonts clíniques` hay una columna «Ús en l'auditoria» con las
referencias empleadas: **ADA 2026, ERAS, ASA, ASCRS**. Es decir, la respuesta del
sistema se comparó contra la guía clínica de referencia, no contra el documento que el
sistema tenía delante. Con esa ancla, una respuesta puede estar **perfectamente bien**
—fiel al documento, sin inventar nada— y puntuar como deficiente porque le falta
contenido que **no está en el documento**. Nos consta que ocurre incluso en respuestas
que vosotros puntuáis alto: a algunas de vuestras 9/10 les pedís contenido que nuestro
corpus no contiene.

Esto no es un reproche al instrumento. Vuestra rúbrica está bien construida: tiene
códigos para fuga de metadatos, contradicción interna, exceso de certeza y riesgo
clínico, que son justo las categorías que hay que vigilar. **El problema no es la
rúbrica, es el ancla.**

**Lo que proponemos:** repetir la evaluación con dos veredictos por pregunta, no uno.
(a) ¿Es correcta **dado el documento**? (b) ¿Es suficiente **para el paciente**? Las dos
cosas importan y miden cosas distintas: la primera evalúa el sistema, la segunda evalúa
el documento. Hoy están fundidas en una sola nota, y el resultado se lee como un fallo
del sistema cuando en buena parte es un límite del material.

---

## 3. Punto 2 — un rechazo solo se puede evaluar conociendo las reglas del sistema

Vuestro código `Sense resposta` penaliza abstenerse cuando la pregunta «es podria
contestar **o derivar**». **En eso tenéis razón y no lo vamos a discutir**: decir «eso
depende de tu caso, coméntalo con tu equipo» no inventa nada, no requiere una palabra
fuera del documento, y es exactamente la tercera salida que el sistema debería tener y
hoy no tiene bien resuelta. Es una de las mejoras que nos llevamos.

Pero para separar un rechazo _correcto_ de un fallo hace falta un dato que la rúbrica no
incorpora: **bajo qué reglas opera el sistema**. Sin él, los 46,3 % de abstenciones son
un bloque indistinguible, y no lo son. Cuando se aplica el criterio de «¿lo resuelve el
documento?», ese bloque se parte en dos:

- **abstenciones correctas** — el documento no lo dice, y responder habría significado
  inventar;
- **sobre-rechazos reales** — el documento sí lo dice y el sistema no lo vio. Es un
  defecto nuestro, lo asumimos, y es donde más hemos trabajado.

En nuestra revisión de las 134, **35 de las abstenciones son correctas** por ese
criterio. Falta un código para ellas en la rúbrica; con él, el mismo conjunto de
respuestas da un resultado muy distinto sin que cambie una sola respuesta.

Un matiz que nos parece importante compartir, porque a nosotros nos costó verlo:
**sobre-rechazo e invención no son dos defectos independientes, son un único umbral.**
Lo medimos con seis variantes de _prompt_ diseñadas para reducir las abstenciones: la
que más las reduce **paga un rechazo correcto por cada sobre-rechazo que gana**. Pedirle
al sistema que sea más servicial es pedirle que invente más. Por eso la solución no
estaba en el _prompt_, y por eso este correo habla tanto de los documentos.

---

## 4. Punto 3 — el corpus auditado era material de desarrollo

Los tres documentos que visteis son un **primer pase de desarrollo**, no material de
producción. Es una distinción que no estaba clara en la entrega, y eso es cosa nuestra.

El caso extremo es **hemorroides**, y conviene decirlo con precisión:

- son **1,1 KB de texto** — seis secciones de viñetas, unas 150 palabras en total;
- **no hay ningún documento fuente de hemorroides**: ese resumen _es_ todo el material
  que existe;
- llegó ya en esa forma, titulado «RESUMEN Cirugía de hemorroides». No es el resultado
  de un proceso de compresión nuestro.

Con ese documento, las 31 preguntas de hemorroides preguntan casi todas por cosas que no
están escritas en ninguna parte. Vuestro 61,3 % de respuestas nulas es, en su mayor
parte, **el sistema comportándose como debe** ante un documento que no cubre el
postoperatorio domiciliario. Volvemos sobre lo que falta en el punto 6.

---

## 5. Punto 4 — el documento de 1,1 KB es la causa de las respuestas telegráficas

Éste es vuestro hallazgo más citable —«96,8 % de doce palabras o menos»— y también el
que más cambia de significado cuando se mira de cerca. Los ejemplos que citáis, **«HbA1c
≥6,5 %»**, **«Anestesia regional»** y **«Duración: 30–60 minutos»**, son literalmente
líneas del documento: en el original están escritas así, como viñetas sin verbo ni
sujeto. El sistema devuelve frases cortas porque **el documento son frases cortas**.

No os pedimos que nos creáis. Lo hemos medido de dos maneras.

**(a) La telegrafía está concentrada en el documento delgado, no repartida por el
sistema.** Sobre 1.206 generaciones, respuestas de menos de 80 caracteres:

| documento                 |     tamaño | respuestas telegráficas |
| ------------------------- | ---------: | ----------------------: |
| cirugía abdominal         |     7,6 KB |                 **2 %** |
| diabetes                  |    13,0 KB |                    11 % |
| **hemorroides**           | **1,1 KB** |                **56 %** |
| total, las 134            |            |                    18 % |
| **total sin hemorroides** |            |                 **7 %** |

Hemorroides aporta **el 71 % de todas las respuestas telegráficas con el 23 % de las
preguntas**. Si fuera una propiedad del sistema, estaría repartida.

**(b) Y es causal, no una correlación.** Reescribimos ese documento **sin cambiar un
solo hecho** —misma información, otra redacción— y volvimos a pasar las mismas 31
preguntas con el mismo modelo y el mismo _prompt_: las respuestas telegráficas cayeron
**del 56 % al 23 %**. Ampliándolo desde fuentes existentes, al 16 %. Y de paso se
arregló el peor defecto que teníamos, el de los anticoagulantes, que había sobrevivido a
seis variantes de _prompt_, a dos ediciones de corpus y a un cambio de modelo.

De ahí sale el resultado que más nos importa de todo este trabajo, y que va mucho más
allá de responderos: **hemos derivado unas guidelines de redacción de documentos
clínicos**, con siete reglas, cada una respaldada por un experimento. No son estilo. Por
ejemplo: «escribe frases completas» **por sí solo empeora el resultado** —lo probamos y
bajó—; lo que funciona es que **el contenido accionable esté en la oración principal** y
que **cada frase lleve dentro su condición, su actor y su alcance**. Ese documento es,
para nosotros, el entregable más valioso de esta auditoría, y fija cómo hay que preparar
el material de las especialidades que vengan.

Con un límite que hemos medido y que preferimos deciros nosotros: **estas reglas reparan
material telegráfico, no mejoran cualquier texto.** Las aplicamos también a los otros
dos documentos. En el de diabetes, que estaba en viñetas, mejoraron las dos cosas. En el
de cirugía abdominal, **que ya estaba bien escrito en prosa, empeoraron el resultado**:
al trocear los párrafos en frases autosuficientes, el sistema pasó a devolver una frase
corta donde antes daba un párrafo con contexto. Así que la conclusión no es
«reescribirlo todo», sino que **el requisito es sobre el material que llega en forma de
resumen esquemático**.

---

## 6. Punto 5 — con esos criterios, el 9 % es 84 %

Hemos releído a mano las 134 respuestas, con dos veredictos por pregunta y citando en
cada una la línea del documento que la decide.

|                                | corrección<br>(¿hace lo que debe, dado el documento?) | presentable<br>(¿además se la enseñarías a un paciente?) |
| ------------------------------ | ----------------------------------------------------: | -------------------------------------------------------: |
| **sistema actual**             |                                              **84 %** |                                                 **78 %** |
| versión que auditasteis (v1.1) |                                                  63 % |                                                     50 % |
| vuestra evaluación             |                                                     — |                                                      9 % |

Las dos columnas están ahí a propósito. La primera es la que responde a «¿está el
sistema haciendo bien su trabajo?». La segunda es más dura y **la incluimos porque nos
incluye a nosotros**: hay respuestas correctas que no son presentables, y son un
problema aunque no sean un error.

El salto de 63 % a 84 % no es un ajuste de criterio: es un **cambio de modelo**, medido
contra estas mismas 134 preguntas. El acierto en la decisión de responder-vs-abstenerse
pasa de 78 % a 91 %, y —esto importa para vuestro séptimo punto— **la inestabilidad cae
del 22 % al 1 %**: en la versión que auditasteis, casi una de cada cuatro preguntas
podía cambiar de respuesta entre dos ejecuciones idénticas. Muchos veredictos
individuales, los vuestros y los nuestros, eran una sola tirada de un proceso ruidoso.
Ya no lo son.

Sobre **las preguntas emocionales**, que es vuestro sexto punto: lo confirmamos y lo
cuantificamos. En la v1.1, las preguntas con carga emocional se abstenían el **70 %** de
las veces frente al **39 %** del resto. Y encontramos la prueba limpia: pares de
preguntas que piden **la misma información** con distinto registro, donde el sistema
responde a una y se abstiene de la otra. Con el sistema actual eso está mayormente
resuelto. Lo que queda —y esto sí es instructivo— **no es el registro emocional**: el
sistema responde bien «tengo miedo, ¿esto cambia mi vida para siempre?» y mal «¿es culpa
mía?». La diferencia es que para la primera **existe una frase en el documento** que la
responde y para la segunda no. Es, otra vez, el documento.

---

## 7. Lo que hay que decidir en vuestro lado

Tres cosas que hemos encontrado y que **no podemos resolver nosotros**, porque no son
defectos del sistema sino del contenido:

1. **Un defecto clínico en el documento de diabetes.** Vuestra crítica más dura —«la
   resposta més perillosa del bloc»— apunta a una respuesta sobre la medicación durante
   una enfermedad aguda. Al revisarla, la frase está **literalmente en nuestro
   documento**:
   `Nunca suspenda medicación de la diabetes (pastillas o insulina); dosis habitual salvo indicación médica`.
   El sistema reprodujo el documento con fidelidad. Y tenéis razón en el fondo: las
   reglas de días de enfermedad de ADA exigen pausar los SGLT2i. **Es el documento el
   que hay que corregir**, y esa decisión es clínica, no técnica. (Esa misma respuesta
   tenía _además_ un defecto nuestro, que sí hemos arreglado: fundía «si hay fiebre,
   paracetamol» con «consulte si la fiebre supera 39 °C» y convertía un criterio de
   alarma en un umbral de tratamiento.)
2. **Una contradicción entre dos fuentes.** La guía de práctica clínica dice que en la
   primera defecación tras la cirugía «no es habitual que se produzcan dolor ni
   hemorragia»; el documento de hemorroides clasifica el dolor al defecar como
   complicación **frecuente**. Las dos no pueden ser la referencia a la vez.
3. **Un hueco de cobertura en hemorroides que ninguna reescritura puede tapar.** Son
   ocho preguntas previsibles sin material posible: vuelta al trabajo, conducir y
   deporte; ingreso y días; puntos y cura de la herida; cómo hacer los baños de asiento;
   señales de alarma para acudir a urgencias; qué analgésico tomar en casa y cada
   cuánto; laxantes; si hay que ir acompañado. **Es el postoperatorio domiciliario
   entero.** Mientras no exista ese material, el sistema seguirá —correctamente—
   diciendo que no tiene información.

---

## 8. Qué vamos a hacer

- **Cambio de modelo**, ya medido contra vuestras 134 preguntas: 91 % de acierto en la
  decisión y estabilidad del 1 %.
- **Reescritura del corpus** con las guidelines. Validado ya en hemorroides; en curso en
  diabetes y cirugía abdominal.
- **La tercera salida** («esto depende de tu caso, coméntalo con tu equipo»), que es
  vuestra aportación más directamente accionable.
- **Guidelines de documentación como requisito de entrada** del material clínico, para
  las especialidades que vengan.
- **Nueva versión entregable**, con el modelo, el corpus y el empaquetado nuevos.

Y una petición: cuando esté, **reevaluad con los dos veredictos separados** —correcta
dado el documento / suficiente para el paciente—. Es el cambio que hace que la
evaluación mida el sistema y el material por separado, que es lo que necesitamos los dos
para saber dónde trabajar.

---

## Pendientes antes de enviar

1. **Recuperar las «ideas sueltas»** que el usuario discutió en otra conversación y que
   no están en este repositorio. Sin eso, este borrador está incompleto.
2. **Re-registrar los veredictos por pregunta de diabetes** (las 55 se juzgaron a mano
   pero solo se guardaron los totales). Si el 84 % sale del equipo, tiene que ser
   auditable pregunta a pregunta como lo son las otras 79.
3. **Cerrar A4** (guidelines aplicadas a diabetes y cirugía) y decidir si el número que
   se manda es el actual o el posterior a la reescritura. **Si se manda el posterior,
   hay que re-derivar la respondibilidad** o el resultado contará mejoras como
   regresiones.
4. **Decidir el número de versión** del entregable (§FASE B) — el texto dice «nueva
   versión entregable» a propósito, sin nombre.
5. Revisar si el punto 7.1 se envía en este correo o en uno separado a la parte clínica.
   Es un defecto de contenido y puede merecer su propio canal.
6. **Tono:** releer buscando cualquier frase que suene a defensa. El criterio acordado
   es que el objetivo no es defendernos, es que el sistema funcione.
