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

Gracias por la revisión, y a Ramon por el set de preguntas: lo hemos adoptado como
banco de pruebas propio. Vuestras cifras cuadran, las hemos recalculado una a una. Lo
que cambia es qué miden.

**1. El criterio de evaluación tiene que ser el de diseño.** El sistema responde
**exclusivamente** desde un documento por procedimiento, con prohibición explícita de
usar conocimiento propio. No es una limitación que estemos disculpando: es el
requisito de seguridad del que parte el producto. De ahí, dos cosas:

- Vuestra hoja `Fonts clíniques` lista ADA 2026, ERAS, ASA y ASCRS. Contra esas guías,
  una respuesta puede ser impecable —fiel al documento, sin inventar nada— y puntuar
  como deficiente porque le falta contenido que **no está en el documento que el
  sistema tenía delante**. Eso mide el material, no el sistema.
- El «No tengo información» es una función, no un fallo, y hay que leerlo junto al
  system prompt, que fija cuándo debe abstenerse **incluso pudiendo** responder. De
  vuestras 62 abstenciones, **36 son correctas por diseño**. Falta un código para ellas
  en la rúbrica.

**2. El corpus auditado es material de desarrollo, y se ha evaluado como si fuera
definitivo.** El de Aiciblock son textos generados para poder desarrollar; el único
confirmado por Joima es el de diabetes de Glucowise. Hemorroides son **28 líneas, 164
palabras**, viñetas sin verbo ni sujeto, y no existe documento fuente: ese resumen es
todo el material que hay.

De vuestros tres ejemplos, dos son **literalmente líneas del documento**: «HbA1c
≥6,5 %» está así en el de diabetes y «Duración: 30–60 minutos» en el de hemorroides.
Cuando el dato está aislado en una viñeta, el sistema no puede añadir lo que no está.
El tercero es un defecto nuestro: el documento dice «anestesia regional o general» y
el sistema devolvió sólo «Anestesia regional», cerrando en una la alternativa. Está
corregido en la v2.

El mismo origen tienen la mayoría de las recomendaciones condicionadas presentadas
como universales: la condición, el actor o el alcance faltaban ya en el documento, o
estaban en una viñeta separada de la regla que acotaban.

Un caso os lo devolvemos porque **no lo podemos resolver nosotros**: vuestra crítica
más dura —la medicación durante enfermedad aguda— apunta a una frase que está
literalmente en nuestro documento de diabetes. El sistema lo reprodujo con fidelidad y
vosotros tenéis razón en el fondo. **Es el documento el que hay que corregir, y esa
decisión es clínica.**

**3. De aquí ha salido lo que más vale: guías de redacción del corpus,** siete reglas,
cada una respaldada por un experimento. Reescribimos hemorroides **sin cambiar un solo
hecho** y las respuestas telegráficas cayeron del 56 % al 23 %. El caso que mejor lo
resume es el de los anticoagulantes: la v1.1 decía «debes ajustar tu medicación»; con
el modelo nuevo pero el documento viejo seguía diciendo «se debe ajustar», sin decir
quién; sólo al reescribir el documento aparece «el equipo médico le ajusta la
medicación». **El defecto sobrevivió al cambio de modelo y lo arregló la redacción.**

Con un límite: estas reglas **reparan material esquemático, no mejoran cualquier
texto**. Aplicadas al documento de cirugía, que ya estaba en prosa, empeoraron el
resultado.

**4. La v1.1 que auditasteis ya no es la que vale.** Hemos evaluado las 134 con un solo
criterio: **¿es correcta, dado el documento?** —responde lo que se pregunta apoyada en
el texto, sin inventar y sin fundir ni des-acotar una regla; o se abstiene donde el
documento no da material—.

| sobre las 134      |      correctas |
| ------------------ | -------------: |
| **v2** (adjunta)   | **120 = 90 %** |
| v1.1 (auditada)    |     83 = 62 %  |
| vuestra evaluación |          9 %   |

El acierto en responder-vs-abstenerse pasa del 79 % al 91 %. Y vuestro punto sobre las
preguntas emocionales lo confirmamos: en la v1.1 se abstenían el **81 %** de las veces
teniendo el material en el corpus; en la v2 esa lista pasa a ser **la más fuerte de las
tres, con un 92 %**.

Dos precisiones sobre ese número: la mejora **no es sólo del modelo** —dos de los tres
documentos están reescritos con las guías del punto 3, que es la prueba de que
funcionan—, y es **una sola ejecución**, como lo era la vuestra.

**5. Lo que no tenemos resuelto: derivar.** Vuestro código `Sense resposta` penaliza
abstenerse cuando se podía **derivar**, y tenéis razón. Lo hemos probado y no nos
convence todavía: la variante que deriva convierte 14 de las 45 abstenciones, pero sólo
4 de las 21 emocionales, siempre con la misma frase de plantilla, y **en 7 de esas 14
deriva preguntas que el documento sí responde** —las causas de la diabetes, el ajuste
de la medicación—, con lo que pierde hasta 3 puntos de acierto. No os la entregamos
así. Lo que sí vamos a hacer es (a) cambiar el texto de la abstención para que
reconozca la pregunta y remita al equipo, que alcanza a las 37 y no puede desplazar
ninguna respuesta correcta, y (b) **añadir material sobre miedo, vergüenza y ansiedad a
los documentos de cirugía y hemorroides**: las emocionales que hoy se responden bien
son las de diabetes, el único documento con sección psicológica. Otra vez el documento.

**Van dos adjuntos**, con los mismos números de pregunta:

- `auditoria_134_evaluacion.md` — vuestra lista tal cual, con nuestra evaluación al lado.
- `auditoria_134_v2.md` — las 134 respondidas por la v2, con nuestra
  evaluación. Para que podáis verlo sin ejecutarlo.

**Y dos cosas que os ofrecemos:**

- La v2 está **desplegada en un servidor a vuestra disposición** para que la probéis
  vosotros. Decidnos y os damos acceso.
- Tenemos el **instalador de la v2 listo**. Quedamos a la espera de que nos dijerais
  dónde lo ibais a instalar para adaptarlo a ese entorno y no hemos tenido respuesta,
  así que os mandamos una **versión general**, que funciona tal cual; si al instalarlo
  surge alguna duda del entorno concreto, preguntadnos y lo ajustamos.

Una petición para la próxima ronda: separad **correcta dado el documento** de
**suficiente para el paciente**. Es lo que hace que la evaluación mida el sistema y el
material por separado, que es lo que necesitamos los dos.

Un abrazo,
Marcelo
