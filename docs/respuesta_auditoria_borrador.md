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
banco de pruebas propio. Vuestras cifras cuadran, las hemos recalculado. Lo que
cambia es qué miden. Va en cuatro puntos, con el detalle pregunta a pregunta
adjunto.

**1. El criterio de evaluación tiene que ser el de diseño.** El sistema responde
**exclusivamente** desde un documento por procedimiento, con prohibición explícita
de usar conocimiento propio. No es una limitación que estemos disculpando: es el
requisito de seguridad del que parte el producto. De ahí, dos cosas:

- Vuestra hoja `Fonts clíniques` lista ADA 2026, ERAS, ASA y ASCRS. Contra esas
  guías, una respuesta puede ser impecable —fiel al documento, sin inventar nada— y
  puntuar como deficiente porque le falta contenido que **no está en el documento
  que el sistema tenía delante**. Eso mide el material, no el sistema.
- El «No tengo información» es una función, no un fallo, y hay que leerlo junto al
  system prompt, que fija cuándo debe abstenerse **incluso pudiendo** responder. De
  vuestro 46,3 % de abstenciones, **35 son correctas por diseño**. Falta un código
  para ellas en la rúbrica.

Donde sí tenéis razón y no lo discutimos: cuando se puede **derivar** («esto depende
de tu caso, coméntalo con tu equipo»), abstenerse es peor. Nos lo llevamos como
mejora.

**2. El corpus auditado es material de desarrollo, y se ha evaluado como si fuera
definitivo.** El de Aiciblock son textos generados para poder desarrollar; el único
confirmado por Joima es el de diabetes de Glucowise. Hemorroides son **28 líneas,
164 palabras**, viñetas sin verbo ni sujeto, y no existe documento fuente. Vuestros
propios ejemplos —«HbA1c ≥6,5 %», «Anestesia regional», «Duración: 30–60 minutos»—
son **literalmente líneas de ese documento**. De ahí sale casi toda la telegrafía, y
también la mayoría de las recomendaciones condicionadas presentadas como
universales: la condición, el actor o el alcance faltaban ya en el texto.

Un caso os lo devolvemos porque es clínico y no técnico: vuestra crítica más dura
—la medicación durante enfermedad aguda— apunta a una frase que está literalmente en
nuestro documento de diabetes. El sistema lo reprodujo con fidelidad, y vosotros
tenéis razón en el fondo. **Es el documento el que hay que corregir.**

**3. De aquí ha salido lo que más vale: guías de redacción del corpus.** Siete
reglas, cada una respaldada por un experimento. Reescribimos hemorroides sin cambiar
un solo hecho y las respuestas telegráficas cayeron del 56 % al 23 %. Con un límite
que preferimos deciros nosotros: reparan material esquemático, no mejoran cualquier
texto — aplicadas al documento de cirugía, que ya estaba en prosa, lo empeoraron.

**4. El sistema que auditasteis ya no es el que vale.** Auditasteis la v1.1. Desde
entonces han cambiado el modelo y dos de los tres documentos. Hemos evaluado las 134
una a una con un solo criterio: **¿es correcta, dado el documento?** —es decir,
responde lo que se pregunta apoyada en el texto, sin inventar y sin fundir ni
des-acotar una regla; o se abstiene donde el documento no da material—.

| sobre las 134                       |      correctas |
| ----------------------------------- | -------------: |
| **versión actual** (aún no enviada) |  **120 = 90 %** |
| versión que auditasteis (v1.1)      |     83 = 62 %  |
| vuestra evaluación                  |          9 %   |

El acierto en responder-vs-abstenerse pasa del 79 % al 91 %. Dos honestidades: la
mejora **no es solo del modelo** —dos de los tres documentos están reescritos con las
guías del punto 3, que es justamente la prueba de que funcionan—, y es **una sola
ejecución**, como lo era la vuestra.

Lo que sigue sin estar bien son las respuestas demasiado escuetas, y ahí os damos la
razón: quedan sobre todo en hemorroides, por lo del punto 2.

Vuestro punto sobre las preguntas emocionales lo confirmamos y lo cuantificamos: en
la v1.1 se abstenían el **81 %** de las veces, teniendo el material en el corpus. En
la versión actual esa lista pasa a ser **la más fuerte de las tres, con un 92 %**.

**Van dos adjuntos**, con los mismos números de pregunta para poder cruzarlos:

- `auditoria_134_evaluacion.md` — vuestra lista tal cual, con nuestra evaluación de
  cada respuesta añadida al lado.
- `auditoria_134_sistema_actual.md` — las 134 respondidas por el sistema actual, con
  nuestra evaluación. Va para que podáis verlo sin tener que ejecutarlo vosotros.

Es discutible pregunta a pregunta, que es como creemos que hay que discutirlo.

**Lo que os ofrecemos ahora:**

- El sistema está **desplegado en un servidor a vuestra disposición** para que lo
  probéis vosotros mismos, con las 134 o con lo que queráis. Decidnos y os damos
  acceso.
- Tenemos el **instalador nuevo listo**. Quedamos a la espera de que nos dijerais
  dónde lo ibais a instalar, para adaptarlo a ese entorno, y no hemos tenido
  respuesta; así que os mandamos directamente una **versión general**, que funciona
  tal cual. Si al instalarlo surge cualquier duda del entorno concreto,
  preguntadnos y lo ajustamos.

Y una petición para la próxima ronda: **evaluad con los dos veredictos separados**,
correcta dado el documento / suficiente para el paciente. Es lo que hace que la
evaluación mida el sistema y el material por separado, que es lo que necesitamos los
dos para saber dónde trabajar.

Un abrazo,
Marcelo
