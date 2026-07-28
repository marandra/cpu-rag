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

Estuvimos revisando el informa y las respuesas, y las hemos reproducido a todas. Hay
varios puntos a comentar, principalmente para alinearnos en que esperar y como evaluar:

1. El criterio de evaluación tiene que ser el de diseño.

El sistema responde solo desde un documento por procedimiento, con prohibición explícita
de usar conocimiento propio como un requisito de seguridad del que parte el RAG. De ahí,
dos cosas.

Las respuestas tienens que evaluarse contra el corpus. Vuestra hoja "Fonts clíniques"
lista ADA 2026, ERAS, ASA y ASCRS como guia de evaluacion. Contra esas guías, una
respuesta puede ser impecable (fiel al documento, sin inventar nada) y puntuar como
deficiente porque le falta contenido que no está en el documento que el sistema tenía
delante. Eso mide el material, no el sistema.

Y el "No tengo información" es una función, no un fallo. Hay que leerlo junto al system
prompt, que fija cuándo debe abstenerse incluso pudiendo responder. De vuestras 62
abstenciones, 36 son correctas por diseño. Falta un código para ellas en la rúbrica.

2. El corpus auditado es material de desarrollo, y se ha evaluado como si fuera
   definitivo.

El de Aiciblock (hemorroides y cirugia abdominal) son textos generados para poder
avanzar. El único confirmado por Joima es el de diabetes de Glucowise. Hemorroides son
160 palabras (para las ~30 preguntas), viñetas sin verbo ni sujeto, y no existe
documento fuente: ese resumen es todo el material que hay.

De vuestros tres ejemplos, dos son literalmente líneas del documento: "HbA1c ≥6,5 %"
está así en el de diabetes, y "Duración: 30-60 minutos" en el de hemorroides. Cuando el
dato está aislado en una viñeta, el sistema no puede añadir lo que no está.

El mismo origen tienen la mayoría de las recomendaciones condicionadas presentadas como
universales: la condición, el actor o el alcance faltaban ya en el documento, o estaban
en una viñeta separada de la regla que acotaban.

Vuestra critica mas dura (la medicación durante enfermedad aguda), apunta a una frase
que está literalmente en el documento de diabetes. El sistema lo reprodujo con
fidelidad, per es el documento el que hay que corregir.

3. La version que habeis usado para la evaluacion es la v1.1, pero la version nueva (la
   de multiinstancia) ya incluye la solucion a varios problemas que fuimos detectando,
   qeu os habieis encontrado arriba.

Dicho todo lo anterior, el 9% que mencionas pasarí a un 60%. La evaluacion respuste por
respuesta esta en el adjunto (auditoria_134_evaluacion.md) para que lo puedas ver en
detalle y entender de donde sale. Luego, el 40% de respuestas incorrectas sale de dos
partes, la mitad es un tema de como esta redactado el corpus. Hay varias sutilezas que
influyen en la salida (lo que te comentaba arriba respecto de hemorroides por ejemplo).
Revisando el estilo de escritura de la informacion las respuestas se corrigen y pasan
del 60% anterior al 80%. De aquí ha salido unas guías de redacción del corpus que ya
estan implementadas en la version 2. Y finalmente con un cambio de modelos hemos podido
pasar de ese 80% a un 90%, medido en vuestro set de 134 preguntas.

5. Lo que no tenemos resuelto: derivar.

Vuestro código "Sense resposta" penaliza abstenerse cuando se podía derivar, y tenéis
razón. Lo hemos probado y no nos convence todavía. La variante que deriva convierte 14
de las 45 abstenciones, pero solo 4 de las 21 emocionales, siempre con la misma frase de
plantilla, y en 7 de esas 14 deriva preguntas que el documento sí responde (las causas
de la diabetes, el ajuste de la medicación), con lo que pierde hasta 3 puntos de
acierto. No os la entregamos así.

De las dos cosas que íbamos a hacer, una ya está hecha y va en la versión 2.2: la
abstención ya no dice "No tengo información sobre eso." sino "Esto no lo recoge la
información que tengo, pero tu equipo sanitario podrá orientarte según tu caso.", que
alcanza a las 42 abstenciones, 24 de ellas emocionales, sin bajar el acierto. Esa misma
versión pasa los tres documentos a tutear al paciente, por coherencia con el resto del
sistema. La otra cosa, añadir material sobre miedo y vergüenza a los documentos de
cirugía y hemorroides, sigue pendiente: las emocionales que hoy se responden bien son
las de diabetes, el único documento con sección psicológica. Otra vez el documento.

Van dos adjuntos, con los mismos números de pregunta:

auditoria_134_evaluacion.md, que es vuestra lista tal cual con nuestra evaluación al
lado, y auditoria_134_v22.md, que son las 134 respondidas por la v2.2, la versión que os
entregamos, también con nuestra evaluación, para que podáis verlo sin tener que
ejecutarlo.

Y dos cosas que os ofrecemos. La v2 está desplegada en un servidor a vuestra disposición
para que la probéis vosotros: decidnos y os damos acceso. Y tenemos el instalador de la
v2 listo. Quedamos a la espera de que nos dijerais dónde lo ibais a instalar para
adaptarlo a ese entorno y no hemos tenido respuesta, así que os mandamos una versión
general, que funciona tal cual; si al instalarlo surge alguna duda del entorno concreto,
preguntadnos y lo ajustamos.

Una petición para la próxima ronda: separad "correcta dado el documento" de "suficiente
para el paciente". Es lo que hace que la evaluación mida el sistema y el material por
separado, que es lo que necesitamos los dos.

Saludos, Marcelo
