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

(Texto plano. Sin negritas, sin rayas y sin comillas angulares: va tal cual en el cuerpo
del correo.)

Hola Xavi,

Hemos reproducido las 134 preguntas y las hemos leído una a una. Antes de los números va
el criterio, porque ahí está la mayor parte de la diferencia.

Lo primero es contra qué se evalúa. El sistema responde solo desde el documento de cada
procedimiento y tiene prohibido usar conocimiento propio, por seguridad, así que solo se
puede comparar contra ese documento y no contra ADA, ERAS, ASA o ASCRS.

Lo segundo es que el "No tengo información" es una instrucción, no un fallo. Está
escrito literalmente en el prompt: "Si la respuesta concreta está escrita en la
INFORMACIÓN, responde lo justo. Si no, responde EXACTAMENTE y sin añadidos: 'No tengo
información sobre eso.' Aplica aunque la información hable del mismo tema o creas saber
la respuesta por tu conocimiento general." Abstenerse donde el documento no dice nada es
la conducta pedida, no una laguna de cobertura.

Con ese criterio (correcta = responde lo que se pregunta apoyada en el documento, sin
inventar y sin des-acotar una regla, o se abstiene donde el documento no da material)
hemos evaluado vuestras 134 respuestas. El 9 % pasa a un 62 %, 83 de 134. La lectura
completa, pregunta a pregunta y con el motivo de cada una, va en el adjunto
auditoria_134_evaluacion.md.

Dicho eso, el 38 % restante son fallos reales, y los peores están donde los señaláis:
las preguntas de miedo, culpa y vergüenza. Agrupando nosotros las 37 que nos parecen
emocionales, la versión que auditasteis acierta 23; el corte exacto lo podéis rehacer
sobre el adjunto. Hay además invención pura, como la hipoglucemia nocturna que "puede
llevar a riesgo vital" y no está en ningún sitio, y condiciones que se pierden, como la
premedicación prometida sin el "cuando el grado de ansiedad y temor sea elevado".

Eso está arreglado, en una versión que no teníais. Auditasteis la v1.1. La v2.2 cambia
el modelo y la redacción de los documentos. Sobre las mismas 134 y con el mismo
criterio: 119 de 134, un 89 %. En las emocionales, de 23 a 32. Va también respuesta a
respuesta en auditoria_134_v22.md.

Para que lo comprobéis vosotros, el instalador está listo; convendría saber dónde lo
ibais a instalar o probar para adaptarlo a ese entorno, así que os mandamos una versión
general que funciona tal cual pero va un poco más lenta. También tenemos la v2.2
desplegada en un servidor a vuestra disposición: decidnos y os damos acceso.

Saludos, Marcelo
