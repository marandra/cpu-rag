# Los corpus de este A/B, preservados

`corpus/markdown/` está en `.gitignore`, así que las reescrituras vivirían solo
en el portátil y en el cluster. Se copian aquí porque **son el objeto del
experimento**: sin el texto, los JSON de al lado no son interpretables ni
reproducibles.

- `diabetes.v4.md` — sesión ciega, desde `docs/corpus_rewrite_brief_diabetes.md`
- `cirugia-abdominal.v4.md` — sesión ciega, desde `docs/corpus_rewrite_brief_cirugia.md`
- `diabetes.v4-mio.md` — control de contaminación (escrito conociendo las preguntas)

Para volver a correrlo hay que copiarlos a `corpus/markdown/` primero.
