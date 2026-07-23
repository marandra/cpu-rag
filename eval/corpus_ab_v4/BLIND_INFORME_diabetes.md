# Informe ciego — reescritura de `diabetes.md` → `diabetes.v4.md`

Sesión independiente. No se ha abierto ni buscado en `eval/` (salvo para escribir
este fichero), `reports/`, `ARRANQUE.md`, `docs/corpus_guidelines.md`,
`tools/audit_*.py` ni ningún `.xlsx`. Los únicos ficheros leídos han sido
`docs/corpus_rewrite_brief_diabetes.md` y `corpus/markdown/diabetes.md`.

---

## 1. Transformaciones aplicadas

- **R6 + R5**: todas las viñetas nominales del original («HbA1c ≥6,5 %.»,
  «Síntomas: sed, cansancio…», «Ayuno…», «Adherencia: tomar según pauta») se han
  convertido en oraciones con verbo conjugado y sujeto explícito. El sujeto por
  defecto del paciente es «usted»; el del equipo clínico, «su equipo sanitario».
  Los infinitivos sueltos y los imperativos sin destinatario han desaparecido.
- **R5, caso crítico**: donde el original elidía el actor y la acción podía
  reasignarse al paciente, se ha explicitado el actor clínico. Ver §5.
- **R2**: el acto va siempre en la principal. Ejemplos:
  «Su equipo sanitario individualiza su tratamiento farmacológico según…» en vez
  de «Se individualiza según…»; «Combinar sus hidratos de carbono con proteína…
  le reduce los picos de glucosa» (el acto, no «es importante combinar»);
  «Perder entre un 5 % y un 10 % de su peso le mejora la glucemia…».
- **R1**: cada frase repite su ancla. En «Objetivos de control» cada una de las
  cinco cifras lleva dentro «un objetivo que su equipo sanitario puede
  individualizar para usted», porque el matiz «(individualizables)» vivía solo en
  el título de sección. En «Prevención de complicaciones» cada medida repite
  «Usted previene las complicaciones…». En «Días de enfermedad» cada frase repite
  «mientras está enfermo», porque el condicionante vivía solo en el encabezado de
  la FAQ. En «Ejercicio» las precauciones repiten «si se trata con insulina o con
  ciertos antidiabéticos» y «cuando hace ejercicio».
- **R7 vs R1**: se ha resuelto siempre a favor de R1. No hay frases de relleno ni
  cortesías, pero sí palabras repetidas a propósito (el ancla de cada sección).
- **Cifras, umbrales y fármacos**: intactos. Solo se han desplegado los símbolos
  a palabras conservando el umbral («≥126 mg/dl» → «es igual o superior a
  126 mg/dl», «<7 %» → «debe ser inferior a 7 %»). Nombres de fármaco tocados:
  ninguno (metformina, sacarina, aspartamo, ciclamato, paracetamol, glucagón).

## 2. Contenido movido de sección (R4)

Se han **eliminado por completo** las secciones «Preguntas frecuentes» y
«Creencias frecuentes (mitos)». Cada hecho se enuncia ahora una sola vez en su
sección temática:

| Origen | Hecho | Destino |
|---|---|---|
| Mitos | No hay «poco» o «mucho» azúcar; se tiene diabetes según cifras científicas | Conceptos generales → Diagnóstico |
| Mitos | La DM2 inicial no suele dar síntomas; se descubre por análisis | Conceptos generales → Síntomas |
| Mitos | Sin insulina también se es diabético; se puede tratar solo con dieta y ejercicio, o con pastillas y/o insulina | Tratamiento farmacológico |
| Mitos | No hay diabetes «buena» o «mala», sino bien o mal controlada | Objetivos del tratamiento |
| Mitos | La dieta del diabético es la dieta equilibrada general; en exceso de peso, reducir calorías | Alimentación (entradilla) |
| Mitos | El plan de alimentación siempre acompaña, también con pastillas o insulina | Alimentación (entradilla) |
| Mitos | La insulina inyectada es como la del páncreas; reduce complicaciones, incluida la ceguera | Insulina (entradilla) |
| Mitos | La glucemia alta actúa silenciosamente, favoreciendo complicaciones crónicas | Prevención de complicaciones (entradilla) |
| Mitos | Bastan 15-20 g de azúcar; más cantidad da problemas posteriores | Hipoglucemia → tratamiento (fusionado con los 15-20 g que ya estaban allí) |
| Mitos | No hay alimentos prohibidos (legumbres, pan, melón); se ajustan las cantidades | Alimentación (fusionado con «ningún alimento prohibido; importan cantidad, frecuencia y calidad») |
| FAQ «¿Puedo ir de vacaciones?» | Todo el bloque de viaje | Sección temática nueva **«Viajes y vacaciones»** |
| FAQ «¿Y si estoy enfermo?» | Todo el bloque de días de enfermedad | Sección temática nueva **«Días de enfermedad»** |

Las dos secciones nuevas no aportan ningún hecho: son el mismo contenido
promovido de FAQ a sección temática, como exige R4. Su posición: «Días de
enfermedad» tras «Hipoglucemia» (situaciones agudas) y «Viajes y vacaciones»
tras «Autoanálisis» (por el material de autocontrol y la conservación de la
insulina).

**Deduplicaciones** (R4: «enuncia cada hecho una sola vez»):

- «Es normal sentir preocupación, miedo o frustración» (Educación terapéutica) y
  «tras el diagnóstico es normal sentir negación, frustración o miedo» (Aspectos
  psicológicos) se han fusionado en una sola frase que conserva las cuatro
  emociones y el matiz «tras el diagnóstico». Vive en Aspectos psicológicos.
- «Resolver dudas con profesionales sanitarios» y «preguntar dudas al equipo
  sanitario y anotarlas entre consultas» se han fusionado en Aspectos
  psicológicos. Nota: «consultar dudas» sigue apareciendo además en Tratamiento
  farmacológico → adherencia, porque allí es una regla distinta (dudas sobre la
  medicación) y estaba en el original.
- «Contar con apoyo familiar/social ayuda» y «buscar apoyo en el entorno y en
  asociaciones» se han fusionado en una frase.
- «Puede no dar síntomas durante años» aparece en la Introducción y en Síntomas
  **igual que en el original**; se ha mantenido en ambos sitios por fidelidad
  estructural (es la única repetición deliberada heredada).

## 3. Preguntas de paciente previstas (R3) y dónde las responde el texto

Cubiertas por una frase explícita:

| Pregunta prevista | Sección que la responde con una frase |
|---|---|
| ¿Cuándo se dice que tengo diabetes? ¿Qué cifra? | Diagnóstico (4 criterios, uno por frase) |
| ¿Por qué me ha dado diabetes? ¿Es hereditaria? | Causas de la DM2 |
| ¿Qué síntomas da? ¿Puedo tenerla sin notar nada? | Síntomas |
| ¿En qué se diferencia el tipo 1 del tipo 2? ¿Y la gestacional? | Tipos |
| ¿Se puede prevenir? | Prevención |
| ¿Qué glucemia debo tener? ¿Y en ayunas? ¿Y después de comer? | Objetivos de control |
| ¿Qué HbA1c, tensión y colesterol debo tener? | Objetivos de control |
| ¿Son fijos esos objetivos? | Objetivos de control (matiz en cada frase) |
| ¿Hay alimentos prohibidos? ¿Puedo comer pan / legumbres / melón? | Alimentación |
| ¿Cuánto peso tengo que perder? | Alimentación |
| ¿Cuántas comidas al día? | Alimentación |
| ¿Puedo tomar fruta? ¿Y zumo? | Grupos de alimentos |
| ¿Puedo comer patata? ¿Frita? | Grupos de alimentos |
| ¿Puedo comer chocolate? ¿Frutos secos? ¿Embutido? | Grupos de alimentos |
| ¿Puedo usar sacarina y otros edulcorantes? | Grupos de alimentos |
| ¿Sirven los suplementos o los productos «naturales»? | Grupos de alimentos |
| ¿Qué es el método del plato? | Dieta mediterránea y método del plato |
| ¿Tengo que quitar los hidratos de carbono? | Hidratos de carbono |
| ¿Qué es una ración? | Raciones |
| ¿Puedo beber alcohol? ¿En ayunas? | Alcohol |
| ¿Cómo debo cocinar? ¿Puedo reutilizar el aceite? | Técnicas culinarias |
| ¿Cuánto ejercicio tengo que hacer? ¿Vale caminar? | Ejercicio físico |
| ¿Las tareas de casa cuentan como ejercicio? | Ejercicio físico |
| ¿El ejercicio me puede bajar el azúcar? ¿Qué llevo encima? | Ejercicio físico (precauciones) |
| ¿Tengo que consultar antes de hacer deporte intenso? | Ejercicio físico (precauciones) |
| ¿Cómo evito las complicaciones? ¿Importa fumar? | Prevención de complicaciones |
| ¿Cómo me cuido los pies? ¿Puedo andar descalzo? ¿Puedo usar callicidas? | Pie diabético |
| ¿Cuándo debo consultar por una herida en el pie? | Pie diabético |
| ¿Qué le hace la diabetes al corazón, a los ojos, al riñón y a los nervios? | Complicaciones por órgano |
| ¿Cada cuánto reviso la vista? | Complicaciones oculares («periódicas», sin intervalo) |
| ¿Qué me van a mirar en las revisiones? | Seguimiento sanitario |
| ¿Cómo se trata la diabetes tipo 2? ¿Siempre con pastillas? | Tratamiento farmacológico |
| ¿Para qué sirve la metformina? | Tratamiento farmacológico |
| ¿Puedo dejar la medicación si me encuentro bien? ¿Cambiar la dosis? | Tratamiento farmacológico (adherencia) |
| ¿Qué efectos secundarios tienen los antidiabéticos? | Tratamiento farmacológico |
| ¿Voy a necesitar insulina? ¿Es un fracaso? | Insulina |
| ¿Qué tipos de insulina hay? | Insulina |
| ¿Dónde me pincho? ¿Por qué hay que rotar? | Insulina |
| ¿Cómo guardo la insulina? ¿Y si hace calor? | Insulina |
| ¿La insulina engorda? ¿Qué son las lipodistrofias? | Insulina |
| ¿Qué es una hipoglucemia y cómo la noto? | Hipoglucemia |
| ¿Qué hago si me baja el azúcar? ¿Cuánto azúcar tomo? | Hipoglucemia |
| ¿Qué hago si alguien se queda inconsciente? | Hipoglucemia |
| ¿Qué hago si me pongo enfermo, con gripe o vómitos? | Días de enfermedad |
| ¿Suspendo la medicación si no como? | Días de enfermedad |
| ¿Puedo tomar paracetamol? ¿Y jarabes? | Días de enfermedad |
| ¿Qué como si tengo diarrea? ¿Y si vomito? | Días de enfermedad |
| ¿Cuándo tengo que llamar al médico estando enfermo? | Días de enfermedad (6 criterios) |
| ¿Cada cuánto me pincho el dedo? ¿Cómo guardo las tiras? | Autoanálisis |
| ¿Puedo viajar? ¿Qué llevo? ¿Pasa algo con los rayos X del aeropuerto? | Viajes y vacaciones |
| ¿Y si cambio de horario en un vuelo largo? | Viajes y vacaciones |
| ¿Es normal sentirme mal anímicamente? ¿Dónde busco apoyo? | Aspectos psicológicos |

**Previstas SIN material en el documento (no rellenadas, R3):**

1. **¿Qué es la prediabetes y con qué cifras se define?** El título del documento
   la menciona, pero el cuerpo no la define ni le da umbrales en ningún punto.
2. **¿Qué es la HbA1c?** Se usa como criterio y como objetivo, pero el documento
   no la define en ninguna frase.
3. **¿La diabetes tipo 2 tiene cura?** Solo hay «la DM2 es progresiva» y «ningún
   producto milagro la cura»; no hay frase que lo afirme o lo niegue.
4. **¿Qué hago si me olvido una dosis de la medicación?**
5. **¿Puedo conducir teniendo diabetes o tratándome con insulina?**
6. **¿Cómo se usa el glucagón y quién lo pone?** Solo aparece como material que
   llevar de viaje.
7. **¿Cómo se mide la acetona en orina?** Solo aparece como criterio de consulta.
8. **¿Debo vacunarme de la gripe?**
9. **Diabetes y embarazo / lactancia en una mujer ya diabética** (solo se define
   la diabetes gestacional).
10. **Salud bucodental.**
11. **Sexualidad y disfunción eréctil** (la neuropatía menciona «problemas
    urinarios», nada más).
12. **¿Cada cuánto exactamente son las revisiones?** El documento dice
    «periódicas» sin ningún intervalo, ni para la analítica ni para el fondo de
    ojo ni para los pies.
13. **¿Qué hago si me sube mucho el azúcar fuera de una enfermedad aguda?** El
    umbral de 300 mg/dl solo está enunciado dentro de los días de enfermedad, y
    se ha mantenido acotado allí para no ampliar su alcance.
14. **¿Cuántos gramos de hidratos por comida / cuántas raciones al día?** Se
    define la ración, no la cantidad diaria.
15. **¿Qué es el índice glucémico?** No aparece.

## 4. Qué he dejado fuera por dudar entre reformulación y añadido

- **«Mezclas» de insulina**: el original solo dice «mezclas». No la he definido
  como «combinación de dos tipos de insulina en un mismo preparado» porque sería
  conocimiento propio. Queda «Existen también mezclas de insulinas».
- **«Insulinas intermedias»**: el original no les asigna descripción (a
  diferencia de rápidas y basales). No he inventado ninguna.
- **Prediabetes**: no he añadido definición ni umbrales pese a estar en el
  título.
- **Hipoglucemia**: no he añadido «repita la glucemia a los 15 minutos» ni
  «tome después un hidrato de absorción lenta», que serían el complemento
  clínico esperable de la regla de los 15-20 g. No están en el original.
- **Glucagón para la inconsciencia**: el original lo menciona solo en el bloque
  de viaje. No lo he conectado con «si está inconsciente, no dar nada por boca».
- **Objetivo «<7 %»**: no he añadido que sea equivalente a ninguna glucemia media.
- **Tabaco**: no he añadido ninguna vía de deshabituación; solo está «no fumar».
- **«Se descubre por análisis»**: he escrito «en un análisis», sin calificarlo de
  rutinario ni de sangre, para no precisar más que el original.
- **Terminología**: no he cambiado «persona diabética» / «persona con diabetes»
  donde el original usaba una u otra, salvo la conversión gramatical necesaria.

## 5. Choques entre reglas y cómo los resolví

1. **R5 vs. fidelidad, en «Prevención de la hipoglucemia»**. El original dice
   «Prevención: horarios regulares, ajuste de medicación, adaptar alimentación y
   ejercicio», sin sujeto. Atribuir el «ajuste de medicación» al paciente crearía
   exactamente el defecto que R5 describe (convertir en instrucción al paciente
   algo del equipo clínico) y además contradiría la regla de adherencia del mismo
   documento («no modificar dosis sin indicación»). Resuelto atribuyéndolo al
   equipo: «con el ajuste de su medicación que le indique su equipo sanitario».
2. **R5 en «Diagnóstico»**. Los cuatro criterios no tienen actor. Resuelto con
   doble anclaje: encabezado «Su equipo sanitario diagnostica que usted tiene
   diabetes con cualquiera de estos cuatro criterios» y cada criterio en forma
   «Usted tiene diabetes si su … es …». El sujeto de la cifra es el paciente; el
   del acto de diagnosticar, el equipo.
3. **R5 en «Objetivos de control»**. «Individualizables» no tiene actor. Se ha
   atribuido al equipo sanitario, no al paciente.
4. **R1 vs. R7 en «Objetivos de control» y «Prevención de complicaciones»**.
   Repetir el anclaje en cada frase produce un texto redundante de leer. Aplicada
   la regla del brief: gana R1.
5. **R1 vs. R7 en «Días de enfermedad»**. El condicionante «mientras está
   enfermo» se repite en casi todas las frases del bloque, incluidas las seis de
   criterios de consulta, para que ninguna pueda extraerse como regla general.
   Coste: el umbral de 300 mg/dl queda deliberadamente acotado a la enfermedad
   aguda (ver §3, pregunta sin material nº 13).
6. **R2 vs. R6 en «Grupos de alimentos»**. Una sola frase con toda la lista de
   «a diario» sería más corta (R7) pero dejaría cada alimento sin su matiz
   propio. Se ha partido en una frase por alimento cuando el alimento lleva
   matiz (fruta entera vs. zumo, verduras crudas o cocidas, frutos secos en
   cantidad moderada, patata no frita), agrupando el resto.
7. **R4 vs. estructura del original**. Vaciar «Preguntas frecuentes» y «Mitos»
   deja el documento sin ninguna sección de FAQ. Se ha aceptado: R4 es explícita
   en que ningún hecho clínico puede vivir solo ahí, y mantener una FAQ vacía o
   duplicada violaría «enuncia cada hecho una sola vez».
8. **R7 vs. fidelidad, en la Introducción**. «Puede no dar síntomas durante
   años» está en la Introducción y en Síntomas en el original. Se ha conservado
   la duplicación por fidelidad estructural, aunque R7 sugeriría eliminarla.

## 6. Dudas clínicas detectadas y NO tocadas

Se anotan por encargo del brief; **ninguna se ha corregido ni matizado en el
texto**.

1. **«Nunca suspenda la medicación de la diabetes» en días de enfermedad.** La
   práctica habitual con deshidratación, vómitos o diarrea es justo la contraria
   para algunos antidiabéticos orales (riesgo renal y de acidosis). El documento
   lo enuncia sin excepción por fármaco; se ha reproducido tal cual, con el
   único matiz que el propio original añade («salvo indicación médica»).
2. **«Algo más de HC de lo habitual» + «té con poco azúcar» + «zumo de fruta»**
   en días de enfermedad, junto al criterio de consultar si la glucemia supera
   300 mg/dl. Las dos indicaciones conviven sin jerarquía en el original.
3. **Hipoglucemia: 15-20 g y nada más.** No hay control a los 15 minutos, ni
   repetición de la toma si no se resuelve, ni hidrato lento posterior.
4. **«Más cantidad de azúcar da problemas posteriores»** se enuncia como hecho
   absoluto, sin cuantificar ni explicar; procede de la lista de mitos.
5. **Inconsciencia por hipoglucemia**: se indica «no dar nada por boca y avisar
   a urgencias», pero no se menciona el glucagón, que sí aparece en el bloque de
   viaje.
6. **LDL <100 mg/dl** como objetivo único, sin distinguir prevención primaria de
   secundaria pese a que el documento reconoce el alto riesgo cardiovascular.
7. **Solape entre umbrales**: el objetivo de glucemia en ayunas llega a
   130 mg/dl mientras el criterio diagnóstico es ≥126 mg/dl. No es un error, pero
   puede confundir al paciente y el documento no lo aclara.
8. **Paracetamol si fiebre**, sin dosis, sin límite de días y sin mención de la
   función hepática o renal.
9. **Acetona en orina**: se usa como criterio de consulta sin explicar qué es,
   cómo se mide ni que hace falta material para medirla.
10. **El título anuncia «prediabetes»** y el cuerpo no la trata en ningún punto.
11. **Diabetes tipo 1**: se dice que «necesita insulina desde el inicio», lo cual
    es correcto, pero el documento es de tipo 2 y no vuelve a mencionarla.
12. **Ejercicio**: se pide consultar antes de ejercicio intenso con retinopatía o
    neuropatía, pero no se limita ningún tipo concreto de ejercicio.
