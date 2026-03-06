# Prompt: Generate realistic patient evaluation queries

Use this prompt with any LLM (GPT-4o, Claude, etc.) to generate realistic
patient queries for evaluating retrieval quality. Paste the source documents
as context after the prompt.

---

## Prompt

```
Eres un generador de consultas realistas de pacientes para evaluar un sistema
de búsqueda de información médica en español.

Tu tarea: generar consultas que simulen cómo pacientes REALES escribirían
preguntas en un chat o buscador sobre su cirugía. NO generes preguntas
perfectas de libro de texto.

CONTEXTO: Los documentos del sistema cubren estos temas:
- Cirugía de hemorroides (hemorroidectomía)
- Cirugía de fisura anal (esfinterotomía)
- Vía clínica RICA (recuperación intensificada en cirugía del adulto)
- Cuidados perioperatorios en cirugía mayor abdominal

PERFILES DE PACIENTE (alterna entre ellos):
1. Persona mayor (65-80 años): escribe despacio, sin tildes, frases cortas,
   puede confundir términos médicos, usa "operación" en vez de "cirugía"
2. Adulto ansioso (35-50 años): preguntas largas y algo desorganizadas,
   mezcla varias dudas en una sola pregunta, usa signos pero comete errores
3. Joven (18-30 años): escribe como en WhatsApp, abreviaciones, sin signos
   de puntuación, sin ¿¡, puede tener faltas de ortografía
4. Persona con español como segunda lengua: errores gramaticales, mezcla
   de género, preposiciones incorrectas, vocabulario limitado
5. Persona con baja alfabetización: errores ortográficos frecuentes,
   frases muy simples, puede describir síntomas en vez de usar términos

REGLAS OBLIGATORIAS:
- NUNCA uses "¿" ni "¡" de apertura (los pacientes reales no los usan)
- Al menos 30% de las consultas deben tener errores ortográficos reales
  (ej: "operasion", "emorroide", "sirugia", "cuanto tienpo", "anestezia")
- Al menos 20% deben ser coloquiales/informales
  (ej: "me van a operar del culo y tengo miedo", "que pasa si no me opero")
- Incluye consultas vagas que un paciente real haría
  (ej: "dolor despues", "cuando puedo comer", "es peligroso")
- Algunas deben ser sobre temas que SÍ están en los documentos pero
  formuladas de forma tan diferente que sean difíciles de buscar
- Incluye 3-5 consultas que NO se pueden responder con los documentos
  (para evaluar cómo el sistema maneja preguntas fuera de alcance)

FORMATO DE SALIDA (JSON):
[
  {
    "query": "la consulta tal como la escribiría el paciente",
    "intent": "qué información busca realmente (en lenguaje técnico, 1 línea)",
    "answerable": true/false,
    "profile": "mayor|ansioso|joven|L2|baja_alfabetización",
    "relevant_sources": ["archivo1.md", ...],
    "expected_keywords": ["palabra1", "palabra2", ...],
    "category": "procedimiento|riesgos|preparación|recuperación|general|alternativas|dolor|nutrición|fuera_de_alcance",
    "difficulty": "easy|medium|hard"
  }
]

DIFFICULTY CRITERIA:
- easy: la consulta contiene palabras clave que aparecen literalmente en los
  documentos (ej: "hemorroides", "ayuno")
- medium: la consulta usa sinónimos o lenguaje coloquial que requiere
  comprensión semántica (ej: "me van a abrir" = cirugía)
- hard: la consulta es muy vaga, tiene muchos errores, o describe el concepto
  sin usar ninguna palabra clave (ej: "eso que te meten por la nariz para
  respirar" = espirómetro incentivador)

Genera exactamente 40 consultas con esta distribución:
- 8 easy, 20 medium, 12 hard
- ~8 por perfil de paciente
- 3-5 fuera de alcance (answerable=false)
- Al menos 5 con errores ortográficos graves
- Al menos 5 extremadamente coloquiales
- Cubrir todas las categorías

IMPORTANTE: Las expected_keywords son las palabras que DEBERÍAN aparecer en
los chunks recuperados (no en la consulta del paciente). Representan la
información correcta que el sistema debe encontrar.
```

---

## Usage notes

- Review generated queries manually: LLMs tend to be "too clean" even when
  asked to be messy. Delete or rewrite queries that feel artificial.
- The `relevant_sources` and `expected_keywords` should be validated against
  actual document content.
- `answerable: false` queries should have `relevant_sources: []` and
  `expected_keywords: []`.
- You can regenerate with different seeds/temperatures for variety.
- Consider adding real queries from user testing sessions as they become
  available — those are always more realistic than synthetic ones.
