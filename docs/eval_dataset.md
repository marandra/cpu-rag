# Dataset de Evaluación

Este documento describe el dataset utilizado para evaluar el sistema RAG de preguntas sobre cirugía perioperatoria.

## Resumen

- **Total de preguntas**: 55
- **In-scope** (respuesta en los documentos): 36
- **Out-of-scope** (sin respuesta disponible): 19
- **Formato**: JSON con metadatos estructurados por pregunta

## Categorías In-Scope (36 preguntas)

Preguntas cuya respuesta **sí está** en los documentos del sistema.

| Categoría | N | Descripción | Ejemplos |
|-----------|---|-------------|----------|
| **procedimiento** | 6 | Descripción de cirugías, técnicas, duración | "cuanto dura la operacion", "que es la anestesia epidural" |
| **recuperación** | 8 | Post-operatorio, movilización, alimentación, alta | "cuando me dejan comer", "es normal que no haga caca" |
| **preparación** | 8 | Ayuno, hábitos, ejercicio, condiciones previas | "puedo tomar agua antes", "tengo diabetes hay algo especial" |
| **riesgos** | 3 | Complicaciones, efectos secundarios | "me pueden quedar problemas para aguantar" |
| **general** | 3 | Información del programa, consentimiento | "que es el programa RICA", "puedo retirar el consentimiento" |
| **alternativas** | 2 | Opciones no quirúrgicas | "hay otra cosa antes de operar", "que pasa si no me opero" |
| **nutrición** | 2 | Alimentación preoperatoria | "que como antes de operarme", "que es el cribado nutricional" |
| **dolor** | 2 | Manejo del dolor postoperatorio | "que me dan para el dolor", "si los calmantes no me hacen nada" |

## Categorías Out-of-Scope (19 preguntas)

Preguntas que el sistema **debe rechazar** con "No tengo información sobre eso."

| Categoría | N | Descripción | Ejemplos |
|-----------|---|-------------|----------|
| **fuera_de_alcance** | 15 | Temas no cubiertos o no médicos | "cuanto cuesta", "que es Python", "como se hace una paella" |
| **zona_gris** | 3 | Relacionadas pero sin respuesta en docs | "puedo tomar ibuprofeno", "puedo conducir después" |
| **otro_procedimiento** | 1 | Pregunta sobre procedimiento diferente | "tengo fisura, me tienen que operar?" (OOS en contexto hemorroides) |

### Subtipos de fuera_de_alcance

- **Administrativo/legal**: costes, baja laboral, segunda opinión, público vs privado
- **Completamente irrelevante**: geografía, política, tecnología, cocina
- **Médico no cubierto**: dolor de cabeza, rodilla, fiebre niño, tensión arterial
- **Meta**: "quién eres tú"

## Perfiles de Paciente

Cada pregunta simula un tipo de paciente real:

| Perfil | Características | Ejemplo de pregunta |
|--------|-----------------|---------------------|
| **ansioso** | Preguntas largas, contexto emocional, miedos | "me van a operar y estoy muy nervioso que me van a hacer exactamente" |
| **baja_alfabetización** | Errores ortográficos, lenguaje coloquial | "cuanto tienpo de ayuno antes de la sirugia" |
| **joven** | Preguntas directas, cortas | "riesgos", "q pasa si no me opero" |
| **mayor** | Preguntas sobre familiares, dudas prácticas | "mi madre tiene 72 años y la van a operar" |

### Errores ortográficos simulados

Para evaluar robustez ante input real de pacientes:
- "operasion" → operación
- "sirugia" → cirugía  
- "tienpo" → tiempo
- "anestezia" → anestesia
- "avdomen" → abdomen
- "diabetis" → diabetes

## Metadatos por Pregunta

Cada entrada del dataset incluye:

```json
{
  "query": "texto de la pregunta",
  "intent": "descripción del intent real del paciente",
  "answerable": true/false,
  "profile": "ansioso|baja_alfabetización|joven|mayor",
  "category": "procedimiento|recuperación|...",
  "difficulty": "easy|medium|hard",
  "relevant_sources": ["documento1.md", ...],
  "expected_keywords": ["palabra1", "palabra2", ...],
  "relevant_spans": ["texto exacto esperado 1", ...],
  "procedure": "hemorroides|fisura"  // opcional, solo para preguntas específicas
}
```

### Campo `procedure` (opcional)

Algunas preguntas son específicas de un procedimiento:

| Procedure | Preguntas | Descripción |
|-----------|-----------|-------------|
| `hemorroides` | 6 | Específicas de hemorroidectomía (duración, alternativas, riesgos, recuperación) |
| `fisura` | 3 | Específicas de esfinterotomía (indicación, descripción, incontinencia) |
| *(sin campo)* | 45 | Genéricas (guidelines RICA/GPC, preparación, dolor, OOS) |

El script `demo_rag.py` filtra automáticamente según `--procedure`.

### Uso de metadatos en evaluación

- **expected_keywords**: scoring automático de cobertura de respuesta
- **relevant_spans**: verificación de que la respuesta usa el contenido correcto
- **relevant_sources**: validación de que el retrieval encuentra los documentos correctos
- **difficulty**: análisis de rendimiento por complejidad

## Cobertura del Flujo Perioperatorio

El dataset cubre el ciclo completo de atención:

```
PRE-OPERATORIO          INTRA-OPERATORIO       POST-OPERATORIO
─────────────────       ────────────────       ─────────────────
• Ayuno                 • Descripción          • Movilización
• Nutrición               cirugía              • Alimentación
• Tabaco/alcohol        • Anestesia            • Dolor
• Ejercicio             • Duración             • Recuperación
• Diabetes              • Técnicas             • Alta
• Ansiedad                (laparoscopia)       • Función intestinal
• Consentimiento                               • Cuidadores
```

## Documentos Fuente

Las respuestas provienen de 4 documentos:

1. **resumen-hemorroides.md** — FAQ hemorroidectomía
2. **resumen-fisura-anal.md** — FAQ esfinterotomía
3. **via-clinica-cirugia-adulto-rica-2021-paciente.md** — Guía RICA para pacientes
4. **gpc_555_cma_iacs_compl-pacientes.md** — Guía de cirugía mayor ambulatoria

## Métricas de Evaluación

### In-scope

| Resultado | Significado |
|-----------|-------------|
| **GOOD** | Respuesta correcta usando los fragmentos |
| **PARTIAL** | Respuesta incompleta pero relevante |
| **MISS** | Respuesta incorrecta o no aborda la pregunta |
| **FALSE_REFUSAL** | Rechaza cuando sí había información disponible |

### Out-of-scope

| Resultado | Significado |
|-----------|-------------|
| **OK_REF** | Rechaza correctamente con "No tengo información" |
| **LEAK** | Hallucina una respuesta usando conocimiento del modelo |
