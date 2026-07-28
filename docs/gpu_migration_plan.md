# Plan — migración a GPU y hosting en spot (cliente)

**Estado: PROPUESTA. Nada de lo que hay aquí sobre GPU está medido.**

Este documento planifica (a) portar el servicio de CPU a GPU y (b) la arquitectura
de hosting en instancias spot que **recomendamos al cliente**, que es quien
hospeda. El entregable sigue siendo el bundle; lo nuevo es que además le damos
una arquitectura de despliegue y unos requisitos de hardware.

> ### Advertencia epistémica — leer antes de citar cualquier número
>
> Todas las cifras de GPU de este documento son **extrapolaciones** del hecho
> medido de que el decode de gemma-4-26B-A4B está limitado por ancho de banda de
> memoria (ver `docs/ec2_test_env.md` §4 y la memoria `gemma-single-socket-bw-ceiling`).
> **No hemos ejecutado este modelo en una GPU ni una sola vez.** Están marcadas
> como `[EST]`. El §1 existe precisamente para convertirlas en `[MED]` antes de
> que nadie tome una decisión de dinero.
>
> Lo único medido aquí es el lado CPU: **4,83 tok/s** con N=1 × nT=8 en
> r7i.2xlarge (`docs/ec2_test_env.md:135`).

---

## 0. La decisión en una frase

El modelo pesa **17 GB (Q4_K_M)**, así que el suelo de hardware es una GPU de
**24 GB de VRAM** — por debajo no entra y se cae a velocidad de CPU. Una vez
pagada esa tarjeta, la concurrencia sale del hueco sobrante y es casi gratis.
La caja más barata que cumple es la **g6.xlarge (L4 24 GB)**, y en spot cuesta
**menos que la caja CPU actual**.

**Lo caro no es el port (≈2 días de código). Lo caro es revalidar las 134
preguntas**, porque cambiar de backend cambia las respuestas.

---

## 1. Relevamiento previo — los números que faltan

Nada del §3 en adelante se decide sin esto. Cinco bloques independientes; los
bloques A, D y E no necesitan tocar código y pueden empezar ya.

### 1.A — Carga real (dato del cliente; es el que más pesa)

Sin esto no se puede decidir si la GPU se justifica siquiera. Sale de los logs de
nginx del despliegue actual (`nginx/` en el bundle) o de la instrumentación del
servicio.

| Nº | Dato a obtener | Cómo | Qué decide |
|---|---|---|---|
| A1 | Consultas/hora: media, p95, pico absoluto | Contar líneas de acceso por hora sobre ≥4 semanas | Si la GPU se justifica |
| A2 | Distribución horaria y por día de semana | Histograma por hora local | Si el apagado fuera de horario es viable (~3× de ahorro) |
| A3 | Concurrencia real observada | Peticiones solapadas en ventanas de 60 s | Cuántos slots hacen falta de verdad |
| A4 | Latencia aceptable acordada | Preguntar al cliente, no inferirla | Si los ~67 s de la CPU actual ya son un problema |
| A5 | Tolerancia a un hueco de servicio | Preguntar: ¿1-2 min de caída al mes es aceptable? | Si spot es viable sin fallback |
| A6 | Reparto de carga entre los tres procedimientos | Agrupar por `procedure` | Si conviene un proceso único o separar |

> **Criterio de abandono:** si A1 da una carga que una sola réplica CPU absorbe
> y A4 dice que un minuto por respuesta está bien, **la migración no se hace**.
> Hay que estar dispuestos a que este bloque mate el proyecto.

### 1.B — Banco de pruebas en GPU (media hora alquilada, ~$0,60)

Una `g6.xlarge` on-demand, el bundle con `n_gpu_layers=-1`, y estas seis medidas.
Es el bloque que convierte todos los `[EST]` en `[MED]`.

| Nº | Medida | Cómo | Qué decide |
|---|---|---|---|
| B1 | tok/s decode, 1 stream | Una consulta, leer `usage` | El factor de mejora real vs 4,83 |
| B2 | **KV bytes/token** | `nvidia-smi` con el prefijo cargado, restar los pesos, dividir por tokens | **Cuántos slots caben** — es el número más importante del bloque |
| B3 | VRAM total con los 3 prefijos cargados | `nvidia-smi` tras calentar los tres | Si un solo proceso sirve ambos perfiles |
| B4 | Rampa de concurrencia 1→2→4→8→16 | Carga sintética, medir tok/s por usuario y agregado | La curva real de degradación del MoE |
| B5 | Prefill con caché de prefijo caliente | Tiempo a primer token de una pregunta nueva | Si la maquinaria de snapshots se puede tirar |
| B6 | Tiempo de carga del GGUF a VRAM | Cronometrar el arranque, por origen (EBS gp3, NVMe local) | El tiempo de recuperación tras una interrupción |

> **Sobre B4 — el matiz MoE.** En un modelo denso, batchear es casi gratis porque
> los pesos se leen una vez para todo el lote. En un MoE, cada token del lote
> puede enrutar a expertos distintos, así que la unión de expertos tocados crece
> con el batch y se acaba leyendo buena parte de los 26B en vez de los ~4B
> activos. `[EST]` la degradación por usuario a batch 8 es de 2-3×, no de 8× —
> pero **es exactamente lo que B4 tiene que confirmar o desmentir**. Es la
> estimación más frágil del documento.

### 1.C — Revalidación de calidad (el coste dominante)

| Nº | Tarea | Qué decide |
|---|---|---|
| C1 | Correr las 134 contra el backend GPU | El nuevo denominador de decisión |
| C2 | Diff contra las 123/134 de la v2.2 en CPU | Cuántas respuestas se mueven |
| C3 | **Lectura a mano** de las que se mueven | Si la calidad se mantiene, mejora o empeora |

> Las respuestas **van a cambiar**: ya lo comprobamos entre builds portable y
> native de la *misma* CPU (83 vs 88 tokens sobre el mismo snapshot,
> `docs/ec2_test_env.md` §1). Cambiar a CUDA altera el orden de acumulación
> mucho más. Y por política del proyecto la calidad se juzga **leyendo**, no con
> el scorer (memoria `feedback-eval-by-reading`) — así que C3 es trabajo humano y
> no se puede automatizar. **Es el mayor coste del proyecto, no el código.**

### 1.D — Viabilidad de spot (no toca código)

Spot es capacidad sobrante que AWS vende con descuento y **puede reclamar en
cualquier momento**, avisando 2 minutos antes. Todo el argumento económico del
§7 —los ~$0,30-0,45/h, por debajo de la caja CPU actual— se apoya en que ese
descuento sea real y estable **en la región del cliente**. Si no lo es, la
estrategia se cae y hay que volver a on-demand.

Lo que se suele tratar como un solo riesgo son en realidad **tres independientes**,
y por eso son tres datos distintos:

| Nº | Riesgo | Dato a obtener | Cómo | Qué decide |
|---|---|---|---|---|
| D1 | **Me la quitan** | Frecuencia de interrupción de g6/g5.xlarge en la región del cliente, por tramos (<5%, 5-10%, … >20% al mes) | AWS Spot Instance Advisor | Cruzado con A5: si sale >20%, el diseño sin fallback del §6 no se sostiene |
| D2 | **El descuento no es el que creo** | Precio spot de los últimos 90 días: descuento medio **y volatilidad** | `describe-spot-price-history` (abajo) | Si spot sigue ganando a la caja CPU; y si el gasto mensual es predecible |
| D3 | **No consigo ninguna** | Spot placement score por región y AZ | Spot placement score (consola / API) | Si el *reemplazo* tras una interrupción es realista — todo el capacity rebalancing del §4 lo asume |

```bash
aws ec2 describe-spot-price-history \
  --instance-types g6.xlarge g5.xlarge g6.2xlarge g5.2xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -d '90 days ago' -Iseconds)" \
  --region <la-del-cliente>
```

> **Corrección a la estimación del §7.** El «60-70% de descuento» que se maneja
> como típico es de instancias **generales**. En GPU los descuentos suelen ser
> bastante **menores (30-50%)**, precisamente porque hay cola para ellas. Si en la
> región del cliente g6.xlarge spot está solo un 30% por debajo, sale a ~$0,57 —
> por *encima* de la caja CPU de $0,50, y la frase «spot cuesta menos que la CPU
> actual» deja de ser cierta. **D2 es el dato que valida o mata el §7.**

**Cómo se leen los tres juntos:**

| D1 interrupción | D2 descuento real | Decisión |
|---|---|---|
| <10% | >50% | Spot puro sin fallback — el plan tal cual |
| <10% | 20-40% | Spot sigue ganando, pero por poco; comparar contra on-demand + apagado horario |
| >20% | cualquiera | Spot solo con fallback (y entonces aplica la pega de gobernanza del §6), o directamente on-demand |

**Dos matices que evitan una conclusión precipitada:**

- **Spot y el apagado fuera de horario son palancas independientes.** El apagado
  por horario (sale de A2) ya da ~3× de ahorro, no tiene riesgo de interrupción y
  es mucho más simple de operar. Si el bloque D vuelve con malos números **no se
  pierde la mejora de coste**, se pierde la mitad de ella y queda la palanca
  fiable.
- **Estos tres datos se miden en la región y la cuenta del cliente**, no en las
  nuestras. Precio spot, frecuencia de interrupción y capacidad varían muchísimo
  entre regiones, y es él quien hospeda.

### 1.E — Prerrequisito administrativo: cuotas de GPU

No es un riesgo de spot, es un **bloqueo duro previo**: sin cuota no arranca
nada, ni siquiera on-demand. Va aparte porque afecta a dos cuentas distintas —
la nuestra (para el banco de pruebas de F1) y la del cliente (para producción).

Cuatro cosas que hay que saber:

1. **Son dos cuotas separadas**, y tener una no da la otra:

   | Cuota | Código | Para qué |
   |---|---|---|
   | Running On-Demand G and VT instances | `L-DB2E81BA` | El banco de pruebas de F1, y el fallback on-demand |
   | All G and VT Spot Instance Requests | `L-3819A6DF` | La arquitectura del §4 |

2. **Se miden en vCPU, no en instancias.** Una g6.xlarge son 4 vCPU. Y como el
   ASG del §4 tiene `max=2` (capacity rebalancing arranca la sustituta **antes**
   de que muera la vieja), hay que cubrir dos instancias a la vez, del tipo más
   grande de la lista diversificada: g5.2xlarge son 8 vCPU → **16 vCPU de cuota
   spot como mínimo**. Pedir ~24 para tener margen.

3. **Son por región.** Tenerla en Fráncfort no sirve en Irlanda.

4. **El trámite es pedir y esperar.** Se solicita por Service Quotas y **no puedes
   lanzar hasta que se apruebe**. Subidas pequeñas pueden auto-aprobarse en
   minutos; las de GPU suelen pasar por revisión humana y tardan de horas a varios
   días laborables. Por eso está en F0 y no más tarde.

```bash
# ¿Qué tengo hoy? (por región)
aws service-quotas list-service-quotas --service-code ec2 --region <region> \
  --query "Quotas[?contains(QuotaName,'G and VT')].[QuotaCode,QuotaName,Value]" --output table

# Pedir subida
aws service-quotas request-service-quota-increase --service-code ec2 \
  --quota-code L-3819A6DF --desired-value 24 --region <region>

# Estado de lo pedido
aws service-quotas list-requested-service-quota-change-history --service-code ec2 --region <region>
```

> **[MED] 2026-07-28 — nuestra cuenta (092768957966) ya está cubierta.** Medido en
> eu-central-1 y us-east-1, idéntico en ambas:
>
> | Cuota | Valor aplicado | ¿Basta? |
> |---|---|---|
> | On-Demand G and VT | **768 vCPU** | Sí, de sobra |
> | Spot G and VT | **64 vCPU** | Sí — son 16 × g6.xlarge; necesitamos 16 vCPU |
>
> **F1 no está bloqueado por cuota**, el banco de pruebas se puede alquilar
> cuando se quiera. Esto corrige la afirmación anterior de este documento de que
> «la cuota de GPU es 0 por defecto»: es falsa para esta cuenta, y en general el
> valor por defecto varía — **hay que consultarlo, no asumirlo**.
>
> **La cuenta del cliente es otra pregunta y sigue abierta.** Que la nuestra esté
> cubierta no dice nada de la suya.

---

## 2. Cambios en el código

### 2.1 Los tres cambios literales

| Qué | Dónde | Cambio |
|---|---|---|
| Offload a GPU | `src/llm.py:73-79` | añadir `n_gpu_layers=-1` a los kwargs de `Llama(...)` |
| Build CUDA | `Dockerfile:38` | `CMAKE_ARGS="-DGGML_CUDA=on"` en vez del bloque AVX2 |
| Imagen base + acceso a la GPU | `Dockerfile:44`, `docker-compose.yml` | base `nvidia/cuda:12.x-runtime`, `gpus: all` en el servicio `rag` |

### 2.2 Las cuatro trampas (esto es el trabajo de verdad)

1. **`compute_key()` no incluye el backend.** `app/snapshot_cache.py:26-46` hashea
   modelo, `n_ctx`, `flash_attn`, prompt y fulldoc — **no** el backend ni
   `n_gpu_layers`. Un build CUDA cargaría en silencio un pickle construido en CPU
   con la misma clave. **Hay que meter el backend en el hash antes de arrancar
   nada en GPU**, o se corrompen resultados sin ningún error visible.

2. **La maquinaria de snapshots probablemente sobra.** Existe para no pagar ~80 s
   de prefill. En GPU ese prefill son `[EST]` 2-4 s, y cargar un pickle de
   400-700 MB de disco tarda más que rehacerlo. Si B5 lo confirma, se elimina
   `app/snapshot_cache.py` de la ruta GPU y se calienta en el arranque. Simplifica
   mucho el despliegue (desaparece el paso `rag-generate`).

3. **`flash_attn=True` (`src/llm.py:78`) se queda**, y en CUDA rinde mejor que en
   CPU. Verificar que la build lo soporta.

4. **Se rompe la portabilidad del entregable.** Hoy el tarball corre en cualquier
   x86 AVX2 (esa fue una decisión explícita, ver `docs/ec2_test_env.md` §1, con un
   SIGILL ya sufrido). Con GPU el cliente necesita NVIDIA + driver + container
   toolkit. **Es la restricción más cara del cambio y no es técnica**: cambia
   quién puede desplegar el producto.

### 2.3 Estructura del cambio

No forkear el repo. Igual que con los perfiles y los flavors de imagen, esto es
**un flavor más**: la ruta CPU (portable AVX2) tiene que seguir viva y
construible, porque es lo que está entregado y auditado hoy.

---

## 3. Decisión de motor (gated por 1.B)

| Opción | Concurrencia real | Esfuerzo | Cuándo |
|---|---|---|---|
| **A. llama-cpp `n_gpu_layers=-1`**, arquitectura intacta | **1** (cada réplica = otra copia de 17 GB, no caben dos en 24 GB) | ~2 días | Si A3 dice que la concurrencia real es 1 y solo se busca latencia |
| **B. llama-server `--parallel N`** | 3-5 `[EST]` (cada slot duplica el KV del prefijo) | ~1 semana | Punto medio; sigue siendo GGUF, mismo modelo, misma cuantización |
| **C. vLLM / SGLang con prefix caching paginado** | 8-16 `[EST]` (el prefijo compartido se guarda **una vez**) | 1-2 semanas | Si A1/A3 piden concurrencia de verdad |

**El argumento a favor de C es específico de esta aplicación**: los tres fulldocs
son prefijos compartidos por todos los usuarios de ese procedimiento (diabetes
~22 KB, cirugía ~7,8 KB, hemorroides ~2,4 KB de markdown). Con caché de prefijo
paginada, el prefijo se almacena una vez y cada usuario adicional solo cuesta su
pregunta + los `max_tokens=320` de salida — unos 400 tokens de KV. La concurrencia
sale casi gratis en VRAM.

**Contra C**: vLLM no consume GGUF bien, así que habría que servir el modelo en
AWQ/FP8 — otra cuantización, **otro conjunto de respuestas**, y el bloque 1.C se
repite entero. No elegir C sin haber presupuestado esa segunda lectura a mano.

**Recomendación provisional:** si A3 ≤ 2, ir a **A** y parar ahí. El salto a C
solo se justifica con demanda de concurrencia demostrada.

---

## 4. Arquitectura de hosting en spot (recomendación al cliente)

```
ASG (min=1, max=2)
├── Mixed instances: [g6.xlarge, g5.xlarge, g6.2xlarge, g5.2xlarge]   ← todas 24 GB
├── Allocation strategy: price-capacity-optimized
├── Capacity rebalancing: ON            ← la pieza clave
├── ≥3 AZs
└── AMI horneada: driver NVIDIA + container toolkit + imagen Docker + GGUF de 17 GB
        ↓
ALB, health check en /health
        ↓
Sidecar drenador: sondea IMDS /latest/meta-data/spot/instance-action cada 5 s
```

Cuatro decisiones y por qué:

1. **AMI horneada con el modelo dentro.** Sin esto la recuperación es de 5-8 min:
   leer 17 GB desde un gp3 con throughput por defecto (125 MB/s) son más de dos
   minutos *solo en eso*, más el pull de la imagen CUDA (~3-5 GB). Con el modelo
   ya en el disco: `[EST]` **90-120 s** de arranque a `/health` verde (B6 lo mide).
   Alternativa: la g6.xlarge trae 250 GB de NVMe local, servir el modelo desde ahí.

2. **Diversificación de tipos y AZs.** Es la palanca más importante y la que más
   se ignora. Un spot de un solo tipo en una sola AZ es frágil; cuatro tipos
   compatibles (todos 24 GB, todos sirven el modelo) en tres AZ hace caer mucho la
   probabilidad de quedarse sin capacidad. Que a veces toque una g5.xlarge más
   cara sigue siendo spot y sigue estando por debajo del on-demand.

3. **Capacity rebalancing.** Con el aviso temprano (*rebalance recommendation*,
   anterior al de 2 minutos), el ASG **lanza la sustituta antes de que muera la
   vieja**. Si hay capacidad, el hueco es cero. Es la respuesta directa a «¿qué
   pasa cuando retiran la máquina?».

4. **Drenado explícito.** Al recibir el aviso: marcar `/health` como fallido → el
   ALB deja de enviar → esperar a que terminen las generaciones en vuelo → apagar.

---

## 5. Qué pasa exactamente en una interrupción

| t | Evento | Acción |
|---|---|---|
| −(minutos) | Rebalance recommendation | ASG lanza la sustituta y la calienta |
| −120 s | Interruption notice (IMDS + EventBridge) | Sidecar marca `/health` fallido; ALB deja de enrutar |
| −120 s → −5 s | Ventana de drenado | Terminan las generaciones en vuelo |
| 0 | Terminación | La sustituta ya sirve |

**Esta aplicación es un caso casi ideal para spot**: las peticiones son sin
estado (memoria `stateless-design`) y duran `[EST]` ~8 s en GPU. Dos minutos de
aviso son enormes para drenar ~8 s de trabajo. **Cero peticiones fallidas** en
una interrupción con aviso. Lo único que no se evita es el hueco si el
rebalancing no consiguió capacidad a tiempo.

---

## 6. Fallback a CPU — y por qué probablemente NO

Lo tentador: dejar la caja CPU actual como red de seguridad, GPU cuando está sana
y CPU cuando no. Nunca cae, solo va lento un rato.

**El problema no es de infraestructura, es de gobernanza.** Las respuestas de los
dos backends no son idénticas. Con fallback tendríais dos conjuntos de respuestas
y un paciente podría recibir una u otra según qué máquina le atendió, sin
trazabilidad. Con una auditoría clínica encima defendiendo un **123/134**
concreto, eso no es un detalle de implementación.

Si aun así se quiere fallback, entonces **hay que auditar los dos backends** (el
bloque 1.C por duplicado) y registrar en cada respuesta cuál la sirvió.

**Recomendación:** sin fallback. ASG bien diversificado y aceptar algún hueco de
1-2 minutos al mes (lo valida A5). Un backend, un número auditado.

---

## 7. Coste

`[EST]` — pendiente de D2. Precios on-demand us-east-1, orientativos.

| Escenario | $/h | Concurrencia | Notas |
|---|---|---|---|
| CPU hoy, un perfil | 0,50 | 1 | 4,83 tok/s `[MED]` |
| CPU hoy, ambos perfiles | ~1,00 | 1+1 | Dos cajas, o una partida 4+4 hilos (más lenta) |
| g6.xlarge on-demand | ~0,81 | según motor | L4 24 GB, 300 GB/s |
| g5.2xlarge on-demand | ~1,21 | según motor | A10G 24 GB, **600 GB/s** |
| **g6.xlarge spot + ALB + EBS** | **~0,30-0,45** | según motor | Recomendada |
| Lo anterior, solo en horario laboral | ~$70-100/mes | | Si A2 lo permite |

Dos notas sobre la comparación:

- **La comparación justa no es 0,50 vs 1,21.** Hoy se sirven dos perfiles y cada
  uno levanta su propia copia de 17 GB. Una sola GPU sirve los tres
  procedimientos y ambos perfiles en un proceso, porque los fulldocs son solo
  prefijos distintos en la misma caché (B3 lo confirma).
- **Contraintuitivo: para esta carga el A10G (g5, más antiguo) bate al L4 (g6,
  más nuevo y barato)** porque tiene el doble de ancho de banda, y el decode está
  limitado por ancho de banda. No elegir por generación.

---

## 8. Fases

| Fase | Contenido | Esfuerzo | Depende de |
|---|---|---|---|
| **F0** | Relevamiento 1.A y 1.D. Comprobar cuotas (1.E) en la cuenta del **cliente** y tramitarlas si faltan | 2-3 días (mucho es espera) | — |
| **F1** | Banco de pruebas 1.B en g6.xlarge alquilada | 1 día | — (cuota nuestra ya verificada, ver 1.E) |
| **F2** | **Punto de decisión**: ¿se hace? ¿con qué motor? | — | F0 + F1 |
| **F3** | Port: los 3 cambios + arreglar `compute_key` + flavor CUDA | 2 días | F2 |
| **F4** | Revalidación 1.C, incluida lectura a mano | **3-5 días** | F3 |
| **F5** | AMI horneada + módulo OpenTofu del ASG spot | 3-4 días | F3 |
| **F6** | Documentación para el cliente: requisitos de hardware, runbook, procedimiento de actualización | 2 días | F4 + F5 |

**F4 es la fase más larga y la que no se puede comprimir.** Si el plan se
presenta con una estimación, que sea esa la que se defienda.

`[EST]` total si se sigue adelante: **3-4 semanas**, de las cuales solo ~2 días
son el port en sí.

---

## 9. Cuándo NO hacerlo

- **A1 + A4** dicen que la carga la absorbe una réplica CPU y nadie se queja de
  esperar un minuto. → No se hace. Se ahorra todo.
- El cliente no puede garantizar NVIDIA en su entorno. → No se hace; se pierde la
  portabilidad que fue una decisión de diseño explícita.
- No hay presupuesto para F4. → **No se hace.** Desplegar un backend nuevo sin
  revalidar las 134 tira por tierra la respuesta a la auditoría, que es el activo
  más caro del proyecto ahora mismo.
- B4 sale mucho peor de lo estimado (degradación cercana a la serie). → Reevaluar:
  la GPU seguiría dando latencia, pero el argumento de concurrencia caería.

---

## 10. Entregables al cliente

1. **Requisitos de hardware**: GPU NVIDIA ≥24 GB VRAM, driver ≥ X, container
   toolkit, 4-8 vCPU, RAM ≥1,5× VRAM (32 GB cómodo, 16 GB justo), 60 GB de disco.
2. **Módulo OpenTofu** del ASG spot, siguiendo las convenciones de
   `project_aws_shared_account_conventions` si aplica a su cuenta.
3. **Receta de la AMI** (Packer o snapshot documentado).
4. **Runbook**: despliegue, actualización de versión (adaptar
   `docs/actualizacion_version_desplegada.md`, cuyo corte de ~70 s cambia si
   desaparecen los snapshots), y qué mirar cuando el ASG no consigue capacidad.
5. **Nota de calidad**: el número auditado del backend GPU, resultado de F4, y la
   constancia explícita de que **no es el mismo backend que produjo el 123/134 en
   CPU**.

---

## Referencias internas

- `docs/ec2_test_env.md` — flavors de imagen, por qué native no aporta en gemma,
  techo de ancho de banda en single-socket
- `docs/actualizacion_version_desplegada.md` — procedimiento actual de
  actualización; cambia si se eliminan los snapshots
- `src/llm.py:65-97` — `load_model()`, donde entra `n_gpu_layers`
- `app/snapshot_cache.py:26-46` — `compute_key()`, la trampa del backend
- `app/config.py` — `n_ctx=8192`, `max_tokens=320`, ruta del GGUF, perfiles
- Memorias: `gemma-single-socket-bw-ceiling`, `gemma-native-no-decode-gain`,
  `feedback-eval-by-reading`, `stateless-design`
