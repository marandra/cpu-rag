# Actualizar una versión ya desplegada

Procedimiento ejecutado y cronometrado de punta a punta en la EC2 de pruebas el
2026-07-27 (r7i.2xlarge, 8 vCPU, gemma-4-26B, un perfil por vez). Cubre el caso
real: **el cliente ya tiene el sistema funcionando y queremos cambiarle la
versión sin reinstalar de cero.**

Lo que hay que entender antes de tocar nada está en §1. Si solo quieres los
comandos, ve a §3.

## 1. Qué obliga a regenerar qué

El servicio no responde con el modelo en crudo: sirve desde un **snapshot**, un
pickle con el KV del prefijo ya calculado. Su nombre es un SHA256 sobre
*(modelo, n_ctx, flash_attn, system prompt, texto del fulldoc)* —
`app/snapshot_cache.py:compute_key`.

De ahí salen tres reglas, y son toda la operación:

| Cambias… | ¿Imagen nueva? | ¿Regenerar snapshot? |
| --- | --- | --- |
| El prompt (`app/prompt.py`, `PROMPT_VARIANT`) | **Sí** — `app/` va horneado | **Sí** — cambia la clave |
| El corpus (`corpus/markdown/*.md`) | No — va montado desde el host | **Sí** — cambia la clave |
| Réplicas, hilos, puertos, API key | No | No |

Dos consecuencias que no son obvias:

- **El nombre del fichero del corpus no entra en la clave**, solo su contenido.
  Renombrar un fulldoc no invalida nada.
- **Los snapshots de dos versiones conviven** en el mismo directorio, porque sus
  claves difieren. Eso es lo que hace barata la vuelta atrás (§4) y lo que
  permite actualizar casi sin corte (§3). Presupuesta ~1,9 GB por procedimiento
  grande (diabetes) y ~0,9-1,1 GB por los pequeños: **~3,9 GB por versión** para
  los dos perfiles.

Si dos versiones comparten prompt y corpus, la clave es la misma **y los bytes
también**: la generación es determinista en la misma caja y con la misma imagen
(verificado: los tres pickles regenerados salieron idénticos a los de la v2, y
sirviéndolos reprodujeron las respuestas carácter a carácter). Regenerar por
encima no destruye nada en ese caso, pero tampoco sirve de nada.

## 2. Tiempos medidos

| Paso | Tiempo |
| --- | ---: |
| `docker build` (solo cambia `app/`) | **< 1 s** — el resto de capas están cacheadas |
| `docker save` + `gzip -1` + subida (118 MB) | **~10 s** a ~24 MB/s |
| `docker load` en destino | pocos segundos |
| Regenerar snapshot — diabetes (22 375 car.) | **~180 s** |
| Regenerar snapshot — hemorroides (2 515 car.) | ~42 s |
| Regenerar snapshot — cirugía (7 652 car.) | ~50 s |
| Arranque hasta `healthy` | **~70 s** |

Total realista para los dos perfiles: **~8 minutos**, de los cuales solo el
arranque (~70 s por perfil) tiene por qué ser corte de servicio si se sigue §3.

El tiempo de warm-up escala con el tamaño del fulldoc, no con el del modelo.

## 3. El procedimiento

Con corte mínimo: se genera el snapshot nuevo **mientras el servicio sigue
sirviendo el viejo**, y solo al final se reinicia apuntando a la versión nueva.
Funciona porque las claves difieren y los dos pickles conviven.

### 3.1 En la máquina de desarrollo

```bash
# 1. Construir. La flavor portable es la que se sirve (en gemma la nativa no
#    aporta decode y además cambia respuestas).
docker build -t cpu-rag-api:<version>-portable .

# 2. Comprobar que la imagen lleva lo que crees. No te fíes del tag.
docker run --rm -e RAG_API_KEY=x --entrypoint python \
  cpu-rag-api:<version>-portable -c \
  "from app.prompt import PROMPT_VARIANTS as P; print(sorted(P))"

# 3. Enviar.
docker save cpu-rag-api:<version>-portable | gzip -1 \
  | ssh <host> 'cat > /opt/rag/images/cpu-rag-api-<version>-portable.tar.gz'
```

### 3.2 En la máquina desplegada

```bash
cd /opt/rag
docker load -i images/cpu-rag-api-<version>-portable.tar.gz

# 4. Generar el snapshot nuevo SIN parar el servicio. Es un contenedor
#    one-shot; el pool sigue sirviendo el pickle viejo, que tiene otra clave.
for p in glucowise aiciblock; do
  RAG_IMAGE=cpu-rag-api:<version>-portable PROMPT_VARIANT=<variante> \
    docker compose --env-file profiles/$p.env -p rag-$p \
    --profile generate run --rm rag-generate
done

# 5. Verificar ANTES de cortar: tiene que haber un pkl nuevo por procedimiento.
find snapshots -name '*.pkl' -newermt '-1 hour'

# 6. Cambiar la versión servida. Aquí está el único corte (~70 s por perfil).
#    Fija RAG_IMAGE y PROMPT_VARIANT en .env para que sobrevivan al reinicio.
for p in glucowise aiciblock; do ./load_and_run.sh $p; done

# 7. Comprobar que sirve lo que debe, no solo que responda.
curl -s localhost:8001/health -H "X-API-Key: $RAG_API_KEY"
docker exec rag-glucowise-rag-1 printenv PROMPT_VARIANT
```

El paso 7 no es ceremonia: `running` y `healthy` no dicen **qué** versión se está
sirviendo. Comprueba imagen, variante y el bind de snapshots.

## 4. Vuelta atrás

Barata, porque el pickle viejo sigue en disco: se revierte `RAG_IMAGE` y
`PROMPT_VARIANT` en `.env` y se relanza `load_and_run.sh`. Cuesta lo que cuesta
arrancar (~70 s), sin regenerar nada.

Por eso **no se borra el snapshot de la versión anterior** hasta que la nueva
lleve un tiempo aceptada. Cuando toque limpiar, borra por clave, no el
directorio.

## 5. Experimentos: no compartir directorio de snapshots

Para probar variantes usa el overlay `infra/ec2-test/docker-compose.exp.yml`,
que exige `SNAP_DIR` y `PROMPT_VARIANT` explícitos y aborta si faltan.

El motivo es real y ya ocurrió: al regenerar la variante servida contra el
directorio de producción, la clave calculada fue **el mismo nombre de fichero**
que el pickle que sirve el pool. Con otra imagen o otro hardware detrás, eso
habría sobrescrito la única forma de reproducir los números de la versión
entregada. Un experimento escribe en `snapshots-exp/<brazo>/`, nunca en
`snapshots/`.

Al terminar, devuelve la caja al estado conocido **antes** de apagarla: los
contenedores llevan `restart: unless-stopped`, así que la máquina reencendería
sirviendo el último brazo experimental. Baja los brazos, relanza
`load_and_run.sh` de cada perfil y verifica con el paso 7.
