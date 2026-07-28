"""Invariantes de una lectura a mano. Impide volver a publicar una lectura parcial.

De dónde sale este fichero
--------------------------

`audit_hand` declaraba que 37 de las 134 «no necesitan lectura» porque se
transferían de la pasada de Ministral: si las dos versiones se abstenían y el
predicado automático decía que la decisión coincidía, se daba el veredicto por
bueno sin leer la respuesta. El razonamiento parece sólido y es falso, por dos
sitios a la vez:

1. El predicado comparaba la abstención contra un `MUST_REFUSE` **precalculado**,
   no contra el documento servido. Nunca preguntaba «¿tiene el documento material
   para esto?». Por ahí pasaron la **85** y la **104**, que se abstienen teniendo
   la respuesta en el documento —y que el propio modelo contesta en la 76 y en la
   109 con esas mismas frases.
2. «Ambas se abstienen» no implica «ambas responden lo mismo». La **54** se
   abstiene en las dos versiones, pero en la v2.2 **responde antes de
   abstenerse**. La comparación ni siquiera miraba el texto.

La regla que sale de ahí, y que este módulo hace cumplir:

    Todo número que salga del equipo tiene que venir de leer las 134, y cada
    veredicto tiene que estar ESCRITO, no deducido de otro.

Leer una parte solo vale dentro de un experimento acotado, donde lo que se sigue
es un delta y no se publica un absoluto.

Qué hace, y qué se quitó de aquí
--------------------------------

Hace una sola cosa: `require_complete()`, que impide publicar una lectura a la
que le falten preguntas.

Hubo una versión de este fichero con dos funciones más: `same_content()`, que
medía si dos respuestas decían lo mismo, y `share()`, que copiaba el veredicto
de un run a otro cuando la similitud pasaba de 0.93. Es el mismo error otra vez
con mejor disfraz —un umbral decidiendo lo que solo se resuelve leyendo— y por
eso no están. Si hace falta saber qué respuestas cambiaron entre dos runs, eso
se mira leyendo un volcado lado a lado (`audit_movers.py`), no con un ratio.

Los veredictos viven escritos, uno por pregunta, en `audit_hand` (para
`eval/ec2`) y en `audit_hand_v22` (para `eval/d1c-tu`). Los scripts los cuentan
y los formatean; no los infieren.
"""

from __future__ import annotations


def require_complete(verdicts: dict[int, tuple], ids, que: str) -> dict[int, tuple]:
    """Una lectura que se publica cubre todas las preguntas, explícitamente."""
    faltan = sorted(set(ids) - set(verdicts))
    if faltan:
        raise AssertionError(
            f"{que}: faltan por leer {len(faltan)} preguntas {faltan}. Una lectura "
            f"parcial no se publica — ver la cabecera de tools/audit_reading.py")
    return verdicts
