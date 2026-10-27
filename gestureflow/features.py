"""Landmarks (21x3) -> características, y clasificador por reglas (línea base y respaldo)."""
import numpy as np

LABELS = ["reposo", "pinch", "palma", "puntero"]
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9
FINGERS = {"indice": (6, 8), "medio": (10, 12), "anular": (14, 16), "menique": (18, 20)}  # (pip, tip)
KNUCKLES = [5, 9, 13, 17]
KNUCKLE_LIFT = 0.14  # en data/raw la punta del índice queda ~0.145 (imagen) por encima de los nudillos al apuntar


def palm_size(lm):
    return float(np.linalg.norm(lm[MIDDLE_MCP, :2] - lm[WRIST, :2])) or 1e-6


def hand_center(lm):
    return lm[[WRIST] + KNUCKLES, :2].mean(axis=0)


def pinch_gap(lm):
    """Separación pulgar–índice en palmas (0 = puntas tocándose)."""
    return float(np.linalg.norm(lm[THUMB_TIP, :2] - lm[INDEX_TIP, :2])) / palm_size(lm)


def pointer(lm, anchor="nudillos"):
    """Punto de la mano que mueve el cursor (coordenadas de imagen sin espejar).

    "nudillos": centro de los 4 nudillos, subido KNUCKLE_LIFT hasta donde suele quedar la punta del índice al
    apuntar (la zona activa, el recentrado y las configuraciones guardadas siguen valiendo igual). Medido en
    data/raw: al pasar de puntero a pellizco la punta del índice se desplaza ~0.68 palmas (~330 px de pantalla)
    y los nudillos ~0.11 (~57 px), con la mitad de temblor. "indice": la punta del índice (modo anterior).
    """
    if anchor == "indice":
        return float(lm[INDEX_TIP, 0]), float(lm[INDEX_TIP, 1])
    k = lm[KNUCKLES, :2].mean(axis=0)
    return float(k[0]), float(k[1]) - KNUCKLE_LIFT


def normalize(lm):
    """Invariante a posición y distancia a la cámara: relativo a la muñeca, escalado por la palma."""
    lm = np.asarray(lm, dtype=np.float32)
    p = palm_size(lm)
    d = lambda a, b: float(np.linalg.norm(lm[a, :2] - lm[b, :2]))
    # extras que las reglas ya usan bien: pinch, extensión de cada dedo y apertura del pulgar
    extras = [d(THUMB_TIP, INDEX_TIP) / p, d(THUMB_TIP, 17) / p]
    extras += [d(tip, WRIST) / max(d(pip, WRIST), 1e-6) for pip, tip in FINGERS.values()]
    return np.concatenate([((lm - lm[WRIST]) / p).reshape(-1), np.array(extras, np.float32)])


def _extended(lm, pip, tip):
    d = lambda i: np.linalg.norm(lm[i, :2] - lm[WRIST, :2])
    return d(tip) > d(pip) + 0.1 * palm_size(lm)


def classify_rules(lm):
    lm = np.asarray(lm, dtype=np.float32)
    if pinch_gap(lm) < 0.25:
        return "pinch"
    ext = {k: _extended(lm, *v) for k, v in FINGERS.items()}
    if all(ext.values()):
        return "palma"
    if ext["indice"] and not (ext["medio"] or ext["anular"] or ext["menique"]):
        return "puntero"
    return "reposo"
