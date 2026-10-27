"""Configuración global única (Fase 1): config.json junto al proyecto, con valores por defecto."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / "config.json"

DEFAULTS = {
    "camera_index": 0,
    "fast_camera": True,            # exposición corta + ganancia alta: 30 FPS aun con poca luz
    "exposure": -6,                 # log2 de segundos (-6 ≈ 16 ms: la mitad de desenfoque al mover la mano que -5; en esta
                                    # webcam la imagen apenas se oscurece, 91 vs 96/255); se acorta sola si no llega a 30 FPS
    "hand": "Derecha",              # mano preferida si hay dos en cámara
    "zone_center": [0.5, 0.47],     # centro de la zona activa (imagen espejada, 0..1)
    "sens": 5,                      # 1..10: mayor = zona activa más pequeña = menos movimiento de brazo
    "anchor": "nudillos",           # punto que mueve el cursor: nudillos (estable al pellizcar) | indice (punta)
    "max_speed": 2400,              # px/s, evita saltos fuera de pantalla
    "dead_px": 4,                   # zona muerta: no mover si el cambio es menor
    "filter": True,                 # One Euro activado
    "min_cutoff": 1.0,              # menor = más suavizado
    "beta": 0.007,                  # mayor = menos retraso al mover rápido
    "pinch_click": True,
    "dwell_click": True,
    "pinch_action": "left",         # left | right | double | drag | none
    "dwell_action": "left",         # left | right | double | none
    "pinch_conf": 0.9,              # confianza mínima del modelo para aceptar un pinch
    "dwell_ms": 1200,
    "dwell_radius": 30,             # px
    "debounce_ms": 250,             # bloqueo tras un clic (evita dobles involuntarios)
    "audio": True,
    "visual": True,
    "mode": 0,                      # 0 neuro asistido, 1 precisión, 2 reposo
    "vote": 5,                      # frames para estabilizar el gesto
    "lost_ms": 500,                 # sin mano más de esto => inactivo
    "tutorial_done": False,
    "auto_train": True,             # entrenar el modelo personal al terminar cada grabación guiada
}

# perfiles rápidos (menú de bandeja)
MODES = [
    {"name": "Neuro Asistido", "icon": "accessibility_new",
     "set": {"filter": True, "min_cutoff": 0.6, "beta": 0.005, "dwell_click": True, "dwell_ms": 1200, "sens": 6}},
    {"name": "Precisión Dibujo", "icon": "draw",
     "set": {"filter": True, "min_cutoff": 1.6, "beta": 0.012, "dwell_click": False, "pinch_click": True, "sens": 4}},
    {"name": "Reposo Pausa", "icon": "bedtime", "set": {}},
]


def load():
    cfg = dict(DEFAULTS)
    if PATH.exists():
        cfg.update({k: v for k, v in json.loads(PATH.read_text(encoding="utf-8")).items() if k in DEFAULTS})
    return cfg


def save(cfg):
    PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# los nudillos se mueven la mitad que la punta al girar la muñeca (palanca 0.91 vs 1.81 palmas en data/raw): sin
# compensar, el cursor se siente pesado. Tiemblan la mitad (8 vs 14 px), así que x1.3 sigue temblando menos que antes.
KNUCKLE_GAIN = 1.3
ZONE_BOTTOM = 0.70  # con el índice más abajo, la muñeca (≈ 1.9 palmas por debajo) sale del cuadro y se pierde la mano


def zone(cfg):
    """Zona activa (x0, y0, x1, y1) en coordenadas de imagen espejada, a partir del centro y la sensibilidad.

    Es más ancha que alta (la mano sube y baja menos de lo que se desplaza de lado) y nunca baja de
    ZONE_BOTTOM; los bordes pueden asomar un poco fuera de imagen porque la detección se hace con margen.
    """
    w = 0.9 - (cfg["sens"] - 1) * 0.07
    if cfg.get("anchor") != "indice":
        w /= KNUCKLE_GAIN
    h = w * 0.75
    cx = min(max(cfg["zone_center"][0], w / 2 - 0.06), 1 - w / 2 + 0.06)
    cy = min(max(cfg["zone_center"][1], h / 2 - 0.04), ZONE_BOTTOM - h / 2)
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
