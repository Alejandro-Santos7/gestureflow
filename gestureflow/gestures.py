"""Máquina de gestos pura (sin cámara ni Windows): etiqueta + punto -> acciones. Fácil de probar."""
import math
from collections import Counter, deque

PALMA_HOLD = 24  # fotogramas seguidos de palma quieta (~0.8 s) para pausar
PALMA_STILL = 60  # px: la palma solo pausa si la mano está quieta; al moverla más que esto, el cursor la sigue.
                  # Una mano relajada se ve como palma (en la grabación del 26/10, 250/250 fotogramas de «reposo»):
                  # con 0.27 s y sin exigir quietud, mover la mano relajada congelaba el cursor.
REWIND = 12      # historial (~0.4 s) para volver al punto en que la pinza empezó a cerrarse
PINCH_IN, PINCH_OUT = 0.35, 0.5  # histéresis en palmas: entra al cerrar por debajo de IN, sale al abrir por encima de OUT
REOPEN = 0.15     # ... o al reabrir esto sobre lo más cerrado de ese pellizco (con la mano relajada, en data/raw,
                  # la separación queda en ~0.33 y nunca llegaba a OUT: el cursor se quedaba pegado tras el clic)
OPEN_FRAMES = 2   # fotogramas seguidos abriendo para soltar: un fotograma ruidoso no suelta ni repite el clic
PINCH_CONFIRM = 2  # fotogramas seguidos de pellizco antes del clic (~66 ms): uno suelto no hace clic
SETTLE_S = 0.15    # al soltar, el cursor espera a que los dedos se reabran antes de volver a moverse


class Engine:
    """update() devuelve (cursor|None, clic: None|'pinch'|'dwell', progreso_dwell 0..1, gesto_estable|None).

    El cursor NO depende de que el modelo acierte «puntero» en cada fotograma (con ~87 % de acierto eso lo
    congelaba a ratos): sigue a la mano siempre, y solo se congela mientras
    - se cierra, se mantiene o se reabre un pellizco (para que el clic no se desvíe), o
    - hay una palma quieta sostenida >= PALMA_HOLD fotogramas (pausa de descanso; moverla la quita).
    El pellizco es un estado con histéresis (gap = separación pulgar–índice en palmas): el clic sale una sola vez
    al entrar, y se coloca donde estaba el cursor antes de empezar a cerrar los dedos.
    El dwell solo cuenta con el gesto estable «puntero». El antirrebote bloquea cualquier clic durante
    debounce_ms tras el anterior. Un corte de mano breve (< lost_ms) conserva el cursor.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.votes = deque(maxlen=cfg["vote"])
        self.stable = None
        self.cursor = None
        self.anchor = None
        self.anchor_t = 0.0
        self.armed = True   # evita repetir dwell sin salir antes del radio
        self.last_seen = None
        self.last_click = -1e9
        self.hist = deque(maxlen=REWIND)  # (cursor, gap) de los fotogramas en que el cursor se movía
        self.pinching, self.pinch_run, self.settle_until = False, 0, 0.0
        self.pinch_min, self.open_run = 0.0, 0
        self.palma_run, self.pre_palma, self.palma_at = 0, None, None
        self.reason = "perdida"  # por qué no se mueve ahora: ok | pinch | palma | perdida

    def _reset_dwell(self, pos, t):
        self.anchor, self.anchor_t = pos, t

    def _before_closing(self):
        """Cursor justo antes de que la pinza empezara a cerrarse: se recorre el historial hacia atrás mientras
        la separación siga creciendo (hacia delante en el tiempo, se estaba cerrando). Se para en cuanto deja
        de crecer: mejor quedarse un poco corto que volver a un punto por el que la mano solo pasaba."""
        pos, last = None, None
        for p, g in reversed(self.hist):
            if g is not None:
                if last is not None and g < last + 0.02:  # aquí los dedos aún no se cerraban
                    break
                last = g
            pos = p  # sin medida (pruebas por etiqueta): lo más antiguo guardado
        return pos

    def update(self, label, point, t, gap=None):
        c = self.cfg
        if label is None:  # sin mano: conservar el cursor un rato por si es un corte breve
            self.reason = "perdida"
            if self.last_seen is not None and (t - self.last_seen) * 1000 > c["lost_ms"]:
                self.votes.clear(); self.hist.clear()
                self.stable = self.cursor = self.anchor = None
                self.palma_run, self.pre_palma = 0, None
                self.pinching, self.pinch_run = False, 0
            return self.cursor, None, 0.0, self.stable
        self.last_seen = t
        self.votes.append(label)
        new = Counter(self.votes).most_common(1)[0][0]

        # --- pausa por palma sostenida ---
        if label == "palma":
            if self.palma_run == 0 or math.dist(point, self.palma_at) > PALMA_STILL:  # empieza o la mano se mueve
                self.palma_run, self.palma_at = 0, point
                self.pre_palma = self.cursor  # dónde estaba el cursor justo antes de abrir la mano
            self.palma_run += 1
        else:
            self.palma_run = 0
        palma_hold = self.palma_run >= PALMA_HOLD
        if self.palma_run == PALMA_HOLD and self.pre_palma is not None:
            self.cursor = self.pre_palma  # deshace lo que se movió mientras se abría la mano

        # --- pellizco: estado con histéresis; un clic al entrar ---
        click = None
        if self.pinching:
            if gap is not None:
                self.pinch_min = min(self.pinch_min, gap)
                opened = gap > PINCH_OUT or gap > self.pinch_min + REOPEN
            else:
                opened = label != "pinch"
            self.open_run = self.open_run + 1 if opened else 0
            if self.open_run >= (OPEN_FRAMES if gap is not None else 1):
                self.pinching, self.settle_until = False, t + SETTLE_S
        else:
            closed = label == "pinch" and (gap is None or gap < PINCH_IN)
            self.pinch_run = self.pinch_run + 1 if closed else 0
            if self.pinch_run >= PINCH_CONFIRM:
                self.pinching, self.pinch_run, self.open_run = True, 0, 0
                self.pinch_min = gap if gap is not None else 0.0
                self.cursor = self._before_closing() or self.cursor
                if c["pinch_click"]:
                    click, self.armed = "pinch", False  # sin dwell duplicado hasta mover el cursor
        entering_pointer = new == "puntero" and self.stable != "puntero"
        self.stable = new

        # --- movimiento ---
        if label == "pinch" or self.pinching or self.pinch_run or t < self.settle_until:
            self.reason = "pinch"
        elif palma_hold:
            self.reason = "palma"
        else:
            self.reason = "ok"
            self.cursor = point
        if self.reason == "ok":
            self.hist.append((self.cursor, gap))

        # --- dwell ---
        progress = 0.0
        if new == "puntero" and self.cursor is not None:
            if entering_pointer or self.anchor is None:
                self._reset_dwell(self.cursor, t)
            d = math.dist(self.cursor, self.anchor)
            if d > c["dwell_radius"]:
                self._reset_dwell(self.cursor, t)
                self.armed = True
            elif self.armed and c["dwell_click"]:
                progress = min((t - self.anchor_t) * 1000 / c["dwell_ms"], 1.0)
                if progress >= 1.0:
                    click, self.armed, progress = "dwell", False, 0.0
        if click and (t - self.last_click) * 1000 < c["debounce_ms"]:
            click = None
        if click:
            self.last_click = t
        return self.cursor, click, progress, self.stable
