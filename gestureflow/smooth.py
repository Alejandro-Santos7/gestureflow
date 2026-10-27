"""Cursor fluido: la detección llega a ~30 Hz pero el cursor se mueve a ~125 Hz.

set_target() recibe cada posición detectada; step() (llamada ~cada 8 ms desde otro hilo) devuelve la
posición a inyectar: persigue el objetivo con una constante de tiempo corta y lo adelanta con la
velocidad medida (tope de 1.5 fotogramas) para que no se note el retardo de la detección.
Sin dependencias de Windows ni de Qt: se prueba con números.
"""
import math


def edge_pin(u, v, pos, screen, margin=0.08):
    """Si la mano estaba junto al borde de la zona (u, v en 0..1; pueden salirse), devuelve la posición del borde
    de pantalla hacia el que iba; si no, None. Sirve para no quedarse corto cuando la mano sale del cuadro."""
    near = u < margin or u > 1 - margin or v < margin or v > 1 - margin
    if not near or pos is None:
        return None
    x = 0.0 if u < margin else screen[0] - 1.0 if u > 1 - margin else pos[0]
    y = 0.0 if v < margin else screen[1] - 1.0 if v > 1 - margin else pos[1]
    return x, y


class CursorRender:
    def __init__(self, tau=0.022, lead_max=0.045):
        self.tau, self.lead_max = tau, lead_max
        self.reset()

    def reset(self):
        self.target = self.prev = self.cur = None
        self.vx = self.vy = 0.0
        self.t_target = self.t_prev = self.t_step = 0.0

    def set_target(self, p, t, still=False):
        """p = (x, y) detectado en el instante t, o None si no hay cursor. still=True: el motor lo congeló
        (pellizco, palma); sin velocidad, para que la predicción no lo empuje tras volver al punto del clic."""
        if p is None:
            self.reset(); return
        if still or p == self.target:
            self.vx = self.vy = 0.0
        elif self.target is not None and 0 < t - self.t_target < 0.25:
            dt = t - self.t_target
            self.vx = 0.5 * self.vx + 0.5 * (p[0] - self.target[0]) / dt
            self.vy = 0.5 * self.vy + 0.5 * (p[1] - self.target[1]) / dt
        self.target, self.t_target = p, t
        if self.cur is None:
            self.cur, self.t_step = p, t

    def step(self, now):
        """-> (x, y) o None. Llamar con tiempos crecientes."""
        if self.target is None:
            return None
        lead = min(max(now - self.t_target, 0.0), self.lead_max)
        gx, gy = self.target[0] + self.vx * lead, self.target[1] + self.vy * lead
        k = 1 - math.exp(-max(now - self.t_step, 0.0) / self.tau)
        self.t_step = now
        self.cur = (self.cur[0] + (gx - self.cur[0]) * k, self.cur[1] + (gy - self.cur[1]) * k)
        return self.cur

    def snap(self):
        """Posición exacta del objetivo (para hacer clic donde está el dedo, sin retardo)."""
        if self.target is not None:
            self.cur = self.target
        return self.cur
