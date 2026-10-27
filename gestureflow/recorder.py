"""Grabación guiada de landmarks etiquetados (sin imágenes). La usan la app y collect.py."""
import csv
import time
from datetime import datetime

from . import config
from .features import LABELS

PROMPTS = {
    "reposo": "REPOSO: puño cerrado o mano relajada",
    "pinch": "PINCH: junta la punta del pulgar con el índice",
    "palma": "PALMA: mano abierta, dedos bien separados y extendidos",
    "puntero": "PUNTERO: solo el índice extendido",
}
# condiciones sugeridas por ronda: la variedad es lo que hace que el modelo generalice
TIPS = ["distancia normal, luz normal", "un poco más cerca de la cámara", "un poco más lejos",
        "mano girada ligeramente", "otra luz (más oscura o lateral)", "mano más alta y más baja"]
HEADER = ["session", "user", "label", "ts"] + [f"{a}{i}" for i in range(21) for a in "xyz"]


class Recorder:
    def __init__(self, user="usuario", rounds=6, frames=250, prep=4.0, out_dir=None):
        self.rounds, self.frames, self.prep, self.user = rounds, frames, prep, user
        self.plan = [(r, l) for r in range(1, rounds + 1) for l in LABELS]
        self.i = self.got = self.n = 0
        self.t0 = None
        out = out_dir or config.ROOT / "data" / "raw"
        out.mkdir(parents=True, exist_ok=True)
        self.day = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = out / f"{self.day}_{user}.csv"
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(HEADER)

    def step(self, lm):
        """Procesa un frame. -> dict de estado; status['done'] es True al terminar."""
        now = time.time()
        if self.i >= len(self.plan):
            self.close()
            return self._status("done", 0)
        r, label = self.plan[self.i]
        if self.t0 is None:
            self.t0 = now
        left = self.prep - (now - self.t0)
        if left > 0:
            return self._status("prep", left)
        if lm is not None:
            self._w.writerow([f"{self.day}-{self.user}-r{r}", self.user, label, f"{now:.3f}", *lm.reshape(-1).tolist()])
            self.got += 1
            self.n += 1
        if self.got >= self.frames:
            self.i += 1; self.got = 0; self.t0 = None
        if self.i >= len(self.plan):
            self.close()
            return self._status("done", 0)
        return self._status("rec", 0)

    def _status(self, phase, left):
        r, label = self.plan[min(self.i, len(self.plan) - 1)]
        done = phase == "done"
        return {"phase": phase, "done": done, "round": r, "rounds": self.rounds, "label": label,
                "left": left, "got": self.got, "frames": self.frames,
                "progress": (self.i * self.frames + self.got) / (len(self.plan) * self.frames),
                "tip": TIPS[(r - 1) % len(TIPS)], "text": self.text(phase, r, label, left)}

    def text(self, phase, r, label, left):
        if phase == "done":
            return f"¡Listo! {self.n} muestras guardadas"
        head = f"Ronda {r}/{self.rounds} · {PROMPTS[label]}"
        if phase == "prep":
            return f"{head}\nPrepárate… {max(left, 0):.0f} s   (esta ronda: {TIPS[(r - 1) % len(TIPS)]})"
        return f"{head}\nGRABANDO {self.got}/{self.frames} — mueve la mano un poco"

    def close(self):
        if not self._f.closed:
            self._f.close()
