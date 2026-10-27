"""Inferencia del MLP entrenado con numpy puro (sin torch/sklearn en tiempo de ejecución)."""
import numpy as np

from .config import ROOT
from .features import LABELS, classify_rules, normalize

MODEL_PATH = ROOT / "models" / "latest.npz"
PINCH_MIN_CONF = 0.9  # medido con validación por sesión: falsos pinch 210 -> 106 (ver runs/)


class Classifier:
    """Usa models/latest.npz si existe; si no, cae a las reglas geométricas."""

    def __init__(self, path=MODEL_PATH):
        self.w = None
        if path.exists():
            z = np.load(path, allow_pickle=False)
            n = int(z["n_layers"])
            self.mean, self.scale = z["mean"], z["scale"]
            self.w = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
            self.classes = [str(c) for c in z["classes"]]
            if len(self.mean) != len(normalize(np.zeros((21, 3), np.float32) + np.arange(21)[:, None])):
                self.w = None  # modelo viejo con otras características: usar reglas hasta reentrenar

    @property
    def source(self):
        return "modelo" if self.w else "reglas"

    def predict(self, lm, pinch_min=PINCH_MIN_CONF):
        if not self.w:
            return classify_rules(lm), 1.0
        a = (normalize(lm) - self.mean) / self.scale
        for i, (W, b) in enumerate(self.w):
            a = a @ W + b
            if i < len(self.w) - 1:
                a = np.maximum(a, 0)
        if a.shape[0] == 1:  # salida binaria (sigmoide)
            p1 = 1 / (1 + np.exp(-a[0]))
            a = np.array([1 - p1, p1])
        else:
            e = np.exp(a - a.max())
            a = e / e.sum()
        k = int(a.argmax())
        label, conf = self.classes[k], float(a[k])
        if label == "pinch" and conf < pinch_min:  # un clic fantasma cuesta más que uno perdido
            label = "reposo"
        return label, conf
