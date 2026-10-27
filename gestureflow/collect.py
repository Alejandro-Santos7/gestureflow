"""Recolección guiada por consola (la app hace lo mismo desde la pestaña Entrenamiento).

Uso:  python -m gestureflow.collect --user ana [--rounds 6] [--frames 250]
"""
import argparse

import cv2

from . import config
from .recorder import Recorder
from .vision import HandTracker, draw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="user")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--prep", type=float, default=4.0, help="segundos de preparación por gesto")
    a = ap.parse_args()
    rec = Recorder(a.user, a.rounds, a.frames, a.prep)
    tr = HandTracker(config.load()["camera_index"])
    while True:
        frame, lm = tr.read()
        if frame is None:
            continue
        st = rec.step(lm)
        cv2.imshow("GestureFlow - recoleccion (q para salir)", draw(cv2.flip(frame, 1), None, st["text"].split("\n")[0]))
        if st["done"] or cv2.waitKey(1) & 0xFF == ord("q"):
            break
    rec.close(); tr.close(); cv2.destroyAllWindows()
    print(f"{rec.n} muestras en {rec.path}")


if __name__ == "__main__":
    main()
