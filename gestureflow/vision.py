"""Cámara + MediaPipe Hands (CNN preentrenada, local en CPU) -> landmarks 21x3.

Fluidez: un hilo captura sin parar (siempre el fotograma más reciente, sin cola) y la cámara se
configura con exposición corta + ganancia máxima: con poca luz la exposición automática baja la
cámara a 4–15 FPS; con exposición ≤ 31 ms y ganancia electrónica alta mantiene 30 FPS.
"""
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .config import ROOT
from .features import hand_center, palm_size

MODEL_FILE = ROOT / "models" / "hand_landmarker.task"  # descarga única (la app la ofrece si falta)
PAD = 0.18  # margen negro alrededor del fotograma: la mano puede salir un poco y sigue siguiéndose
JUMP = 0.3        # mientras se sigue una mano, una detección a más de esto (imagen) no es ella (cara, otra mano)
RECENT_S = 0.5    # memoria de dónde estaba la mano
ACQUIRE = 3       # una mano nueva (sin memoria) debe verse quieta en su sitio estos fotogramas seguidos
RELOCK = 45       # ~1.5 s rechazando lo que MediaPipe sigue: rehacer el detector (una vez; cuesta ~0.65 s) para que busque de nuevo
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
             "float16/latest/hand_landmarker.task")


class HandTracker:
    def __init__(self, camera_index=0, fast=True, exposure=-5):
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Falta {MODEL_FILE} (modelo preentrenado de MediaPipe, ver README)")
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("la cámara está ocupada o desconectada")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # sin cola: latencia mínima
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.fast, self.exposure, self.brightness = fast, exposure, 0
        self.gain = self.gain_max = None  # se descubren al pedir 255: el driver recorta a su máximo
        self._settings_done = False
        self.reapply()
        self.t0 = time.perf_counter()
        self.hands = self._landmarker()
        self.score, self.side = 0.0, None  # confianza de detección y mano ("Derecha"/"Izquierda")
        self._last, self._last_t = None, -1e9  # centro de la mano seguida y cuándo se aceptó por última vez
        self._palm = None                       # tamaño típico de la palma seguida (imagen)
        self._cand, self._acq, self._reject, self._relocked = None, 0, 0, False
        self.other_side = None  # mano no preferida vista de forma sostenida (para avisar en la UI)
        self.t_frame = time.perf_counter()
        self._frame, self._t, self._seq, self._seen = None, 0.0, 0, 0
        self._new = threading.Event()
        self._run = True
        self._th = threading.Thread(target=self._grab, daemon=True, name="cam-grab")
        self._th.start()

    @staticmethod
    def _landmarker():
        """Una sola mano: MediaPipe solo ejecuta el detector de palmas (~21 ms medidos en este equipo) cuando
        no sigue ninguna. Con num_hands=2 y una mano en cámara lo ejecutaba en cada fotograma, se pasaba de los
        33 ms de un fotograma y se saltaban fotogramas (tirones); además encontraba la cara como «segunda mano»."""
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODEL_FILE)),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.35))

    def _accept(self, lm, prefer, side):
        """¿Es esta detección la mano del usuario? Mientras se sigue una, no puede teletransportarse; una nueva
        debe ser del lado preferido, de tamaño parecido a la anterior y mantenerse ACQUIRE fotogramas."""
        ctr, size = hand_center(lm), palm_size(lm)
        if self._t - self._last_t < RECENT_S:
            return np.linalg.norm(ctr - self._last) <= JUMP
        ok = side == prefer and (self._palm is None or 0.6 < size / self._palm < 1.6)
        same = self._cand is not None and np.linalg.norm(ctr - self._cand) < 0.1
        self._acq = (self._acq + 1 if same else 1) if ok else 0
        self._cand = ctr
        return self._acq >= ACQUIRE

    def _apply_exposure(self):
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual (DirectShow)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)

    def reapply(self):
        """Aplica exposición, ganancia y brillo. Algunos drivers las ignoran si se piden antes de que
        la cámara entregue fotogramas (p. ej. al reabrirla justo después de cerrarla): se repite luego."""
        if not self.fast:
            return
        self._apply_exposure()
        if self.gain_max is None:
            self.cap.set(cv2.CAP_PROP_GAIN, 255)
            g = self.cap.get(cv2.CAP_PROP_GAIN)
            self.gain_max = g if g and g > 0 else 64
            self.gain = self.gain_max
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)

    def _grab(self):
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                break
            self._frame, self._t = f, time.perf_counter()
            self._seq += 1
            self._new.set()
        self._run = False
        self._new.set()  # despierta a read() para que informe del fallo

    def read(self, prefer="Derecha"):
        """-> (frame_bgr, landmarks|None). landmarks en coordenadas de imagen normalizadas 0..1.

        Siempre procesa el fotograma más reciente. Una mano nueva debe ser la preferida (etiqueta de MediaPipe
        verificada en vivo con la cámara sin espejar); ver _accept.
        """
        if not self._new.wait(1.0) or self._seq == self._seen:
            return None, None  # la cámara dejó de entregar fotogramas (o se cerró)
        self._new.clear()
        frame, self._seen, self.t_frame = self._frame, self._seq, self._t
        if not self._settings_done and self._seq >= 4:  # ya hay imagen: ahora sí se respetan
            self._settings_done = True
            self.reapply()
        h, w = frame.shape[:2]
        px, py = int(w * PAD), int(h * PAD)
        padded = cv2.copyMakeBorder(frame, py, py, px, px, cv2.BORDER_CONSTANT)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
        res = self.hands.detect_for_video(img, int((time.perf_counter() - self.t0) * 1000))
        if not res.hand_landmarks:
            self.score, self.side = 0.0, None
            return frame, None
        side = "Derecha" if res.handedness[0][0].category_name == "Right" else "Izquierda"
        W, H = w + 2 * px, h + 2 * py  # a coordenadas del fotograma original (pueden quedar fuera de 0..1)
        lm = np.array([[(p.x * W - px) / w, (p.y * H - py) / h, p.z] for p in res.hand_landmarks[0]], np.float32)
        if not self._accept(lm, prefer, side):
            self._reject += 1
            self.other_side = side if side != prefer and self._reject >= 30 else None
            if self._reject >= RELOCK and not self._relocked:  # enganchado a otra cosa: que vuelva a buscar
                self.hands.close(); self.hands = self._landmarker(); self._reject, self._relocked = 0, True
            self.score, self.side = 0.0, None
            return frame, None
        self._reject, self._relocked, self.other_side = 0, False, None
        self._last, self._last_t = hand_center(lm), self._t
        size = palm_size(lm)
        self._palm = size if self._palm is None else 0.95 * self._palm + 0.05 * size
        self.score, self.side = float(res.handedness[0][0].score), side
        return frame, lm

    def shorten_exposure(self):
        """Si la cámara no llega a ~28 FPS, acorta la exposición un paso (hasta -8 ≈ 4 ms)."""
        if self.fast and self.exposure > -8:
            self.exposure -= 1
            self.reapply()
            return True
        return False

    def nudge_exposure(self, brightness):
        """Un paso de autocalibración de luminosidad en los dos sentidos, sin alargar la exposición
        (así no baja el FPS). -> True cuando la imagen ya está en rango.

        Muy oscura: más ganancia, luego más brillo. Muy clara: menos brillo, luego menos ganancia y,
        como último recurso, exposición más corta."""
        if 95 <= brightness <= 165:
            return True
        step = max(self.gain_max / 8, 1)
        if brightness < 95:
            if self.gain < self.gain_max:
                self.gain = min(self.gain + step, self.gain_max)
            elif self.brightness < 64:
                self.brightness = min(self.brightness + 8, 64)
        else:
            if self.brightness > 0:
                self.brightness = max(self.brightness - 8, 0)
            elif self.gain > step:
                self.gain -= step
            else:
                self.shorten_exposure()
        self.cap.set(cv2.CAP_PROP_GAIN, self.gain)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        return False

    def close(self):
        self._run = False
        try:
            self.cap.release()  # también desbloquea un read() colgado del hilo de captura
        finally:
            self._th.join(timeout=2)
            self.hands.close()


def draw(frame, lm, label=None):
    h, w = frame.shape[:2]
    if lm is not None:
        for x, y, _ in lm:
            cv2.circle(frame, (int(x * w), int(y * h)), 3, (80, 220, 200), -1)
    if label:
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame
