"""Hilo de seguimiento: cámara -> landmarks -> clasificador -> filtro -> motor -> SendInput."""
import base64
import ctypes
import subprocess
import threading
import time
import winsound
from collections import Counter, deque

import cv2
import numpy as np
import psutil
from PySide6.QtCore import QThread, Signal

from . import config, inject
from .features import pinch_gap, pointer
from .filters import OneEuro
from .gestures import Engine
from .model import Classifier
from .smooth import CursorRender, edge_pin
from .vision import HandTracker

ACTION_NAMES = {"left": "Clic izquierdo", "right": "Clic derecho", "double": "Doble clic",
                "middle": "Clic central", "drag": "Arrastrar", "none": "Sin acción"}


def beep():
    threading.Thread(target=winsound.Beep, args=(880, 60), daemon=True).start()


class Worker(QThread):
    tick = Signal(dict)       # telemetría (+ jpeg base64 si alguien la pide)
    failed = Signal(str)
    rec_done = Signal(str, int)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # dict compartido con la UI: los cambios se aplican en vivo
        self.paused = False
        self._stop = False
        self.tr = None
        self.render = CursorRender()
        self.inject_on = False   # el hilo del cursor solo mueve el ratón cuando esto es True
        self.moves = 0           # movimientos inyectados (para medir los Hz reales del cursor)
        self._mover = None
        self.rec = None          # Recorder activo: mientras graba no se inyecta ratón
        self.cancel_rec = False
        self.reload = False      # recargar el modelo recién entrenado
        self.want_frames = 0     # nº de vistas que muestran la cámara
        self.armed = None        # acción armada para el próximo clic (barra de acciones)
        self.dragging = False
        self.calib = 0           # pasos restantes de autocalibración de exposición
        self.recenter = False
        self.clicks = 0
        self.last_event = ("En espera", 0)
        self.calib_result = None

    def stop(self, timeout=3000):
        """Detiene el hilo y libera la cámara. Si read() está colgado, la suelta a la fuerza."""
        self._stop = True
        if self.wait(timeout):
            return True
        tr = getattr(self, "tr", None)
        if tr is not None:
            try:
                tr.cap.release()  # desbloquea un read() colgado desde otro hilo
            except Exception:  # noqa: BLE001
                pass
        return self.wait(2000)

    def start_recording(self, rec):
        self.cancel_rec = False
        self.rec = rec

    def _mover_loop(self):
        """Hilo del cursor: ~125 Hz, independiente de los FPS de la cámara. También vigila F9 / Ctrl+Alt+P:
        aquí se mira cada 8 ms (por fotograma se perdían pulsaciones cortas cuando la cámara daba un tirón)."""
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)  # temporizador de 1 ms (por defecto Windows duerme en pasos de ~15 ms)
        last, hot_prev = None, False
        try:
            while not self._stop:
                time.sleep(0.008)
                hot = inject.hotkey_down()
                if hot and not hot_prev:
                    self.paused = not self.paused
                hot_prev = hot
                p = self.render.step(time.perf_counter()) if self.inject_on else None
                if p is None:
                    last = None
                    continue
                q = (round(p[0]), round(p[1]))
                if q != last:
                    inject.move(*q); last = q; self.moves += 1
        finally:
            winmm.timeEndPeriod(1)

    def _do(self, kind, src):
        pos = self.render.snap()  # clic exactamente donde está el dedo, no donde el cursor aún llega
        if pos:
            inject.move(round(pos[0]), round(pos[1]))
        if kind == "drag":
            (inject.up if self.dragging else inject.down)()
            self.dragging = not self.dragging
            if not self.dragging:
                self.armed = None
            name = "Soltar" if not self.dragging else "Arrastrar: sujetando"
        else:
            if kind == "none":
                return
            {"left": inject.click, "right": lambda: inject.click("right"),
             "middle": lambda: inject.click("middle"), "double": inject.double}[kind]()
            self.armed = None
            name = ACTION_NAMES[kind]
        self.clicks += 1
        self.last_event = (f"{name} ({'pellizco' if src == 'pinch' else 'dwell'})", self.last_event[1])
        if self.cfg["audio"]:
            beep()

    def run(self):
        try:
            self.tr = HandTracker(self.cfg["camera_index"], self.cfg["fast_camera"], self.cfg["exposure"])
        except Exception as e:  # noqa: BLE001
            self.tr = None
            self.failed.emit(f"No se pudo abrir la cámara ({e})"); return
        try:
            self._loop(self.tr)
        except Exception as e:  # noqa: BLE001  cualquier fallo: avisar, nunca dejar la cámara abierta
            self.failed.emit(f"{type(e).__name__}: {e}")
        finally:
            self._stop, self.inject_on = True, False  # detiene también el hilo del cursor
            if self.dragging:
                inject.up(); self.dragging = False
            try:
                self.tr.close()
            except Exception:  # noqa: BLE001
                pass
            self.tr = None
            if self.rec:  # cierre durante una grabación: no dejar el CSV abierto
                self.rec.close(); self.rec = None

    def _loop(self, tr):
        cfg = self.cfg
        self._mover = threading.Thread(target=self._mover_loop, daemon=True, name="cursor-mover")
        self._mover.start()
        t_tune, last_moves, t_hz, hz = time.perf_counter() + 12, 0, time.perf_counter(), 0.0
        t_chk, seq_chk = None, 0
        clf, eng = Classifier(), Engine(cfg)
        fx, fy = OneEuro(), OneEuro()
        sw, sh = inject.screen_size()
        proc, ncpu = psutil.Process(), psutil.cpu_count() or 1
        proc.cpu_percent(None)
        cpu = ram = 0.0
        last_stats = last_frame = time.perf_counter()
        fps, n = 0.0, 0
        out = last_move = None
        pin = lost_t = None  # posición de borde a mantener si la mano sale del cuadro por ahí
        trail = deque(maxlen=15)
        reasons = deque(maxlen=90)  # ~3 s: cuánto del tiempo el cursor estuvo congelado y por qué
        prev_stable = None

        while not self._stop:
            frame, lm = tr.read(cfg["hand"])
            if frame is None:
                self.failed.emit("No se pudo leer la cámara"); break
            t = time.perf_counter()
            dt = max(t - last_frame, 1e-3)
            fps = 0.9 * fps + 0.1 / dt
            last_frame = t
            brightness = float(frame[::8, ::8].mean())

            if self.reload:
                clf, self.reload = Classifier(), False
            if t < t_tune and not self.calib:  # ajuste automático de la cámara durante los primeros segundos
                if t_chk is None:
                    if n >= 8:  # empezar a medir cuando ya llegan fotogramas
                        t_chk, seq_chk = t, tr._seq
                elif t - t_chk >= 1.5:  # ¿llega a 30 FPS? si no, exposición más corta (o reaplicar ajustes)
                    cam_fps = (tr._seq - seq_chk) / (t - t_chk)
                    t_chk, seq_chk = t, tr._seq
                    if cam_fps < 26 and not tr.shorten_exposure():
                        tr.reapply()
                elif n % 6 == 0:
                    tr.nudge_exposure(brightness)
            if self.calib and n % 3 == 0:
                self.calib -= 1
                if tr.nudge_exposure(brightness) or not self.calib:
                    self.calib_result, self.calib = brightness, 0

            label = point = gap = None
            conf = pressure = 0.0
            tip = None
            if lm is not None:
                label, conf = clf.predict(lm, cfg["pinch_conf"])
                gap = pinch_gap(lm)
                pressure = min(max((0.9 - gap) / 0.7, 0), 1)
                px_, py_ = pointer(lm, cfg["anchor"])
                tip = (1 - px_, py_)  # espejado
                if self.recenter:
                    cfg["zone_center"], self.recenter = [tip[0], tip[1]], False
                x0, y0, x1, y1 = config.zone(cfg)
                uu, vv = (tip[0] - x0) / (x1 - x0), (tip[1] - y0) / (y1 - y0)
                u, v = min(max(uu, 0), 1), min(max(vv, 0), 1)
                if cfg["filter"]:
                    fx.min_cutoff = fy.min_cutoff = cfg["min_cutoff"]
                    fx.beta = fy.beta = cfg["beta"]
                    p = (fx(u * sw, t), fy(v * sh, t))
                else:
                    p = (u * sw, v * sh)
                if out is not None:  # zona muerta + velocidad máxima
                    d = float(np.hypot(p[0] - out[0], p[1] - out[1]))
                    if d < cfg["dead_px"]:
                        p = out
                    elif d > cfg["max_speed"] * dt:
                        k = cfg["max_speed"] * dt / d
                        p = (out[0] + (p[0] - out[0]) * k, out[1] + (p[1] - out[1]) * k)
                out = point = p
                pin, lost_t = edge_pin(uu, vv, p, (sw, sh)), None
            else:
                lost_t = lost_t or t
                if t - lost_t > 0.3:  # un corte breve (1-9 fotogramas) no reinicia filtro ni posición
                    fx.reset(); fy.reset()
                    if pin is None or t - lost_t > 1.5:
                        out = pin = None

            cursor, click, progress, stable = eng.update(label, point, t, gap)
            reasons.append(eng.reason)
            non_ok = [r for r in reasons if r != "ok"]
            freeze = {"pct": round(100 * len(non_ok) / len(reasons)),
                      "why": Counter(non_ok).most_common(1)[0][0] if non_ok else ""}
            if stable != prev_stable and stable:
                self.last_event = (f"Gesto: {stable}", round(conf * 100))
            prev_stable = stable
            if stable == "puntero" and cursor:
                trail.append(cursor)
            else:
                trail.clear()
            jitter = float(np.std(np.array(trail), axis=0).max()) if len(trail) > 5 else 0.0

            rec, rec_st = self.rec, None
            if rec:
                if self.cancel_rec:
                    rec.close(); self.rec = rec = None
                else:
                    rec_st = rec.step(lm)
                    if rec_st["done"]:
                        self.rec = None; self.rec_done.emit(str(rec.path), rec.n)
                cursor, click, progress = None, None, 0.0
            active = not self.paused and not rec
            target = (pin if lm is None else None) or cursor  # mano fuera por un borde: seguir hasta el borde
            self.render.set_target(target if active else None, t, still=eng.reason != "ok")
            self.inject_on = bool(active and target)
            if active and click:
                self._do(self.armed or cfg[f"{click}_action"], click)
            if self.paused and self.dragging:  # nunca dejar el botón pegado
                inject.up(); self.dragging = False
            latency = (time.perf_counter() - tr.t_frame) * 1000

            if t - t_hz >= 1:
                hz, last_moves, t_hz = (self.moves - last_moves) / (t - t_hz), self.moves, t
            if t - last_stats > 1:
                cpu, ram, last_stats = proc.cpu_percent(None) / ncpu, proc.memory_info().rss / 2**20, t
            n += 1
            info = {
                "label": stable, "raw": label, "conf": round(conf * 100), "det": round(tr.score * 100), "side": tr.side,
                "progress": progress, "cursor": cursor, "clicked": bool(active and click), "clicks": self.clicks,
                "fps": round(fps, 1), "hz": round(hz), "exposure": tr.exposure, "latency": round(latency), "bright": round(brightness),
                "cpu": round(cpu, 1), "ram": round(ram), "paused": self.paused, "source": clf.source,
                "freeze": freeze, "pressure": round(pressure * 100), "jitter": round(jitter), "zone": config.zone(cfg),
                "screen": [sw, sh], "event": self.last_event, "armed": self.armed, "dragging": self.dragging,
                "rec": rec_st, "calibrating": bool(self.calib), "calib_result": self.calib_result,
                "ptr": tip, "other_side": tr.other_side, "lm": None if lm is None else [[round(1 - float(x), 4), round(float(y), 4)] for x, y, _ in lm],
            }
            self.calib_result = None
            if self.want_frames and n % 2 == 0:  # video a media tasa: la UI no necesita más
                small = cv2.resize(cv2.flip(frame, 1), (480, 360))
                info["jpeg"] = base64.b64encode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1]).decode()
            self.tick.emit(info)


def open_osk():
    """Teclado en pantalla nativo de Windows (admite escritura por permanencia)."""
    subprocess.Popen(["cmd", "/c", "start", "", "osk.exe"], creationflags=subprocess.CREATE_NO_WINDOW)


