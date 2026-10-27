"""Puente QWebChannel: la interfaz HTML llama a estos slots y recibe telemetría por señales."""
import csv
import json
import shutil
from datetime import datetime

from PySide6.QtCore import QObject, QThread, Signal, Slot

from . import config
from .recorder import Recorder
from .worker import open_osk

RAW = config.ROOT / "data" / "raw"
DISCARDED = config.ROOT / "data" / "discarded"
RUNS = config.ROOT / "runs"


class TrainThread(QThread):
    done = Signal(dict)

    def run(self):
        try:
            from .train import train  # scikit-learn tarda en cargar: solo al entrenar
            _, m = train(name="app")
            self.done.emit({"ok": True, "m": m})
        except BaseException as e:  # noqa: BLE001  (train usa SystemExit para "faltan datos")
            self.done.emit({"ok": False, "error": str(e) or "Faltan datos: graba al menos 2 gestos distintos."})


def _files():
    out = []
    for p in sorted(RAW.glob("*.csv")) if RAW.exists() else []:
        with open(p, newline="") as f:
            rows = list(csv.reader(f))[1:]
        out.append({"name": p.name, "samples": len(rows), "rounds": len({r[0] for r in rows})})
    return out


def _runs():
    idx = RUNS / "index.csv"
    if not idx.exists():
        return []
    rows = list(csv.reader(open(idx)))[1:]
    return [{"run": r[0], "acc": float(r[4]), "rules": float(r[5]), "promoted": r[6] == "True"} for r in reversed(rows)]


class Backend(QObject):
    tick = Signal(str)
    state = Signal(str)
    toast = Signal(str)
    trained = Signal(str)
    recDone = Signal(str)
    camError = Signal(str)  # "" = cámara bien

    def __init__(self, app):
        super().__init__()
        self.app = app

    # ---------- estado ----------
    def snapshot(self):
        a = self.app
        return {"cfg": a.cfg, "cameras": a.camera_names(), "files": _files(), "runs": _runs(),
                "modes": [{"name": m["name"], "icon": m["icon"]} for m in config.MODES],
                "overlays": a.overlay_state()}

    def push(self):
        self.state.emit(json.dumps(self.snapshot()))

    @Slot(result=str)
    def getState(self):
        return json.dumps(self.snapshot())

    # ---------- configuración ----------
    @Slot(str, str)
    def setConfig(self, key, value):
        if key in config.DEFAULTS:
            self.app.cfg[key] = json.loads(value)
            if key == "camera_index":
                self.app.restart_worker()
            self.push()

    @Slot(result=bool)
    def saveConfig(self):
        try:
            config.save(self.app.cfg)
        except OSError as e:
            self.toast.emit(f"No se pudo guardar config.json ({e.strerror or e}). Cierra otros programas que lo usen y vuelve a intentarlo.")
            return False
        self.toast.emit("Configuración guardada")
        return True

    @Slot()
    def resetConfig(self):
        keep = {k: self.app.cfg[k] for k in ("camera_index", "tutorial_done")}
        self.app.cfg.update(config.DEFAULTS); self.app.cfg.update(keep)
        self.push(); self.toast.emit("Valores por defecto restaurados (pulsa Guardar para conservarlos)")

    @Slot(int)
    def setMode(self, i):
        m = config.MODES[i]
        self.app.cfg.update(m["set"]); self.app.cfg["mode"] = i
        w = self.app.worker
        if w:
            w.paused = m["name"].startswith("Reposo")
        self.push(); self.toast.emit("Modo: " + m["name"])

    # ---------- seguimiento ----------
    @Slot()
    def toggleTracking(self):
        w = self.app.worker
        if w:
            w.paused = not w.paused
            self.toast.emit("Seguimiento pausado (F9)" if w.paused else "Seguimiento activo")

    @Slot()
    def restartCamera(self):
        self.toast.emit("Reiniciando cámara…")
        self.app.restart_camera(manual=True)

    @Slot(int)
    def wantFrames(self, delta):
        w = self.app.worker
        if w:
            w.want_frames = max(0, w.want_frames + delta)

    @Slot()
    def recenter(self):
        if self.app.worker:
            self.app.worker.recenter = True
            self.toast.emit("Zona activa centrada en tu mano")

    @Slot()
    def calibrate(self):
        if self.app.worker:
            self.app.worker.calib = 30

    @Slot(str)
    def arm(self, kind):
        w = self.app.worker
        if not w:
            return
        if kind == "kbd":
            open_osk(); self.toast.emit("Teclado en pantalla de Windows abierto")
        elif kind == "lock":
            self.toggleTracking()
        else:
            w.armed = None if w.armed == kind else kind

    # ---------- entrenamiento ----------
    @Slot(str, int)
    def startRecording(self, user, rounds):
        w = self.app.worker
        if not (w and w.isRunning()):
            self.toast.emit("La cámara no está activa"); return
        w.start_recording(Recorder(user.strip() or "usuario", rounds))

    @Slot()
    def cancelRecording(self):
        if self.app.worker:
            self.app.worker.cancel_rec = True
        self.toast.emit("Grabación cancelada (lo grabado se conserva)")
        self.push()

    @Slot(str)
    def discardFile(self, name):
        src = RAW / name
        if src.parent == RAW and src.exists():
            try:
                DISCARDED.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(DISCARDED / name))  # reversible: queda en data/discarded
                self.toast.emit(f"{name} movido a data/discarded")
            except OSError as e:
                self.toast.emit(f"No se pudo descartar {name}: {e.strerror or e}")
        self.push()

    @Slot()
    def train(self):
        if getattr(self, "th", None) is not None and self.th.isRunning():
            self.toast.emit("Ya se está entrenando el modelo"); return
        self.th = TrainThread()
        self.th.done.connect(self._trained)
        self.th.start()

    def _trained(self, res):
        if res["ok"] and self.app.worker:
            self.app.worker.reload = True  # usa el modelo nuevo sin reiniciar
        self.trained.emit(json.dumps(res)); self.push()

    # ---------- ventana y superposiciones ----------
    @Slot(str)
    def startMove(self, name):
        w = self.app.win if name == "main" else self.app.overlays[name]
        w.windowHandle().startSystemMove()

    @Slot(str)
    def startResize(self, edges):
        from PySide6.QtCore import Qt
        e = Qt.Edges()
        for name, flag in (("l", Qt.LeftEdge), ("r", Qt.RightEdge), ("t", Qt.TopEdge), ("b", Qt.BottomEdge)):
            if name in edges:
                e |= flag
        self.app.win.windowHandle().startSystemResize(e)

    @Slot()
    def minimize(self):
        self.app.win.showMinimized()

    @Slot()
    def quitApp(self):
        self.app.quit()

    @Slot()
    def toggleMax(self):
        w = self.app.win
        w.showNormal() if w.isMaximized() else w.showMaximized()

    @Slot()
    def closeWindow(self):
        self.app.quit()  # un solo clic: cierra todo y libera la cámara

    @Slot()
    def openSuite(self):
        self.app.show_suite()

    @Slot(str)
    def showView(self, view):
        self.app.show_suite(view)

    @Slot(str)
    def toggleOverlay(self, name):
        if name == "kbd":
            open_osk(); self.toast.emit("Teclado en pantalla de Windows abierto")
        else:
            self.app.toggle_overlay(name)
        self.push()

    @Slot()
    def tutorialDone(self):
        self.app.cfg["tutorial_done"] = True
        config.save(self.app.cfg)

    @Slot(str, result=bool)
    def saveInterest(self, data):
        p = config.ROOT / "data" / "interes_fase2.jsonl"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({**json.loads(data), "fecha": datetime.now().isoformat()}, ensure_ascii=False) + "\n")
        except OSError as e:
            self.toast.emit(f"No se pudo guardar el registro: {e.strerror or e}")
            return False
        return True
