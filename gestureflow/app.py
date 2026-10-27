"""GestureFlow Suite: ventana sin marco con interfaz web (ui/index.html), paneles flotantes y bandeja."""
import ctypes
import json
import os
import sys
import time
import urllib.request

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QGuiApplication, QIcon, QPainter, QPen
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QMessageBox, QSystemTrayIcon, QWidget

from . import config
from .bridge import Backend
from .vision import MODEL_FILE, MODEL_URL
from .worker import Worker

SINGLE = "GestureFlow.Suite.single"
UI = config.ROOT / "gestureflow" / "ui"
COLORS = {"puntero": "#4cd7f6", "pinch": "#4edea3", "palma": "#adc6ff", "reposo": "#869397", None: "#869397"}


class Download(QThread):
    done = Signal()
    failed = Signal(str)

    def run(self):
        try:
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = MODEL_FILE.with_suffix(".part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(MODEL_FILE)
            self.done.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class Ring(QWidget):
    """Anillo click-through que sigue al cursor: progreso del dwell y halo al hacer clic."""
    S = 72

    def __init__(self):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
                         | Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.resize(self.S, self.S)
        self.progress, self.flash, self.color = 0.0, 0, QColor(COLORS[None])

        self.on = False
        self.follow_timer = QTimer(self); self.follow_timer.timeout.connect(self.follow); self.follow_timer.start(16)

    def update_state(self, t, visual):
        self.on = bool(t["cursor"]) and not t["rec"]
        if not self.on:
            self.hide(); return
        self.progress = t["progress"]
        self.flash = 6 if (t["clicked"] and visual) else max(self.flash - 1, 0)
        self.color = QColor("#ef4444" if t["paused"] else "#f59e0b" if t["armed"] else COLORS.get(t["label"], COLORS[None]))
        self.follow(); self.show(); self.update()

    def follow(self):
        """El anillo sigue al cursor real a 60 Hz (no a los FPS de la cámara)."""
        if self.on:
            p = QCursor.pos()
            self.move(p.x() - self.S // 2, p.y() - self.S // 2)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(14, 14, self.S - 28, self.S - 28)
        c = QColor(self.color); c.setAlpha(70)
        p.setPen(QPen(c, 3)); p.drawEllipse(r)
        if self.progress > 0:
            pen = QPen(self.color, 5); pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen); p.drawArc(r, 90 * 16, int(-self.progress * 360 * 16))
        if self.flash:  # halo expansivo
            h = QColor("#4cd7f6"); h.setAlpha(40 * self.flash)
            g = (6 - self.flash) * 2.5
            p.setPen(QPen(h, 3)); p.drawEllipse(r.adjusted(-g, -g, g, g))


def web_view(backend, fragment, parent=None, transparent=False):
    v = QWebEngineView(parent)
    s = v.settings()
    s.setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, False)  # todo local
    s.setAttribute(QWebEngineSettings.ShowScrollBars, True)
    ch = QWebChannel(v.page())
    ch.registerObject("backend", backend)
    v.page().setWebChannel(ch)
    if transparent:
        v.page().setBackgroundColor(Qt.transparent)
    else:
        v.page().setBackgroundColor(QColor("#0b1326"))
    v.setContextMenuPolicy(Qt.NoContextMenu)
    url = QUrl.fromLocalFile(str(UI / "index.html")); url.setFragment(fragment)
    v.load(url)
    return v


class Overlay(QWidget):
    """Panel flotante siempre visible (HUD, barra de acciones, menú de bandeja)."""

    def __init__(self, backend, name, hide_on_blur=False):
        super().__init__(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.name, self.hide_on_blur = name, hide_on_blur
        self.view = web_view(backend, name, self, transparent=True)

    def resizeEvent(self, e):
        self.view.resize(self.size())

    def changeEvent(self, e):
        if self.hide_on_blur and e.type() == e.Type.ActivationChange and not self.isActiveWindow():
            self.hide()
        super().changeEvent(e)


class Main(QMainWindow):
    def __init__(self, backend):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Window)
        self.backend = backend
        self.setWindowTitle("GestureFlow")
        self.setMinimumSize(900, 600)
        self.resize(1360, 860)
        self.setCentralWidget(web_view(backend, "main"))

    def closeEvent(self, e):  # cerrar = salir del todo (Alt+F4 incluido) y liberar la cámara
        e.accept(); self.backend.app.quit()


class App:
    def __init__(self):
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GestureFlow.Suite")  # icono propio en la barra de tareas
        self.qt = QApplication(sys.argv)
        self.qt.setQuitOnLastWindowClosed(False)
        probe = QLocalSocket(); probe.connectToServer(SINGLE)
        if probe.waitForConnected(300):  # ya hay una GestureFlow abierta: avisarle y salir
            probe.write(b"show"); probe.flush(); probe.waitForBytesWritten(300)
            os._exit(0)
        QLocalServer.removeServer(SINGLE)
        self.server = QLocalServer(); self.server.listen(SINGLE)
        self.server.newConnection.connect(lambda: (self.server.nextPendingConnection(), self.show_suite()))
        self._quitting = False
        self.last_tick = self.last_try = time.monotonic()
        self.retries, self.restarting = 0, False
        self.icon = QIcon(str(UI / "logo.svg"))
        self.qt.setWindowIcon(self.icon)
        self.cfg = config.load()
        self.worker = None
        self.backend = Backend(self)
        self.ring = Ring()
        self.win = Main(self.backend)
        self.overlays = {"hud": Overlay(self.backend, "hud"), "actions": Overlay(self.backend, "actions"),
                         "tray": Overlay(self.backend, "tray", hide_on_blur=True)}
        self.tray = QSystemTrayIcon(self.icon)
        menu = QMenu()
        for text, fn in (("Abrir GestureFlow", self.show_suite), ("Pausar / Reanudar", self.backend.toggleTracking), ("Salir", self.quit)):
            a = QAction(text, menu); a.triggered.connect(fn); menu.addAction(a)
        self.tray.setContextMenu(menu); self.tray.setToolTip("GestureFlow")
        self.tray.activated.connect(lambda r: self.toggle_overlay("tray") if r == QSystemTrayIcon.Trigger else None)
        self.tray.show()
        self.win.show()
        self.start_worker() if MODEL_FILE.exists() else self.offer_download()
        self.wd = QTimer(); self.wd.timeout.connect(self.watchdog); self.wd.start(1500)

    # ---------- motor ----------
    def offer_download(self):
        r = QMessageBox.question(
            self.win, "Modelo de manos",
            "Falta el modelo preentrenado de detección de manos de MediaPipe (hand_landmarker.task, ~7.8 MB).\n\n"
            "Se descarga una sola vez desde storage.googleapis.com. ¿Descargarlo ahora?")
        if r != QMessageBox.Yes:
            self.backend.toast.emit("Sin el modelo de manos no se puede usar la cámara"); return
        self.backend.toast.emit("Descargando modelo de manos…")
        self.dl = Download()
        self.dl.done.connect(self.start_worker)
        self.dl.failed.connect(lambda m: QMessageBox.warning(self.win, "Descarga fallida", m))
        self.dl.start()

    def start_worker(self):
        if self.worker and self.worker.isRunning():
            return
        want = self.worker.want_frames if self.worker else 0
        self.worker = Worker(self.cfg)
        self.worker.want_frames = want
        self.worker.tick.connect(self.on_tick)
        self.worker.failed.connect(self.on_failed)
        self.worker.rec_done.connect(self.on_rec_done)
        self.worker.start()

    def restart_camera(self, manual=False):
        """Reinicia la cámara (suelta la anterior aunque esté colgada) conservando pausa y vistas."""
        if self._quitting or self.restarting:
            return
        self.restarting = True
        if manual:
            self.retries = 0
        paused = self.worker.paused if self.worker else False
        if self.worker:
            self.worker.stop()
        self.start_worker()
        if self.worker:
            self.worker.paused = paused
        self.last_try = self.last_tick = time.monotonic()
        self.restarting = False

    restart_worker = restart_camera  # lo usa el cambio de cámara en Configuración

    def on_rec_done(self, path, n):
        self.backend.recDone.emit(json.dumps({"path": path, "n": n}))
        if self.cfg["auto_train"]:  # el modelo personal se rehace solo con todos los datos de data/raw
            self.backend.train()

    def on_failed(self, msg):
        self.backend.camError.emit(msg)
        self.backend.toast.emit("Cámara: " + msg)

    def watchdog(self):
        """Cada 1.5 s: si la cámara cayó o dejó de enviar datos, la reinicia (máx. 3 intentos seguidos)."""
        if self._quitting or self.restarting or not MODEL_FILE.exists():
            return
        now = time.monotonic()
        dead = self.worker is None or not self.worker.isRunning()
        if (dead or now - self.last_tick > 5) and now - self.last_try > 3:
            if self.retries < 3:
                self.retries += 1
                self.backend.toast.emit(f"Cámara sin señal: reiniciando (intento {self.retries}/3)…")
                self.restart_camera()
            elif self.retries == 3:
                self.retries = 4
                self.backend.camError.emit("Sin señal de la cámara. Revisa que no la use otra app y pulsa «Reiniciar cámara».")

    def on_tick(self, t):
        self.last_tick, self.retries = time.monotonic(), 0
        self.ring.update_state(t, self.cfg["visual"])
        self.backend.tick.emit(json.dumps(t))

    def camera_names(self):
        return [d.description() for d in QMediaDevices.videoInputs()] or ["Cámara 0"]

    # ---------- ventanas ----------
    def show_suite(self, view=None):
        self.win.showNormal(); self.win.raise_(); self.win.activateWindow()
        if view:
            self.win.centralWidget().page().runJavaScript(f"ui.view={json.dumps(view)};render()")
        self.overlays["tray"].hide()

    def overlay_state(self):
        return {k: o.isVisible() for k, o in self.overlays.items()}

    def toggle_overlay(self, name):
        o = self.overlays[name]
        if o.isVisible():
            o.hide(); return
        g = QGuiApplication.primaryScreen().availableGeometry()
        if name == "actions":
            o.setGeometry(g.right() - 420, g.top() + 20, 400, min(g.height() - 40, 880))
        elif name == "hud":
            o.setGeometry(g.center().x() - 390, g.top() + 30, 780, 640)
        else:  # tray: sobre la bandeja, esquina inferior derecha
            o.setGeometry(g.right() - 460, g.bottom() - 700, 440, 690)
        o.show(); o.raise_(); o.activateWindow()

    def quit(self):
        if self._quitting:
            return
        self._quitting = True
        self.wd.stop()
        self.ring.hide(); self.win.hide()  # la ventana desaparece al instante
        for o in self.overlays.values():
            o.hide()
        self.tray.hide()
        if self.worker:
            self.worker.stop()  # libera la cámara (con liberación forzada si read() se colgó)
        self.server.close()
        self.qt.quit()

    def run(self):
        return self.qt.exec()


def main():
    code = App().run()
    os._exit(code)  # garantía final: ningún hilo ni proceso hijo mantiene la cámara encendida


if __name__ == "__main__":
    main()
