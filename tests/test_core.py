"""Pruebas del núcleo con manos sintéticas. Ejecutar: python tests/test_core.py"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gestureflow import config  # noqa: E402
from gestureflow.features import classify_rules  # noqa: E402
from gestureflow.filters import OneEuro  # noqa: E402
from gestureflow.gestures import Engine  # noqa: E402
from gestureflow.train import train  # noqa: E402

XS = {"indice": 0.0, "medio": 0.3, "anular": 0.6, "menique": 0.9}
BASE = {"indice": 5, "medio": 9, "anular": 13, "menique": 17}


def synth(gesture, rng=None):
    """Mano sintética 21x3 (palma = 1, dedos hacia arriba, y negativo = arriba)."""
    lm = np.zeros((21, 3), np.float32)
    ext = {"palma": 4, "puntero": 1, "reposo": 0, "pinch": 0}[gesture]
    for k, i in BASE.items():
        e = ext == 4 or (ext == 1 and k == "indice")
        x = XS[k] - 0.45
        lm[i] = (x, -1.0, 0)
        lm[i + 1], lm[i + 2], lm[i + 3] = ((x, -1.4, 0), (x, -1.65, 0), (x, -1.9, 0)) if e else ((x, -1.3, 0), (x, -1.0, 0), (x, -0.7, 0))
    lm[1:5] = [(-0.5, -0.3, 0), (-0.8, -0.6, 0), (-1.0, -0.9, 0), (-1.2, -1.1, 0)]
    if gesture == "pinch":
        lm[8] = (-0.45, -1.2, 0); lm[4] = (-0.5, -1.25, 0)
    if rng is not None:
        lm += rng.normal(0, 0.02, lm.shape).astype(np.float32)
    return lm


def test_rules():
    for g in ("reposo", "pinch", "palma", "puntero"):
        assert classify_rules(synth(g)) == g, g


def test_filter_smooths_jitter():
    f = OneEuro(min_cutoff=1.0, beta=0.0)
    rng = np.random.default_rng(0)
    out = [f(100 + rng.normal(0, 5), i / 30) for i in range(120)]
    assert np.std(out[30:]) < 3


def feed(eng, seq, t0=0.0, dt=1 / 30):
    ev = []
    for i, (label, pt) in enumerate(seq):
        ev.append(eng.update(label, pt, t0 + i * dt))
    return ev


def test_dwell_clicks_once():
    cfg = dict(config.DEFAULTS)
    eng = Engine(cfg)
    ev = feed(eng, [("puntero", (500, 500))] * 60)  # 2 s quieto
    assert sum(bool(e[1]) for e in ev) == 1
    assert max(e[2] for e in ev) > 0.9  # el HUD vio progreso


def test_dwell_resets_on_move():
    eng = Engine(dict(config.DEFAULTS))
    ev = feed(eng, [("puntero", (500 + 40 * (i % 2), 500)) for i in range(60)])  # se mueve > radio
    assert not any(e[1] for e in ev)


def test_pinch_clicks_and_freezes_cursor():
    eng = Engine(dict(config.DEFAULTS))
    seq = [("puntero", (300, 300))] * 10 + [("pinch", (340, 320))] * 10
    ev = feed(eng, seq)
    assert sum(bool(e[1]) for e in ev) == 1
    assert ev[-1][0] == (300, 300)  # no deriva al cerrar la mano


def test_palma_pauses_and_lost_hand_resets():
    eng = Engine(dict(config.DEFAULTS))
    ev = feed(eng, [("puntero", (100, 100))] * 10 + [("palma", (900, 900))] * 30)
    assert ev[-1][0] == (100, 100) and not any(e[1] for e in ev)
    ev = feed(eng, [(None, None)] * 30, t0=10)
    assert ev[-1][0] is None


def test_train_pipeline_end_to_end():
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "data").mkdir()
        with open(d / "data" / "synth.csv", "w") as f:
            f.write("session,user,label,ts," + ",".join(f"{a}{i}" for i in range(21) for a in "xyz") + "\n")
            for s in range(3):
                for g in ("reposo", "pinch", "palma", "puntero"):
                    for _ in range(60):
                        row = synth(g, rng).reshape(-1)
                        f.write(f"s{s},t,{g},0," + ",".join(f"{v:.5f}" for v in row) + "\n")
        run, m = train(d / "data", d / "runs", d / "models", name="test", max_iter=200)
        assert m["split"] == "leave-one-session-out" and m["accuracy"] > 0.95, m["accuracy"]
        for name in ("config.json", "metrics.json", "data_manifest.json", "env.txt", "model.npz"):
            assert (run / name).exists(), name
        assert (d / "runs" / "index.csv").exists()
        if m["promoted_to_latest"]:
            from gestureflow.model import Classifier
            c = Classifier(d / "models" / "latest.npz")
            assert c.source == "modelo" and c.predict(synth("pinch", rng))[0] == "pinch"


def test_debounce_and_dwell_toggle():
    cfg = dict(config.DEFAULTS); cfg["debounce_ms"] = 500
    eng = Engine(cfg)
    # pinch, suelta y vuelve a pinch en < 500 ms: solo un clic
    seq = [("puntero", (300, 300))] * 6 + [("pinch", (300, 300))] * 5 + [("puntero", (300, 300))] * 5 + [("pinch", (300, 300))] * 5
    assert sum(bool(e[1]) for e in feed(eng, seq)) == 1
    cfg = dict(config.DEFAULTS); cfg["dwell_click"] = False
    assert not any(e[1] for e in feed(Engine(cfg), [("puntero", (500, 500))] * 90))


def test_cursor_render_is_fluid_and_exact():
    from gestureflow.smooth import CursorRender
    r = CursorRender()
    # detección a 30 Hz moviendo el dedo a 600 px/s; el cursor se pinta a 125 Hz
    moves, t, x = [], 0.0, 100.0
    for frame in range(30):
        t_det = frame / 30
        x = 100 + 600 * t_det
        r.set_target((x, 500.0), t_det)
        for k in range(4):  # ~4 pasos de 8 ms entre detecciones
            p = r.step(t_det + 0.008 * (k + 1))
            moves.append(p[0])
    steps = [b - a for a, b in zip(moves, moves[1:])]
    assert min(steps) > -1e-6, "el cursor nunca retrocede"
    assert len(moves) / (30 / 30) >= 100, "≥ 100 movimientos por segundo con detección a 30 Hz"
    assert max(steps) < 12, "pasos pequeños (sin saltos de 20 px = un fotograma)"
    assert abs(moves[-1] - x) < 40, "sigue al dedo sin quedarse atrás"
    r.set_target((x, 500.0), 1.0); r.set_target((x, 500.0), 1.033)  # congelado
    assert r.snap() == (x, 500.0)


def test_cursor_does_not_stick_when_model_misclassifies():
    """Regresión: con ~87 % de acierto, fotogramas sueltos de reposo/palma congelaban el cursor a ratos."""
    eng = Engine(dict(config.DEFAULTS))
    labels = ["puntero", "reposo", "puntero", "palma", "puntero", "puntero", "reposo", "reposo", "puntero", "palma"] * 6
    seq = [(lab, (100 + 10 * i, 300)) for i, lab in enumerate(labels)]
    ev = feed(eng, seq)
    stuck = sum(1 for a, b in zip(ev, ev[1:]) if a[0] == b[0])
    assert stuck == 0, f"el cursor se quedó quieto {stuck} fotogramas aunque la mano se movía"
    assert ev[-1][0] == seq[-1][1]


def test_sustained_palma_still_pauses_and_brief_dropouts_keep_cursor():
    eng = Engine(dict(config.DEFAULTS))
    feed(eng, [("puntero", (100 + 5 * i, 200)) for i in range(10)])
    ev = feed(eng, [(None, None)] * 5, t0=0.33)              # corte de 5 fotogramas (~0.17 s)
    assert all(e[0] is not None for e in ev), "un corte breve de la mano no debe borrar el cursor"
    ev = feed(eng, [("palma", (900, 900))] * 30, t0=0.5)      # palma quieta sostenida: pausa y no se aleja
    assert ev[-1][0] == ev[26][0] and ev[-1][0][0] < 200


def test_pinch_click_lands_where_finger_was_before_closing():
    eng = Engine(dict(config.DEFAULTS))
    seq = [("puntero", (300, 300))] * 10 + [("puntero", (300 + 15 * i, 300 + 10 * i)) for i in range(1, 4)] + [("pinch", (400, 380))] * 6
    ev = feed(eng, seq)
    click = next(e for e in ev if e[1] == "pinch")
    assert math.dist(click[0], (300, 300)) < 60, "el clic no debe irse con el movimiento de cerrar los dedos"


def test_knuckle_anchor_does_not_move_when_pinching():
    from gestureflow.features import pointer
    a, b = synth("puntero"), synth("pinch")  # misma mano: solo se doblan el índice y el pulgar
    assert math.dist(pointer(a, "nudillos"), pointer(b, "nudillos")) < 1e-6
    assert math.dist(pointer(a, "indice"), pointer(b, "indice")) > 0.5  # la punta se va ~0.7 palmas


def feed_gap(eng, seq, t0=0.0, dt=1 / 30):
    return [eng.update(lab, pt, t0 + i * dt, gap) for i, (lab, pt, gap) in enumerate(seq)]


def test_pinch_rewinds_to_closing_start_clicks_once_and_settles():
    eng = Engine(dict(config.DEFAULTS))
    seq = [("puntero", (100 + 20 * i, 300), 0.8) for i in range(10)]          # se acerca al objetivo
    seq += [("puntero", (300, 300), 0.8)] * 5                                  # se para encima
    seq += [("reposo", (300, 300 + 15 * k), g) for k, g in ((1, 0.7), (2, 0.55), (3, 0.4))]  # cierra y deriva
    seq += [("pinch", (300, 350), 0.2)] * 4 + [("reposo", (300, 352), 0.3)] + [("pinch", (300, 350), 0.2)] * 4  # parpadeo
    seq += [("puntero", (300, 360), 0.7)] * 2                                  # reabre (dentro de SETTLE_S)
    seq += [("puntero", (300, 360), 0.8)] * 8
    ev = feed_gap(eng, seq)
    clicks = [e for e in ev if e[1] == "pinch"]
    assert len(clicks) == 1, "un pellizco = un clic, aunque el modelo parpadee"
    assert clicks[0][0] == (300, 300), f"clic donde estaba antes de cerrar, no en {clicks[0][0]}"
    assert ev[-9][0] == (300, 300), "al reabrir los dedos el cursor espera"
    assert ev[-1][0] == (300, 360), "después vuelve a seguir a la mano"


def test_pinch_rewind_does_not_go_back_past_a_moving_hand():
    eng = Engine(dict(config.DEFAULTS))
    seq = [("puntero", (100 + 20 * i, 300), 0.8) for i in range(12)]          # moviéndose, pulgar abierto
    seq += [("pinch", (340, 300), 0.2)] * 3
    click = next(e for e in feed_gap(eng, seq) if e[1])
    assert click[0] == (320, 300), "sin cierre previo: el último punto, no 0.4 s atrás"


def test_single_frame_pinch_does_not_click():
    eng = Engine(dict(config.DEFAULTS))
    seq = [("puntero", (300, 300), 0.8)] * 6 + [("pinch", (300, 300), 0.2)] + [("puntero", (300, 300), 0.8)] * 6
    assert not any(e[1] == "pinch" for e in feed_gap(eng, seq))


def test_pinch_releases_with_relaxed_hand_and_ignores_noise():
    """Regresión: la mano relajada deja el pulgar a ~0.33 palmas del índice; soltar exigía > 0.5 y el cursor se pegaba."""
    eng = Engine(dict(config.DEFAULTS))
    seq = [("puntero", (300, 300), 0.9)] * 6 + [("pinch", (300, 300), 0.15)] * 6
    seq += [("pinch", (300, 300), 0.35)] + [("pinch", (300, 300), 0.15)] * 3   # un fotograma ruidoso: sigue
    seq += [("reposo", (400 + 10 * i, 300), 0.33) for i in range(15)]          # se relaja y mueve la mano
    ev = feed_gap(eng, seq)
    assert sum(e[1] == "pinch" for e in ev) == 1, "el fotograma ruidoso no repite el clic"
    assert ev[-1][0] == (540, 300), "con la mano relajada el cursor vuelve a moverse"


def test_tracker_never_teleports_and_confirms_new_hands():
    """Regresión: el cursor saltaba «de la nada al medio» cuando MediaPipe tomaba la cara por la mano."""
    from gestureflow.features import hand_center
    from gestureflow.vision import HandTracker
    tr = object.__new__(HandTracker)  # solo la lógica de aceptación, sin cámara
    tr._last, tr._last_t, tr._palm, tr._cand, tr._acq = None, -1e9, None, None, 0
    hand = synth("puntero") * 0.1 + np.array([0.5, 0.8, 0], np.float32)       # mano abajo
    face = synth("palma") * 0.15 + np.array([0.5, 0.35, 0], np.float32)        # «mano» en la cara, arriba
    tr._t = 0.0
    assert not tr._accept(face, "Derecha", "Izquierda"), "una mano nueva del otro lado no se acepta"
    got = []
    for k in range(3):
        tr._t = k / 30; got.append(tr._accept(hand, "Derecha", "Derecha"))
    assert got == [False, False, True], "una mano nueva se confirma en 3 fotogramas"
    tr._last, tr._last_t, tr._t = hand_center(hand), 0.1, 0.2
    assert not tr._accept(face, "Derecha", "Derecha"), "mientras se sigue la mano, la cara (lejos) se rechaza"
    assert tr._accept(hand + np.array([0.05, 0, 0], np.float32), "Derecha", "Izquierda"), "la mano que se mueve sí"


def test_moving_relaxed_open_hand_is_not_a_pause():
    """Regresión: la mano relajada se clasifica como palma; moverla así no debe congelar el cursor."""
    eng = Engine(dict(config.DEFAULTS))
    ev = feed(eng, [("palma", (100 + 8 * i, 300)) for i in range(60)])         # 240 px/s, mano abierta
    assert sum(1 for a, b in zip(ev, ev[1:]) if a[0] == b[0]) <= 1
    assert ev[-1][0] == (100 + 8 * 59, 300)
    ev = feed(eng, [("palma", (600, 300))] * 30, t0=3)                          # quieta: pausa
    assert ev[-1][0] == ev[-2][0]
    ev = feed(eng, [("palma", (600 + 40 * i, 300)) for i in range(1, 6)], t0=4)  # se mueve: la quita
    assert ev[-1][0] == (800, 300)


def test_zone_never_needs_wrist_out_of_frame():
    for cy in (0.3, 0.47, 0.9, 1.5):          # incluida tu configuración antigua (0.47) y centros absurdos
        for sens in (1, 5, 10):
            cfg = dict(config.DEFAULTS, zone_center=[0.5, cy], sens=sens)
            x0, y0, x1, y1 = config.zone(cfg)
            assert y1 <= config.ZONE_BOTTOM + 1e-9, (cy, sens, y1)   # el índice nunca tiene que bajar de aquí
            assert x1 - x0 > 0.1 and y1 - y0 > 0.1
    # con sensibilidad media la zona sigue siendo cómoda: > 40 % del alto del cuadro
    z = config.zone(dict(config.DEFAULTS, zone_center=[0.5, 0.47], anchor="indice"))
    assert z[3] - z[1] > 0.4
    k = config.zone(dict(config.DEFAULTS, zone_center=[0.5, 0.47]))  # nudillos: más ganancia
    assert abs((z[2] - z[0]) / (k[2] - k[0]) - config.KNUCKLE_GAIN) < 1e-6


def test_edge_pin_keeps_cursor_at_edge_when_hand_leaves():
    from gestureflow.smooth import edge_pin
    screen = (1920, 1080)
    assert edge_pin(0.5, 0.97, (900, 1000), screen) == (900, 1079)      # salió por abajo: llegar al borde
    assert edge_pin(0.02, 0.5, (30, 500), screen) == (0.0, 500)         # por la izquierda
    assert edge_pin(1.2, 1.1, (1900, 1070), screen) == (1919, 1079)     # esquina, ya fuera de la zona
    assert edge_pin(0.5, 0.5, (960, 540), screen) is None               # en el centro: no retener nada


def test_padding_coordinate_roundtrip():
    # un punto en el centro del fotograma original queda en el centro del acolchado y vuelve igual
    w, h, pad = 640, 480, 0.18
    px, py = int(w * pad), int(h * pad)
    W, H = w + 2 * px, h + 2 * py
    for ox, oy in ((0.5, 0.5), (0.0, 1.0), (-0.1, 1.1)):               # incluso fuera del cuadro
        xp, yp = (ox * w + px) / W, (oy * h + py) / H
        assert abs((xp * W - px) / w - ox) < 1e-9 and abs((yp * H - py) / h - oy) < 1e-9
        assert 0 <= xp <= 1 and 0 <= yp <= 1, "MediaPipe recibe coordenadas válidas aun con la mano fuera"


def test_recorder_writes_sessions_and_finishes():
    from gestureflow.recorder import Recorder
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder("t", rounds=2, frames=3, prep=0, out_dir=Path(d))
        n = 0
        while True:
            st = rec.step(synth("palma")); n += 1
            if st["done"]:
                break
            assert n < 100
        rows = Path(rec.path).read_text().strip().splitlines()[1:]
        assert len(rows) == 2 * 4 * 3 and len({r.split(",")[0] for r in rows}) == 2
        assert st["progress"] == 1.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)

