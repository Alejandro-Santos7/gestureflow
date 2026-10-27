"""Entrena el MLP sobre landmarks y deja un registro reproducible en runs/<id>/.

Uso:  python -m gestureflow.train [--data data/raw] [--name mlp-v1]
Validación: dejar sesiones enteras fuera (los frames vecinos son casi idénticos; un split
aleatorio inflaría la precisión). Con 1 sola sesión cae a split aleatorio y lo marca.
El modelo solo se promueve a models/latest.npz si no es peor que las reglas geométricas.
"""
import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from . import config
from .features import classify_rules, normalize


def load_data(data_dir):
    raws, y, sessions, manifest = [], [], [], []
    for p in sorted(Path(data_dir).glob("*.csv")):
        rows = list(csv.reader(open(p, newline="")))[1:]
        for r in rows:
            sessions.append(r[0]); y.append(r[2]); raws.append(np.array(r[4:], np.float32).reshape(21, 3))
        manifest.append({"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "rows": len(rows)})
    return np.array(raws), np.array(y), np.array(sessions), manifest


def _fit(X, y, cfg):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    m = MLPClassifier(cfg["hidden"], max_iter=cfg["max_iter"], random_state=cfg["seed"], early_stopping=False)
    m.fit((X - mean) / scale, y)
    return m, mean, scale


def train(data_dir=None, runs_dir=None, models_dir=None, name="mlp", seed=0, hidden=(64, 32), max_iter=300):
    data_dir = Path(data_dir or config.ROOT / "data" / "raw")
    runs_dir = Path(runs_dir or config.ROOT / "runs")
    models_dir = Path(models_dir or config.ROOT / "models")
    cfg = {"name": name, "seed": seed, "hidden": list(hidden), "max_iter": max_iter}

    raws, y, sessions, manifest = load_data(data_dir)
    if len(set(y)) < 2:
        raise SystemExit(f"Hacen falta >= 2 gestos en {data_dir}; ejecuta gestureflow.collect primero.")
    X = np.array([normalize(lm) for lm in raws])

    # --- validación ---
    uniq = sorted(set(sessions))
    if len(uniq) >= 2:
        split, folds = "leave-one-session-out", [(sessions != s, sessions == s) for s in uniq]
    else:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        split = "random 80/20 (1 sesion: optimista)"
        folds = [(np.isin(idx, tr), np.isin(idx, te))]
    pred = np.empty(len(y), dtype=y.dtype)
    seen = np.zeros(len(y), bool)
    fold_acc = []
    for trm, tem in folds:
        m, mean, scale = _fit(X[trm], y[trm], cfg)
        pred[tem] = m.predict((X[tem] - mean) / scale)
        seen |= tem
        fold_acc.append(float((pred[tem] == y[tem]).mean()))
    labels = sorted(set(y))
    rules_pred = np.array([classify_rules(lm) for lm in raws[seen]])
    acc = float((pred[seen] == y[seen]).mean())
    rules_acc = float((rules_pred == y[seen]).mean())

    # --- modelo final con todos los datos ---
    m, mean, scale = _fit(X, y, cfg)
    promoted = acc >= rules_acc
    metrics = {
        "split": split, "n_samples": int(len(y)), "n_sessions": len(uniq),
        "samples_per_class": {k: int((y == k).sum()) for k in labels},
        "accuracy": acc, "fold_accuracy": fold_acc, "rules_baseline_accuracy": rules_acc,
        "labels": labels, "confusion_matrix": confusion_matrix(y[seen], pred[seen], labels=labels).tolist(),
        "report": classification_report(y[seen], pred[seen], labels=labels, output_dict=True, zero_division=0),
        "promoted_to_latest": promoted,
    }

    run_id = f"{datetime.now():%Y-%m-%d_%H%M%S}_{name}"
    run = runs_dir / run_id
    run.mkdir(parents=True)
    arrays = {"mean": mean, "scale": scale, "n_layers": len(m.coefs_), "classes": np.array(m.classes_)}
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        arrays[f"W{i}"], arrays[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
    np.savez(run / "model.npz", **arrays)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=config.ROOT).stdout.strip()
    except OSError:
        commit = ""
    (run / "env.txt").write_text(
        f"python {platform.python_version()}\nnumpy {np.__version__}\nsklearn {sklearn.__version__}\n"
        f"platform {platform.platform()}\ngit {commit or 'n/a'}\n")

    if promoted:
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(run / "model.npz", models_dir / "latest.npz")
    idx_path = runs_dir / "index.csv"
    new = not idx_path.exists()
    with open(idx_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["run", "split", "samples", "sessions", "accuracy", "rules_accuracy", "promoted"])
        w.writerow([run_id, split, len(y), len(uniq), f"{acc:.4f}", f"{rules_acc:.4f}", promoted])
    return run, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--name", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run, m = train(a.data, name=a.name, seed=a.seed)
    print(f"Registro: {run}\nsplit: {m['split']}  sesiones: {m['n_sessions']}  muestras: {m['n_samples']}")
    print(f"accuracy modelo: {m['accuracy']:.3f}   reglas: {m['rules_baseline_accuracy']:.3f}")
    for k in m["labels"]:
        r = m["report"][k]
        print(f"  {k:8s} precision {r['precision']:.3f}  recall {r['recall']:.3f}")
    print("promovido a models/latest.npz" if m["promoted_to_latest"] else "NO promovido: las reglas rinden mejor")


if __name__ == "__main__":
    sys.exit(main())
