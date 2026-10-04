"""One preregistered geometry classifier; the final split is never scored."""

import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "representation-probe-falsifier-v1" / "development_screen.py"


def geometry(hidden):
    """Summarize relative layer changes, independent of hidden-vector basis."""
    x = np.asarray(hidden, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError("expected finite batch x layers x width")
    # Exclude embeddings and the final normalization boundary.
    x = x[:, 1:-1]
    norms = np.linalg.norm(x, axis=-1)
    if np.any(norms <= 1e-12):
        raise ValueError("degenerate representation")
    unit = x / norms[..., None]
    angular = 1 - np.clip(np.sum(unit[:, 1:] * unit[:, :-1], axis=-1), -1, 1)
    radial = np.diff(np.log(norms), axis=1)
    columns = []
    for indices in np.array_split(np.arange(angular.shape[1]), 3):
        a, r = angular[:, indices], radial[:, indices]
        columns.extend([a.mean(1), a.std(1), r.mean(1), r.std(1)])
    steps = np.diff(unit, axis=1)
    lengths = np.linalg.norm(steps, axis=-1)
    path = lengths.sum(1)
    direct = np.linalg.norm(unit[:, -1] - unit[:, 0], axis=-1)
    curvature = np.linalg.norm(np.diff(steps, axis=1), axis=-1)
    columns.extend([path, direct, direct / np.maximum(path, 1e-12),
                    curvature.mean(1), curvature.std(1)])
    result = np.column_stack(columns)
    if result.shape != (len(x), 17) or not np.isfinite(result).all():
        raise ValueError("invalid geometry features")
    return result


def check():
    rng = np.random.default_rng(20260905)
    x = rng.normal(size=(3, 12, 8))
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    actual = geometry(x)
    np.testing.assert_allclose(actual, geometry(7.0 * x @ q), atol=1e-10)
    np.testing.assert_allclose(geometry(np.ones((2, 12, 8))), 0, atol=1e-10)
    print("PASS: geometry shape, finiteness, scale and basis invariance")


def main():
    signal.alarm(60)
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location("frozen_aimo_screen", OLD)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    hidden, labels, hashes = screen.load_inputs()
    development, tuning, final = screen.split_indices(labels)
    assert screen.split_digest(final, hashes) == "b07ae0d639b4a759271529ab6e14d21ec087ebbf75579b183d54a0390096c5b2"
    assert screen.split_digest(development, hashes) == "f58a48e9d6e2dbe579040b50c8f5eaef354cd740295244ecdb7cafbc39c4b98b"
    # No final representations are materialized or predictions generated.
    train_x = geometry(hidden[development])
    tune_x = geometry(hidden[tuning])
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", solver="liblinear",
        max_iter=1000, random_state=20260905))
    model.fit(train_x, labels[development])
    prediction = model.predict(tune_x)
    majority_label = int(labels[development].mean() >= 0.5)
    baseline = float(accuracy_score(labels[tuning], np.full(len(tuning), majority_label)))
    accuracy = float(accuracy_score(labels[tuning], prediction))
    balanced = float(balanced_accuracy_score(labels[tuning], prediction))
    passed = accuracy >= 0.74 and balanced >= 0.65 and accuracy - baseline >= 0.03
    result = {
        "mechanism": "relative-layer-geometry-v1", "feature_count": 17,
        "candidate_count": 1, "training_count": len(development), "tuning_count": len(tuning),
        "tuning_previously_exposed": True, "final_count": len(final),
        "final_status": "unscored", "final_predictions_generated": False,
        "tuning_accuracy": accuracy, "tuning_balanced_accuracy": balanced,
        "majority_accuracy": baseline, "delta_vs_majority": accuracy - baseline,
        "gate_passed": passed, "elapsed_seconds": time.monotonic() - started,
        "cost_usd": 0, "model_downloads": 0, "platform_submissions": 0,
        "interpretation": "Development falsifier only; no private leaderboard or cross-model generalization claim."
    }
    output = ROOT / "result.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    output.chmod(0o600)
    print(json.dumps(result))


if __name__ == "__main__":
    check() if "--check" in sys.argv else main()
