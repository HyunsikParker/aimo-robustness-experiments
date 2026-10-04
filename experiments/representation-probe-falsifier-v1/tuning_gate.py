#!/usr/bin/env python3
"""Evaluate five frozen representation candidates on tuning; keep final unopened."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import development_screen as screen


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_RESULTS = ROOT / "development_results.json"
ENSEMBLE_RESULTS = ROOT / "development_ensemble_results.json"
OUTPUT = ROOT / "tuning_results.json"
EXPECTED_DEVELOPMENT_SHA256 = (
    "dc26c91d51430d5701ac2087154b463bbfd317f288c8451cca837a9b365d6730"
)
EXPECTED_ENSEMBLE_SHA256 = (
    "46dee4570bf870198d52959bf960d03c03297bf2ac18b504f797fd051a4c005c"
)
MIN_TUNING_ACCURACY = 0.63
MIN_TUNING_BALANCED_ACCURACY = 0.60
FROZEN_CANDIDATES = (
    {
        "candidate_id": "single-layer19-balanced-c0.01",
        "c": 0.01,
        "class_weight": "balanced",
        "layers": (19,),
        "mode": "single",
    },
    {
        "candidate_id": "margin-trio-18-20-c0.01",
        "c": 0.01,
        "class_weight": "balanced",
        "layers": (18, 19, 20),
        "mode": "mean-margin",
    },
    {
        "candidate_id": "mean-features-18-19-20-c0.01",
        "c": 0.01,
        "class_weight": "balanced",
        "layers": (18, 19, 20),
        "mode": "mean-features",
    },
    {
        "candidate_id": "single-layer19-none-c0.001",
        "c": 0.001,
        "class_weight": None,
        "layers": (19,),
        "mode": "single",
    },
    {
        "candidate_id": "concatenate-19-20-c0.001",
        "c": 0.001,
        "class_weight": "balanced",
        "layers": (19, 20),
        "mode": "concatenate",
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimator(candidate: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=float(candidate["c"]),
                    class_weight=candidate["class_weight"],
                    max_iter=5000,
                    random_state=screen.CV_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def predict(
    candidate: dict[str, Any],
    features: np.ndarray,
    labels: np.ndarray,
    development: np.ndarray,
    tuning: np.ndarray,
) -> np.ndarray:
    layers = tuple(candidate["layers"])
    mode = candidate["mode"]
    if mode == "single":
        model = estimator(candidate)
        layer = layers[0]
        model.fit(features[development, layer, :], labels[development])
        return model.predict(features[tuning, layer, :]).astype(np.int64)
    if mode == "mean-margin":
        margins = []
        for layer in layers:
            model = estimator(candidate)
            model.fit(features[development, layer, :], labels[development])
            margins.append(model.decision_function(features[tuning, layer, :]))
        return (np.mean(np.stack(margins), axis=0) >= 0.0).astype(np.int64)
    if mode == "mean-features":
        train = np.mean(features[development][:, layers, :], axis=1, dtype=np.float32)
        test = np.mean(features[tuning][:, layers, :], axis=1, dtype=np.float32)
    elif mode == "concatenate":
        train = features[development][:, layers, :].reshape(len(development), -1)
        test = features[tuning][:, layers, :].reshape(len(tuning), -1)
    else:
        raise RuntimeError("unknown frozen candidate mode")
    model = estimator(candidate)
    model.fit(train, labels[development])
    return model.predict(test).astype(np.int64)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("tuning results already exist")
    if (
        sha256(DEVELOPMENT_RESULTS) != EXPECTED_DEVELOPMENT_SHA256
        or sha256(ENSEMBLE_RESULTS) != EXPECTED_ENSEMBLE_SHA256
    ):
        raise RuntimeError("development evidence changed")
    development_report = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    ensemble_report = json.loads(ENSEMBLE_RESULTS.read_text(encoding="utf-8"))
    if (
        development_report["tuning"]["status"] != "unopened"
        or development_report["final"]["status"] != "unopened"
        or ensemble_report["tuning"] != "unopened"
        or ensemble_report["final"] != "unopened"
    ):
        raise RuntimeError("development evidence did not preserve holdouts")

    source, labels, _ = screen.load_inputs()
    development, tuning, _ = screen.split_indices(labels)
    features = np.asarray(source, dtype=np.float32)
    if not np.isfinite(features[development]).all() or not np.isfinite(features[tuning]).all():
        raise RuntimeError("development or tuning representations are non-finite")

    tuning_labels = labels[tuning]
    results: list[dict[str, Any]] = []
    for frozen in FROZEN_CANDIDATES:
        prediction = predict(frozen, features, labels, development, tuning)
        result = {
            **frozen,
            "accuracy": float(accuracy_score(tuning_labels, prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(tuning_labels, prediction)
            ),
            "false_negative": int(((tuning_labels == 1) & (prediction == 0)).sum()),
            "false_positive": int(((tuning_labels == 0) & (prediction == 1)).sum()),
            "f1": float(f1_score(tuning_labels, prediction)),
            "predicted_positive": int(prediction.sum()),
        }
        result["passes_gate"] = bool(
            result["accuracy"] >= MIN_TUNING_ACCURACY
            and result["balanced_accuracy"] >= MIN_TUNING_BALANCED_ACCURACY
            and 0 < result["predicted_positive"] < len(tuning)
        )
        results.append(result)
    results.sort(
        key=lambda row: (
            not row["passes_gate"],
            -row["balanced_accuracy"],
            -row["accuracy"],
            row["candidate_id"],
        )
    )
    selected = results[0] if results and results[0]["passes_gate"] else None
    report = {
        "development_ensemble_sha256": EXPECTED_ENSEMBLE_SHA256,
        "development_results_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "final": "unopened",
        "frozen_candidate_count": len(FROZEN_CANDIDATES),
        "gate": {
            "minimum_accuracy": MIN_TUNING_ACCURACY,
            "minimum_balanced_accuracy": MIN_TUNING_BALANCED_ACCURACY,
        },
        "open_final": selected is not None,
        "results": results,
        "schema_version": 1,
        "selected_candidate": selected["candidate_id"] if selected else None,
        "tuning_count": len(tuning),
        "tuning_positive": int(tuning_labels.sum()),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "final": "unopened",
                "open_final": report["open_final"],
                "results": results,
                "selected_candidate": report["selected_candidate"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
