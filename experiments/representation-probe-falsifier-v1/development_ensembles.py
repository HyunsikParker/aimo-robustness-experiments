#!/usr/bin/env python3
"""Test a small frozen family of adjacent-layer ensembles on development only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import development_screen as screen


ROOT = Path(__file__).resolve().parent
DEVELOPMENT_RESULTS = ROOT / "development_results.json"
OUTPUT = ROOT / "development_ensemble_results.json"
EXPECTED_DEVELOPMENT_SHA256 = (
    "dc26c91d51430d5701ac2087154b463bbfd317f288c8451cca837a9b365d6730"
)
LAYER_SETS = {
    "pair-19-20": (19, 20),
    "trio-18-20": (18, 19, 20),
    "quad-17-20": (17, 18, 19, 20),
    "stable-10-19-20": (10, 19, 20),
    "stable-10-17-20": (10, 17, 18, 19, 20),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimator(c_value: float) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=screen.CV_SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )


def summarize(values: list[float], name: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {f"{name}_mean": float(array.mean()), f"{name}_std": float(array.std())}


def evaluate_margin_ensemble(
    features: np.ndarray,
    labels: np.ndarray,
    layers: tuple[int, ...],
    c_value: float,
    cv: RepeatedStratifiedKFold,
) -> dict[str, float]:
    accuracy: list[float] = []
    balanced: list[float] = []
    f1: list[float] = []
    base = estimator(c_value)
    indices = np.arange(len(labels))
    for train, test in cv.split(indices, labels):
        margins = []
        for layer in layers:
            model = clone(base)
            model.fit(features[train, layer, :], labels[train])
            margins.append(model.decision_function(features[test, layer, :]))
        prediction = (np.mean(np.stack(margins), axis=0) >= 0.0).astype(np.int64)
        accuracy.append(float(accuracy_score(labels[test], prediction)))
        balanced.append(float(balanced_accuracy_score(labels[test], prediction)))
        f1.append(float(f1_score(labels[test], prediction)))
    return {
        **summarize(accuracy, "accuracy"),
        **summarize(balanced, "balanced_accuracy"),
        **summarize(f1, "f1"),
    }


def evaluate_feature_projection(
    features: np.ndarray,
    labels: np.ndarray,
    layers: tuple[int, ...],
    c_value: float,
    mode: str,
    cv: RepeatedStratifiedKFold,
) -> dict[str, float]:
    if mode == "mean-features":
        matrix = np.mean(features[:, layers, :], axis=1, dtype=np.float32)
    elif mode == "concatenate":
        matrix = features[:, layers, :].reshape(len(features), -1)
    else:
        raise RuntimeError("unknown projection mode")
    scores = cross_validate(
        estimator(c_value),
        matrix,
        labels,
        cv=cv,
        scoring=("accuracy", "balanced_accuracy", "f1"),
        n_jobs=4,
        error_score="raise",
        return_train_score=False,
    )
    return screen.summarize_scores(scores)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("development ensemble results already exist")
    if sha256(DEVELOPMENT_RESULTS) != EXPECTED_DEVELOPMENT_SHA256:
        raise RuntimeError("development screen result changed")
    base_report = json.loads(DEVELOPMENT_RESULTS.read_text(encoding="utf-8"))
    if base_report["tuning"]["status"] != "unopened" or base_report["final"]["status"] != "unopened":
        raise RuntimeError("tuning or final was already opened")

    source, labels, _ = screen.load_inputs()
    development, _, _ = screen.split_indices(labels)
    features = np.asarray(source[development], dtype=np.float32)
    development_labels = labels[development]
    if not np.isfinite(features).all():
        raise RuntimeError("development representations are non-finite")

    results: list[dict[str, Any]] = []
    for name, layers in LAYER_SETS.items():
        for c_value in (0.001, 0.01):
            cv = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=5, random_state=screen.CV_SEED
            )
            results.append(
                {
                    "c": c_value,
                    "candidate_id": f"margin-{name}-c{c_value:g}",
                    "layers": list(layers),
                    "mode": "mean-margin",
                    **evaluate_margin_ensemble(
                        features,
                        development_labels,
                        layers,
                        c_value,
                        cv,
                    ),
                }
            )
    for mode, layers, c_values in (
        ("mean-features", (18, 19, 20), (0.001, 0.01)),
        ("concatenate", (19, 20), (0.0001, 0.001)),
    ):
        for c_value in c_values:
            cv = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=5, random_state=screen.CV_SEED
            )
            results.append(
                {
                    "c": c_value,
                    "candidate_id": f"{mode}-{'-'.join(map(str, layers))}-c{c_value:g}",
                    "layers": list(layers),
                    "mode": mode,
                    **evaluate_feature_projection(
                        features,
                        development_labels,
                        layers,
                        c_value,
                        mode,
                        cv,
                    ),
                }
            )
    results.sort(
        key=lambda row: (
            -row["balanced_accuracy_mean"],
            -row["accuracy_mean"],
            row["balanced_accuracy_std"],
            row["candidate_id"],
        )
    )
    report = {
        "development_count": len(development),
        "development_results_sha256": EXPECTED_DEVELOPMENT_SHA256,
        "final": "unopened",
        "results": results,
        "schema_version": 1,
        "tuning": "unopened",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"candidates": len(results), "top": results[:5]}, sort_keys=True))


if __name__ == "__main__":
    main()
