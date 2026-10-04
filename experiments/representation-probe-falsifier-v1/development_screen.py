#!/usr/bin/env python3
"""Screen representation probes on development only; tuning/final stay unopened."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments" / "representation-extraction-output-v1-20260823"
ARRAY = SOURCE / "qwen_prompt_representations_fp16.npy"
ROWS = SOURCE / "representation_rows.json"
MANIFEST = SOURCE / "extraction_manifest.json"
OUTPUT = Path(__file__).resolve().parent / "development_results.json"
ARRAY_SHA256 = "2d9a93e3d804fb28254310072ec2040bbe2aa73d7dfcba823b5d24b04bdfe04e"
ROWS_SHA256 = "c6d42509b1b0bb40a6d4a9441bdfb778d9bfe4157b4cd770ad67d1b1d6d7e15c"
EXPECTED_SHAPE = (137, 37, 4096)
FINAL_SIZE = 28
TUNING_SIZE = 27
SPLIT_SEED = 20260823
TUNING_SEED = 20260824
CV_SEED = 20260825


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs() -> tuple[np.ndarray, np.ndarray, list[str]]:
    if sha256(ARRAY) != ARRAY_SHA256 or sha256(ROWS) != ROWS_SHA256:
        raise RuntimeError("validated representation inputs changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("array_sha256") != ARRAY_SHA256
        or manifest.get("rows_sha256") != ROWS_SHA256
        or manifest.get("array_shape") != list(EXPECTED_SHAPE)
    ):
        raise RuntimeError("representation manifest identity changed")
    features = np.load(ARRAY, allow_pickle=False, mmap_mode="r")
    rows = json.loads(ROWS.read_text(encoding="utf-8"))
    if features.shape != EXPECTED_SHAPE or features.dtype != np.float16:
        raise RuntimeError("representation array shape or dtype changed")
    if type(rows) is not list or len(rows) != EXPECTED_SHAPE[0]:
        raise RuntimeError("representation row mapping changed")
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    hashes = [row["problem_hash"] for row in rows]
    if labels.sum() != 36 or len(set(hashes)) != len(hashes) or hashes != sorted(hashes):
        raise RuntimeError("representation labels or hashes changed")
    return features, labels, hashes


def split_indices(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    development_tuning, final = train_test_split(
        indices,
        test_size=FINAL_SIZE,
        random_state=SPLIT_SEED,
        stratify=labels,
    )
    development, tuning = train_test_split(
        development_tuning,
        test_size=TUNING_SIZE,
        random_state=TUNING_SEED,
        stratify=labels[development_tuning],
    )
    development = np.sort(development)
    tuning = np.sort(tuning)
    final = np.sort(final)
    if (
        len(development) != 82
        or len(tuning) != TUNING_SIZE
        or len(final) != FINAL_SIZE
        or set(development) & set(tuning)
        or set(development) & set(final)
        or set(tuning) & set(final)
    ):
        raise RuntimeError("three-way split is invalid")
    return development, tuning, final


def split_digest(indices: np.ndarray, hashes: list[str]) -> str:
    payload = "\n".join(hashes[int(index)] for index in indices).encode()
    return hashlib.sha256(payload).hexdigest()


def configs() -> list[tuple[str, Pipeline]]:
    built: list[tuple[str, Pipeline]] = []
    for weight_name, class_weight in (("none", None), ("balanced", "balanced")):
        for c_value in (0.001, 0.01, 0.1, 1.0):
            built.append(
                (
                    f"logreg-{weight_name}-c{c_value:g}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                LogisticRegression(
                                    C=c_value,
                                    class_weight=class_weight,
                                    max_iter=5000,
                                    random_state=CV_SEED,
                                    solver="liblinear",
                                ),
                            ),
                        ]
                    ),
                )
            )
        for alpha in (1.0, 10.0, 100.0):
            built.append(
                (
                    f"ridge-{weight_name}-a{alpha:g}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                RidgeClassifier(
                                    alpha=alpha,
                                    class_weight=class_weight,
                                    solver="lsqr",
                                ),
                            ),
                        ]
                    ),
                )
            )
        for c_value in (0.001, 0.01, 0.1):
            built.append(
                (
                    f"svc-{weight_name}-c{c_value:g}",
                    Pipeline(
                        [
                            ("scale", StandardScaler()),
                            (
                                "model",
                                LinearSVC(
                                    C=c_value,
                                    class_weight=class_weight,
                                    dual="auto",
                                    max_iter=5000,
                                    random_state=CV_SEED,
                                ),
                            ),
                        ]
                    ),
                )
            )
    return built


def summarize_scores(scores: dict[str, Any]) -> dict[str, float]:
    return {
        "accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "accuracy_std": float(np.std(scores["test_accuracy"])),
        "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
        "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
        "f1_mean": float(np.mean(scores["test_f1"])),
        "f1_std": float(np.std(scores["test_f1"])),
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("development results already exist")
    features, labels, hashes = load_inputs()
    development, tuning, final = split_indices(labels)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=CV_SEED)
    y_development = labels[development]
    baseline_accuracy: list[float] = []
    baseline_balanced: list[float] = []
    for _, test in cv.split(development, y_development):
        prediction = np.zeros(len(test), dtype=np.int64)
        baseline_accuracy.append(float(accuracy_score(y_development[test], prediction)))
        baseline_balanced.append(
            float(balanced_accuracy_score(y_development[test], prediction))
        )

    results: list[dict[str, Any]] = []
    warnings.filterwarnings("error", category=ConvergenceWarning)
    for layer in range(EXPECTED_SHAPE[1]):
        matrix = np.asarray(features[development, layer, :], dtype=np.float32)
        if not np.isfinite(matrix).all():
            raise RuntimeError("development representation is non-finite")
        for config_id, estimator in configs():
            scores = cross_validate(
                estimator,
                matrix,
                y_development,
                cv=cv,
                scoring=("accuracy", "balanced_accuracy", "f1"),
                n_jobs=4,
                error_score="raise",
                return_train_score=False,
            )
            results.append(
                {
                    "config_id": config_id,
                    "layer": layer,
                    **summarize_scores(scores),
                }
            )

    results.sort(
        key=lambda row: (
            -row["accuracy_mean"],
            -row["balanced_accuracy_mean"],
            row["accuracy_std"],
            row["layer"],
            row["config_id"],
        )
    )
    report = {
        "baseline": {
            "accuracy_mean": float(np.mean(baseline_accuracy)),
            "accuracy_std": float(np.std(baseline_accuracy)),
            "balanced_accuracy_mean": float(np.mean(baseline_balanced)),
        },
        "cv": {"folds": 5, "repeats": 5, "seed": CV_SEED},
        "development": {
            "count": len(development),
            "positive": int(labels[development].sum()),
            "split_digest": split_digest(development, hashes),
        },
        "final": {
            "count": len(final),
            "positive": int(labels[final].sum()),
            "split_digest": split_digest(final, hashes),
            "status": "unopened",
        },
        "input": {
            "array_sha256": ARRAY_SHA256,
            "rows_sha256": ROWS_SHA256,
            "shape": list(EXPECTED_SHAPE),
        },
        "results": results,
        "schema_version": 1,
        "sklearn_version": sklearn.__version__,
        "split_seeds": {"final": SPLIT_SEED, "tuning": TUNING_SEED},
        "tuning": {
            "count": len(tuning),
            "positive": int(labels[tuning].sum()),
            "split_digest": split_digest(tuning, hashes),
            "status": "unopened",
        },
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "baseline_accuracy": report["baseline"]["accuracy_mean"],
                "development_results": len(results),
                "final": "unopened",
                "top": results[:5],
                "tuning": "unopened",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
