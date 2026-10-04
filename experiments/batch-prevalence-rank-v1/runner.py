"""Fixed grouped falsifier for batch prevalence allocation; never reads holdout files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"
RESULT_PATH = HERE / "result.json"
REQUIRED_COLUMNS = {"model_id", "reasoning_effort", "problem", "is_robust", "problem_group"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_input() -> tuple[dict, pd.DataFrame, Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    source = (HERE / config["development_path"]).resolve()
    if sha256(source) != config["development_sha256"]:
        raise RuntimeError("development hash mismatch")
    raw = pd.read_json(source)
    if not REQUIRED_COLUMNS.issubset(raw.columns):
        raise RuntimeError("development schema mismatch")
    frame = raw.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    if len(frame) != config["rows"] or frame["problem_group"].nunique() != config["groups"]:
        raise RuntimeError("development cardinality mismatch")
    if frame["is_robust"].isna().any():
        raise RuntimeError("development contains unlabelled rows")
    return config, frame, source


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "problem": frame["problem"].astype(str),
            "model_id": frame["model_id"].astype(str),
            "reasoning_effort": frame["reasoning_effort"].astype(str),
            "source": "current",
        }
    )


def pipeline(config: dict) -> Pipeline:
    vec = config["vectorizer"]
    clf = config["classifier"]
    features = ColumnTransformer(
        [
            (
                "text",
                TfidfVectorizer(
                    analyzer=vec["analyzer"],
                    ngram_range=tuple(vec["ngram_range"]),
                    min_df=vec["min_df"],
                    sublinear_tf=vec["sublinear_tf"],
                    max_features=vec["max_features"],
                    dtype=np.float64,
                ),
                "problem",
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["model_id", "reasoning_effort", "source"],
            ),
        ]
    )
    model = LogisticRegression(
        solver=clf["solver"],
        C=clf["C"],
        class_weight=clf["class_weight"],
        max_iter=clf["max_iter"],
        random_state=config["seed"],
    )
    return Pipeline([("features", features), ("model", model)])


def metric_set(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "predicted_negative": int((y_pred == 0).sum()),
        "predicted_positive": int((y_pred == 1).sum()),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def posterior_prevalence(labels: np.ndarray) -> float:
    return float((labels.sum() + 1.0) / (len(labels) + 2.0))


def ranked_allocation(probabilities: np.ndarray, prevalence: float, stable_indices: np.ndarray) -> np.ndarray:
    size = len(probabilities)
    if size == 0:
        return np.empty(0, dtype=int)
    positives = math.floor(prevalence * size + 0.5)
    if size >= 2:
        positives = min(size - 1, max(1, positives))
    else:
        positives = int(prevalence >= 0.5)
    order = np.lexsort((stable_indices, -probabilities))
    result = np.zeros(size, dtype=int)
    result[order[:positives]] = 1
    return result


def check_only() -> None:
    config, frame, source = load_input()
    if config["seed"] != 20260921 or config["folds"] != 7:
        raise RuntimeError("fixed fold configuration mismatch")
    print(
        json.dumps(
            {
                "status": "PASS",
                "rows": len(frame),
                "groups": int(frame["problem_group"].nunique()),
                "source_sha256": sha256(source),
                "config_sha256": sha256(CONFIG_PATH),
                "sklearn": sklearn.__version__,
            },
            sort_keys=True,
        )
    )


def run() -> dict:
    started = time.monotonic()
    config, frame, source = load_input()
    x = prepare_features(frame)
    y = frame["is_robust"].astype(bool).to_numpy(dtype=int)
    groups = frame["problem_group"].astype(str).to_numpy()
    keys = list(zip(x["model_id"], x["reasoning_effort"]))
    splitter = StratifiedGroupKFold(
        n_splits=config["folds"], shuffle=True, random_state=config["seed"]
    )
    predictions = {
        "candidate": np.full(len(frame), -1, dtype=int),
        "per_row_threshold": np.full(len(frame), -1, dtype=int),
        "configuration_prior": np.full(len(frame), -1, dtype=int),
    }
    fold_summaries = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), 1):
        model = pipeline(config)
        model.fit(x.iloc[train_idx], y[train_idx])
        probabilities = model.predict_proba(x.iloc[test_idx])[:, 1]
        predictions["per_row_threshold"][test_idx] = (probabilities >= 0.5).astype(int)
        global_prevalence = posterior_prevalence(y[train_idx])
        subgroup_summaries = []
        for key in sorted(set(keys[i] for i in test_idx)):
            local_positions = np.array(
                [position for position, row_index in enumerate(test_idx) if keys[row_index] == key],
                dtype=int,
            )
            train_labels = y[
                np.array([row_index for row_index in train_idx if keys[row_index] == key], dtype=int)
            ]
            prevalence = posterior_prevalence(train_labels) if len(train_labels) else global_prevalence
            allocated = ranked_allocation(
                probabilities[local_positions], prevalence, test_idx[local_positions]
            )
            target_indices = test_idx[local_positions]
            predictions["candidate"][target_indices] = allocated
            predictions["configuration_prior"][target_indices] = int(prevalence >= 0.5)
            subgroup_summaries.append(
                {
                    "model_id": key[0],
                    "reasoning_effort": key[1],
                    "test_rows": int(len(target_indices)),
                    "train_rows": int(len(train_labels)),
                    "allocated_positive": int(allocated.sum()),
                }
            )
        fold_summaries.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_groups": int(len(set(groups[train_idx]))),
                "test_groups": int(len(set(groups[test_idx]))),
                "subgroups": subgroup_summaries,
            }
        )
    if any((values < 0).any() for values in predictions.values()):
        raise RuntimeError("incomplete out-of-fold predictions")
    measured = {name: metric_set(y, values) for name, values in predictions.items()}
    candidate = measured["candidate"]
    references = [measured["per_row_threshold"], measured["configuration_prior"]]
    gate = config["gate"]
    gates = {
        "accuracy": candidate["accuracy"] >= gate["accuracy_min"],
        "balanced_accuracy": candidate["balanced_accuracy"] >= gate["balanced_accuracy_min"],
        "nondegenerate": candidate["predicted_negative"] >= gate["predicted_each_class_min"]
        and candidate["predicted_positive"] >= gate["predicted_each_class_min"],
        "accuracy_gain": candidate["accuracy"] - max(r["accuracy"] for r in references)
        >= gate["accuracy_gain_over_each_reference_min"],
        "balanced_accuracy_gain": candidate["balanced_accuracy"]
        - max(r["balanced_accuracy"] for r in references)
        >= gate["balanced_accuracy_gain_over_each_reference_min"],
    }
    elapsed = time.monotonic() - started
    gates["runtime"] = elapsed <= gate["runtime_seconds_max"]
    result = {
        "stage": "completed",
        "screen": "development_grouped_cross_validation",
        "mechanism": "configuration-batch-prevalence-ranked-text-v1",
        "gate_passed": all(gates.values()),
        "rows": int(len(frame)),
        "problem_groups": int(frame["problem_group"].nunique()),
        "folds": fold_summaries,
        "metrics": measured,
        "accuracy_gain_over_best_reference": float(
            candidate["accuracy"] - max(r["accuracy"] for r in references)
        ),
        "balanced_accuracy_gain_over_best_reference": float(
            candidate["balanced_accuracy"]
            - max(r["balanced_accuracy"] for r in references)
        ),
        "gates": gates,
        "runtime_seconds": elapsed,
        "cost_usd": 0,
        "model_inference_calls": 0,
        "holdout_files_opened": 0,
        "holdout_groups_processed": 0,
        "platform_submissions": 0,
        "input_hashes": {
            "development": sha256(source),
            "config": sha256(CONFIG_PATH),
            "experiment": sha256(HERE / "experiment.json"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "failure_scope": None
        if all(gates.values())
        else "The fixed batch-prevalence development gate failed; no holdout or platform action is authorized.",
    }
    temporary = RESULT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, RESULT_PATH)
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--wait-for", type=Path)
    args = parser.parse_args()
    if args.check:
        check_only()
        return
    if args.wait_for:
        deadline = time.monotonic() + 60
        while not args.wait_for.exists():
            if time.monotonic() > deadline:
                raise RuntimeError("run gate timeout")
            time.sleep(0.1)
    run()


if __name__ == "__main__":
    main()
