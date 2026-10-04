"""Exactly one CPU fit, after the MLX extraction process has exited."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def main():
    config = json.loads((ROOT / "run-config.json").read_text())
    labels_path = ROOT / "labels.json"
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == config["labels_sha256"]
    labels = {x["problem_hash"]: x for x in json.loads(labels_path.read_text())}
    source = json.loads((ROOT / "features.json").read_text())
    rows = source["rows"]
    assert len(rows) == 109 and len({r["problem_hash"] for r in rows}) == 109
    assert {r["problem_hash"] for r in rows} == set(config["allowed_hashes"])
    assert not (set(labels) & set(config["forbidden_final_hashes"]))
    assert max(r["identity_sham_jsd"] for r in rows) <= 1e-8
    for r in rows:
        assert r["split"] == labels[r["problem_hash"]]["split"]
    train = [r for r in rows if r["split"] == "train"]
    tune = [r for r in rows if r["split"] == "tuning"]
    assert len(train) == 82 and len(tune) == 27
    x = lambda values: np.array([[r["jsd"]] for r in values], dtype=np.float64)
    y = lambda values: np.array([labels[r["problem_hash"]]["label"] for r in values])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1,
        class_weight="balanced", solver="liblinear", max_iter=1000,
        random_state=20260906))
    model.fit(x(train), y(train))
    prediction = model.predict(x(tune))
    majority = int(np.mean(y(train)) >= 0.5)
    baseline = float(accuracy_score(y(tune), np.full(27, majority)))
    accuracy = float(accuracy_score(y(tune), prediction))
    balanced = float(balanced_accuracy_score(y(tune), prediction))
    passed = accuracy >= 0.74 and balanced >= 0.65 and accuracy - baseline >= 0.03
    result = {"mechanism": "matched-prefix-newline-jsd-v1", "candidate_count": 1,
        "feature_count": 1, "train_count": 82, "tuning_count": 27,
        "tuning_previously_exposed": True, "tuning_accuracy": accuracy,
        "tuning_balanced_accuracy": balanced, "majority_accuracy": baseline,
        "delta_vs_majority": accuracy - baseline, "gate_passed": passed,
        "coefficient": float(model[-1].coef_[0, 0]),
        "intercept": float(model[-1].intercept_[0]),
        "max_identity_sham_jsd": max(r["identity_sham_jsd"] for r in rows),
        "mean_intervention_jsd": float(np.mean(x(rows))),
        "extraction_seconds": source["elapsed_seconds"],
        "peak_mlx_bytes": source["peak_mlx_bytes"],
        "final_count": 28, "final_status": "unscored",
        "final_predictions_generated": False, "cross_model_validation": False,
        "platform_submissions": 0, "cloud_compute_used": False, "cost_usd": 0}
    path = ROOT / "result.json"
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + "\n")
    path.chmod(0o600)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        assert np.isfinite(StandardScaler().fit_transform([[0.0], [1.0]])).all()
        print("PASS: isolated CPU fitting dependencies")
    else:
        main()
