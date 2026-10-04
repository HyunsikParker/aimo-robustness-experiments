"""Audit official development data without exposing protected problem labels."""
import hashlib
import io
import json
import os
import re
import unicodedata
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PROJECT = HERE.parent.parent
REVISION = "23791d9b120132087877b0459f953d5959fc8455"
URL = f"https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/{REVISION}/data/train-00000-of-00001.parquet"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def save(name, value):
    path = HERE / name
    payload = value if isinstance(value, bytes) else (json.dumps(value, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(payload)


def main():
    plan = json.loads((HERE / "split-plan.json").read_text())
    assert plan["revision"] == REVISION
    prior_path = PROJECT / "experiments/intervention-jsd-native-v1/run-config.json"
    prior = json.loads(prior_path.read_text())
    exposed = set(prior["allowed_hashes"])
    protected = set(prior["forbidden_final_hashes"])
    assert len(exposed) == 109 and len(protected) == 28 and not exposed & protected
    old_path = PROJECT / "baselines/data/math-robust-agg.csv"
    # Read text only, never old labels or model features.
    old = pd.read_csv(old_path, usecols=["original_problem"])
    exposed_norm, protected_norm = set(), set()
    for problem in old.original_problem.unique():
        exact = digest(problem.encode())
        group = digest(normalized(problem).encode())
        if exact in exposed:
            exposed_norm.add(group)
        if exact in protected:
            protected_norm.add(group)
    assert len(exposed_norm) == 109 and len(protected_norm) == 28
    if (HERE / "source.parquet").exists():
        payload = (HERE / "source.parquet").read_bytes()
        correction = json.loads((HERE / "data-discrepancy.json").read_text())
        assert digest(payload) == correction["source_sha256"]
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read(10 * 1024 * 1024 + 1)
        save("source.parquet", payload)
    assert len(payload) <= 10 * 1024 * 1024
    # First pass uses problem text only. No labels affect selection.
    text = pd.read_parquet(io.BytesIO(payload), columns=["problem"])
    assert len(text) == 82 and text.problem.map(type).eq(str).all()
    group_ids = text.problem.map(lambda s: digest(normalized(s).encode()))
    groups = set(group_ids)
    assert len(groups) == 36  # Pinned bytes differ from the card's claimed 28.
    quarantined = groups & protected_norm
    fresh = groups - exposed_norm - protected_norm
    ranked = sorted(fresh, key=lambda h: digest(("aimo-main-v2-holdout-20260920:" + h).encode()))
    holdout = set(ranked[:8]) if len(ranked) >= 8 else set()
    development = groups - quarantined - holdout
    assert not development & holdout and not development & quarantined
    assert not holdout & exposed_norm and not holdout & protected_norm
    manifest = {
        "source_sha256": digest(payload),
        "split_plan_sha256": digest((HERE / "split-plan.json").read_bytes()),
        "prior_config_sha256": digest(prior_path.read_bytes()),
        "prior_text_source_sha256": digest(old_path.read_bytes()),
        "development_groups": sorted(development),
        "holdout_groups": sorted(holdout),
        "quarantined_groups": sorted(quarantined),
        "fresh_holdout_available": bool(holdout),
    }
    save("split-manifest.json", manifest)
    frame = pd.read_parquet(io.BytesIO(payload))
    assert set(frame.columns) == {"model_id", "reasoning_effort", "problem", "is_robust", "max_drop"}
    assert len(frame) == 82
    frame["problem_group"] = group_ids
    assert not frame.duplicated(["model_id", "reasoning_effort", "problem_group"]).any()
    dev = frame[frame.problem_group.isin(development)].copy()
    assert dev.is_robust.dropna().isin([True, False]).all()
    labelled = dev[dev.is_robust.notna()].copy()
    counts = []
    for (model, effort), sub in labelled.groupby(["model_id", "reasoning_effort"]):
        counts.append({"model_id": model, "reasoning_effort": effort,
                       "rows": len(sub), "positive": int(sub.is_robust.sum()),
                       "negative": int((~sub.is_robust.astype(bool)).sum())})
    save("development-labelled.json", labelled.to_dict(orient="records"))
    save("development-unlabelled.json", dev[dev.is_robust.isna()].drop(columns=["is_robust"]).to_dict(orient="records"))
    result = {
        "dataset": plan["dataset"], "revision": REVISION, "source_sha256": digest(payload),
        "total_rows": 82, "total_problem_groups": len(groups),
        "dataset_card_problem_count": 28,
        "previously_exposed_groups": len(groups & exposed_norm),
        "old_final_groups_quarantined": len(quarantined),
        "fresh_groups": len(fresh), "holdout_groups": len(holdout),
        "development_groups": len(development), "development_labelled_rows": len(labelled),
        "development_unlabelled_rows": int(dev.is_robust.isna().sum()),
        "development_configuration_counts": counts,
        "fresh_holdout_available": bool(holdout),
        "inference_calls": 0, "fitted_models": 0, "platform_submissions": 0,
        "holdout_labels_reported": False, "cost_usd": 0,
        "limitations": ["Exact and NFKC/whitespace-normalized duplicate detection only; mathematical equivalence not established.",
                         "Dataset and development audit is preparation, not model performance or a first-place claim."]
    }
    save("receipt.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
