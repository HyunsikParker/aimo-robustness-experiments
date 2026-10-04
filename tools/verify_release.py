"""Verify immutable release identities and recorded claims with no model execution."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text())


def main():
    checks = 0

    def check(condition, name):
        nonlocal checks
        if not condition:
            raise RuntimeError(name)
        checks += 1

    manifest = read("release-manifest.json")
    expected = {r["path"] for r in manifest["files"]} | {"release-manifest.json"}
    observed = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                if p.is_file() and not any(x in p.relative_to(ROOT).parts for x in (".git", "__pycache__"))}
    check(observed == expected, "release file allowlist changed; run on a clean source checkout")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        check(not path.is_symlink(), "symlink in source release")
        payload = path.read_bytes()
        check(len(payload) == record["bytes"] and hashlib.sha256(payload).hexdigest() == record["sha256"], "release identity changed")
        if path.suffix == ".py":
            ast.parse(payload, filename=record["path"])
            checks += 1
        text = payload.decode("utf-8")
        private_markers = tuple("/" + folder + "/" for folder in ("Volumes", "Users")) + ("bounty-ops" + "/runtime", "authorization" + "_ref")
        check(not any(marker in text for marker in private_markers), "private operations path in release")
        check(re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)", text) is None, "credential marker in release")

    probe = read("experiments/representation-probe-falsifier-v1/development_results.json")
    ensemble = read("experiments/representation-probe-falsifier-v1/development_ensemble_results.json")
    tuning = read("experiments/representation-probe-falsifier-v1/tuning_results.json")
    check(len(probe["results"]) == 740 and len(ensemble["results"]) == 14, "development search cardinality")
    check(tuning["open_final"] is False and tuning["selected_candidate"] is None and tuning["final"] == "unopened", "final scoring boundary")
    check(len(tuning["results"]) == 5, "frozen candidate count")
    for result in tuning["results"]:
        check(math.isclose(result["accuracy"], 16 / 27) and result["passes_gate"] is False, "frozen tuning result")
        check(result["false_positive"] + result["false_negative"] == 11, "tuning confusion arithmetic")
    for relative, correct in [("experiments/layer-geometry-v1/result.json", 16), ("experiments/intervention-jsd-native-v1/result.json", 19)]:
        result = read(relative)
        check(math.isclose(result["tuning_accuracy"], correct / 27) and result["gate_passed"] is False, "follow-up result")
        check(result["final_predictions_generated"] is False, "follow-up final boundary")

    batch = read("experiments/batch-prevalence-rank-v1/verified-result.json")
    check(batch["gate_passed"] is False and batch["rows"] == 53 and batch["problem_groups"] == 27, "batch gate")
    for metrics in batch["metrics"].values():
        c = metrics["confusion"]
        total = sum(c.values())
        check(total == 53 and math.isclose(metrics["accuracy"], (c["tn"] + c["tp"]) / total), "batch accuracy arithmetic")
        balanced = .5 * (c["tp"] / (c["tp"] + c["fn"]) + c["tn"] / (c["tn"] + c["fp"]))
        check(math.isclose(metrics["balanced_accuracy"], balanced), "batch balanced arithmetic")
    check(math.isclose(batch["metrics"]["candidate"]["accuracy"], 41 / 53), "batch 41/53")
    check(batch["holdout_groups_processed"] == 0 and batch["model_inference_calls"] == 0 and batch["platform_submissions"] == 0, "batch execution boundary")

    pilot = read("experiments/answer-agreement-pilot-v1/verified-result.json")
    samples = read("experiments/answer-agreement-pilot-v1/samples-summary.json")
    check(pilot["samples"] == len(samples) == 6 and pilot["complete_answers"] == 0 and pilot["gate_passed"] is False, "feasibility gate")
    check(sum(s["tokens"] for s in samples) == pilot["generated_tokens"] == 10367, "generation token arithmetic")
    check(sum(s["termination"] == "token_cap" for s in samples) == 5 and sum(s["termination"] == "time_cap" for s in samples) == 1, "sample termination counts")
    check(pilot["robustness_accuracy_measured"] is False and pilot["holdout_cases_processed"] == 0, "feasibility scope")

    for label in ("local", "host"):
        result = read(f"experiments/tokenizer-fidelity-v1/{label}-result.json")
        check(result["loaders"]["auto"]["unchanged_interventions"] == 109, "automatic tokenizer control")
        check(result["loaders"]["generic"]["unchanged_interventions"] == 0 and result["loaders"]["generic"]["fidelity_and_manipulation_passed"] is True, "native-compatible tokenizer")
        check(result["final_cases_processed"] == 0 and result["model_inference"] is False, "tokenizer execution boundary")
    parity = read("experiments/tokenizer-fidelity-v1/parity/result.json")
    check(parity["gate_passed"] is True and all(c["passed"] for c in parity["checks"]), "stored synthetic runtime parity")
    check(parity["real_weights_or_competition_data_read"] is False, "parity synthetic-only boundary")
    public = read("public-inputs.json")
    check(len(public["legacy_exposed"]) == 109 and len(public["current_development_groups"]) == 27 and len(public["pilot_groups"]) == 3, "public preparation cardinality")
    deployment = read("deployment-checkpoint.json")
    check([r["submission_id"] for r in deployment["controls"]] == [959561, 959564], "hosted control identities")
    check(deployment["uncertainty_upload"]["submission_id"] is None and deployment["uncertainty_upload"]["worker_ran"] is False, "HTTP failure is not a score")
    print(json.dumps({"status": "PASS", "checks": checks, "release_files": len(expected),
        "model_weights_loaded": False, "models_fitted": 0, "model_inference_calls": 0,
        "local_holdout_files_opened": 0, "platform_submissions": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
