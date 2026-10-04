"""Rebuild exposed public inputs and native token IDs; never fit or load weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "public-inputs.json").read_text())


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def checked(payload: bytes, expected: str, limit: int) -> bytes:
    if len(payload) > limit or digest(payload) != expected:
        raise RuntimeError("public input size or identity mismatch")
    return payload


def download(url: str, expected: str, limit: int) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = response.read(limit + 1)
    return checked(payload, expected, limit)


def place(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise RuntimeError("refusing a symlink output")
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing changed existing input: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def json_bytes(value, *, compact=False, newline=True) -> bytes:
    return (json.dumps(value, indent=None if compact else 2) + ("\n" if newline else "")).encode()


def normalized_hash(problem: str) -> str:
    normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", problem)).strip()
    return digest(normalized.encode())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="download pinned public data and tokenizer files, never weights")
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument("--current-parquet", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    args = parser.parse_args()
    if not args.fetch and not all((args.baseline_csv, args.current_parquet, args.tokenizer_dir)):
        parser.error("use --fetch, or provide all three offline sources")

    baseline_rev = SPEC["baseline_revision"]
    csv_url = f"https://raw.githubusercontent.com/aimo-interp/baselines/{baseline_rev}/data/math-robust-agg.csv"
    baseline = checked(args.baseline_csv.read_bytes(), SPEC["baseline_csv_sha256"], 2 * 1024**2) if args.baseline_csv else download(csv_url, SPEC["baseline_csv_sha256"], 2 * 1024**2)
    rows = {}
    allowed = {r["problem_hash"] for r in SPEC["legacy_exposed"]}
    for row in csv.DictReader(io.StringIO(baseline.decode())):
        problem = row["original_problem"]
        key = digest(problem.encode())
        if key not in allowed:
            continue  # Do not interpret or export final-row labels.
        label = row["model_is_robust"].lower()
        if label not in {"true", "false"}:
            raise RuntimeError("unexpected public label encoding")
        value = (problem, int(label == "true"))
        if key in rows and rows[key] != value:
            raise RuntimeError("conflicting duplicate public input")
        rows[key] = value
    if set(rows) != allowed or len(allowed) != 109:
        raise RuntimeError("exposed input identity mismatch")
    inputs, labels = [], []
    for item in SPEC["legacy_exposed"]:
        key, split = item["problem_hash"], item["split"]
        if split not in {"train", "tuning"}:
            raise RuntimeError("final split is not permitted")
        problem, label = rows[key]
        inputs.append({"problem_hash": key, "split": split, "problem": problem})
        labels.append({"problem_hash": key, "split": split, "label": label})
    input_bytes, label_bytes = json_bytes(inputs), json_bytes(labels)
    checked(input_bytes, SPEC["expected_outputs"]["intervention_inputs"], 1024**2)
    checked(label_bytes, SPEC["expected_outputs"]["intervention_labels"], 1024**2)
    for experiment in ("intervention-jsd-v1", "intervention-jsd-native-v1"):
        place(ROOT / "experiments" / experiment / "inputs.json", input_bytes)
        place(ROOT / "experiments" / experiment / "labels.json", label_bytes)

    import pandas as pd
    revision = SPEC["current_dataset_revision"]
    current_url = f"https://huggingface.co/datasets/aimo-interp/train-main-v2/resolve/{revision}/data/train-00000-of-00001.parquet"
    current = checked(args.current_parquet.read_bytes(), SPEC["current_parquet_sha256"], 10 * 1024**2) if args.current_parquet else download(current_url, SPEC["current_parquet_sha256"], 10 * 1024**2)
    # Select exposed groups from public problem text before requesting label columns.
    text = pd.read_parquet(io.BytesIO(current), columns=["problem"])
    groups = text.problem.map(normalized_hash)
    selected = groups.isin(SPEC["current_development_groups"])
    frame = pd.read_parquet(io.BytesIO(current)).loc[selected].copy()
    frame["problem_group"] = groups[selected]
    labelled = frame.loc[frame.is_robust.notna()]
    if len(labelled) != 53 or labelled.problem_group.nunique() != 27:
        raise RuntimeError("current development cardinality mismatch")
    development_bytes = json_bytes(labelled.to_dict(orient="records"))
    checked(development_bytes, SPEC["expected_outputs"]["current_development"], 1024**2)
    place(ROOT / "experiments/main-v2-intake-20260920/development-labelled.json", development_bytes)
    lookup = dict(zip(frame.problem_group, frame.problem))
    pilot = [{"problem_group": key, "problem": lookup[key]} for key in SPEC["pilot_groups"]]
    pilot_bytes = json_bytes(pilot, newline=False)
    checked(pilot_bytes, SPEC["expected_outputs"]["pilot_inputs"], 1024**2)
    place(ROOT / "experiments/answer-agreement-pilot-v1/inputs.json", pilot_bytes)

    checkpoint = ROOT / "experiments/tokenizer-fidelity-v1/checkpoint"
    for name, expected in SPEC["tokenizer_hashes"].items():
        url = f"https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B/resolve/{SPEC['model_revision']}/{name}"
        payload = checked((args.tokenizer_dir / name).read_bytes(), expected, 32 * 1024**2) if args.tokenizer_dir else download(url, expected, 32 * 1024**2)
        place(checkpoint / name, payload)
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    native = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    prepared = []
    for case in inputs:
        pair = []
        for content in (case["problem"], "\n" + case["problem"] + "\n"):
            messages = [{"role": "user", "content": content}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
            if ids != native.encode(rendered, add_special_tokens=False).ids or not 0 < len(ids) <= 2048:
                raise RuntimeError("native tokenization mismatch")
            pair.append(ids)
        if pair[0] == pair[1]:
            raise RuntimeError("newline manipulation was erased")
        prepared.append({"problem_hash": case["problem_hash"], "split": case["split"], "original_ids": pair[0], "intervention_ids": pair[1]})
    prepared_bytes = json_bytes(prepared, compact=True)
    config = json.loads((ROOT / "experiments/intervention-jsd-native-v1/run-config.json").read_text())
    checked(prepared_bytes, config["prepared_sha256"], 1024**2)
    place(ROOT / "experiments/intervention-jsd-native-v1/prepared-inputs.json", prepared_bytes)
    print(json.dumps({"status": "PASS", "legacy_cases": 109, "current_development_rows": 53,
        "pilot_cases": 3, "tokenizer_cases": len(prepared), "all_output_hashes_match": True,
        "local_holdout_files_opened": 0, "weights_loaded": False, "models_fitted": 0,
        "inference_calls": 0, "platform_submissions": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
