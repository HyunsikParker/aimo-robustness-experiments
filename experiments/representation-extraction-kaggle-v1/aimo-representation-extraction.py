"""Extract pinned Qwen prompt representations for leakage-safe probe research."""

from __future__ import annotations

import gc
import os
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
MODEL_REVISION = "6e8885a6ff5c1dc5201574c8fd700323f23c25fa"
BASELINE_REVISION = "7e8839966750059b6b1d247a12ab56552c79b342"
CSV_SHA256 = "07774d862388ae4fcd40af472528c87b5c0a646b56bc378165372cb51c170679"
PROMPT_FILE_SHA256 = "2455c1a0cb69aef5bd3fc47b7665fdaa1cfcf50b3da6db02d24a2d6e5dc7c234"
PROMPT_TEXT_SHA256 = "72283c58f504d7c3d32d66066fcb06aecc20d14e93136cd6b1589f7f3ce7b002"
EXPECTED_ROWS = 141
EXPECTED_UNIQUE_PROBLEMS = 137
EXPECTED_LAYERS = 37
EXPECTED_HIDDEN_DIMENSION = 4096
MAX_PROMPT_TOKENS = 2048
WORKING = Path(os.environ.get("AIMO_EXTRACTION_OUTPUT", str(Path(__file__).resolve().parents[1] / "representation-extraction-output-v1-20260823")))
TEMP = WORKING / "scratch"


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_download(payload: bytes, expected_sha256: str, limit: int) -> bytes:
    if len(payload) > limit or digest_bytes(payload) != expected_sha256:
        raise RuntimeError("public research input failed size or digest validation")
    return payload


def read_bounded_response(response, limit: int) -> bytes:
    if response.status_code != 200:
        raise RuntimeError("public input download failed")
    length = response.headers.get("Content-Length")
    if length is not None and (not length.isdigit() or int(length) > limit):
        raise RuntimeError("public input declared an invalid size")
    chunks: list[bytes] = []
    total = 0
    for block in response.iter_content(chunk_size=64 * 1024):
        if not block:
            continue
        total += len(block)
        if total > limit:
            raise RuntimeError("public input exceeded the download bound")
        chunks.append(block)
    return b"".join(chunks)


def download_csv() -> bytes:
    response = requests.get(
        "https://raw.githubusercontent.com/aimo-interp/baselines/7e8839966750059b6b1d247a12ab56552c79b342/data/math-robust-agg.csv",
        timeout=60,
        allow_redirects=False,
        stream=True,
    )
    try:
        payload = read_bounded_response(response, 2 * 1024 * 1024)
    finally:
        response.close()
    return validate_download(payload, CSV_SHA256, 2 * 1024 * 1024)


def download_prompt() -> bytes:
    response = requests.get(
        "https://raw.githubusercontent.com/aimo-interp/baselines/7e8839966750059b6b1d247a12ab56552c79b342/prompts/solve.txt",
        timeout=60,
        allow_redirects=False,
        stream=True,
    )
    try:
        payload = read_bounded_response(response, 64 * 1024)
    finally:
        response.close()
    return validate_download(payload, PROMPT_FILE_SHA256, 64 * 1024)


def load_public_rows(csv_payload: bytes) -> tuple[pd.DataFrame, list[str]]:
    csv_path = TEMP / "math-robust-agg.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(csv_payload)
    frame = pd.read_csv(csv_path)
    required = {
        "model_id",
        "dataset_id",
        "problem_id",
        "original_problem",
        "model_is_robust",
    }
    if len(frame) != EXPECTED_ROWS or not required.issubset(frame.columns):
        raise RuntimeError("public training table shape is invalid")
    if frame[list(required)].isna().any().any() or set(frame["model_id"]) != {
        "qwen3-8b:low"
    }:
        raise RuntimeError("public training table identity is invalid")
    grouped = frame.groupby("original_problem", sort=False, dropna=False)
    if grouped["model_is_robust"].nunique().max() != 1:
        raise RuntimeError("one public problem has conflicting labels")
    unique = grouped.agg(
        problem_id=("problem_id", "first"),
        model_is_robust=("model_is_robust", "first"),
    ).reset_index()
    if len(unique) != EXPECTED_UNIQUE_PROBLEMS:
        raise RuntimeError("public problem grain is invalid")
    unique["problem_hash"] = unique["original_problem"].map(
        lambda value: digest_bytes(value.encode("utf-8"))
    )
    if unique["problem_hash"].duplicated().any():
        raise RuntimeError("public problem hashes are not unique")
    unique = unique.sort_values("problem_hash").reset_index(drop=True)
    return unique, unique["original_problem"].tolist()


def build_prompt(tokenizer, system_prompt: str, problem: str) -> str:
    try:
        formatted = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except (AttributeError, TypeError, ValueError):
        return f"{system_prompt}\n\n{problem}"
    return formatted if isinstance(formatted, str) else f"{system_prompt}\n\n{problem}"


def input_device(model) -> torch.device:
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("a free Kaggle GPU is required")
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("no CUDA device is visible")
    max_memory = {index: "13GiB" for index in range(gpu_count)}
    max_memory["cpu"] = "24GiB"
    cache = TEMP / "huggingface"
    offload = TEMP / "offload"
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache,
        local_files_only=False,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache,
        dtype=torch.float16,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=offload,
        low_cpu_mem_usage=True,
        local_files_only=False,
        trust_remote_code=False,
        use_safetensors=True,
    )
    model.eval()
    return model, tokenizer, gpu_count


def extract(model, tokenizer, system_prompt: str, problems: list[str]) -> np.ndarray:
    representations = np.empty(
        (len(problems), EXPECTED_LAYERS, EXPECTED_HIDDEN_DIMENSION), dtype=np.float16
    )
    device = input_device(model)
    for index, problem in enumerate(problems):
        prompt = build_prompt(tokenizer, system_prompt, problem)
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_TOKENS,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            output = model(
                **encoded,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != EXPECTED_LAYERS:
            raise RuntimeError("model hidden-state stack has an unexpected shape")
        row = np.stack(
            [state[0, -1, :].detach().float().cpu().numpy() for state in hidden_states]
        )
        if row.shape != (EXPECTED_LAYERS, EXPECTED_HIDDEN_DIMENSION) or not np.isfinite(
            row
        ).all():
            raise RuntimeError("model produced malformed representations")
        representations[index] = row.astype(np.float16)
        del output, hidden_states, row, encoded
        if (index + 1) % 10 == 0 or index + 1 == len(problems):
            print(json.dumps({"encoded": index + 1, "total": len(problems)}))
    return representations


def main() -> None:
    if TEMP.exists():
        shutil.rmtree(TEMP)
    TEMP.mkdir(parents=True)
    csv_payload = download_csv()
    prompt_payload = download_prompt()
    system_prompt = prompt_payload.decode("utf-8").strip()
    if digest_bytes(system_prompt.encode("utf-8")) != PROMPT_TEXT_SHA256:
        raise RuntimeError("public prompt normalization changed")
    unique, problems = load_public_rows(csv_payload)
    model, tokenizer, gpu_count = load_model()
    representations = extract(model, tokenizer, system_prompt, problems)

    array_path = WORKING / "qwen_prompt_representations_fp16.npy"
    np.save(array_path, representations, allow_pickle=False)
    rows = [
        {
            "problem_hash": row.problem_hash,
            "label": bool(row.model_is_robust),
        }
        for row in unique.itertuples(index=False)
    ]
    rows_path = WORKING / "representation_rows.json"
    rows_path.write_text(
        json.dumps(rows, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "array_bytes": array_path.stat().st_size,
        "array_dtype": "float16",
        "array_sha256": digest_file(array_path),
        "array_shape": list(representations.shape),
        "baseline_revision": BASELINE_REVISION,
        "csv_sha256": CSV_SHA256,
        "gpu_count": gpu_count,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "prompt_file_sha256": PROMPT_FILE_SHA256,
        "prompt_text_sha256": PROMPT_TEXT_SHA256,
        "rows_sha256": digest_file(rows_path),
        "schema_version": 1,
        "torch_version": torch.__version__,
    }
    (WORKING / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"complete": True, "rows": len(rows), "shape": manifest["array_shape"]}))

    del model, tokenizer, representations
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(TEMP)


if __name__ == "__main__":
    main()
