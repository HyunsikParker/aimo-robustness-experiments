"""One frozen, matched-prefix intervention; no labels or final cases are read."""

import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm.utils import load_model
from mlx_lm.models.cache import make_prompt_cache

from metrics import normalized_mean_jsd

ROOT = Path(__file__).resolve().parent
K = 8
MAX_PROMPT = 2048
MEMORY_CAP = 24 * 1024**3
SHAM_TOLERANCE = 1e-8


def atomic(name, value):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def continuation(model, prompt, eos_ids):
    cache = make_prompt_cache(model)
    step = mx.array([prompt])
    generated = []
    for _ in range(K):
        scores = model(step, cache=cache)[:, -1, :]
        next_token = mx.argmax(scores, axis=-1)
        mx.eval(next_token)
        token = int(next_token.item())
        generated.append(token)
        if token in eos_ids:
            break
        step = mx.array([[token]])
    del cache, scores, next_token, step
    return generated


def teacher_logits(model, prompt, generated):
    # Position len(prompt)-1 predicts generated[0], with the same prefix in
    # every arm. We do not compare independently generated continuations.
    inputs = mx.array([prompt + generated[:-1]])
    logits = model(inputs)[:, len(prompt) - 1 : len(prompt) - 1 + len(generated), :]
    logits = logits.astype(mx.float32)
    mx.eval(logits)
    result = np.array(logits[0])
    del inputs, logits
    return result


def measure(model, original, changed, eos_ids):
    generated = continuation(model, original, eos_ids)
    baseline = teacher_logits(model, original, generated)
    sham = teacher_logits(model, original, generated)
    intervention = teacher_logits(model, changed, generated)
    sham_score = normalized_mean_jsd(baseline, sham)
    if sham_score > SHAM_TOLERANCE:
        raise RuntimeError("identity-sham numerical control failed")
    score = normalized_mean_jsd(baseline, intervention)
    return score, sham_score, generated


def check():
    from mlx_lm.models.qwen3 import Model, ModelArgs

    mx.random.seed(20260906)
    model = Model(ModelArgs(model_type="qwen3", hidden_size=32,
        num_hidden_layers=2, intermediate_size=64, num_attention_heads=2,
        rms_norm_eps=1e-6, vocab_size=32, num_key_value_heads=2,
        max_position_embeddings=128, rope_theta=10000, head_dim=16,
        tie_word_embeddings=False))
    model.eval()
    value, sham, tokens = measure(model, [1, 2, 3], [1, 4, 2, 3], set())
    assert len(tokens) == K and 0 <= value <= 1 and sham <= SHAM_TOLERANCE
    assert np.array_equal(teacher_logits(model, [1, 2, 3], tokens),
                          teacher_logits(model, [1, 2, 3], tokens))
    assert mx.default_device() == mx.gpu
    print("PASS: actual Qwen3 class, matched-prefix indexing, identity sham, GPU")


def main():
    os.umask(0o077)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("run cap")))
    signal.alarm(1800)
    config = json.loads((ROOT / "run-config.json").read_text())
    inputs_path = ROOT / "inputs.json"
    assert digest(inputs_path) == config["input_sha256"]
    cases = json.loads(inputs_path.read_text())
    assert len(cases) == 109
    assert not ({c["problem_hash"] for c in cases} & set(config["forbidden_final_hashes"]))
    assert {c["problem_hash"] for c in cases} == set(config["allowed_hashes"])
    prepared_path = ROOT / "prepared-inputs.json"
    assert digest(prepared_path) == config["prepared_sha256"]
    prepared_rows = json.loads(prepared_path.read_text())
    assert [c["problem_hash"] for c in prepared_rows] == [c["problem_hash"] for c in cases]
    prepared = []
    for case, row in zip(cases, prepared_rows):
        assert case["split"] == row["split"]
        original, changed = row["original_ids"], row["intervention_ids"]
        if original == changed or max(len(original), len(changed)) > MAX_PROMPT:
            raise RuntimeError("frozen native tokenization or prompt bound failed")
        assert all(isinstance(x, int) and 0 <= x < 151936 for x in original + changed)
        prepared.append((case, original, changed))
    scratch = ROOT / "scratch"
    scratch.mkdir(mode=0o700, exist_ok=False)
    os.environ["HF_HOME"] = str(scratch / "hf")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    atomic("progress.json", {"stage": "download", "completed": 0, "total": 109})
    model_path = snapshot_download(config["model_id"], revision=config["model_revision"],
        token=False, local_dir=str(scratch / "model"), max_workers=2,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "model.safetensors.index.json", "*.safetensors", "LICENSE"])
    for item in config["weights"]:
        path = Path(model_path) / item["path"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise RuntimeError("pinned checkpoint digest mismatch")
    atomic("progress.json", {"stage": "load", "completed": 0, "total": 109})
    mx.set_memory_limit(MEMORY_CAP)
    mx.set_cache_limit(1024**3)
    # Token IDs were verified against the pinned native tokenizer and the
    # official generic-loader path in both local and host tokenizer versions.
    # Do not invoke the incompatible automatic legacy Llama converter here.
    model, _ = load_model(Path(model_path), lazy=True)
    model.eval()
    # Preserve checkpoint precision; no quantization or model substitution.
    from mlx.utils import tree_flatten
    dtypes = sorted({str(x.dtype) for _, x in tree_flatten(model.parameters())})
    eos = set(config["eos_token_ids"])
    outputs = []
    extraction_start = time.monotonic()
    for index, (case, original, changed) in enumerate(prepared):
        case_start = time.monotonic()
        value, sham, tokens = measure(model, original, changed, eos)
        outputs.append({"problem_hash": case["problem_hash"], "split": case["split"],
                        "jsd": value, "identity_sham_jsd": sham,
                        "continuation_ids": tokens, "prompt_tokens": len(original),
                        "changed_prompt_tokens": len(changed),
                        "seconds": time.monotonic() - case_start})
        atomic("features.partial.json", outputs)
        elapsed = time.monotonic() - started
        peak = mx.get_peak_memory()
        atomic("progress.json", {"stage": "extract", "completed": index + 1,
              "total": 109, "elapsed_seconds": elapsed, "peak_mlx_bytes": peak})
        if peak > MEMORY_CAP:
            raise RuntimeError("memory cap exceeded")
        if index == 2:
            projected = (time.monotonic() - extraction_start) * 109 / 3
            if projected + (extraction_start - started) > 1680:
                raise TimeoutError("three-case timing preflight exceeds run cap")
        gc.collect()
        mx.clear_cache()
    atomic("features.json", {"rows": outputs, "model_id": config["model_id"],
          "revision": config["model_revision"], "parameter_dtypes": dtypes,
          "elapsed_seconds": time.monotonic() - started,
          "peak_mlx_bytes": mx.get_peak_memory(), "final_cases_processed": 0,
          "normalization": "mean full-vocabulary JSD divided by ln(2)", "K": K})
    print("PASS: 109 development/tuning features; final cases excluded", flush=True)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        try:
            main()
        except BaseException as error:
            # Provider exceptions may carry signed download URLs. Never log them.
            import traceback
            frames = traceback.extract_tb(error.__traceback__)
            atomic("extraction-error.json", {"failure_class": "operational",
                  "error_class": type(error).__name__, "details_suppressed": True,
                  "frames": [{"file": Path(f.filename).name, "line": f.lineno,
                              "function": f.name} for f in frames[-5:]]})
            print("FAIL: extraction operational error; see private error class", flush=True)
            raise SystemExit(2)
