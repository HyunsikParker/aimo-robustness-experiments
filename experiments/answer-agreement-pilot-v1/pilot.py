"""Frozen, label-free feasibility screen for sampled final-answer agreement."""
import hashlib
import json
import os
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).parent
os.umask(0o077)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(name, value):
    temp = ROOT / (name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(ROOT / name)


def answer(text, terminal=False):
    # Ignore boxes in private reasoning. Require a completed final-answer box.
    if "</think>" in text:
        final = text.rsplit("</think>", 1)[1]
    elif "<think>" in text or not terminal:
        return None
    else:
        final = text
    positions = list(re.finditer(r"\\boxed\s*\{", final))
    if not positions:
        return None
    start = positions[-1].end()
    depth = 1
    for end in range(start, len(final)):
        depth += (final[end] == "{") - (final[end] == "}")
        if depth == 0:
            value = re.sub(r"\s+", "", final[start:end]).replace(r"\,", "")
            if not value:
                return None
            fraction = re.fullmatch(r"\\(?:d?frac)\{(-?\d+)\}\{(-?\d+)\}", value)
            try:
                if fraction:
                    return str(Fraction(int(fraction[1]), int(fraction[2])))
                if re.fullmatch(r"-?\d+(?:\.\d+)?", value):
                    return str(Fraction(value))
            except ZeroDivisionError:
                return None
            return value  # Exact LaTeX only; no claim of symbolic equivalence.
    return None


def check():
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    cfg = json.loads((ROOT / "config.json").read_text())
    cases = json.loads((ROOT / "inputs.json").read_text())
    assert len(cases) == 3 and digest(ROOT / "inputs.json") == cfg["inputs_sha256"]
    assert not {c["problem_group"] for c in cases} & set(cfg["holdout_groups"])
    checkpoint = ROOT.parent / "tokenizer-fidelity-v1" / "checkpoint"
    for name, sha in cfg["tokenizer_hashes"].items():
        assert digest(checkpoint / name) == sha
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint, local_files_only=True)
    native = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    prepared = []
    for case in cases:
        content = case["problem"] + cfg["prompt_suffix"]
        messages = [{"role": "user", "content": content}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        assert ids == native.encode(rendered, add_special_tokens=False).ids
        assert 0 < len(ids) <= 2048 and rendered.endswith("<｜Assistant｜>")
        prepared.append((case["problem_group"], ids))
    assert answer("<think>\\boxed{2}</think>\\boxed{\\frac{2}{4}}") == "1/2"
    assert answer("<think>\\boxed{2}") is None
    assert answer("</think>\\boxed{\\frac{2}{") is None
    assert answer("</think>\\boxed{0.5}") == "1/2"
    assert answer("</think>\\boxed{\\frac{1}{0}}") is None
    assert answer("\\boxed{3}", terminal=True) == "3"
    assert answer("<think>\\boxed{3}", terminal=True) is None
    return cfg, native, prepared


def main():
    cfg, tokenizer, prepared = check()
    if "--check" in sys.argv:
        print("PASS: pinned native tokenizer parity; three development inputs; holdout exclusion; boxed-answer controls")
        return
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    started = time.monotonic()
    scratch = ROOT / "scratch"
    scratch.mkdir(exist_ok=False, mode=0o700)
    os.environ["HF_HOME"] = str(scratch / "hf")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    disable_progress_bars()
    atomic("progress.json", {"stage": "download", "pid": os.getpid(), "completed_samples": 0})
    model_path = Path(snapshot_download(cfg["model_id"], revision=cfg["model_revision"], token=False,
        local_dir=str(scratch / "model"), max_workers=2,
        allow_patterns=["config.json", "model.safetensors.index.json", "*.safetensors", "LICENSE"]))
    for item in cfg["weights"]:
        path = model_path / item["path"]
        assert path.stat().st_size == item["size"] and digest(path) == item["sha256"]
    atomic("progress.json", {"stage": "load", "pid": os.getpid(), "completed_samples": 0})
    mx.set_memory_limit(24 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    model, _ = load_model(model_path, lazy=True)
    model.eval()
    mx.eval(model.parameters())
    inference_started = time.monotonic()
    samples = []
    for case_index, (group, prompt) in enumerate(prepared):
        for sample_index in range(2):
            seed = 20260920 + case_index * 2 + sample_index
            mx.random.seed(seed)
            tokens = []
            sample_start = time.monotonic()
            termination = "token_cap"
            value = None
            generator = generate_step(mx.array(prompt), model, max_tokens=2048,
                sampler=make_sampler(temp=0.6, top_p=0.95), prefill_step_size=512)
            for token, _ in generator:
                tokens.append(int(token))
                elapsed = time.monotonic() - inference_started
                if len(tokens) % 32 == 0 or int(token) in cfg["eos_token_ids"]:
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    value = answer(text, terminal=int(token) in cfg["eos_token_ids"])
                    atomic("progress.json", {"stage": "inference", "pid": os.getpid(),
                        "case_index": case_index, "sample_index": sample_index,
                        "sample_tokens": len(tokens), "completed_samples": len(samples),
                        "inference_seconds": elapsed})
                    if value is not None:
                        termination = "final_box"
                        break
                if int(token) in cfg["eos_token_ids"]:
                    termination = "eos"
                    break
                if elapsed > 1800:
                    termination = "time_cap"
                    break
            generator.close()
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            value = answer(text, terminal=termination == "eos")
            sample = {"problem_group": group, "sample": sample_index, "seed": seed,
                "tokens": len(tokens), "elapsed_seconds": time.monotonic() - sample_start,
                "termination": termination, "answer": value}
            atomic(f"sample-{case_index}-{sample_index}.json", {**sample, "text": text, "token_ids": tokens})
            samples.append(sample)
            atomic("samples-summary.json", samples)
            mx.clear_cache()
            if termination == "time_cap":
                break
        if samples[-1]["termination"] == "time_cap":
            break
    complete = sum(s["answer"] is not None for s in samples)
    inference_seconds = time.monotonic() - inference_started
    passed = len(samples) == 6 and complete >= 5 and inference_seconds <= 1800
    result = {"stage": "completed", "screen": "feasibility_only", "gate_passed": passed,
        "samples": len(samples), "complete_answers": complete,
        "generated_tokens": sum(s["tokens"] for s in samples),
        "inference_seconds": inference_seconds, "total_seconds": time.monotonic() - started,
        "peak_mlx_bytes": mx.get_peak_memory(), "cost_usd": 0,
        "holdout_cases_processed": 0, "robustness_labels_loaded": False,
        "robustness_accuracy_measured": False, "platform_submissions": 0,
        "failure_scope": None if passed else "Fixed 2048-token/two-sample local feasibility gate failed; does not establish robustness prediction accuracy.",
        "target_gpu_runtime_verified": False}
    atomic("result.json", result)
    atomic("progress.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
