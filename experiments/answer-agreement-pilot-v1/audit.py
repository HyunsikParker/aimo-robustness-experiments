"""Verify completed generation from saved token IDs; never rerun inference."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser():
    spec = importlib.util.spec_from_file_location("frozen_pilot", ROOT / "pilot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.answer


def self_test():
    extract = parser()
    # A direct answer ending at EOS can be labelled final_box by the generator.
    # Terminal state must come from actual saved token IDs, not that label.
    direct = r"The final result is \boxed{0.5}."
    assert extract(direct, terminal=True) == "1/2"
    assert extract(direct, terminal=False) is None
    assert extract(r"<think>\boxed{7}", terminal=True) is None
    assert extract(r"<think>\boxed{7}</think>\boxed{\frac{2}{4}}") == "1/2"
    assert extract(r"</think>\boxed{\frac{2}{4}") is None
    assert extract(r"</think>\boxed{}") is None
    print("PASS: direct-EOS answer, reasoning-box exclusion, rational normalization, truncated box")


def main():
    self_test()
    if "--self-test" in sys.argv:
        return
    from tokenizers import Tokenizer
    terminal = json.loads((ROOT / "supervisor-complete.json").read_text())
    assert terminal["exit_code"] == 0, "Worker did not finish normally"
    original = json.loads((ROOT / "result.json").read_text())
    config = json.loads((ROOT / "config.json").read_text())
    # Private preflight receipts are excluded from this source release.
    # Check release identity with tools/verify_release.py before reproducing.
    checkpoint = ROOT.parent / "tokenizer-fidelity-v1" / "checkpoint"
    for name, expected in config["tokenizer_hashes"].items():
        assert sha(checkpoint / name) == expected
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    extract = parser()
    cases = json.loads((ROOT / "inputs.json").read_text())
    samples, corrections, sample_hashes = [], [], {}
    for case_index, case in enumerate(cases):
        assert case["problem_group"] not in config["holdout_groups"]
        for sample_index in range(2):
            path = ROOT / f"sample-{case_index}-{sample_index}.json"
            if not path.exists():
                continue
            sample_hashes[path.name] = sha(path)
            data = json.loads(path.read_text())
            assert data["problem_group"] == case["problem_group"]
            assert data["seed"] == 20260920 + case_index * 2 + sample_index
            tokens = data["token_ids"]
            assert 0 < len(tokens) <= 2048 and len(tokens) == data["tokens"]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            assert text == data["text"]
            is_eos = tokens[-1] in config["eos_token_ids"]
            verified_answer = extract(text, terminal=is_eos)
            if verified_answer != data["answer"]:
                corrections.append({"file": path.name, "reason": "Recompute final-answer availability from actual EOS token rather than termination label",
                    "original_complete": data["answer"] is not None, "verified_complete": verified_answer is not None})
            samples.append({"problem_group": case["problem_group"], "sample": sample_index,
                            "tokens": len(tokens), "answer": verified_answer})
    assert len(samples) == original["samples"]
    assert sum(s["tokens"] for s in samples) == original["generated_tokens"]
    complete = sum(s["answer"] is not None for s in samples)
    passed = len(samples) == 6 and complete >= 5 and original["inference_seconds"] <= 1800
    assert original["peak_mlx_bytes"] <= 24 * 1024**3
    verified = {**original, "complete_answers": complete, "gate_passed": passed,
        "original_result_sha256": sha(ROOT / "result.json"), "sample_hashes": sample_hashes,
        "audit_corrections": corrections, "source_and_token_checks_passed": True,
        "accuracy_claim": "None; generation feasibility only"}
    if passed:
        verified["failure_scope"] = None
    target = ROOT / "verified-result.json"
    with target.open("x") as handle:
        json.dump(verified, handle, indent=2)
        handle.write("\n")
    print(json.dumps({key: verified[key] for key in ["gate_passed", "samples", "complete_answers", "generated_tokens", "inference_seconds", "audit_corrections"]}))


if __name__ == "__main__":
    main()
