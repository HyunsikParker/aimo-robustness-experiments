"""Compare actual tokenization with the pinned native tokenizer, without weights."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from tokenizers import Tokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "intervention-jsd-v1"


def main():
    os.umask(0o077)
    label = sys.argv[1]
    assert label in ("local", "host")
    inputs = OLD / "inputs.json"
    config = json.loads((OLD / "run-config.json").read_text())
    assert hashlib.sha256(inputs.read_bytes()).hexdigest() == config["input_sha256"]
    cases = json.loads(inputs.read_text())
    assert len(cases) == 109
    assert {x["problem_hash"] for x in cases} == set(config["allowed_hashes"])
    assert not ({x["problem_hash"] for x in cases} & set(config["forbidden_final_hashes"]))
    native = Tokenizer.from_file(str(ROOT / "checkpoint/tokenizer.json"))
    loaders = {
        "auto": AutoTokenizer.from_pretrained(ROOT / "checkpoint", local_files_only=True,
                                              trust_remote_code=False),
        "generic": PreTrainedTokenizerFast.from_pretrained(ROOT / "checkpoint",
                                                           local_files_only=True),
    }
    summaries = {}
    for name, tokenizer in loaders.items():
        rows = []
        counts = {"native_mismatch_original": 0, "native_mismatch_intervention": 0,
                  "unchanged_interventions": 0, "rendered_unchanged": 0,
                  "over_length": 0, "max_tokens": 0}
        for case in cases:
            problem = case["problem"]
            pair = []
            rendered = []
            for content in (problem, "\n" + problem + "\n"):
                messages = [{"role": "user", "content": content}]
                text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
                ids = tokenizer.apply_chat_template(messages, tokenize=True,
                    return_dict=False, add_generation_prompt=True)
                assert isinstance(ids, list) and all(isinstance(x, int) for x in ids)
                native_ids = native.encode(text, add_special_tokens=False).ids
                pair.append((ids, native_ids))
                rendered.append(text)
            counts["native_mismatch_original"] += pair[0][0] != pair[0][1]
            counts["native_mismatch_intervention"] += pair[1][0] != pair[1][1]
            counts["unchanged_interventions"] += pair[0][0] == pair[1][0]
            counts["rendered_unchanged"] += rendered[0] == rendered[1]
            counts["over_length"] += max(len(pair[0][0]), len(pair[1][0])) > 2048
            counts["max_tokens"] = max(counts["max_tokens"], len(pair[0][0]), len(pair[1][0]))
            rows.append({"problem_hash": case["problem_hash"], "split": case["split"],
                         "original_ids": pair[0][0], "intervention_ids": pair[1][0]})
        passed = not any(counts[k] for k in counts if k != "max_tokens")
        summaries[name] = {"class": type(tokenizer).__name__, "cases": len(cases),
                           **counts, "fidelity_and_manipulation_passed": passed}
        if passed:
            output = ROOT / (label + "-" + name + "-prepared.json")
            output.write_text(json.dumps(rows) + "\n")
            summaries[name]["prepared_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    result = {"environment": label, "packages": {k: importlib.metadata.version(k)
              for k in ("transformers", "tokenizers", "huggingface-hub")},
              "loaders": summaries, "final_cases_processed": 0,
              "weights_downloaded": False, "model_inference": False}
    output = ROOT / (label + "-result.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
