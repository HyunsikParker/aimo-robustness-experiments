#!/usr/bin/env python3
"""One-time restricted conversion of the reviewed official forest artifact."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import joblib.numpy_pickle as numpy_pickle
import numpy as np
from safetensors.numpy import save_file
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "getting-started"
    / "solutions"
    / "uncertainty-profiling"
    / "uncertainty_artifacts"
    / "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B.joblib"
)
DESTINATION = (
    ROOT
    / "candidates"
    / "uncertainty-safe-forest-v1"
    / "uncertainty_forest.safetensors"
)
EXPECTED_SOURCE_SHA256 = (
    "bc90bd8eed4d9f6383cbb9ae1626896afea90b0e71c46d9a56bc87d376f85592"
)
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
FEATURE_NAMES = (
    "generation_mean_logprob",
    "generation_mean_prob",
    "generation_mean_nll",
    "generation_ppl",
    "generation_min_logprob",
    "generation_max_logprob",
    "generation_std_logprob",
    "generation_min_k_logprob",
    "generation_mean_entropy",
    "generation_mean_top1_prob",
    "generation_mean_top2_margin",
    "generation_frac_selected_is_top1",
    "generation_frac_high_conf_tokens",
    "generation_frac_low_conf_tokens",
)

ALLOWED_GLOBALS = frozenset(
    {
        ("builtins", "slice"),
        ("joblib.numpy_pickle", "NumpyArrayWrapper"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("sklearn.ensemble._forest", "RandomForestRegressor"),
        ("sklearn.impute._base", "SimpleImputer"),
        ("sklearn.pipeline", "Pipeline"),
        ("sklearn.tree._classes", "DecisionTreeRegressor"),
        ("sklearn.tree._tree", "Tree"),
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RestrictedNumpyUnpickler(numpy_pickle.NumpyUnpickler):
    """Allow only the exact numeric and scikit-learn classes in the reviewed file."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) == ("torch.torch_version", "TorchVersion"):
            return str
        if (module, name) not in ALLOWED_GLOBALS:
            raise RuntimeError(f"forbidden serialized global: {module}.{name}")
        return super().find_class(module, name)


def restricted_load(path: Path) -> object:
    """Load through joblib's array reader with a strict class allowlist."""

    with path.open("rb") as stream:
        with numpy_pickle._validate_fileobject_and_memmap(stream, str(path), None) as (
            file_object,
            mmap_mode,
        ):
            if isinstance(file_object, str):
                raise RuntimeError("legacy joblib compatibility format is forbidden")
            return RestrictedNumpyUnpickler(
                str(path),
                file_object,
                ensure_native_byte_order=True,
                mmap_mode=mmap_mode,
            ).load()


def validated_forest(
    payload: object,
) -> tuple[SimpleImputer, RandomForestRegressor, float]:
    """Validate the complete executable object graph before exporting arrays."""

    if type(payload) is not dict:
        raise RuntimeError("artifact root must be a dictionary")
    expected_keys = {
        "schema_version",
        "artifact_type",
        "model_id",
        "feature_names",
        "generation_config",
        "estimator",
        "decision_threshold",
        "selected_params",
        "cv_results",
        "training_provenance",
        "library_versions",
    }
    if set(payload) != expected_keys:
        raise RuntimeError("artifact root schema is not exact")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "uncertainty_regressor"
    ):
        raise RuntimeError("artifact identity is invalid")
    if (
        payload["model_id"] != EXPECTED_MODEL_ID
        or tuple(payload["feature_names"]) != FEATURE_NAMES
    ):
        raise RuntimeError("artifact model or feature order is invalid")
    expected_config = {
        "max_prompt_length": 2048,
        "max_new_tokens": 4096,
        "batch_size": 8,
        "cache_implementation": "static",
        "chat_template_role": "user",
        "do_sample": False,
        "min_k_fraction": 0.2,
        "high_conf_threshold": 0.9,
        "low_conf_threshold": 0.1,
    }
    if payload["generation_config"] != expected_config:
        raise RuntimeError("artifact generation configuration is invalid")

    threshold = payload["decision_threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(float(threshold)):
        raise RuntimeError("artifact decision threshold is invalid")
    pipeline = payload["estimator"]
    if type(pipeline) is not Pipeline or [name for name, _ in pipeline.steps] != [
        "imputer",
        "regressor",
    ]:
        raise RuntimeError("artifact estimator pipeline is invalid")
    imputer = pipeline.named_steps["imputer"]
    forest = pipeline.named_steps["regressor"]
    if type(imputer) is not SimpleImputer or imputer.strategy != "median":
        raise RuntimeError("artifact imputer is invalid")
    if type(forest) is not RandomForestRegressor:
        raise RuntimeError("artifact regressor is invalid")
    expected_params = {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": 1.0,
    }
    actual_params = forest.get_params(deep=False)
    if any(actual_params[key] != value for key, value in expected_params.items()):
        raise RuntimeError("artifact forest parameters are invalid")
    if len(forest.estimators_) != 500 or any(
        type(tree) is not DecisionTreeRegressor for tree in forest.estimators_
    ):
        raise RuntimeError("artifact forest trees are invalid")
    return imputer, forest, float(threshold)


def main() -> None:
    if (
        SOURCE.is_symlink()
        or not SOURCE.is_file()
        or SOURCE.stat().st_size != 1_007_536
    ):
        raise RuntimeError("reviewed source artifact is unsafe or missing")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("reviewed source artifact hash mismatch")
    imputer, forest, threshold = validated_forest(restricted_load(SOURCE))

    offsets = [0]
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    feature: list[np.ndarray] = []
    thresholds: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for estimator in forest.estimators_:
        tree = estimator.tree_
        nodes = int(tree.node_count)
        offsets.append(offsets[-1] + nodes)
        left.append(np.asarray(tree.children_left, dtype=np.int64))
        right.append(np.asarray(tree.children_right, dtype=np.int64))
        feature.append(np.asarray(tree.feature, dtype=np.int64))
        thresholds.append(np.asarray(tree.threshold, dtype=np.float64))
        values.append(np.asarray(tree.value[:, 0, 0], dtype=np.float64))

    tensors = {
        "imputer_statistics": np.asarray(imputer.statistics_, dtype=np.float64),
        "tree_offsets": np.asarray(offsets, dtype=np.int64),
        "children_left": np.concatenate(left),
        "children_right": np.concatenate(right),
        "feature": np.concatenate(feature),
        "threshold": np.concatenate(thresholds),
        "value": np.concatenate(values),
    }
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(DESTINATION))
    print(
        json.dumps(
            {
                "destination_sha256": sha256(DESTINATION),
                "destination_bytes": DESTINATION.stat().st_size,
                "decision_threshold": threshold,
                "tree_count": len(forest.estimators_),
                "total_nodes": offsets[-1],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
