"""Data-only random-forest loading and deterministic prediction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from uncertainty_profile.config import FEATURE_NAMES


ARTIFACT_NAME = "uncertainty_forest.safetensors"
ARTIFACT_SHA256 = "c5deba2bffbcac89e909c1b77cb6ae8bc26bc891ca5f55fb12f971b6987c4164"
DECISION_THRESHOLD = 0.5759658815999995
EXPECTED_TREE_COUNT = 500
MAX_TOTAL_NODES = 2_000_000
REQUIRED_TENSORS = frozenset(
    {
        "imputer_statistics",
        "tree_offsets",
        "children_left",
        "children_right",
        "feature",
        "threshold",
        "value",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SafeForest:
    """Strict numeric representation of the reviewed scikit-learn forest."""

    imputer_statistics: np.ndarray
    tree_offsets: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    feature: np.ndarray
    threshold: np.ndarray
    value: np.ndarray
    decision_threshold: float = DECISION_THRESHOLD

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the mean leaf value from every tree."""

        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise RuntimeError("feature matrix has an invalid shape")
        if np.isinf(matrix).any():
            raise RuntimeError("feature matrix contains an infinite value")
        matrix = np.where(np.isnan(matrix), self.imputer_statistics, matrix)
        if not np.isfinite(matrix).all():
            raise RuntimeError("feature matrix is not finite after imputation")

        scores = np.zeros(matrix.shape[0], dtype=np.float64)
        rows = np.arange(matrix.shape[0])
        for tree_index in range(len(self.tree_offsets) - 1):
            start = int(self.tree_offsets[tree_index])
            stop = int(self.tree_offsets[tree_index + 1])
            left = self.children_left[start:stop]
            right = self.children_right[start:stop]
            feature = self.feature[start:stop]
            threshold = self.threshold[start:stop]
            value = self.value[start:stop]

            nodes = np.zeros(matrix.shape[0], dtype=np.int64)
            active = np.ones(matrix.shape[0], dtype=bool)
            for _ in range(stop - start):
                active_rows = rows[active]
                if not active_rows.size:
                    break
                active_nodes = nodes[active_rows]
                leaves = left[active_nodes] == -1
                active[active_rows[leaves]] = False
                branch_rows = active_rows[~leaves]
                if branch_rows.size:
                    branch_nodes = nodes[branch_rows]
                    go_left = (
                        matrix[branch_rows, feature[branch_nodes]]
                        <= threshold[branch_nodes]
                    )
                    nodes[branch_rows] = np.where(
                        go_left,
                        left[branch_nodes],
                        right[branch_nodes],
                    )
            if active.any():
                raise RuntimeError("forest traversal exceeded the validated tree size")
            scores += value[nodes]
        return scores / float(len(self.tree_offsets) - 1)


def _int64(tensors: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = tensors[name]
    if value.dtype != np.int64 or value.ndim != 1:
        raise RuntimeError(f"{name} must be a one-dimensional int64 tensor")
    return value


def _float64(tensors: dict[str, np.ndarray], name: str) -> np.ndarray:
    value = tensors[name]
    if value.dtype != np.float64 or value.ndim != 1:
        raise RuntimeError(f"{name} must be a one-dimensional float64 tensor")
    return value


def _validate_tree(
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
    threshold: np.ndarray,
    value: np.ndarray,
) -> None:
    nodes = len(left)
    arrays = (right, feature, threshold, value)
    if nodes < 1 or not all(len(array) == nodes for array in arrays):
        raise RuntimeError("forest tree arrays have inconsistent lengths")
    leaf = left == -1
    if not np.array_equal(leaf, right == -1):
        raise RuntimeError("forest leaf markers are inconsistent")
    indexes = np.arange(nodes)
    if ((left[~leaf] <= indexes[~leaf]) | (left[~leaf] >= nodes)).any():
        raise RuntimeError("forest left-child index is invalid")
    if ((right[~leaf] <= indexes[~leaf]) | (right[~leaf] >= nodes)).any():
        raise RuntimeError("forest right-child index is invalid")
    if ((feature[~leaf] < 0) | (feature[~leaf] >= len(FEATURE_NAMES))).any():
        raise RuntimeError("forest feature index is invalid")
    if not np.isfinite(threshold[~leaf]).all() or not np.isfinite(value).all():
        raise RuntimeError("forest contains non-finite numeric data")


@lru_cache(maxsize=1)
def load_forest() -> SafeForest:
    """Verify and parse the data-only forest artifact once."""

    path = Path(__file__).resolve().parents[1] / ARTIFACT_NAME
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > 64 * 1024 * 1024
    ):
        raise RuntimeError("forest artifact is unsafe or missing")
    if _sha256(path) != ARTIFACT_SHA256:
        raise RuntimeError("forest artifact integrity check failed")

    tensors = load_file(str(path))
    if set(tensors) != REQUIRED_TENSORS:
        raise RuntimeError("forest artifact tensor schema is invalid")
    imputer = _float64(tensors, "imputer_statistics")
    if imputer.shape != (len(FEATURE_NAMES),) or not np.isfinite(imputer).all():
        raise RuntimeError("forest imputer statistics are invalid")

    offsets = _int64(tensors, "tree_offsets")
    if offsets.shape != (EXPECTED_TREE_COUNT + 1,) or offsets[0] != 0:
        raise RuntimeError("forest tree offsets are invalid")
    if (np.diff(offsets) <= 0).any() or int(offsets[-1]) > MAX_TOTAL_NODES:
        raise RuntimeError("forest tree sizes are invalid")

    left = _int64(tensors, "children_left")
    right = _int64(tensors, "children_right")
    feature = _int64(tensors, "feature")
    threshold = _float64(tensors, "threshold")
    value = _float64(tensors, "value")
    arrays = (left, right, feature, threshold, value)
    if not all(len(array) == int(offsets[-1]) for array in arrays):
        raise RuntimeError("forest artifact array lengths are invalid")

    for tree_index in range(EXPECTED_TREE_COUNT):
        start = int(offsets[tree_index])
        stop = int(offsets[tree_index + 1])
        _validate_tree(
            left[start:stop],
            right[start:stop],
            feature[start:stop],
            threshold[start:stop],
            value[start:stop],
        )
    return SafeForest(imputer, offsets, left, right, feature, threshold, value)
