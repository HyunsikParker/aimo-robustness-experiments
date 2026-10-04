"""Synthetic parity and integrity tests; no transformer loading or scoring."""
from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "candidates/uncertainty-safe-forest-v1"))
from audit import convert_official_forest as converter
from uncertainty_profile import safe_forest


class NumericForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = converter.SOURCE
        if path.is_symlink() or path.stat().st_size != 1_007_536 or converter.sha256(path) != converter.EXPECTED_SOURCE_SHA256:
            raise RuntimeError("source forest identity failed before deserialization")
        payload = converter.restricted_load(path)
        cls.imputer, cls.reference, cls.threshold = converter.validated_forest(payload)
        safe_forest.load_forest.cache_clear()
        cls.numeric = safe_forest.load_forest()

    def test_exact_artifact_and_no_metadata(self):
        path = converter.DESTINATION
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), safe_forest.ARTIFACT_SHA256)
        with safe_open(path, framework="numpy") as handle:
            self.assertIsNone(handle.metadata())
            self.assertEqual(set(handle.keys()), safe_forest.REQUIRED_TENSORS)

    def test_synthetic_scores_and_decisions(self):
        rng = np.random.default_rng(20260823)
        matrix = rng.normal(size=(1024, len(converter.FEATURE_NAMES)))
        matrix[rng.random(matrix.shape) < .08] = np.nan
        expected = self.reference.predict(self.imputer.transform(matrix))
        actual = self.numeric.predict(matrix)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(actual < self.numeric.decision_threshold, expected < self.threshold)
        self.assertEqual(self.numeric.decision_threshold, self.threshold)

    def test_changed_hash_fails_before_parse(self):
        original = safe_forest.ARTIFACT_SHA256
        try:
            safe_forest.ARTIFACT_SHA256 = "0" * 64
            safe_forest.load_forest.cache_clear()
            with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                safe_forest.load_forest()
        finally:
            safe_forest.ARTIFACT_SHA256 = original
            safe_forest.load_forest.cache_clear()

    def test_invalid_feature_shapes_and_infinity(self):
        for matrix in (np.ones((2, 13)), np.full((2, 14), np.inf)):
            with self.subTest(shape=matrix.shape):
                with self.assertRaises(RuntimeError):
                    self.numeric.predict(matrix)


if __name__ == "__main__":
    unittest.main()
