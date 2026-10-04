import unittest

import numpy as np

from metrics import normalized_mean_jsd


class NormalizedMeanJSDTests(unittest.TestCase):
    def test_identity(self):
        logits = np.array([[1.0, 2.0, -3.0], [-50.0, 0.0, 50.0]])
        self.assertAlmostEqual(normalized_mean_jsd(logits, logits), 0.0, places=14)

    def test_symmetry(self):
        a = np.array([[1.0, 2.0, 3.0], [-2.0, 4.0, 0.0]])
        b = np.array([[-1.0, 0.0, 5.0], [2.0, -4.0, 1.0]])
        self.assertAlmostEqual(normalized_mean_jsd(a, b), normalized_mean_jsd(b, a))

    def test_extreme_disjoint_distributions_without_overflow(self):
        limit = np.finfo(np.float64).max
        a = np.array([[limit, -limit, -limit], [10000.0, -10000.0, -10000.0]])
        b = np.array([[-limit, limit, -limit], [-10000.0, 10000.0, -10000.0]])
        with np.errstate(all="raise"):
            self.assertAlmostEqual(normalized_mean_jsd(a, b), 1.0, places=14)

    def test_per_row_translation_invariance(self):
        rng = np.random.default_rng(314)
        a, b = rng.normal(size=(2, 5, 17))
        shifts_a = np.array([100.0, -100.0, 10.0, -10.0, 0.0])[:, None]
        shifts_b = np.array([-200.0, 200.0, -20.0, 20.0, 1.0])[:, None]
        self.assertAlmostEqual(
            normalized_mean_jsd(a, b),
            normalized_mean_jsd(a + shifts_a, b + shifts_b), places=13,
        )

    def test_random_range(self):
        rng = np.random.default_rng(2718)
        for scale in (0.001, 1.0, 100.0, 10000.0):
            a, b = rng.normal(size=(2, 8, 101)) * scale
            result = normalized_mean_jsd(a, b)
            self.assertIsInstance(result, float)
            self.assertTrue(0.0 <= result <= 1.0)

    def test_known_full_vocabulary_value_and_equal_row_weighting(self):
        p = np.array([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]])
        q = np.array([[0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])
        mixture = (p + q) / 2
        expected = np.mean(np.sum(
            (p * np.log(p / mixture) + q * np.log(q / mixture)) / 2,
            axis=1,
        )) / np.log(2)
        self.assertGreater(expected, 0.0)  # Same top token, different tails.
        self.assertAlmostEqual(normalized_mean_jsd(np.log(p), np.log(q)), expected)

    def test_rejected_inputs(self):
        valid = np.zeros((2, 3))
        invalid = [
            np.zeros((1, 3)), np.zeros((2, 4)), np.zeros((0, 3)),
            np.zeros((2, 1)), np.zeros(3), np.zeros((2, 3, 1)),
            [[0.0, np.nan, 0.0]] * 2,
            [[0.0, np.inf, 0.0]] * 2,
            [[0.0, -np.inf, 0.0]] * 2,
            [[1.0], [1.0, 2.0]], [["a"] * 3] * 2,
            np.ones((2, 3), dtype=complex), None,
        ]
        for bad in invalid:
            for a, b in ((bad, valid), (valid, bad)):
                with self.subTest(bad=repr(bad)):
                    with self.assertRaises(ValueError):
                        normalized_mean_jsd(a, b)


if __name__ == "__main__":
    unittest.main()
