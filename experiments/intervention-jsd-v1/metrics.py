"""Full-vocabulary, normalized Jensen–Shannon divergence for matched logits."""

import numpy as np


def normalized_mean_jsd(logits_a, logits_b):
    """Return mean row-wise JSD / ln(2) for finite, matching K × V logits.

    All vocabulary entries participate. Rows are equally weighted; inputs must
    have K >= 1 and V >= 2. Invalid inputs raise ValueError.
    """
    arrays = []
    for logits in (logits_a, logits_b):
        try:
            raw = np.asarray(logits)
            if raw.dtype.kind not in "fiu":
                raise ValueError("logits must be real numeric arrays")
            with np.errstate(over="ignore", invalid="ignore"):
                array = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("logits must be real numeric arrays") from exc
        if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
            raise ValueError("logits must have shape K × V with K >= 1, V >= 2")
        if not np.isfinite(array).all():
            raise ValueError("logits must be finite")
        arrays.append(array)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("logit shapes must match")

    def log_softmax(array):
        # Subtraction of opposite finite float64 extremes can yield -inf;
        # that is a legitimate zero-probability limit, not an overflow failure.
        with np.errstate(over="ignore", under="ignore"):
            shifted = array - np.max(array, axis=1, keepdims=True)
            return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    log_p, log_q = map(log_softmax, arrays)
    ln2 = np.log(2.0)
    with np.errstate(under="ignore"):
        log_m = np.logaddexp(log_p, log_q) - ln2

        def kl_terms(log_prob):
            prob = np.exp(log_prob)
            terms = np.zeros_like(prob)
            positive = prob > 0
            # Explicit masking implements 0 log(0 / m) = 0, including m = 0.
            terms[positive] = prob[positive] * (
                log_prob[positive] - log_m[positive]
            )
            return terms.sum(axis=1)

        jsd = (kl_terms(log_p) + kl_terms(log_q)) / (2.0 * ln2)
    return float(np.clip(jsd.mean(), 0.0, 1.0))
