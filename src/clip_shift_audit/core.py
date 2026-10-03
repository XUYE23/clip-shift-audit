"""Known-class calibration and held-out open-set evaluation.

Higher scores always mean 'more likely in-distribution'.
Calibration contains known-class data only; test labels use -1 for unknowns.
"""
import hashlib
import math
import numpy as np


def matrix(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or not all(a.shape) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite 2-D matrix")
    return a


def normalize(value, name):
    a = matrix(value, name)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError(f"{name} has invalid or zero-norm rows")
    return a / norms


def cosine_logits(images, texts):
    x, t = normalize(images, "images"), normalize(texts, "texts")
    if x.shape[1] != t.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    return np.clip(x @ t.T, -1.0, 1.0)


def confidence(logits, method="cosine", temperature=0.01):
    z = matrix(logits, "logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if method == "cosine":
        return z.max(axis=1)
    if method == "margin":
        if z.shape[1] < 2:
            raise ValueError("margin requires at least two classes")
        top = np.partition(z, -2, axis=1)[:, -2:]
        return top[:, 1] - top[:, 0]
    if method == "msp":
        p = np.exp((z - z.max(axis=1, keepdims=True)) / temperature)
        return 1.0 / p.sum(axis=1)
    raise ValueError("method must be cosine, msp, or margin")


def vector(value, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or not a.size or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return a


def fit_threshold(known_scores, alpha=0.05):
    """Reject score < kth calibration order statistic.

    k=floor(alpha*(n+1)). When k=0, return None and accept all samples.
    For exchangeable known-class scores and fixed scoring function, the marginal
    false-rejection probability is at most k/(n+1) <= alpha. This gives no
    guarantee on unknown-class detection or under distribution shift.
    """
    s = vector(known_scores, "known_scores")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    k = math.floor(alpha * (len(s) + 1))
    return None if k == 0 else float(np.sort(s)[k - 1])


def auroc(known_scores, unknown_scores):
    """Mann-Whitney AUROC with average ranks for ties; ID is positive."""
    pos = vector(known_scores, "known_scores")
    neg = vector(unknown_scores, "unknown_scores")
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2
        i = j
    return float((ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) /
                 (len(pos)*len(neg)))


def risk_coverage(scores, correct):
    """Tie-grouped selective risk on known classes."""
    order = np.argsort(-scores, kind="stable")
    s, ok = scores[order], np.asarray(correct)[order]
    cumulative_errors = np.cumsum(~ok)
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]) + 1, len(s)]
    return [{"coverage": float(n / len(s)),
             "risk": float(cumulative_errors[n-1] / n),
             "threshold": float(s[n-1])} for n in ends]


def fingerprint(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.asarray(a, dtype="<f8")
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes(order="C"))
    return h.hexdigest()


def audit(text_embeddings, calibration_embeddings, test_embeddings, labels,
          *, alpha=0.05, temperature=0.01):
    text = matrix(text_embeddings, "text_embeddings")
    cal = cosine_logits(calibration_embeddings, text)
    test = cosine_logits(test_embeddings, text)
    y = np.asarray(labels)
    if y.ndim != 1 or len(y) != len(test) or y.dtype.kind not in "iu":
        raise ValueError("labels must be an integer vector matching test rows")
    if (y < -1).any() or (y >= len(text)).any():
        raise ValueError("labels must be -1 (unknown) or a text row index")
    known = y >= 0
    if not known.any() or known.all():
        raise ValueError("test split must contain both known and unknown samples")
    pred = test.argmax(axis=1)
    report = {"schema_version": 1, "positive_class": "known",
              "protocol": "known-only calibration; fixed scores; held-out test",
              "input_sha256": fingerprint(text, calibration_embeddings,
                                           test_embeddings, y),
              "n_calibration": len(cal), "n_known": int(known.sum()),
              "n_unknown": int((~known).sum()), "alpha": alpha,
              "temperature": temperature,
              "closed_set_accuracy": float(np.mean(pred[known] == y[known])),
              "methods": {}}
    for method in ("cosine", "msp", "margin"):
        if method == "margin" and len(text) < 2:
            continue
        cs = confidence(cal, method, temperature)
        scores = confidence(test, method, temperature)
        threshold = fit_threshold(cs, alpha)
        accept = np.ones(len(scores), dtype=bool) if threshold is None else scores >= threshold
        accepted_known = accept & known
        target = np.sort(scores[known])[::-1][math.ceil(.95 * known.sum())-1]
        report["methods"][method] = {
            "threshold": threshold,
            "accept_all_due_to_small_calibration": threshold is None,
            "auroc": auroc(scores[known], scores[~known]),
            "fpr_at_95_tpr": float(np.mean(scores[~known] >= target)),
            "known_acceptance": float(np.mean(accept[known])),
            "unknown_acceptance": float(np.mean(accept[~known])),
            "selective_accuracy": (float(np.mean(pred[accepted_known] == y[accepted_known]))
                                   if accepted_known.any() else None),
            "risk_coverage": risk_coverage(scores[known], pred[known] == y[known]),
        }
    return report
