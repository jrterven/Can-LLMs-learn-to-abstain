from collections import defaultdict

import numpy as np

from .scoring import reward


def paired_contrast(rows, left="conditioned", right="binary", thresholds=(0.6, 0.8), samples=10000, seed=0):
    """Crossed paired bootstrap: preserve question identity across thresholds and seeds."""
    lookup = {}
    for r in rows:
        if r["arm"] in {left, right} and r["tau"] in thresholds:
            key = (r["arm"], r["seed"], r["id"], r["tau"])
            if key in lookup:
                raise ValueError("Duplicate observation in contrast.")
            lookup[key] = reward(r["outcome"], r["tau"])
    seeds = sorted({k[1] for k in lookup if k[0] == left})
    ids = sorted({k[2] for k in lookup if k[0] == left})
    if not seeds or not ids:
        raise ValueError("Both conditions are required for a paired contrast.")
    right_seeds = {k[1] for k in lookup if k[0] == right}
    baseline = right in {"prompt", "filter", "forced", "warmup"} and right_seeds == {0}
    expected = len(seeds) * len(ids) * len(thresholds)
    if sum(k[0] == left for k in lookup) != expected:
        raise ValueError("Left condition has incomplete seed/question/threshold grid.")
    if not baseline and right_seeds != set(seeds):
        raise ValueError("Training seeds are not paired.")
    differences = np.empty((len(seeds), len(ids)))
    for si, s in enumerate(seeds):
        for qi, q in enumerate(ids):
            try:
                differences[si, qi] = np.mean([lookup[left, s, q, t] - lookup[right, 0 if baseline else s, q, t] for t in thresholds])
            except KeyError as exc:
                raise ValueError("Missing paired question/threshold; do not silently inner-join.") from exc
    # Require identical right-side question population as well.
    if {k[2] for k in lookup if k[0] == right} != set(ids):
        raise ValueError("Right condition has a different question population.")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples)
    for i in range(samples):
        ss = rng.integers(0, len(seeds), len(seeds))
        qq = rng.integers(0, len(ids), len(ids))
        draws[i] = differences[np.ix_(ss, qq)].mean()
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"left": left, "right": right, "delta_utility": float(differences.mean()),
            "ci95": [float(lower), float(upper)], "n_questions": len(ids), "n_seeds": len(seeds),
            "per_seed": {str(s): float(differences[i].mean()) for i, s in enumerate(seeds)},
            "bootstrap_samples": samples, "thresholds": list(thresholds),
            "caution": "Three training seeds give limited information about training variability."}


def group_rows(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["model"], r["dataset"], r["arm"], r["seed"], r["tau"]].append(r)
    return groups
