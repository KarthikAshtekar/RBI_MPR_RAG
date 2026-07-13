from __future__ import annotations

import math
import random
from statistics import NormalDist


def wilson_interval(successes: int, total: int, confidence: float = .95):
    if total == 0: return (None, None)
    z = NormalDist().inv_cdf(1 - (1-confidence)/2)
    p = successes / total; denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return max(0.0, center-half), min(1.0, center+half)


def bootstrap_mean_interval(values, *, resamples=2000, confidence=.95, seed=42):
    values = list(values)
    if not values: return (None, None)
    rng = random.Random(seed); n=len(values)
    means=sorted(sum(rng.choice(values) for _ in range(n))/n for _ in range(resamples))
    alpha=(1-confidence)/2
    return means[int(alpha*resamples)], means[min(resamples-1, int((1-alpha)*resamples)-1)]

