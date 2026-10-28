"""The three heterogeneity measures: collision probability Pr_i, Jaccard similarity j_pop,
and rarity Rare(d_i).

Functions in the "from the draft notebook" section are copied unchanged from
uniqueness-aware-feature-engineering-draft.ipynb (tests/test_measures.py checks this). The rest wraps or vectorises them; the tests check that the wrappers agree with
the originals.
"""
import math

import numpy as np
from sklearn.metrics import jaccard_score


# --------------------------------------------------------------------------------------
# From the draft notebook (unchanged)
# --------------------------------------------------------------------------------------

#1. collision probabilities
def compute_pr_i(a, b):
    """
    Computes the multivariate probability for arbitrary length vectors.

    Parameters:
    a (array-like): The pooled population vector (sum of all vectors)
    b (array-like): An individual vector from vectors
    """
    a = np.array(a, dtype=int)
    b = np.array(b, dtype=int)

    if len(a) != len(b):
        raise ValueError("Vectors 'a' and 'b' must have the same length.")

    sum_a = int(np.sum(a))
    sum_b = int(np.sum(b))

    # Calculate numerator
    numerator = 1
    for ai in a:
        numerator *= math.factorial(ai)
    numerator *= math.factorial(sum_b)
    numerator *= math.factorial(sum_a - sum_b)

    # Calculate denominator
    denominator = 1
    for bi in b:
        denominator *= math.factorial(bi)
    for ai, bi in zip(a, b):
        denominator *= math.factorial(ai - bi)
    denominator *= math.factorial(sum_a)

    return numerator / denominator


def compute_log_factorial_sum(vector, pooled_pop=None, mode="denominator"):
    """
    Computes the sum of log-factorials to avoid overflow/underflow.
    """
    if mode == "numerator":
        # numerator log-factorials
        sum_a = int(np.sum(pooled_pop))
        sum_b = int(np.sum(vector))
        log_sum = 0.0
        for ai in pooled_pop:
            log_sum += math.log(math.factorial(int(ai)))
        log_sum += math.log(math.factorial(sum_b))
        log_sum += math.log(math.factorial(sum_a - sum_b))
        return log_sum
    else:
        # denominator log-factorials
        sum_a = int(np.sum(pooled_pop))
        log_sum = 0.0
        for bi in vector:
            log_sum += math.log(math.factorial(int(bi)))
        for ai, bi in zip(pooled_pop, vector):
            log_sum += math.log(math.factorial(int(ai - bi)))
        log_sum += math.log(math.factorial(sum_a))
        return log_sum

def compute_normalized_pr_i(a, b):
    """
    Computes the collision probability Pr_i normalized by the theoretical maximum
    probability, using log-space arithmetic to prevent numerical underflow.
    """
    a = np.array(a, dtype=int)
    b = np.array(b, dtype=int)
    sum_b = int(np.sum(b))

    # 1. Create the optimal theoretical vector of same size
    sorted_indices = np.argsort(a)[::-1]
    optimal_b = np.zeros_like(b)
    optimal_b[sorted_indices[:sum_b]] = 1

    # 2. Calculate log-probabilities to avoid underflow
    log_num_raw = compute_log_factorial_sum(b, pooled_pop=a, mode="numerator")
    log_den_raw = compute_log_factorial_sum(b, pooled_pop=a, mode="denominator")
    log_raw_pr = log_num_raw - log_den_raw

    log_num_max = compute_log_factorial_sum(optimal_b, pooled_pop=a, mode="numerator")
    log_den_max = compute_log_factorial_sum(optimal_b, pooled_pop=a, mode="denominator")
    log_max_pr = log_num_max - log_den_max

    # 3. Compute ratio in log-space: log(raw / max) = log(raw) - log(max)
    log_normalized = log_raw_pr - log_max_pr

    try:
        return math.exp(log_normalized)
    except OverflowError:
        return 1.0


def compute_similarity(x, reference_representation):
  return jaccard_score(x, reference_representation)


# --------------------------------------------------------------------------------------
# Rarity: the per-vector logic from the draft notebook's MAIN cell, as a function
# --------------------------------------------------------------------------------------

def compute_rarity(ref_rep, pooled_pop, num_reps):
    """Rare(d_i): median rarity (in %) of the features active in d_i.

    Feature rarity is 100 - 100 * (share of reference representations holding it).
    """
    feature_rarity = 100.0 - (100.0 * np.asarray(pooled_pop) / num_reps)
    active_features = (np.asarray(ref_rep) == 1)
    if np.any(active_features):
        return float(np.median(feature_rarity[active_features]))
    return 0.0


# --------------------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------------------

def compute_log10_pr_i(a, b):
    """log10 Pr_i, built from compute_log_factorial_sum.

    compute_pr_i underflows to 0.0 once the pooled population is a few hundred
    individuals, so the pipeline reports Pr_i on a log scale.
    """
    a = np.array(a, dtype=int)
    b = np.array(b, dtype=int)
    log_num = compute_log_factorial_sum(b, pooled_pop=a, mode="numerator")
    log_den = compute_log_factorial_sum(b, pooled_pop=a, mode="denominator")
    return (log_num - log_den) / math.log(10)


def jaccard_matrix(A, B=None):
    """Pairwise Jaccard similarity between the rows of A and the rows of B.

    Vectorised equivalent of calling compute_similarity on every pair (sklearn's
    jaccard_score returns 0 when both vectors are empty, so we do too).
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    intersection = A @ B.T
    union = A.sum(axis=1)[:, None] + B.sum(axis=1)[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def jaccard_rows(A, B):
    """Row-wise Jaccard similarity: J(A[i], B[i]) for every i."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    intersection = (A * B).sum(axis=1)
    union = A.sum(axis=1) + B.sum(axis=1) - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def system_representation(d, selected):
    """The system's view of an individual: d_i restricted to the selected features."""
    return np.asarray(d) * np.asarray(selected, dtype=int)


def compute_system_j(d, selected):
    """j_sys between each d_i and its system representation (works on one vector or a matrix).

    Because the system representation is a subset of d_i, this equals the share of the
    individual's self-reported features that the system keeps.
    """
    d = np.atleast_2d(d)
    return jaccard_rows(system_representation(d, selected), d)
