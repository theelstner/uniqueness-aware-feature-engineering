import ast
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from uafe import data, measures

NOTEBOOK = Path(__file__).parents[1] / "uniqueness-aware-feature-engineering-draft.ipynb"


def _notebook_functions():
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    funcs = {}
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef):
                funcs[node.name] = ast.get_source_segment(src, node)
    return funcs


@pytest.mark.parametrize("module, name", [
    (measures, "compute_pr_i"),
    (measures, "compute_log_factorial_sum"),
    (measures, "compute_normalized_pr_i"),
    (measures, "compute_similarity"),
    (data, "generate_distribution"),
    (data, "generate_mixed_sparsity_distribution"),
])
def test_copied_functions_match_notebook(module, name):
    assert inspect.getsource(getattr(module, name)).strip() == _notebook_functions()[name].strip()


def _small_population(seed=0, n=12, length=8, sparsity=0.4):
    np.random.seed(seed)
    X = data.generate_distribution(num_vectors=n, length=length, sparsity=sparsity)
    return X, X.sum(axis=0)


def test_log10_pr_i_matches_exact_pr_i():
    X, pooled = _small_population()
    for d in X:
        assert measures.compute_log10_pr_i(pooled, d) == pytest.approx(
            math.log10(measures.compute_pr_i(pooled, d)), abs=1e-9)


def test_normalized_pr_i_is_one_for_most_typical_vector():
    _, pooled = _small_population()
    best = np.zeros_like(pooled)
    best[np.argsort(pooled)[::-1][:3]] = 1
    assert measures.compute_normalized_pr_i(pooled, best) == pytest.approx(1.0)


def test_jaccard_matrix_matches_compute_similarity():
    X, _ = _small_population()
    J = measures.jaccard_matrix(X)
    for i in range(len(X)):
        for j in range(len(X)):
            assert J[i, j] == pytest.approx(measures.compute_similarity(X[i], X[j]))


def test_system_j_matches_compute_similarity_and_is_coverage():
    X, _ = _small_population()
    selected = np.array([1, 1, 0, 1, 0, 0, 1, 0], dtype=bool)
    j = measures.compute_system_j(X, selected)
    for d, ji in zip(X, j):
        r = measures.system_representation(d, selected)
        assert ji == pytest.approx(measures.compute_similarity(r, d))
        assert ji == pytest.approx(r.sum() / d.sum())


def test_rarity_matches_notebook_main_cell():
    X, pooled = _small_population()
    feature_rarity = 100.0 - (100.0 * pooled / len(X))
    for d in X:
        assert measures.compute_rarity(d, pooled, len(X)) == pytest.approx(
            np.median(feature_rarity[d == 1]))
