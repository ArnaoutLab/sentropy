from numpy import (
    sqrt,
    all as np_all,
    allclose,
    ndarray,
    array,
    dtype,
    memmap,
    inf,
    isfinite,
    float32,
    zeros,
    empty,
    dot,
    random
)

from pandas import DataFrame
from numpy.linalg import norm
from pytest import fixture, raises, mark
import ray

from sentropy import Set
from sentropy.backend import get_backend
from sentropy.exceptions import InvalidArgumentError
from sentropy.ray import (
    _interset_weighted_abundances_ray,
    SimilarityFromRayFunction,
    SimilarityFromSymmetricRayFunction,
    weighted_similarity_chunk_nonsymmetric
)
from sentropy.similarity import (
    SimilarityFromArray,
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
)

import sentropy.tests.mockray as mockray
from sentropy.tests.base_tests.similarity_test import similarity_from_distance

from sentropy.tests.base_tests.similarity_test import (
    relative_abundance_3by2,
    relative_abundance_3by1,
    X_3by1,
    X_3by2,
)


def ray_fix(monkeypatch):
    monkeypatch.setattr(ray, "put", mockray.put)
    monkeypatch.setattr(ray, "get", mockray.get)
    monkeypatch.setattr(ray, "remote", mockray.remote)
    monkeypatch.setattr(ray, "wait", mockray.wait)


@fixture(autouse=True)
def setup(monkeypatch):
    ray_fix(monkeypatch)


MEASURES = (
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
    "rho_hat",
)

abundances_large = array(
    [
        [45, 23],
        [4, 54],
        [23, 1],
        [623, 0],
        [23, 7],
        [23, 90],
        [1, 1],
        [34, 62],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [96, 5],
        [6, 24],
        [6, 4],
        [65, 91],
        [345, 4],
        [23, 62],
        [62, 73],
        [23, 7],
        [23, 90],
        [1, 1],
        [34, 62],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [96, 5],
        [6, 24],
        [6, 4],
        [65, 91],
        [345, 4],
        [23, 62],
        [62, 73],
        [23, 7],
        [23, 90],
        [1, 1],
        [34, 62],
        [13, 72],
        [23, 23],
        [72, 3],
        [62, 3],
        [623, 4],
        [234, 90],
        [23, 12],
        [96, 5],
        [6, 24],
        [6, 4],
        [65, 91],
        [345, 4],
        [23, 62],
        [62, 73],
        [23, 7],
    ]
)
X_large = array(
    [
        [6, 1, 1],
        [7, 34, 62],
        [8, 13, 72],
        [9, 23, 23],
        [11, 72, 3],
        [12, 62, 3],
        [13, 623, 4],
        [14, 234, 90],
        [34, 62, 3],
        [62, 43, 4],
        [23, 34, 90],
        [1, 23, 12],
        [2, 96, 5],
        [0, 45, 23],
        [1, 4, 54],
        [2, 23, 1],
        [3, 623, 0],
        [4, 23, 7],
        [5, 23, 90],
        [3, 6, 24],
        [5, 6, 4],
        [6, 65, 91],
        [34, 75, 4],
        [1, 23, 62],
        [2, 62, 73],
        [3, 23, 7],
        [15, 23, 12],
        [16, 96, 5],
        [17, 6, 24],
        [18, 6, 4],
        [19, 65, 91],
        [3, 45, 4],
        [20, 23, 62],
        [21, 62, 73],
        [22, 23, 7],
        [23, 84, 90],
        [14, 1, 1],
        [24, 34, 62],
        [25, 13, 72],
        [23, 75, 23],
        [26, 72, 3],
        [27, 62, 3],
        [62, 73, 4],
        [21, 34, 90],
        [21, 23, 12],
        [22, 13, 72],
        [45, 23, 23],
        [23, 72, 3],
        [24, 62, 3],
        [62, 63, 4],
        [24, 34, 90],
        [24, 23, 12],
        [45, 96, 5],
        [24, 6, 24],
        [25, 6, 4],
        [26, 65, 91],
        [35, 45, 4],
        [23, 23, 62],
        [27, 62, 73],
        [28, 23, 7],
        [29, 23, 90],
        [4, 1, 1],
        [29, 34, 62],
        [31, 13, 72],
        [32, 3, 23],
        [33, 2, 3],
    ]
)


def similarity_function(a, b):
    a = a / norm(a)
    b = b / norm(b)
    return dot(a, b)


@mark.parametrize(
    "x, y, expected",
    [
        (array([1, 2, 3]), array([1, 2, 3]), 1.0),
        (array([0, 1, 0]), array([2, 0, 0]), 0.0),
        (array([0, 1, 1]), array([0, 1, 0]), 1 / sqrt(2)),
        (array([0, 1, 1]), array([1, 1, 0]), 1 / 2),
    ],
)
def test_similarity_function(x, y, expected):
    assert allclose(similarity_function(x, y), expected)


@mark.parametrize(
    "relative_abundance, X, chunk_size",
    [
        (relative_abundance_3by2, X_3by2, 2),
        (relative_abundance_3by2, X_3by1, 1),
        (relative_abundance_3by1, X_3by2, 4),
        (relative_abundance_3by1, X_3by2, 2),
    ],
)
def test_weighted_abundances_from_function(relative_abundance, X, chunk_size):
    sim_matrix = zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            sim_matrix[i, j] = similarity_function(X[i], X[j])
    similarity1 = SimilarityFromArray(sim_matrix)
    expected = similarity1.weighted_abundances(relative_abundance=relative_abundance)
    similarities_out = empty(shape=(X.shape[0], X.shape[0]))
    similarity = SimilarityFromRayFunction(
        func=similarity_function,
        X=X,
        similarities_out=similarities_out,
    )
    weighted_abundances = similarity.weighted_abundances(
        relative_abundance=relative_abundance
    )
    assert allclose(weighted_abundances, expected)
    assert allclose(similarities_out, sim_matrix)


def test_comparisons():
    results = []
    for simclass in [
        SimilarityFromFunction,
        SimilarityFromRayFunction,
        SimilarityFromSymmetricFunction,
        SimilarityFromSymmetricRayFunction,
    ]:
        if "Ray" in simclass.__name__:
            similarity = simclass(
                func=similarity_function, X=X_large)
        else:
            similarity = simclass(func=similarity_function, X=X_large, chunk_size=4)
        m = Set(abundances_large, similarity)
        df = m.to_dataframe(qs=[0, 1, 2, 200], ms=MEASURES)
        results.append(df.drop(columns="level"))
    for result in results[1:]:
        assert allclose(results[0].to_numpy(), result.to_numpy())


def test_similarities_out():
    computed_similarity_matrices = []
    similarities_out = empty((X_large.shape[0], X_large.shape[0]))
    for simclass in [
        SimilarityFromFunction,
        SimilarityFromRayFunction,
        SimilarityFromSymmetricFunction,
        SimilarityFromSymmetricRayFunction,
    ]:
        if "Ray" in simclass.__name__:
            similarity = simclass(
                func=similarity_function,
                X=X_large,
                similarities_out=similarities_out,
            )
        else:
            similarity = simclass(
                func=similarity_function,
                X=X_large,
                chunk_size=7,
                similarities_out=similarities_out,
            )
        similarity.weighted_abundances(abundances_large)
        computed_similarity_matrices.append(similarities_out)
    for matrix in computed_similarity_matrices[1:]:
        assert allclose(computed_similarity_matrices[0], matrix)

# ── Helper ──────────────────────────────────────────────

def _dot_similarity(x, y):
    """Simple inner-product similarity for testing."""
    return float(dot(x, y))


# ── 1. _interset_weighted_abundances_ray ─────────────────
def test_interset_weighted_abundances_ray_triggers_backpressure():
    """Force max_inflight_tasks < number of chunks so the
    `if len(futures) >= max_inflight_tasks` branch executes."""
    backend = get_backend("numpy")
    rng = random.default_rng(42)
    X = rng.random((10, 3))
    Y = rng.random((10, 3))
    abundance = rng.random((10, 2))

    # chunk_size=2  → 5 chunks
    # max_inflight=2 → after 2 futures are in flight, backpressure triggers
    result = _interset_weighted_abundances_ray(
        similarity=_dot_similarity,
        X=X,
        Y=Y,
        relative_abundance=abundance,
        chunk_size=2,
        max_inflight_tasks=2,
        backend=backend,
    )

    assert result.shape == (10, 2)
    # Sanity: result should be finite
    assert np_all(isfinite(result))


# ── 2. SimilarityFromRayFunction.weighted_abundances ────

def test_similarity_from_ray_function_backpressure():
    """Override max_inflight_tasks to 1 so backpressure fires on the
    very second chunk."""
    rng = random.default_rng(42)
    X = rng.random((10, 3))

    obj = SimilarityFromRayFunction(func=_dot_similarity, X=X)

    # Override to force small values
    obj.chunk_size = 2          # 5 chunks for 10 rows
    obj.max_inflight_tasks = 1  # backpressure after 1st future

    abundance = rng.random((10, 2))
    result = obj.weighted_abundances(abundance)

    assert result.shape == (10, 2)
    assert np_all(isfinite(result))


# ── 3. SimilarityFromSymmetricRayFunction.weighted_abundances

def test_similarity_from_symmetric_ray_function_backpressure():
    """Same strategy — override max_inflight_tasks and chunk_size."""
    rng = random.default_rng(42)
    X = rng.random((10, 3))

    obj = SimilarityFromSymmetricRayFunction(func=_dot_similarity, X=X)

    obj.chunk_size = 2
    obj.max_inflight_tasks = 1

    abundance = rng.random((10, 2))
    result = obj.weighted_abundances(abundance)

    assert result.shape == (10, 2)
    assert np_all(isfinite(result))

def test_weighted_similarity_chunk_nonsymmetric_with_dataframe_y():
    """Cover the `elif isinstance(Y, DataFrame): Y = Y.to_numpy()` branch."""
    backend = get_backend("numpy")
    rng = random.default_rng(42)

    X = DataFrame(rng.random((6, 3)))
    Y = DataFrame(rng.random((6, 3)))          # <-- DataFrame, not ndarray
    abundance = rng.random((6, 2))

    chunk_index, result, similarities = weighted_similarity_chunk_nonsymmetric(
        similarity=_dot_similarity,
        X=X,
        Y=Y,
        relative_abundance=abundance,
        backend=backend,
        chunk_size=6,        # single chunk covering all rows
        chunk_index=0,
        return_Z=True,
    )

    assert result.shape == (6, 2)
    assert similarities.shape == (6, 6)
    assert np_all(isfinite(result))
