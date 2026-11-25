from numpy import allclose, asarray
from pytest import mark, warns

from sentropy.set import Set
from sentropy.abundance import Abundance
from sentropy.backend import get_backend
from sentropy.components import Components
from sentropy.exceptions import ArgumentWarning
from sentropy.similarity import SimilarityFromArray
from sentropy.tests.base_tests.set_test import set_data


@mark.parametrize(
    "data",
    set_data,
)
def test_make_components(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    assert isinstance(set.components, Components)


@mark.parametrize("data", set_data[2:])
def test_set_similarity(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    assert allclose(
        set.components.set_ordinariness,
        data.set_similarity,
    )


@mark.parametrize("data", set_data[2:])
def test_subset_similarity(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    assert allclose(set.components.subset_ordinariness, data.subset_similarity)


@mark.parametrize("data", set_data[2:])
def test_normalized_subset_similarity(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    assert allclose(
        set.components.normalized_subset_ordinariness,
        data.normalized_subset_similarity,
    )

@mark.parametrize("backend_name", ["numpy", "torch"])
def test_negative_ordinariness_warns(backend_name):
    backend = get_backend(backend_name)
    abundance = Abundance(counts=asarray([[1], [10], [1]]), backend=backend)
    similarity = SimilarityFromArray(
        similarity=asarray([
            [1.0, -0.9, 0.2],
            [-0.9, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]),
        backend=backend,
    )
    with warns(ArgumentWarning, match="Negative ordinariness values"):
        Components(abundance=abundance, similarity=similarity)
