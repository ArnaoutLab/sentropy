from numpy import allclose
from pytest import mark

from sentropy.set import Set
from sentropy.components import Components
from sentropy.abundance import make_abundance
from sentropy.tests.base_tests.set_test import set_data
from sentropy.tests.base_tests.similarity_test import similarity_array_3by3_1


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
    assert allclose(
        set.components.subset_ordinariness, data.subset_similarity
    )


@mark.parametrize("data", set_data[2:])
def test_normalized_subset_similarity(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    assert allclose(
        set.components.normalized_subset_ordinariness,
        data.normalized_subset_similarity,
    )
