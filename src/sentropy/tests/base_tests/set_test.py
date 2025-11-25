"""Tests for diversity.set."""

from dataclasses import dataclass, field
from numpy import allclose, array, ndarray, identity, zeros, inf, maximum, log
from numpy.linalg import norm
from pandas import DataFrame, concat
from pandas.testing import assert_frame_equal
from pytest import mark, raises
from sentropy.exceptions import InvalidArgumentError

from sentropy.log import LOGGER
from sentropy.abundance import Abundance
from sentropy.similarity import (
    Similarity,
    SimilarityIdentity,
    SimilarityFromArray,
    SimilarityFromSymmetricFunction,
    SimilarityFromFunction,
)
from sentropy import Set
from sentropy.tests.base_tests.similarity_test import similarity_dataframe_3by3
from sentropy.tests.base_tests.similarity_test import similarity_array_3by3_1

MEASURES = (
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
    "rho_hat",
    "beta_hat",
)

subset_names = ["subset_1", "subset_2"]
counts_3by2 = DataFrame([[1, 5], [3, 0], [0, 1]], columns=subset_names)
counts_6by2 = DataFrame(
    [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
    columns=subset_names,
)


@dataclass
class FrequencySet6by2:
    description = "frequency-sensitive set; 6 species, 2 subsets"
    viewpoint: float = 0.0
    counts: DataFrame = field(default_factory=lambda: counts_6by2)
    similarity: None = None
    set_similarity: None = None
    subset_similarity: None = None
    normalized_subset_similarity: None = None
    subset_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subset_names,
                "viewpoint": [0.0, 0.0],
                "alpha": [6.0, 6.0],
                "rho": [1.0, 1.0],
                "beta": [1.0, 1.0],
                "gamma": [6.0, 6.0],
                "normalized_alpha": [3.0, 3.0],
                "normalized_rho": [0.5, 0.5],
                "normalized_beta": [2.0, 2.0],
                "rho_hat": [0.0, 0.0],
                "beta_hat": [1.0, 1.0],
            },
        )
    )
    set_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["set"],
                "viewpoint": [0.0],
                "alpha": [6.0],
                "rho": [1.0],
                "beta": [1.0],
                "gamma": [6.0],
                "normalized_alpha": [3.0],
                "normalized_rho": [0.5],
                "normalized_beta": [2.0],
                "rho_hat": [0.0],
                "beta_hat": [1.0],
            },
            index=[0],
        )
    )


@dataclass
class FrequencySet3by2:
    description = "frequency-sensitive set; 3 species, 2 subsets"
    viewpoint: float = 2.0
    counts: DataFrame = field(default_factory=lambda: counts_3by2)
    similarity: None = None
    set_similarity: None = None
    subset_similarity: None = None
    normalized_subset_similarity: None = None
    subset_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subset_names,
                "viewpoint": [2.0, 2.0],
                "alpha": [4.0, 2.30769231],
                "rho": [1.26315789, 1.16129032],
                "beta": [0.79166667, 0.86111111],
                "gamma": [2.66666667, 1.93548387],
                "normalized_alpha": [1.6, 1.38461538],
                "normalized_rho": [0.50526316, 0.69677419],
                "normalized_beta": [1.97916667, 1.43518519],
                "rho_hat": [0.263158, 0.161290],
                "beta_hat": [0.583333, 0.722222],
            }
        )
    )
    set_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["set"],
                "viewpoint": [2.0],
                "alpha": [2.7777777777777777],
                "rho": [1.2],
                "beta": [0.8319209039548022],
                "gamma": [2.173913043478261],
                "normalized_alpha": [1.4634146341463414],
                "normalized_rho": [0.6050420168067228],
                "normalized_beta": [1.612461673236969],
                "rho_hat": [0.190840],
                "beta_hat": [0.659420],
            },
            index=[0],
        )
    )


@dataclass
class SimilaritySet6by2:
    description = "similarity-sensitive set; 6 species, 2 subsets"
    viewpoint: float = 0.0
    counts: ndarray = field(default_factory=lambda: counts_6by2)
    similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [1.0, 0.5, 0.5, 0.7, 0.7, 0.7],
                [0.5, 1.0, 0.5, 0.7, 0.7, 0.7],
                [0.5, 0.5, 1.0, 0.7, 0.7, 0.7],
                [0.7, 0.7, 0.7, 1.0, 0.5, 0.5],
                [0.7, 0.7, 0.7, 0.5, 1.0, 0.5],
                [0.7, 0.7, 0.7, 0.5, 0.5, 1.0],
            ]
        )
    )
    set_similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
                [0.68333333],
            ]
        )
    )
    subset_similarity: ndarray = field(
        default_factory=lambda: (
            array(
                [
                    [0.33333333, 0.35],
                    [0.33333333, 0.35],
                    [0.33333333, 0.35],
                    [0.35, 0.33333333],
                    [0.35, 0.33333333],
                    [0.35, 0.33333333],
                ],
            ),
        )
    )
    normalized_subset_similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [0.66666667, 0.7],
                [0.66666667, 0.7],
                [0.66666667, 0.7],
                [0.7, 0.66666667],
                [0.7, 0.66666667],
                [0.7, 0.66666667],
            ]
        )
    )
    subset_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subset_names,
                "viewpoint": [0.0, 0.0],
                "alpha": [3.0, 3.0],
                "rho": [2.05, 2.05],
                "beta": [0.487805, 0.487805],
                "gamma": [1.463415, 1.463415],
                "normalized_alpha": [1.5, 1.5],
                "normalized_rho": [1.025, 1.025],
                "normalized_beta": [0.97561, 0.97561],
                "rho_hat": [1.05, 1.05],
                "beta_hat": [-0.02439, -0.02439],
            }
        )
    )
    set_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["set"],
                "viewpoint": [0.0],
                "alpha": [3.0],
                "rho": [2.05],
                "beta": [0.487805],
                "gamma": [1.463415],
                "normalized_alpha": [1.5],
                "normalized_rho": [1.025],
                "normalized_beta": [0.97561],
                "rho_hat": [1.05],
                "beta_hat": [-0.02439],
            },
            index=[0],
        )
    )


@dataclass
class SimilaritySet3by2:
    description = "similarity-sensitive set; 3 species, 2 subsets"
    viewpoint: float = 2.0
    counts: DataFrame = field(default_factory=lambda: counts_3by2)
    similarity: ndarray = field(default_factory=lambda: array(similarity_array_3by3_1))
    set_similarity: ndarray = field(
        default_factory=lambda: array([[0.76], [0.62], [0.22]])
    )
    subset_similarity: ndarray = field(
        default_factory=lambda: (
            array(
                [
                    [0.25, 0.51],
                    [0.35, 0.27],
                    [0.07, 0.15],
                ]
            ),
        )
    )
    normalized_subset_similarity: ndarray = field(
        default_factory=lambda: array(
            [
                [0.625, 0.85],
                [0.875, 0.45],
                [0.175, 0.25],
            ]
        )
    )
    subset_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": subset_names,
                "viewpoint": [2.0, 2.0],
                "alpha": [3.07692308, 2.22222222],
                "rho": [1.97775446, 1.48622222],
                "beta": [0.50562394, 0.67284689],
                "gamma": [1.52671756, 1.49253731],
                "normalized_alpha": [1.23076923, 1.33333333],
                "normalized_rho": [0.79110178, 0.89173333],
                "normalized_beta": [1.26405985, 1.12141148],
                "rho_hat": [0.977754, 0.486222],
                "beta_hat": [0.011247877758913116, 0.345694],
            }
        )
    )
    set_results: DataFrame = field(
        default_factory=lambda: DataFrame(
            {
                "community": ["set"],
                "viewpoint": [2.0],
                "alpha": [2.5],
                "rho": [1.6502801833927663],
                "beta": [0.5942352817544037],
                "gamma": [1.5060240963855422],
                "normalized_alpha": [1.2903225806451613],
                "normalized_rho": [0.8485572790897555],
                "normalized_beta": [1.1744247216675028],
                "rho_hat": [0.608604],
                "beta_hat": [0.026811],
            },
            index=[0],
        )
    )


set_data = (
    FrequencySet6by2(),
    FrequencySet3by2(),
    SimilaritySet6by2(),
    SimilaritySet3by2(),
)


@mark.parametrize(
    "data, expected",
    zip(
        set_data,
        (
            SimilarityIdentity,
            SimilarityIdentity,
            SimilarityFromArray,
            SimilarityFromArray,
        ),
    ),
)
def test_set(data, expected):
    set = Set(counts=data.counts, similarity=data.similarity)
    assert isinstance(set, Set)
    assert isinstance(set.abundance, Abundance)
    assert isinstance(set.similarity, expected)


@mark.parametrize("measure", MEASURES)
@mark.parametrize("data", set_data)
def test_set_diversity(data, measure):
    set = Set(counts=data.counts, similarity=data.similarity)
    set_diversity = set.set_diversity(
        measure=measure, viewpoint=data.viewpoint
    )
    assert allclose(set_diversity, data.set_results[measure])


@mark.parametrize("measure", MEASURES)
@mark.parametrize("data", set_data)
def test_subset_diversity(data, measure):
    set = Set(counts=data.counts, similarity=data.similarity)
    subset_diversity = set.subset_diversity(
        measure=measure, viewpoint=data.viewpoint
    )
    assert allclose(subset_diversity, data.subset_results[measure])


def test_subset_diversity_invalid_measure():
    with raises(InvalidArgumentError):
        Set(counts=counts_3by2).subset_diversity(
            measure="omega", viewpoint=0
        )


@mark.parametrize("data", set_data)
def test_subsets_to_dataframe(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    subsets_df = set.subsets_to_dataframe(data.viewpoint)
    assert_frame_equal(subsets_df, data.subset_results)


@mark.parametrize("data", set_data)
def test_metacommunities_to_dataframe(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    set_df = set.set_to_dataframe(
        viewpoint=data.viewpoint
    )
    assert_frame_equal(set_df, data.set_results)


@mark.parametrize("data", set_data)
def test_to_dataframe(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    expected = concat(
        [data.set_results, data.subset_results]
    ).reset_index(drop=True)
    assert_frame_equal(set.to_dataframe(viewpoint=data.viewpoint), expected)


@mark.parametrize("data", set_data)
def test_select_measures(data):
    set = Set(counts=data.counts, similarity=data.similarity)
    selected_measures = [
        "alpha",
        "gamma",
        "normalized_rho",
    ]
    expected_columns = selected_measures + ["community", "viewpoint"]
    df = set.to_dataframe(
        viewpoint=data.viewpoint, measures=selected_measures
    )
    for col in df:
        assert col in expected_columns
    for col in expected_columns:
        assert col in df


def test_effective_counts():
    """
    Test that:
    * passing None, an instance of SimilarityIdentity, or an identity matrix gives the same results
    * normalized_alpha for each subset is appropriate for effective species count for each viewpoint:
      - raw species count at q = 0
      - 1/(proportion of largest species) at q = infinity
      - intermediate values of q yield intermediate values
    * giving the species some similiarity changes all the effective species counts
    """
    counts = DataFrame(
        {"A": [1, 2, 3, 4, 5], "B": [10, 5, 5, 0, 0]},
        index=["apple", "orange", "banana", "pear", "blueberry"],
    )
    viewpoints = [0, 1, 2, 3, 4, 99, inf]
    first_df = None
    for sim in [None, SimilarityIdentity(), identity(5)]:
        m = Set(counts, sim)
        df = m.to_dataframe(
            measures=["alpha", "normalized_alpha"], viewpoint=viewpoints
        )
        df.set_index(["community", "viewpoint"], inplace=True)
        if first_df is None:
            first_df = df
        else:
            for col in first_df:
                for ind in first_df.index:
                    assert df.loc[ind][col] == first_df.loc[ind][col]
        assert df.loc[("A", 0)]["normalized_alpha"] == 5.0
        assert df.loc[("A", inf)]["normalized_alpha"] == 3.0
        assert df.loc[("B", 0)]["normalized_alpha"] == 3.0
        assert df.loc[("B", inf)]["normalized_alpha"] == 2.0
        for community in ["A", "B"]:
            for i in range(1, len(viewpoints)):
                assert (
                    df.loc[(community, viewpoints[i - 1])]["normalized_alpha"]
                    >= df.loc[(community, viewpoints[i])]["normalized_alpha"]
                )
    sim = identity(5)
    for i, j, val in [
        (1, 0, 0.5),
        (1, 2, 0.4),
        (0, 2, 0.4),
        (3, 4, 0.1),
    ]:
        sim[i, j] = sim[j, i] = val
    m = Set(counts, sim)
    df = m.to_dataframe(viewpoint=viewpoints)
    df.set_index(["community", "viewpoint"], inplace=True)
    for col in first_df:
        for ind in first_df.index:
            assert df.loc[ind][col] < first_df.loc[ind][col]
    for community in ["A", "B"]:
        for i in range(1, len(viewpoints)):
            assert (
                df.loc[(community, viewpoints[i - 1])]["normalized_alpha"]
                >= df.loc[(community, viewpoints[i])]["normalized_alpha"]
            )

def test_symmetric_similarity_function():
    X = array([[1, 2], [3, 4], [5, 6]])

    def similarity_function(species_i, species_j):
        return 1 / (1 + norm(species_i - species_j))

    set1 = Set(array([[1, 1], [1, 0], [0, 1]]), \
        similarity=similarity_function, X=X, chunk_size=10)

    set2 = Set(array([[1, 1], [1, 0], [0, 1]]), \
        similarity=similarity_function, X=X, chunk_size=10, symmetric=True)

    assert set1.to_dataframe(viewpoint=[0,1,inf]).equals(\
        set2.to_dataframe(viewpoint=[0,1,inf]))


def test_property1():
    """
    Test elementary property 1 from L&C:
    Symmetry. Diversity is unchanged by the order in which the species happen to be listed.
    """
    X = DataFrame(
        {
            "red": [255, 245, 245, 213, 44],
            "green": [0, 108, 236, 227, 13],
            "blue": [0, 66, 66, 120, 92],
            "core": [100, 0, 0, 100, 0],
            "round": [90, 95, 20, 50, 90],
        },
        index=["apple", "orange", "banana", "pear", "blueberry"],
    )
    communities = DataFrame(
        {"bowl": [3, 1, 5, 1, 0], "fridge": [1, 4, 0, 1, 20]},
        index=["apple", "orange", "banana", "pear", "blueberry"],
    )

    def similarity_function(species_i, species_j):
        return 1 / (1 + norm(species_i - species_j) / 100)

    num_species = communities.shape[0]
    viewpoints = [0, 1, 2, 4, 88, inf]
    measures = [
        "alpha",
        "rho",
        "beta",
        "gamma",
        "normalized_alpha",
        "normalized_rho",
        "normalized_beta",
        "rho_hat",
    ]

    def get_result():
        set = Set(communities,
            similarity=similarity_function, X=X.to_numpy(), symmetric=True)
        return set.to_dataframe(
            viewpoint=viewpoints, measures=measures
        ).set_index(["community", "viewpoint"])

    df1 = get_result()
    X = X.sort_index()
    communities = communities.sort_index()
    df2 = get_result()
    assert allclose(df1.to_numpy(), df2.to_numpy())


def test_property2():
    """
    Test elementary property 2 from L&C:
    Absent species. Diversity is unchanged by adding a new species of abundance 0.
    This test also tests passing a pandas dataframe or a path to a file for the similarity matrix.
    """
    labels_2b = (
        "ladybug",
        "bee",
        "butterfly",
        "lobster",
        "fish",
        "turtle",
        "parrot",
        "llama",
        "orangutan",
    )
    no_species_2b = len(labels_2b)
    S_2b = identity(n=no_species_2b)
    # fmt: off
    S_2b[0][1:9] = (0.60, 0.55, 0.45, 0.25, 0.22, 0.23, 0.18, 0.16)  # ladybug
    S_2b[1][2:9] = (      0.60, 0.48, 0.22, 0.23, 0.21, 0.16, 0.14)  # bee
    S_2b[2][3:9] = (            0.42, 0.27, 0.20, 0.22, 0.17, 0.15)  # bu’fly
    S_2b[3][4:9] = (                  0.28, 0.26, 0.26, 0.20, 0.18)  # lobster
    S_2b[4][5:9] = (                        0.75, 0.70, 0.66, 0.63)  # fish
    S_2b[5][6:9] = (                              0.85, 0.70, 0.70)  # turtle
    S_2b[6][7:9] = (                                    0.75, 0.72)  # parrot
    S_2b[7][8:9] = (                                          0.85)  # llama
    pass                                                             # orangutan
    # fmt: on

    S_2b = maximum(S_2b, S_2b.transpose())
    S_2b_df = DataFrame({labels_2b[i]: S_2b[i] for i in range(no_species_2b)}, index=labels_2b)

    counts = DataFrame({"Community 2b": [1, 1, 1, 1, 1, 1, 1, 1, 0]}, index=labels_2b)
    viewpoints = [0, 1, 2, 3, 4, 5, inf]
    set = Set(counts, similarity=S_2b_df)
    df1 = set.to_dataframe(viewpoint=viewpoints).set_index(
        ["community", "viewpoint"]
    )
    counts = counts[counts["Community 2b"] > 0]
    S_2b = S_2b[:-1, :-1]
    S_2b_df = DataFrame({labels_2b[i]: S_2b[i] for i in range(no_species_2b-1)}, index=labels_2b[:-1])
    S_2b_df.to_csv('S_2b_df_after_removing_zero_abundance_species.csv', index=False)
    set = Set(counts, similarity='S_2b_df_after_removing_zero_abundance_species.csv')
    df2 = set.to_dataframe(viewpoint=viewpoints).set_index(
        ["community", "viewpoint"]
    )
    assert allclose(df1.to_numpy(), df2.to_numpy())


def test_property3():
    """
    Test elementary property 3 from L&C:
    Identical species. If two species are identical, then merging them into one leaves the diversity unchanged.
    """
    labels = ["zucchini", "pumpkin", "eggplant", "aubergine"]
    no_species = len(labels)
    sim = identity(no_species)
    # fmt: off
    sim[0][1:4] = (0.5, 0.3, 0.3)
    sim[1][2:4] = (     0.3, 0.3)
    sim[2][3:4] = (          1.0)
    # fmt: on
    sim = maximum(sim, sim.T)
    counts = DataFrame(
        {"A": [3, 2, 4, 0], "B": [0, 4, 0, 5], "C": [1, 1, 1, 1]}, index=labels
    )
    viewpoints = [0, 1, 2, 3, 4, 5, 90, inf]
    measures = [
        "alpha",
        "rho",
        "beta",
        "gamma",
        "normalized_alpha",
        "normalized_rho",
        "normalized_beta",
        "rho_hat",
    ]
    set = Set(counts, sim)
    df1 = set.to_dataframe(viewpoint=viewpoints, measures=measures).set_index(
        ["community", "viewpoint"]
    )
    labels = labels[:-1]
    sim = sim[:-1, :-1]
    counts = DataFrame({"A": [3, 2, 4], "B": [0, 4, 5], "C": [1, 1, 2]}, index=labels)
    set = Set(counts, sim)
    df2 = set.to_dataframe(viewpoint=viewpoints, measures=measures).set_index(
        ["community", "viewpoint"]
    )
    assert allclose(df1.to_numpy(), df2.to_numpy(), equal_nan=True)


def test_figure_1():
    """
    Test that we get the results described in L&C pp. 482-483 (figure 1)
    """
    before_counts = array([[1], [3], [6]])
    before_sim = None
    naive_counts = array([[1], [3], [3], [3]])
    naive_sim = None
    nonnaive_counts = naive_counts
    nonnaive_sim = array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.9], [0, 0, 0.9, 1]])
    before = Set(before_counts, before_sim)
    naive = Set(naive_counts, naive_sim)
    nonnaive = Set(nonnaive_counts, nonnaive_sim)
    for q in [0, 1, 2, 3, 4, 5, inf]:
        before_alpha = before.set_diversity(viewpoint=q, measure="alpha")
        naive_alpha = naive.set_diversity(viewpoint=q, measure="alpha")
        nonnaive_alpha = nonnaive.set_diversity(viewpoint=q, measure="alpha")
        assert (naive_alpha - before_alpha) >= 1.0
        assert (naive_alpha - nonnaive_alpha) > 0.9
        assert naive_alpha <= 4.0
        assert before_alpha <= 3.0
        assert (nonnaive_alpha - before_alpha) < 0.2