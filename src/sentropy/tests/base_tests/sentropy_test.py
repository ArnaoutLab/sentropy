from sentropy.sentropy import sentropy, interset_ordinariness
from sentropy.exceptions import InvalidArgumentError
from sentropy.similarity import (
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
    SimilarityFromFile,
    SimilarityFromDataFrame,
)
import numpy as np
import pandas as pd
import torch
import pytest

LCR_MEASURES = (
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


def test_single_output():
    P = np.array([0.7, 0.3])
    D1 = sentropy(P)
    assert np.allclose(D1, 1.84202)
    H1 = sentropy(P, eff_no=False)
    assert np.allclose(H1, np.log(D1))


def test_abundance_as_dict():
    # a dataset with two classes, "apples" and "oranges"
    C1 = np.array([5, 3, 0, 0])  # apples; e.g. 5 McIntosh and 3 gala
    C2 = np.array([0, 0, 6, 2])  # oranges; e.g. 6 navel and 2 cara cara
    P = {"apples": C1, "oranges": C2}  # package the classes as P
    S = np.array(
        [  # similarities of all elements, including between classes
            [
                1.0,
                0.8,
                0.2,
                0.1,
            ],  #    note here the non-zero similarity between apples and oranges
            [0.8, 1.0, 0.1, 0.3],
            [0.2, 0.1, 1.0, 0.9],
            [0.1, 0.3, 0.9, 1.0],
        ]
    )

    D1Z = sentropy(P, similarity=S, measure="normalized_rho", level="both")
    R1 = D1Z(which="apples")  # note, q=1. is the default
    R2 = D1Z(which="oranges")
    R3 = D1Z(which="overall")
    assert np.allclose(R1, 0.59125)
    assert np.allclose(R2, 0.58613)
    assert np.allclose(R3, 0.58868)
    D1Z = sentropy(
        P, similarity=S, level="overall", q=[0, 1, np.inf], measure="normalized_rho"
    )
    R4 = D1Z(q=1)
    assert np.allclose(R4, R3)
    D1Z = sentropy(
        P, similarity=S, level="subset", q=[0, 1, np.inf], measure="normalized_rho"
    )
    R5 = D1Z(q=1, which="oranges", measure="normalized_rho")
    assert np.allclose(R5, R2)


def test_no_similarity():
    # Check most inequalities in Table S2.1 of Reeve's paper "how to partition diversity" https://arxiv.org/pdf/1404.6520.

    counts = np.array([[1, 0], [0, 1], [1, 1], [2, 3]])
    similarity = np.array(
        [[1, 0.5, 0, 0], [0.5, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0.1, 1]]
    )
    N, S = counts.shape[1], counts.shape[0]
    weights = np.sum(counts, axis=0).astype(float)
    weights /= np.sum(weights)
    diversity_indices = sentropy(
        counts, q=[1], measure=LCR_MEASURES, similarity=similarity, level="both"
    ).raw_dict

    assert 1 <= diversity_indices["overall_alpha_q=1"] <= N * S
    assert (1 / weights <= diversity_indices["subset_alpha_q=1"]).all() & (
        diversity_indices["subset_alpha_q=1"] <= S / weights
    ).all()
    assert (1 <= diversity_indices["subset_normalized_alpha_q=1"]).all() & (
        diversity_indices["subset_normalized_alpha_q=1"] < S
    ).all()
    assert (1 <= diversity_indices["overall_normalized_alpha_q=1"]) & (
        diversity_indices["overall_normalized_alpha_q=1"] < S
    )
    assert (1 <= diversity_indices["subset_rho_q=1"]).all()
    assert 1 <= diversity_indices["overall_rho_q=1"]
    assert (diversity_indices["subset_beta_q=1"] <= 1).all()
    assert 0 < diversity_indices["overall_beta_q=1"] <= 1
    assert (weights <= diversity_indices["subset_normalized_rho_q=1"]).all()
    assert 0 <= diversity_indices["overall_normalized_rho_q=1"]
    assert (diversity_indices["subset_normalized_beta_q=1"] <= 1 / weights).all()
    assert diversity_indices["overall_normalized_beta_q=1"] <= N
    assert (1 <= diversity_indices["subset_gamma_q=1"]).all() & (
        diversity_indices["subset_gamma_q=1"] <= S / weights
    ).all()
    assert 1 <= diversity_indices["overall_gamma_q=1"] <= S
    assert (
        diversity_indices["subset_gamma_q=1"] <= diversity_indices["subset_alpha_q=1"]
    ).all()
    assert (
        diversity_indices["overall_gamma_q=1"] <= diversity_indices["overall_alpha_q=1"]
    )


def test_return_dataframe():
    # Check the inequalities in Table S2.1 of Reeve's paper that hold in the naive type model (the ones marked with asterisks)
    counts = np.array([[1, 3], [3, 1], [1, 1], [2, 3]])
    N, S = counts.shape[1], counts.shape[0]
    weights = np.sum(counts, axis=0).astype(float)
    weights /= np.sum(weights)
    df = sentropy(counts, q=[1], measure=LCR_MEASURES, return_dataframe=True, level="both")

    assert (df["rho"][1:].to_numpy() <= 1 / weights).all()
    assert df["rho"][0] <= N
    assert (weights <= df["beta"][1:].to_numpy()).all()
    assert (df["normalized_rho"][1:] <= 1).all()
    assert df["normalized_rho"][0] <= 1
    assert (1 <= df["normalized_beta"][1:]).all()
    assert 1 <= df["normalized_beta"][0]
    assert df["alpha"][0] <= N * df["gamma"][0]
    assert df["normalized_alpha"][0] <= df["gamma"][0]


def test_arguments_symmetric_and_parallelize():
    sfargs = np.array([[1, 2], [3, 4], [5, 6]])

    def similarity_function(species_i, species_j):
        return 1 / (1 + np.linalg.norm(species_i - species_j))

    results_1 = sentropy(
        np.array([[1, 1], [1, 0], [0, 1]]),
        q=[1],
        similarity=similarity_function,
        sfargs=sfargs,
        chunk_size=10,
        return_dataframe=True,
    )

    results_2 = sentropy(
        np.array([[1, 1], [1, 0], [0, 1]]),
        q=[1],
        similarity=similarity_function,
        sfargs=sfargs,
        chunk_size=10,
        parallelize=True,
        return_dataframe=True,
    )

    results_3 = sentropy(
        np.array([[1, 1], [1, 0], [0, 1]]),
        q=[1],
        similarity=similarity_function,
        sfargs=sfargs,
        chunk_size=10,
        symmetric=True,
        return_dataframe=True,
    )

    results_4 = sentropy(
        np.array([[1, 1], [1, 0], [0, 1]]),
        q=[1],
        similarity=similarity_function,
        sfargs=sfargs,
        chunk_size=10,
        symmetric=True,
        parallelize=True,
        return_dataframe=True,
    )

    assert results_1.equals(results_2)
    assert results_1.equals(results_3)
    assert results_1.equals(results_4)

def test_torch_backend():
    # a dataset with two classes, "apples" and "oranges"
    P1 = np.array([12, 3, 0, 0])                  # apples; e.g. 12 Granny Smith and 3 McIntosh (but no oranges)
    P2 = np.array([0,  0, 4, 4])                  # oranges; e.g. 4 navel and 4 cara cara (but no apples) 
    P  = {"apples": P1, "oranges": P2}            # package the classes as P
    S = np.array([                                # similarities of all elements, regardless of class
      [1.,  0.7, 0.0, 0.0],                       #    note here the non-zero similarity between apples and oranges
      [0.7, 1.,  0.1, 0.3],
      [0.0, 0.1, 1.,  0.9],
      [0.0, 0.3, 0.9, 1. ],
      ])
    D1Z = sentropy(P, similarity=S,
                   level="subset",                # level="class" is identical; an alias/synonym
                   measure="normalized_rho", backend="torch")
    R1 = D1Z(which="apples")                      # note, no need to pass a measure or a viewpoint
    R2 = D1Z(which="oranges")                     # because D1Z only computed 1 measure and 1 viewpoint anyway

    assert torch.allclose(R1, torch.tensor([0.6695], dtype=R1.dtype), rtol=1e-3, atol=1e-4)
    assert torch.allclose(R2, torch.tensor([0.3750], dtype=R2.dtype), rtol=1e-3, atol=1e-4)

def test_sre_no_similarity():
    counts_1 = np.array([[9 / 25], [12 / 25], [4 / 25]])
    counts_2 = np.array([[1 / 3], [1 / 3], [1 / 3]])

    results_default_viewpoint = sentropy(
        counts_2, counts_1, q=[1], return_dataframe=True, level="both"
    )
    results_viewpoint_2 = sentropy(
        counts_2, counts_1, q=[2], return_dataframe=True, level="both"
    )

    assert np.allclose(results_default_viewpoint[0], 1.1023618416445828, atol=1e-8)
    assert np.allclose(
        results_default_viewpoint[0], results_default_viewpoint[1].iloc[0, 0], rtol=1e-5
    )
    assert results_viewpoint_2[0] > results_default_viewpoint[0]


def test_arguments_eff_no_and_which_in_sre():
    counts_1 = np.array([[9 / 25], [12 / 25], [4 / 25]])
    counts_2 = np.array([[1 / 3], [1 / 3], [1 / 3]])
    results_1 = sentropy(counts_1, counts_2, level="both")
    results_2 = sentropy(counts_1, counts_2)
    results_3 = sentropy(counts_1, counts_2, level="class")
    results_4 = sentropy(counts_1, counts_2, eff_no=False, level="both")

    assert results_2 == results_1[0]
    assert results_3 == results_1[1]
    assert np.allclose(results_4[0], 0.0853)


def test_sre_with_similarity_from_array():
    labels = [
        "owl",
        "eagle",
        "flamingo",
        "swan",
        "duck",
        "chicken",
        "turkey",
        "dodo",
        "dove",
    ]
    no_species = len(labels)
    S = np.identity(n=no_species)

    S[0][1:9] = (0.91, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88)  # owl
    S[1][2:9] = (0.88, 0.89, 0.88, 0.88, 0.88, 0.89, 0.88)  # eagle
    S[2][3:9] = (0.90, 0.89, 0.88, 0.88, 0.88, 0.89)  # flamingo
    S[3][4:9] = (0.92, 0.90, 0.89, 0.88, 0.88)  # swan
    S[4][5:9] = (0.91, 0.89, 0.88, 0.88)  # duck
    S[5][6:9] = (0.92, 0.88, 0.88)  # chicken
    S[6][7:9] = (0.89, 0.88)  # turkey
    S[7][8:9] = 0.88  # dodo
    # dove

    S = np.maximum(S, S.transpose())
    counts_1 = pd.DataFrame({"Community": [1, 1, 1, 1, 1, 1, 1, 1, 1]}, index=labels)
    counts_2 = pd.DataFrame({"Community": [1, 2, 1, 1, 1, 1, 1, 2, 1]}, index=labels)
    result_default_viewpoint = sentropy(
        counts_1, counts_2, similarity=S, q=1, return_dataframe=True, level="both"
    )
    assert np.allclose(result_default_viewpoint[0], 1.0004668803029282)


def test_sre_with_similarity_from_function():
    sfargs = np.array([[1, 2], [3, 4], [5, 6]])

    def similarity_function(species_i, species_j):
        return np.exp(-np.linalg.norm(species_i - species_j))

    counts_1 = pd.DataFrame({"community_1": [1, 1, 0], "community_2": [1, 0, 1]})
    counts_2 = pd.DataFrame({"community_1": [2, 1, 0], "community_2": [2, 0, 1]})

    results = sentropy(
        counts_2, counts_1, similarity=similarity_function, sfargs=sfargs, level="both"
    )

    assert np.allclose(results[0], 1.0655322169685402, atol=1e-8)

def test_sce_no_similarity():
    actual_1 = sentropy(np.array([0.1,0.2,0.7]), np.array([0.5,0.25, 0.25]), measure="sce")
    expected_1 = 3.7321319661472296
    assert np.allclose(actual_1, expected_1)
    actual_2 = sentropy(np.array([0.1,0.2,0.7]), np.array([0.5,0.25, 0.25]), measure="sce", q=2)
    expected_2 = 3.6363636363636362
    assert np.allclose(actual_2, expected_2)

def test_sce_with_similarity_from_array():
    abundance_1 = {'basket_1': [12,3,0,0], 'basket_2': [0,0,4,4]}
    abundance_2 = {'basket_3': [5,0,6,0], 'basket_4': [0,2,0,7]}
    S = np.array([
        [1., 0.7, 0.1, 0.1],
        [0.7, 1., 0.1, 0.3],
        [0.1, 0.1, 1., 0.9],
        [0.1, 0.3, 0.9, 1.],
    ])
    actual = sentropy(abundance_1, abundance_2, similarity=S, measure="sce", level="class", eff_no=False)
    expected = np.array([[0.7374846 , 1.32147731], [0.57451801, 0.24724937]])

def test_vendi_1():
    counts = np.array([[1],[2],[3]])
    Z = np.array([
    [1.0,0.5,0.2],
    [0.5,1.0,0.3],
    [0.2,0.3,1.0]])
    VS = sentropy(counts, similarity=Z, measure="vendi")
    VE = sentropy(counts, similarity=Z, measure="vendi", eff_no=False)
    Z_p = np.diag(np.sqrt([1/6,2/6,3/6])) @ Z @ np.diag(np.sqrt([1/6,2/6,3/6]))
    eigvals = np.linalg.eigvals(Z_p)
    expected_VE = np.sum(-eigvals*np.log(eigvals))
    expected_VS = np.exp(expected_VE)
    assert np.allclose(VS, expected_VS)
    assert np.allclose(VE, expected_VE)

def test_vendi_2():
    """The case where we do not pass any similarity matrix."""
    counts = np.array([[1],[2],[3]])
    VS = sentropy(counts, measure="vendi")
    p = counts/6
    expected_VS = np.exp(-np.sum(p*np.log(p)))
    assert np.allclose(VS, expected_VS)

def test_vendi_3():
    """Test that the Vendi score at viewpoint=2 is same as LCR alpha with the elementwise squared similarity matrix
    as proved by Claude.
    """
    counts = np.array([[10,5],[5,10],[0,2]])
    Z = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
        ])
    VS = sentropy(counts, similarity=Z, measure='vendi', q=2, level='class')
    VS1, VS2 = VS.raw_dict['subset_vendi_q=2'][0], VS.raw_dict['subset_vendi_q=2'][1]
    alpha = sentropy(counts, similarity=Z**2, measure='normalized_alpha', q=2, level='class', return_dataframe=True)
    alpha1, alpha2 = alpha['normalized_alpha'][0], alpha['normalized_alpha'][1]
    assert np.allclose(VS1, alpha1)
    assert np.allclose(VS2, alpha2)

    VE = sentropy(counts, similarity=Z, measure='vendi', q=2, level='class', eff_no=False, backend='torch')
    VE1, VE2 = VE.raw_dict['subset_vendi_q=2'][0], VE.raw_dict['subset_vendi_q=2'][1]
    assert np.allclose(np.exp(VE1), VS1)
    assert np.allclose(np.exp(VE2), VS2)

def test_vendi_4():
    """Test that an error is raised if we pass a function to the similarity argument, when computing the Vendi score"""
    with pytest.raises(InvalidArgumentError):
        counts = np.array([[1],[2],[3]])
        sim_function = lambda x,y : np.exp(-np.abs(x-y))
        VS = sentropy(counts, similarity=sim_function, sfargs=np.array([-1,5,10]), measure='vendi')

def test_vendi_5():
    """Test the Vendi score for negative viewpoint and non-integer viewpoint"""
    p = np.array([[1/6],[2/6],[3/6]])
    VE_1 = sentropy(p, measure="vendi", q=0.5, eff_no=False)
    VS_1 = sentropy(p, measure="vendi", q=0.5)
    expected_VS_1 = np.sum(np.sqrt(p))**2
    expected_VE_1 = np.log(expected_VS_1)
    assert np.allclose(VS_1, expected_VS_1)
    assert np.allclose(VE_1, expected_VE_1)

    VS_2 = sentropy(p, measure="vendi", q=-1)
    expected_VS_2 = np.sqrt(np.sum(1/p))
    assert np.allclose(VS_2, expected_VS_2)

def test_interset_ordinariness():
    # Three objects in X whose ordinariness we want to calculate.
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.7, 0.7],
    ])

    # Four objects in the reference set Y.
    Y = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
    ])

    # Raw counts for the four elements of Y.
    Y_counts = np.array([100, 50, 25, 25])

    # A simple similarity function.
    def similarity(x, y):
        return np.dot(x, y)
    ordinariness = interset_ordinariness(
        X=X,
        Y=Y,
        Y_abundance=Y_counts,
        similarity=similarity,
    )
    expected = np.array([0.7375, 0.2625, 0.7   ])
    assert np.allclose(ordinariness, expected)
