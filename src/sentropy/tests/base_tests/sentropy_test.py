from sentropy.sentropy import sentropy
from sentropy.similarity import SimilarityFromFunction, SimilarityFromSymmetricFunction,\
SimilarityFromFile, SimilarityFromDataFrame
import numpy as np
import pandas as pd

def test_LCR_no_similarity():
    #Check most inequalities in Table S2.1 of Reeve's paper "how to partition diversity" https://arxiv.org/pdf/1404.6520.
    
    counts = np.array([[1,0], [0,1], [1,1], [2,3]])
    similarity = np.array([[1,0.5,0,0],[0.5,1,0,0],[0,0,1,0.1],[0,0,0.1,1]])
    N, S = counts.shape[1], counts.shape[0]
    weights = np.sum(counts, axis=0).astype(float)
    weights /= np.sum(weights)
    diversity_indices = sentropy(counts,qs=[1], similarity=similarity)

    assert 1 <= diversity_indices['set_alpha_q=1'] <= N*S
    assert (1/weights <= diversity_indices['subset_alpha_q=1']).all() & (diversity_indices['subset_alpha_q=1'] <= S/weights).all()
    assert (1 <= diversity_indices['subset_normalized_alpha_q=1']).all() & (diversity_indices['subset_normalized_alpha_q=1'] < S).all() 
    assert (1 <= diversity_indices['set_normalized_alpha_q=1']) & (diversity_indices['set_normalized_alpha_q=1'] < S)
    assert (1 <= diversity_indices['subset_rho_q=1']).all()
    assert (1 <= diversity_indices['set_rho_q=1'])
    assert (diversity_indices['subset_beta_q=1'] <= 1).all()
    assert (0< diversity_indices['set_beta_q=1'] <= 1)
    assert (weights <= diversity_indices['subset_normalized_rho_q=1']).all()
    assert (0 <= diversity_indices['set_normalized_rho_q=1'])
    assert (diversity_indices['subset_normalized_beta_q=1'] <= 1/weights).all()
    assert (diversity_indices['set_normalized_beta_q=1'] <= N)
    assert (1 <= diversity_indices['subset_gamma_q=1']).all() & (diversity_indices['subset_gamma_q=1'] <= S/weights).all()
    assert (1 <= diversity_indices['set_gamma_q=1'] <= S)
    assert (diversity_indices['subset_gamma_q=1'] <= diversity_indices['subset_alpha_q=1']).all()
    assert (diversity_indices['set_gamma_q=1'] <= diversity_indices['set_alpha_q=1'])

def test_LCR_return_dataframe():
    #Check the inequalities in Table S2.1 of Reeve's paper that hold in the naive type model (the ones marked with asterisks)
    counts = np.array([[1,3], [3,1], [1,1], [2,3]])
    N, S = counts.shape[1], counts.shape[0]
    weights = np.sum(counts, axis=0).astype(float)
    weights /= np.sum(weights)
    df = sentropy(counts,qs=[1], return_dataframe=True)

    assert (df['rho'][1:].to_numpy() <= 1/weights).all()
    assert df['rho'][0] <= N
    assert (weights <= df['beta'][1:].to_numpy()).all()
    assert (df['normalized_rho'][1:] <= 1).all()
    assert df['normalized_rho'][0] <= 1
    assert (1 <= df['normalized_beta'][1:]).all()
    assert 1 <= df['normalized_beta'][0]
    assert df['alpha'][0] <= N*df['gamma'][0]
    assert df['normalized_alpha'][0] <= df['gamma'][0]

def test_arguments_symmetric_and_parallelize_of_LCR():
    sfargs = np.array([
      [1, 2], 
      [3, 4], 
      [5, 6]
    ])

    def similarity_function(species_i, species_j):
        return 1 / (1 + np.linalg.norm(species_i - species_j))

    results_1 = sentropy(np.array([[1, 1], [1, 0], [0, 1]]), qs=[1],similarity=similarity_function,
                                            sfargs=sfargs, chunk_size=10, return_dataframe=True)

    results_2 = sentropy(np.array([[1, 1], [1, 0], [0, 1]]), qs=[1],similarity=similarity_function,
                                sfargs=sfargs, chunk_size=10, parallelize=True, return_dataframe=True)

    results_3 = sentropy(np.array([[1, 1], [1, 0], [0, 1]]), qs=[1],similarity=similarity_function,
                                sfargs=sfargs, chunk_size=10, symmetric=True, return_dataframe=True)

    results_4 = sentropy(np.array([[1, 1], [1, 0], [0, 1]]), qs=[1],similarity=similarity_function,
            sfargs=sfargs, chunk_size=10, symmetric=True, parallelize=True, return_dataframe=True)

    assert results_1.equals(results_2)
    assert results_1.equals(results_3)
    assert results_1.equals(results_4)

def test_kl_div_no_similarity():
    counts_1 = np.array([[9/25], [12/25], [4/25]])
    counts_2 = np.array([[1/3], [1/3], [1/3]])

    results_default_viewpoint = sentropy(counts_2, counts_1, qs=[1], return_dataframe=True)
    results_viewpoint_2 = sentropy(counts_2, counts_1, qs=[2], return_dataframe=True)

    assert np.allclose(results_default_viewpoint[0], 1.1023618416445828, atol=1e-8)
    assert np.allclose(results_default_viewpoint[0], results_default_viewpoint[1].iloc[0,0], rtol=1e-5)
    assert results_viewpoint_2[0] > results_default_viewpoint[0]

def test_arguments_eff_no_and_which_in_kl_div():
    counts_1 = np.array([[9/25], [12/25], [4/25]])
    counts_2 = np.array([[1/3], [1/3], [1/3]])
    results_1 = sentropy(counts_1, counts_2)
    results_2 = sentropy(counts_1, counts_2, level='set')
    results_3 = sentropy(counts_1, counts_2, level='subset')
    results_4 = sentropy(counts_1, counts_2, eff_no=False)

    assert results_2 == results_1[0]
    assert results_3 == results_1[1]
    assert np.allclose(results_4[0], 0.0853)


def test_kl_div_with_similarity_from_array():
    labels = ["owl", "eagle", "flamingo", "swan", "duck", "chicken", "turkey", "dodo", "dove"]
    no_species = len(labels)
    S = np.identity(n=no_species)


    S[0][1:9] = (0.91, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88, 0.88) # owl
    S[1][2:9] = (      0.88, 0.89, 0.88, 0.88, 0.88, 0.89, 0.88) # eagle
    S[2][3:9] = (            0.90, 0.89, 0.88, 0.88, 0.88, 0.89) # flamingo
    S[3][4:9] = (                  0.92, 0.90, 0.89, 0.88, 0.88) # swan
    S[4][5:9] = (                        0.91, 0.89, 0.88, 0.88) # duck
    S[5][6:9] = (                              0.92, 0.88, 0.88) # chicken
    S[6][7:9] = (                                    0.89, 0.88) # turkey
    S[7][8:9] = (                                          0.88) # dodo
                                                                    # dove

    S = np.maximum(S, S.transpose() )
    counts_1 = pd.DataFrame({"Community": [1, 1, 1, 1, 1, 1, 1, 1, 1]}, index=labels)
    counts_2 = pd.DataFrame({"Community": [1, 2, 1, 1, 1, 1, 1, 2, 1]}, index=labels)
    result_default_viewpoint = sentropy(counts_1, counts_2, similarity=S, qs=1, return_dataframe=True)
    assert np.allclose(result_default_viewpoint[0], 1.0004668803029282)


def test_kl_div_with_similarity_from_function():
    sfargs = np.array([[1, 2], [3, 4], [5, 6]])

    def similarity_function(species_i, species_j):
        return np.exp(-np.linalg.norm(species_i - species_j))

    counts_1 = pd.DataFrame({'community_1': [1,1,0], 'community_2': [1,0,1]})
    counts_2 = pd.DataFrame({'community_1': [2,1,0], 'community_2': [2,0,1]})

    results = sentropy(counts_2, counts_1, similarity=similarity_function, sfargs=sfargs)

    assert np.allclose(results[0], 1.0655322169685402, atol=1e-8)