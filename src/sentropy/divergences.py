from sentropy.powermean import power_mean
from numpy import prod, power, ndarray, minimum, log as np_log, identity as np_identity, zeros as np_zeros
from pandas import DataFrame
from sentropy.similarity import SimilarityFromArray, SimilarityIdentity, SimilarityFromFile, SimilarityFromSymmetricFunction, SimilarityFromFunction
from sentropy.ray import SimilarityFromSymmetricRayFunction, SimilarityFromRayFunction

def make_normalized_subcommunity_abundance(abundance):
    if type(abundance) == DataFrame:
        abundance = abundance.to_numpy()
    abundance = abundance.astype(float)
    return abundance/abundance.sum(axis=0)

def make_metacommunity_abundance(abundance):
    if type(abundance) == DataFrame:
        abundance = abundance.to_numpy()
    abundance = abundance.astype(float)
    metacommunity_abundance = abundance.sum(axis=1, keepdims=True)
    metacommunity_abundance /= metacommunity_abundance.sum()
    return metacommunity_abundance

def get_exp_renyi_div_from_ords(P, P_ord, Q_ord, viewpoint, atol):
    ord_ratio = P_ord/Q_ord
    if viewpoint != 1:
        exp_renyi_div = power_mean(
            order=viewpoint-1,
            weights=P,
            items=ord_ratio,
            atol=atol,
        )
    else:
        exp_renyi_div = prod(power(ord_ratio, P))
    return exp_renyi_div

def kl_div_effno(P_abundance, Q_abundance, similarity=None, viewpoint=1, symmetric=False, X=None, chunk_size=10, \
    parallelize=False, max_inflight_tasks=64, return_dataframe=False):
    P_meta_ab = make_metacommunity_abundance(P_abundance)
    Q_meta_ab = make_metacommunity_abundance(Q_abundance)
    P_norm_subcom_ab = make_normalized_subcommunity_abundance(P_abundance)
    Q_norm_subcom_ab = make_normalized_subcommunity_abundance(Q_abundance)

    if similarity is None:
        similarity = SimilarityIdentity()
    elif isinstance(similarity, ndarray):
        similarity = SimilarityFromArray(similarity=similarity)
    elif isinstance(similarity, DataFrame):
        similarity = SimilarityFromArray(similarity=similarity.values)
    elif isinstance(similarity, str):
        similarity = SimilarityFromFile(similarity, chunk_size=chunk_size)
    elif callable(similarity):
        if symmetric:
            if parallelize:
                similarity = SimilarityFromSymmetricRayFunction(func=similarity,X=X, chunk_size=chunk_size, max_inflight_tasks=max_inflight_tasks)
            else:
                similarity = SimilarityFromSymmetricFunction(func=similarity,X=X, chunk_size=chunk_size)
        else:
            if parallelize:
                similarity = SimilarityFromRayFunction(func=similarity, X=X, chunk_size=chunk_size, max_inflight_tasks=max_inflight_tasks)
            else:
                similarity = SimilarityFromFunction(func=similarity, X=X, chunk_size=chunk_size)

    P_meta_ord = similarity.weighted_abundances(P_meta_ab)
    P_norm_subcom_ord = similarity.weighted_abundances(P_norm_subcom_ab)
    Q_meta_ord = similarity.weighted_abundances(Q_meta_ab)
    Q_norm_subcom_ord = similarity.weighted_abundances(Q_norm_subcom_ab)

    P_num_subcommunities = P_abundance.shape[1]
    Q_num_subcommunities = Q_abundance.shape[1]

    if type(P_abundance) == DataFrame:
        P_subcommunities_names = P_abundance.columns
    else:
        P_subcommunities_names = [str(i) for i in range(P_num_subcommunities)]
    if type(Q_abundance) == DataFrame:
        Q_subcommunities_names = Q_abundance.columns
    else:
        Q_subcommunities_names = [str(i) for i in range(Q_num_subcommunities)]

    min_count = minimum(1 / P_abundance.sum(), 1e-9)

    exp_renyi_div_meta = get_exp_renyi_div_from_ords(P_meta_ab, P_meta_ord, Q_meta_ord, viewpoint, min_count)

    exp_renyi_divs_subcom = np_zeros(shape=(P_num_subcommunities, Q_num_subcommunities))
    for i in range(P_num_subcommunities):
        for j in range(Q_num_subcommunities):
            P = P_norm_subcom_ab[:,i]
            P_ord = P_norm_subcom_ord[:,i]
            Q_ord = Q_norm_subcom_ord[:,j]
            exp_renyi_div = get_exp_renyi_div_from_ords(P, P_ord, Q_ord, viewpoint, min_count)
            exp_renyi_divs_subcom[i,j] = exp_renyi_div

    if return_dataframe:
        exp_renyi_divs_subcom = DataFrame(exp_renyi_divs_subcom, columns=Q_subcommunities_names, \
            index=P_subcommunities_names)

    return exp_renyi_div_meta, exp_renyi_divs_subcom

def rel_ent(P_abundance, Q_abundance, similarity=None, viewpoint=1, symmetric=False, X=None, chunk_size=10, \
    parallelize=False, max_inflight_tasks=64, return_dataframe=False):
    exp_renyi_div_meta, exp_renyi_divs_subcom = kl_div_effno(P_abundance, Q_abundance, similarity, viewpoint, symmetric, X, chunk_size, \
    parallelize, max_inflight_tasks, return_dataframe)
    return np_log(exp_renyi_div_meta), np_log(exp_renyi_divs_subcom)
