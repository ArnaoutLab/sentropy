from typing import Union, Optional, Callable, Iterable, Tuple
from numpy import inf as np_inf, ndarray, minimum, prod, power, zeros as np_zeros
from pandas import DataFrame

from sentropy.similarity import (
    SimilarityIdentity,
    SimilarityFromArray,
    SimilarityFromFile,
    SimilarityFromSymmetricFunction,
    SimilarityFromFunction,
)

from sentropy.ray import (
    SimilarityFromSymmetricRayFunction,
    SimilarityFromRayFunction,
)

from sentropy.set import Set

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

def LCR_sentropy(counts: Union[DataFrame, ndarray],
    similarity: Optional[Union[ndarray, DataFrame, str, Callable]] = None,
    viewpoint: Union[float, Iterable[float]] = [0,1,np_inf],
    measures: Iterable[str] = MEASURES,
    symmetric: Optional[bool] = False,
    X: Optional[Union[ndarray, DataFrame]] = None,
    chunk_size: Optional[int] = 10,
    parallelize: Optional[bool] = False,
    max_inflight_tasks: Optional[int] = 64,
    return_dataframe: bool = False
    ):

    superset = Set(counts, similarity, symmetric, X, chunk_size, parallelize, max_inflight_tasks)
    
    if return_dataframe:
        sentropies = superset.to_dataframe(viewpoint, measures)
    else:
        sentropies = {}
        for q in viewpoint:
            for measure in measures:
                sentropies[f'set_{measure}_q={q}'] = superset.set_diversity(viewpoint=q, measure=measure)
                sentropies[f'subset_{measure}_q={q}'] = superset.subset_diversity(viewpoint=q, measure=measure)
    return sentropies

def make_normalized_subset_abundance(abundance):
    if type(abundance) == DataFrame:
        abundance = abundance.to_numpy()
    abundance = abundance.astype(float)
    return abundance/abundance.sum(axis=0)

def make_set_abundance(abundance):
    if type(abundance) == DataFrame:
        abundance = abundance.to_numpy()
    abundance = abundance.astype(float)
    set_abundance = abundance.sum(axis=1, keepdims=True)
    set_abundance /= set_abundance.sum()
    return set_abundance

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
    P_meta_ab = make_set_abundance(P_abundance)
    Q_meta_ab = make_set_abundance(Q_abundance)
    P_norm_subcom_ab = make_normalized_subset_abundance(P_abundance)
    Q_norm_subcom_ab = make_normalized_subset_abundance(Q_abundance)

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

    P_num_subsets = P_abundance.shape[1]
    Q_num_subsets = Q_abundance.shape[1]

    if type(P_abundance) == DataFrame:
        P_subsets_names = P_abundance.columns
    else:
        P_subsets_names = [str(i) for i in range(P_num_subsets)]
    if type(Q_abundance) == DataFrame:
        Q_subsets_names = Q_abundance.columns
    else:
        Q_subsets_names = [str(i) for i in range(Q_num_subsets)]

    min_count = minimum(1 / P_abundance.sum(), 1e-9)

    exp_renyi_div_meta = get_exp_renyi_div_from_ords(P_meta_ab, P_meta_ord, Q_meta_ord, viewpoint, min_count)

    exp_renyi_divs_subcom = np_zeros(shape=(P_num_subsets, Q_num_subsets))
    for i in range(P_num_subsets):
        for j in range(Q_num_subsets):
            P = P_norm_subcom_ab[:,i]
            P_ord = P_norm_subcom_ord[:,i]
            Q_ord = Q_norm_subcom_ord[:,j]
            exp_renyi_div = get_exp_renyi_div_from_ords(P, P_ord, Q_ord, viewpoint, min_count)
            exp_renyi_divs_subcom[i,j] = exp_renyi_div

    if return_dataframe:
        exp_renyi_divs_subcom = DataFrame(exp_renyi_divs_subcom, columns=Q_subsets_names, \
            index=P_subsets_names)

    return exp_renyi_div_meta, exp_renyi_divs_subcom


# ----------------------------------------------------------------------
# Relative‑sentropy dispatcher
# ----------------------------------------------------------------------
def relative_sentropy(
    counts_a: Union[DataFrame, ndarray],
    counts_b: Optional[Union[DataFrame, ndarray]] = None,
    *,
    similarity: Optional[Union[ndarray, DataFrame, str, Callable]] = None,
    viewpoint: Union[float, Iterable[float]] = (0, 1, np_inf),
    measures: Iterable[str] = MEASURES,
    symmetric: bool = False,
    X: Optional[Union[ndarray, DataFrame]] = None,
    chunk_size: int = 10,
    parallelize: bool = False,
    max_inflight_tasks: int = 64,
    return_dataframe: bool = False,
) -> Union[dict, Tuple[dict, DataFrame]]:
    """
    Compute either

    * **Leinster‑Cobbold‑Reeve (LCR) diversity indices** when only ``counts_a`` is given, or
    * **Similarity‑sensitive KL‑divergence (and its Rényi generalisations)** when both ``counts_a`` and ``counts_b`` are supplied.

    Parameters
    ----------
    counts_a : DataFrame | ndarray
        First abundance matrix (the “reference” set).
    counts_b : DataFrame | ndarray, optional
        Second abundance matrix.  If ``None`` the function falls back to the
        LCR‐diversity path.
    similarity, viewpoint, measures, symmetric, X, chunk_size,
    parallelize, max_inflight_tasks, return_dataframe
        Forwarded verbatim to the underlying ``sentropy`` or ``kl_div_effno``
        implementations.

    Returns
    -------
    dict
        When ``counts_b`` is ``None`` – exactly the same structure that
        ``sentropy`` returns.
    tuple(dict, DataFrame)
        When ``counts_b`` is provided – a pair consisting of the meta‑level
        Rényi divergence (scalar) and the matrix of subset‑level divergences,
        matching the output of ``kl_div_effno``.
    """

    if counts_b is None:
        return LCR_sentropy(
            counts=counts_a,
            similarity=similarity,
            viewpoint=viewpoint,
            measures=measures,
            symmetric=symmetric,
            X=X,
            chunk_size=chunk_size,
            parallelize=parallelize,
            max_inflight_tasks=max_inflight_tasks,
            return_dataframe=return_dataframe,
        )

    else:
        return kl_div_effno(
            P_abundance=counts_a,
            Q_abundance=counts_b,
            similarity=similarity,
            viewpoint=viewpoint,
            symmetric=symmetric,
            X=X,
            chunk_size=chunk_size,
            parallelize=parallelize,
            max_inflight_tasks=max_inflight_tasks,
            return_dataframe=return_dataframe,
        )
