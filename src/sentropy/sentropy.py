from typing import Union, Optional, Callable, Iterable, Tuple
from numpy import inf as np_inf, ndarray, minimum, prod, power, zeros as np_zeros, log as np_log, sum as np_sum
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
from sentropy.powermean import power_mean

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
    return_dataframe: bool = False,
    which: str = 'both',
    eff_no: bool = True,
    backend: str = 'numpy',
    device: str = 'cpu',
    ):

    superset = Set(counts, similarity, symmetric, X, chunk_size, parallelize, max_inflight_tasks, backend, device)
    
    if return_dataframe:
        sentropies = superset.to_dataframe(viewpoint, measures, which=which, eff_no=eff_no)
    else:
        sentropies = {}
        for q in viewpoint:
            for measure in measures:
                if which in ["both", "set"]:
                    sentropies[f'set_{measure}_q={q}'] = superset.set_diversity(viewpoint=q, measure=measure, eff_no=eff_no)
                if which in ["both", "subset"]:
                    sentropies[f'subset_{measure}_q={q}'] = superset.subset_diversity(viewpoint=q, measure=measure, eff_no=eff_no)
    return sentropies

def kl_div_effno(P_abundance, Q_abundance, similarity=None, viewpoint=1, symmetric=False, X=None, chunk_size=10, \
    parallelize=False, max_inflight_tasks=64, return_dataframe=False, which='both', eff_no=True, backend='numpy', device='cpu'):
    P_superset = Set(P_abundance, similarity, symmetric, X, chunk_size, parallelize, max_inflight_tasks, backend, device)
    Q_superset = Set(Q_abundance, similarity, symmetric, X, chunk_size, parallelize, max_inflight_tasks, backend, device)
    P_set_ab = P_superset.abundance.set_abundance
    Q_set_ab = Q_superset.abundance.set_abundance
    P_norm_subset_ab =  P_superset.abundance.normalized_subset_abundance
    Q_norm_subset_ab =  Q_superset.abundance.normalized_subset_abundance
    P_set_ord = P_superset.components.set_ordinariness
    Q_set_ord = Q_superset.components.set_ordinariness
    P_norm_subset_ord = P_superset.components.normalized_subset_ordinariness
    Q_norm_subset_ord = Q_superset.components.normalized_subset_ordinariness

    P_num_subsets = P_abundance.shape[1]
    Q_num_subsets = Q_abundance.shape[1]

    if type(P_abundance)==DataFrame:
        P_subsets_names = P_abundance.columns
        P_abundance = P_abundance.to_numpy()
    else:
        P_subsets_names = [str(i) for i in range(P_num_subsets)]
        
    if type(Q_abundance)==DataFrame:
        Q_subsets_names = Q_abundance.columns
        Q_abundance = Q_abundance.to_numpy()
    else:
        Q_subsets_names = [str(i) for i in range(Q_num_subsets)]

    min_count = minimum(1 / P_abundance.sum(), 1e-9)

    def get_exp_renyi_div_from_ords(P, P_ord, Q_ord, viewpoint, atol, backend):
        ord_ratio = P_ord/Q_ord
        if viewpoint != 1:
            exp_renyi_div = power_mean(
                order=viewpoint-1,
                weights=P,
                items=ord_ratio,
                atol=atol,
                backend=P_superset.backend,
            )
        else:
            exp_renyi_div = P_superset.backend.prod(P_superset.backend.power(ord_ratio, P))
        return exp_renyi_div

    if which in ["both", "set"]:
        exp_renyi_div_set = get_exp_renyi_div_from_ords(P_set_ab, P_set_ord, Q_set_ord, viewpoint, min_count, backend)
        if eff_no == False:
            exp_renyi_div_set = P_superset.backend.log(exp_renyi_div_set)

    if which in ["both", "subset"]:
        exp_renyi_divs_subset = np_zeros(shape=(P_num_subsets, Q_num_subsets))
        for i in range(P_num_subsets):
            for j in range(Q_num_subsets):
                P = P_norm_subset_ab[:,i]
                P_ord = P_norm_subset_ord[:,i]
                Q_ord = Q_norm_subset_ord[:,j]
                exp_renyi_div = get_exp_renyi_div_from_ords(P, P_ord, Q_ord, viewpoint, min_count, backend)
                exp_renyi_divs_subset[i,j] = exp_renyi_div

        if return_dataframe:
            exp_renyi_divs_subset = DataFrame(exp_renyi_divs_subset, columns=Q_subsets_names, \
                index=P_subsets_names)

        if eff_no == False:
            exp_renyi_divs_subset = P_superset.backend.log(exp_renyi_divs_subset)

    if which=="both":
        return exp_renyi_div_set, exp_renyi_divs_subset
    elif which=="set":
        return exp_renyi_div_set
    elif which=="subset":
        return exp_renyi_divs_subset


# ----------------------------------------------------------------------
# Relative‑sentropy dispatcher
# ----------------------------------------------------------------------
def relative_sentropy(
    counts_a: Union[DataFrame, ndarray],
    counts_b: Optional[Union[DataFrame, ndarray]] = None,
    *,
    similarity: Optional[Union[ndarray, DataFrame, str, Callable]] = None,
    viewpoint: float = 1,
    measures: Iterable[str] = MEASURES,
    symmetric: bool = False,
    X: Optional[Union[ndarray, DataFrame]] = None,
    chunk_size: int = 10,
    parallelize: bool = False,
    max_inflight_tasks: int = 64,
    return_dataframe: bool = False,
    which: str = 'both',
    eff_no: bool = True,
    backend: str = 'numpy',
    device: str = 'cpu',
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
            which=which,
            eff_no = eff_no,
            backend = backend,
            device = device,
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
            which=which,
            eff_no = eff_no,
            backend = backend,
            device = device,
        )
