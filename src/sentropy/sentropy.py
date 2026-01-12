from typing import Union, Optional, Callable, Iterable, Tuple
from numpy import (
    inf as np_inf,
    array,
    ndarray,
    minimum,
    prod,
    power,
    zeros as np_zeros,
    log as np_log,
    sum as np_sum,
    atleast_1d,
    arange,
    column_stack,
)
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


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

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


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------

def _normalize_counts(counts):
    """Convert counts to ndarray and extract subset names."""
    if isinstance(counts, DataFrame):
        return counts.to_numpy(), counts.columns.to_list()
    elif isinstance(counts, dict):
        return column_stack(list(counts.values())), list(counts.keys())
    elif isinstance(counts, ndarray):
        if counts.ndim == 1:
            counts = counts.reshape(-1, 1)
        return counts, list(range(counts.shape[1]))


def _build_superset(
    counts,
    similarity,
    symmetric,
    sfargs,
    chunk_size,
    parallelize,
    max_inflight_tasks,
    backend,
    device,
    subsets_names,
):
    return Set(
        counts,
        similarity,
        symmetric,
        sfargs,
        chunk_size,
        parallelize,
        max_inflight_tasks,
        backend,
        device,
        subsets_names,
    )


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------

class SentropyResult:
    def __init__(self, raw_dict, subsets_names, qs, ms, level):
        self.raw_dict = raw_dict
        self.subsets_names = subsets_names
        self.qs = qs
        self.ms = ms
        self.level = level

    def __call__(self, which=None, q=None, measure=None):
        if which is None and self.level == "overall":
            which = "overall"
        if q is None and len(self.qs) == 1:
            q = self.qs[0]
        if measure is None and len(self.ms) == 1:
            m = self.ms[0]
        else:
            m = measure

        if which == "overall":
            key = f"overall_{m}_q={q}"
            if key not in self.raw_dict:
                key = f"overall_{m}_q={float(q)}"
            return self.raw_dict[key]
        else:
            key = f"subset_{m}_q={q}"
            if key not in self.raw_dict:
                key = f"subset_{m}_q={float(q)}"
            idx = list(self.subsets_names).index(which)
            return self.raw_dict[key][idx]


# ----------------------------------------------------------------------
# LCR helpers
# ----------------------------------------------------------------------

def _compute_lcr_measures(superset, qs, ms, level, eff_no):
    results = {}

    for q in qs:
        for m in ms:
            if level in ("both", "overall"):
                results[f"overall_{m}_q={q}"] = superset.set_diversity(
                    q=q, m=m, eff_no=eff_no
                )
            if level in ("both", "subset"):
                results[f"subset_{m}_q={q}"] = superset.subset_diversity(
                    q=q, m=m, eff_no=eff_no
                )
    return results


# ----------------------------------------------------------------------
# LCR Sentropy
# ----------------------------------------------------------------------

def sentropy_single_abundance(
    counts: Union[DataFrame, ndarray],
    similarity=None,
    qs=1,
    ms=MEASURES,
    symmetric=False,
    sfargs=None,
    chunk_size=10,
    parallelize=False,
    max_inflight_tasks=64,
    return_dataframe=False,
    level="both",
    eff_no=True,
    backend="numpy",
    device="cpu",
):

    counts, subsets_names = _normalize_counts(counts)
    qs = atleast_1d(qs)
    ms = atleast_1d(ms)

    superset = _build_superset(
        counts,
        similarity,
        symmetric,
        sfargs,
        chunk_size,
        parallelize,
        max_inflight_tasks,
        backend,
        device,
        subsets_names,
    )

    if return_dataframe:
        return superset.to_dataframe(qs, ms, level=level, eff_no=eff_no)

    if len(qs) == 1 and len(ms) == 1 and counts.shape[1] == 1:
        return superset.set_diversity(q=qs[0], m=ms[0], eff_no=eff_no)

    results = _compute_lcr_measures(superset, qs, ms, level, eff_no)
    return SentropyResult(results, subsets_names, qs, ms, level)


# ----------------------------------------------------------------------
# KL / Rényi divergence helpers
# ----------------------------------------------------------------------

def _exp_renyi_div(P, P_ord, Q_ord, q, atol, backend):
    ratio = P_ord/Q_ord
    if q != 1:
        return power_mean(
            order=q - 1,
            weights=P,
            items=ratio,
            atol=atol,
            backend=backend,
        )
    return backend.prod(backend.power(ratio, P))


def _compute_renyi_divergences(
    P_superset,
    Q_superset,
    q,
    m,
    level,
    eff_no,
    backend,
):
    P_set_ab = P_superset.abundance.set_abundance
    Q_set_ab = Q_superset.abundance.set_abundance

    P_set_ord = P_superset.components.set_ordinariness
    Q_set_ord = Q_superset.components.set_ordinariness

    P_norm_ab = P_superset.abundance.normalized_subset_abundance
    Q_norm_ab = Q_superset.abundance.normalized_subset_abundance

    nP = P_superset.abundance.num_subsets
    nQ = Q_superset.abundance.num_subsets

    if m == 'normalized':
        P_subset_ord = P_superset.components.normalized_subset_ordinariness
        Q_subset_ord = Q_superset.components.normalized_subset_ordinariness
    else:
        P_subset_ord = P_superset.components.subset_ordinariness
        Q_subset_ord = Q_superset.components.subset_ordinariness

    min_count = min(1 / P_set_ab.sum(), 1e-9)
    backend = P_superset.backend

    results = {}

    if level in ("both", "overall"):
        val = _exp_renyi_div(P_set_ab, P_set_ord, Q_set_ord, q, min_count, backend)
        results["overall"] = backend.log(val) if not eff_no else val

    if level in ("both", "subset"):
        mat = backend.zeros((nP, nQ))

        for i in range(nP):
            for j in range(nQ):
                mat[i, j] = _exp_renyi_div(
                    P_norm_ab[:, i],
                    P_subset_ord[:, i],
                    Q_subset_ord[:, j],
                    q,
                    min_count,
                    backend,
                )

        if not eff_no:
            mat = backend.log(mat)

        results["subset"] = mat

    return results


# ----------------------------------------------------------------------
# KL divergence front-end
# ----------------------------------------------------------------------

def sentropy_two_abundances(
    P_abundance,
    Q_abundance,
    similarity=None,
    q=1,
    m='normalized',
    symmetric=False,
    sfargs=None,
    chunk_size=10,
    parallelize=False,
    max_inflight_tasks=64,
    return_dataframe=False,
    level="both",
    eff_no=True,
    backend="numpy",
    device="cpu",
):

    P, P_names = _normalize_counts(P_abundance)
    Q, Q_names = _normalize_counts(Q_abundance)

    P_superset = Set(
        P, similarity, symmetric, sfargs,
        chunk_size, parallelize, max_inflight_tasks,
        backend, device
    )

    Q_superset = Set(
        Q, similarity, symmetric, sfargs,
        chunk_size, parallelize, max_inflight_tasks,
        backend, device
    )

    results = _compute_renyi_divergences(
        P_superset,
        Q_superset,
        q,
        m,
        level,
        eff_no,
        backend,
    )

    if return_dataframe and "subset" in results:
        results["subset"] = DataFrame(
            results["subset"],
            index=P_names,
            columns=Q_names,
        )

    if level == "both":
        return results["overall"], results["subset"]
    return results[level]



# ----------------------------------------------------------------------
# Public dispatcher.
# API note: the public API uses argument q for viewpoint(s) and m for measure(s), even though
# internally we use q and m (for a single viewpoint/measure) and qs and ms (for possibly multiple viewpoints/measures)
# ----------------------------------------------------------------------

def sentropy(
    counts_a,
    counts_b=None,
    *,
    similarity=None,
    q=1,
    measure=None,
    symmetric=False,
    sfargs=None,
    chunk_size=10,
    parallelize=False,
    max_inflight_tasks=64,
    return_dataframe=False,
    level="overall",
    eff_no=True,
    backend="numpy",
    device="cpu",
):
    if level=="class":
        level="subset"

    if counts_b is None:
        if measure is None:
            measure = 'alpha'
        return sentropy_single_abundance(
            counts=counts_a,
            similarity=similarity,
            qs=q,
            ms=measure,
            symmetric=symmetric,
            sfargs=sfargs,
            chunk_size=chunk_size,
            parallelize=parallelize,
            max_inflight_tasks=max_inflight_tasks,
            return_dataframe=return_dataframe,
            level=level,
            eff_no=eff_no,
            backend=backend,
            device=device,
        )

    q = q if isinstance(q, (int, float)) else q[0]
    if measure is None:
        measure = 'normalized'

    return sentropy_two_abundances(
        P_abundance=counts_a,
        Q_abundance=counts_b,
        similarity=similarity,
        q=q,
        m=measure,
        symmetric=symmetric,
        sfargs=sfargs,
        chunk_size=chunk_size,
        parallelize=parallelize,
        max_inflight_tasks=max_inflight_tasks,
        return_dataframe=return_dataframe,
        level=level,
        eff_no=eff_no,
        backend=backend,
        device=device,
    )

