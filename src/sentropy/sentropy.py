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

from sentropy.abundance import Abundance, joint_ordinariness
from sentropy.backend import get_backend

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

from sentropy.set import Set, build_similarity
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
    "sce", #similarity-sensitive cross-entropy
    "sre", #similarity-sensitive relative entropy
    "vendi", #vendi score
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

    superset = Set(
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
# SRE / SCE helpers
# ----------------------------------------------------------------------


def _compute_srd_from_ordinarinesses(P, P_ord, Q_ord, q, atol, backend):
    """Compute similarity-sensitive relative diversity from P abundance, P ordinariness and Q ordinariness."""
    ratio = P_ord / Q_ord
    if q != 1:
        return power_mean(
            order=q - 1,
            weights=P,
            items=ratio,
            atol=atol,
            backend=backend,
        )
    return backend.prod(backend.power(ratio, P))


def _compute_scd_from_ordinarinesses(P, Q_ord, q, atol, backend):
    """Compute similarity-sensitive cross-diversity from P abundance and Q ordinariness.
    
    Parameters
    ----------
    P : array-like
        Probability distribution (abundance)
    Q_ord : array-like
        Similarity-weighted Q (Z @ Q)
    q : float
        Viewpoint parameter
    atol : float
        Absolute tolerance for numerical stability
    backend : BaseBackend
        Computation backend
        
    Returns
    -------
    float or array-like
        Cross-entropy value
    """
    inv_Q_ord = 1/Q_ord
    if q != 1:
        return power_mean(
            order=1 - q,
            weights=P,
            items=inv_Q_ord,
            atol=atol,
            backend=backend,
        )
    return backend.prod(backend.power(inv_Q_ord, P))

def _compute_sre(
    P_abundance, Q_abundance,
    P_set_ord, Q_set_ord, P_norm_ord, Q_norm_ord,
    q, level, eff_no, backend,
):
    P_set_ab = P_abundance.set_abundance
    P_norm_ab = P_abundance.normalized_subset_abundance
    min_count = min(1 / P_set_ab.sum(), 1e-9)

    results = {}
    if level in ("both", "overall"):
        val = _compute_srd_from_ordinarinesses(P_set_ab, P_set_ord, Q_set_ord, q, min_count, backend)
        results["overall"] = backend.log(val) if not eff_no else val

    if level in ("both", "subset"):
        nP, nQ = P_norm_ab.shape[1], Q_norm_ord.shape[1]
        mat = backend.zeros((nP, nQ))
        for i in range(nP):
            for j in range(nQ):
                mat[i, j] = _compute_srd_from_ordinarinesses(
                    P_norm_ab[:, i], P_norm_ord[:, i], Q_norm_ord[:, j], q, min_count, backend,
                )
        results["subset"] = backend.log(mat) if not eff_no else mat

    return results


def _compute_sce(
    P_abundance, Q_abundance,
    P_set_ord, Q_set_ord, P_norm_ord, Q_norm_ord,
    q, level, eff_no, backend,
):
    """Compute similarity-sensitive cross-entropy between P and Q."""
    P_set_ab = P_abundance.set_abundance
    P_norm_ab = P_abundance.normalized_subset_abundance
    min_count = min(1 / P_set_ab.sum(), 1e-9)

    results = {}
    if level in ("both", "overall"):
        val = _compute_scd_from_ordinarinesses(P_set_ab, Q_set_ord, q, min_count, backend)
        results["overall"] = backend.log(val) if not eff_no else val

    if level in ("both", "subset"):
        nP, nQ = P_norm_ab.shape[1], Q_norm_ord.shape[1]
        mat = backend.zeros((nP, nQ))
        for i in range(nP):
            for j in range(nQ):
                mat[i, j] = _compute_scd_from_ordinarinesses(
                    P_norm_ab[:, i], Q_norm_ord[:, j], q, min_count, backend,
                )
        results["subset"] = backend.log(mat) if not eff_no else mat

    return results

# ----------------------------------------------------------------------
# SRE/SCE front-end
# ----------------------------------------------------------------------

def sentropy_two_abundances(
    P_abundance,
    Q_abundance,
    similarity=None,
    q=1,
    m='sre',
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

    backend_obj = get_backend(backend, device)

    P_ab = Abundance(counts=P, backend=backend_obj)
    Q_ab = Abundance(counts=Q, backend=backend_obj)

    sim = build_similarity(
        similarity=similarity,
        symmetric=symmetric,
        X=sfargs,
        chunk_size=chunk_size,
        parallelize=parallelize,
        max_inflight_tasks=max_inflight_tasks,
        backend=backend_obj,
    )

    (P_set_ord, P_subset_ord, P_norm_ord), (Q_set_ord, Q_subset_ord, Q_norm_ord) = (
        joint_ordinariness(P_ab, Q_ab, sim)
    )

    if m == 'sre':
        results = _compute_sre(
            P_ab, Q_ab, P_set_ord, Q_set_ord, P_norm_ord, Q_norm_ord,
            q, level, eff_no, backend_obj,
        )
    elif m == 'sce':
        results = _compute_sce(
            P_ab, Q_ab, P_set_ord, Q_set_ord, P_norm_ord, Q_norm_ord,
            q, level, eff_no, backend_obj,
        )

    if return_dataframe and "subset" in results:
        results["subset"] = DataFrame(
            results["subset"], index=P_names, columns=Q_names,
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
    if level == "class":
        level = "subset"

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

    else:
        q = q if isinstance(q, (int, float)) else q[0]
        if measure is None:
            measure = 'sre'
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
