from typing import Union
from numpy import (
    ndarray,
    atleast_1d,
)
from pandas import DataFrame

from sentropy.abundance import Abundance, joint_ordinariness, normalize_counts as _normalize_counts
from sentropy.backend import get_backend
from sentropy.exceptions import InvalidArgumentError


from sentropy.set import Set, build_similarity
from sentropy.spectral import vendi_score
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
# Result container
# ----------------------------------------------------------------------


class SentropyResult:
    """Container for diversity results, keyed by level, measure, and viewpoint.

    Returned by LCR() when more than one viewpoint, measure, or subset
    is requested and `return_dataframe` is False. Retrieve individual
    results by calling the object (see `__call__`).

    Attributes
    ----------
    raw_dict : dict
        Maps result keys such as ``"overall_{m}_q={q}"`` and
        ``"subset_{m}_q={q}"`` to their diversity values.
    subsets_names : list
        Names of the subsets (columns of the original counts).
    qs : list
        Viewpoint parameters that were computed.
    ms : list
        Measures that were computed.
    level : {'overall', 'subset', 'both'}
        Level(s) at which the results were computed.
    """
    def __init__(self, raw_dict, subsets_names, qs, ms, level):
        self.raw_dict = raw_dict
        self.subsets_names = subsets_names
        self.qs = qs
        self.ms = ms
        self.level = level

    def __call__(self, which=None, q=None, measure=None):
        """Retrieve a single diversity value from the stored results.

        Convenience accessor allowing a SentropyResult to be called
        directly, e.g. ``result("overall", q=2, measure="alpha")`` or,
        when subset-level results were computed, ``result("gut")`` to
        get the value for the subset named ``"gut"``.

        Parameters
        ----------
        which : str, optional
            Either ``"overall"`` or the name of a subset. Defaults to
            ``"overall"`` when ``self.level == "overall"``; for level
            ``"subset"`` or ``"both"`` a subset name must be supplied
            (an omitted ``which`` is otherwise interpreted as a subset
            name and raises ValueError).
        q : float, optional
            Viewpoint parameter. Defaults to the single computed
            viewpoint when exactly one was requested (``self.qs``);
            must be given otherwise.
        measure : str, optional
            Measure name (see `MEASURES`). Defaults to the single
            computed measure when exactly one was requested
            (``self.ms``); must be given otherwise.

        Returns
        -------
        float or array-like
            For ``which="overall"``, the set-level value stored under
            the key ``"overall_{measure}_q={q}"``. For a subset name,
            the per-subset value (entry at the position of `which` in
            `subsets_names`) from the array stored under
            ``"subset_{measure}_q={q}"``. If the key built with `q` as
            passed is absent, the lookup is retried with the viewpoint
            formatted as a float (e.g. ``"q=1.0"``).

        Raises
        ------
        KeyError
            If the requested (which, measure, q) combination was not
            computed — e.g. several viewpoints or measures were
            requested and `q` or `measure` is not given explicitly.
        ValueError
            If `which` is not a name in `subsets_names` (raised by
            ``list.index``).
        """
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
    """Compute one-community (LCR) diversity measures for every (q, m) pair.

    Loops over viewpoints × measures and delegates to
    `Set.set_diversity` / `Set.subset_diversity`, which compute and
    cache the underlying components once per Set.

    Parameters
    ----------
    superset : Set
        The community whose diversities are computed.
    qs : iterable of float
        Viewpoint parameters.
    ms : iterable of str
        Measure names (subset of `Set.MEASURES`).
    level : {'both', 'overall', 'subset'}
        Which level(s) to compute.
    eff_no : bool
        If True, return effective numbers; if False, entropies.

    Returns
    -------
    dict
        Maps ``"overall_{m}_q={q}"`` and/or ``"subset_{m}_q={q}"`` to
        the corresponding diversity value (scalar for overall entries,
        per-subset array for subset entries).
    """
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


def LCR(
    counts: Union[DataFrame, ndarray],
    similarity=None,
    qs=1,
    ms='alpha',
    symmetric=False,
    sfargs=None,
    chunk_size=10,
    parallelize= True,
    return_dataframe=False,
    level="overall",
    eff_no=True,
    backend="numpy",
    device="cpu",
):
    """Calculate one-set Leinster-Cobbold-Reeve diversity measures.

    Parameters
    ----------
    counts : pandas.DataFrame, numpy.ndarray, dict, or torch.Tensor
        Counts with one row per species and one column per subset
        (a 1-D input is treated as a single subset). For a DataFrame
        the column names become the subset names; for a dict its keys
        do.
    similarity : None, ndarray, DataFrame, scipy sparse matrix, str, or callable, optional
        Pairwise species similarity. None uses the identity
        (frequency-only diversity). A str is a path to a csv/tsv file
        (square similarity matrix, streamed `chunk_size` rows at a
        time). A callable takes two rows of `sfargs` and returns their
        similarity; decorate a vectorizable callable with
        `batch_similarity` to compute whole chunks in one call.
    qs : float or iterable of float, default=1
        Viewpoint parameter(s), controlling the contribution of rare
        species (q=0: all species weigh equally; q→inf: only the most
        similar/frequent dominate). Viewpoints > 100 are treated as
        infinity.
    ms : str or iterable of str, default='alpha'
        Measure(s) to compute: "alpha", "rho", "beta", "gamma",
        "normalized_alpha", "normalized_rho", "normalized_beta",
        "rho_hat", "beta_hat".
    symmetric : bool, default=False
        Whether the callable similarity satisfies
        similarity(i, j) == similarity(j, i). Only relevant when
        `similarity` is a callable; symmetric functions are evaluated
        roughly half as many times.
    sfargs : array-like, optional
        Feature array X (one row per species) on which a callable
        similarity operates.
    chunk_size : int, default=10
        Number of rows of the similarity matrix to produce at once.
        Only relevant when `similarity` is a file or a callable.
    parallelize : bool, default=True
        Whether to parallelize a callable similarity with Ray. Only
        relevant when `similarity` is a callable.
    return_dataframe : bool, default=False
        If True, return a tidy pandas.DataFrame with columns level,
        viewpoint, and one column per measure, instead of the
        container below.
    level : {'overall', 'subset', 'both'}, default="overall"
        Which level(s) to compute.
    eff_no : bool, default=True
        If True, return effective numbers; if False, their natural
        logarithms (entropies).
    backend : {'numpy', 'torch'}, default="numpy"
        Computational backend.
    device : {'cpu', 'cuda', 'mps'}, default="cpu"
        Device for the torch backend.

    Returns
    -------
    pandas.DataFrame
        If `return_dataframe` is True.
    float
        If exactly one viewpoint, one measure, and one subset were
        requested (the bare set-level diversity).
    SentropyResult
        Otherwise; index results with the ``(which, q, measure)``
        call syntax of `SentropyResult`.
    """

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
    """Compute similarity-sensitive relative diversities/sentropies between P and Q.

    From P's abundances and the precomputed ordinarinesses of P and Q,
    evaluates the similarity-sensitive analogue of relative
    diversity/entropy, following https://arxiv.org/abs/1404.6520. The
    per-element ratio P_ordinariness / Q_ordinariness is aggregated as
    a weighted power mean of order q − 1 (or, for q = 1, as
    prod(ratio ** P)).

    Parameters
    ----------
    P_abundance, Q_abundance : Abundance
        Abundance objects of the two communities (P supplies the
        weighting distributions).
    P_set_ord, Q_set_ord : array-like
        Set-level ordinarinesses (Z @ set_abundance) of P and Q.
    P_norm_ord, Q_norm_ord : array-like
        Ordinarinesses of P's and Q's normalized subsets
        (Z @ normalized_subset_abundance), one column per subset.
    q : float
        Viewpoint parameter.
    level : {'both', 'overall', 'subset'}
        Which level(s) to compute.
    eff_no : bool
        If True, return effective numbers; if False, their natural
        logarithms.
    backend : BaseBackend
        Computation backend.

    Returns
    -------
    dict
        Keys "overall" (scalar) and/or "subset" (matrix of shape
        (n_subsets_P, n_subsets_Q), one relative diversity per pair of
        P-subset and Q-subset).
    """
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
    parallelize=True,
    return_dataframe=False,
    level="overall",
    eff_no=True,
    backend="numpy",
    device="cpu",
):
    """Calculate a similarity-sensitive relative or cross entropy between two communities.

    Front-end for the two-input measures: "sre" (similarity-sensitive
    relative entropy, P relative to Q) and "sce" (similarity-sensitive
    cross-entropy). Both communities must describe the same species
    (rows) so that one shared similarity applies; the ordinarinesses
    of both are computed jointly in a single pass over the similarity
    matrix, so an expensive similarity is evaluated only once.

    Parameters
    ----------
    P_abundance, Q_abundance : pandas.DataFrame, numpy.ndarray, dict, or torch.Tensor
        Counts for the two communities, one row per species and one
        column per subset (a 1-D input is treated as a single
        subset). Accepted types are the same as `counts` in LCR().
    similarity : optional
        As in LCR(); applies to the species shared by both inputs.
    q : float, default=1
        Viewpoint parameter.
    m : {'sre', 'sce'}, default='sre'
        Measure: similarity-sensitive relative entropy or
        similarity-sensitive cross-entropy.
    symmetric : bool, default=False
        As in LCR(); only relevant for a callable similarity.
    sfargs : array-like, optional
        As in LCR(); feature array on which a callable similarity
        operates.
    chunk_size : int, default=10
        As in LCR().
    parallelize : bool, default=True
        As in LCR().
    return_dataframe : bool, default=False
        If True (and subset-level results were requested), the subset
        result is returned as a pandas.DataFrame with rows indexed by
        P's subset names and columns labelled by Q's.
    level : {'overall', 'subset', 'both'}, default="overall"
        Which level(s) to compute.
    eff_no : bool, default=True
        If True, return effective numbers; if False, their natural
        logarithms.
    backend : {'numpy', 'torch'}, default="numpy"
        Computational backend.
    device : {'cpu', 'cuda', 'mps'}, default="cpu"
        Device for the torch backend.

    Returns
    -------
    tuple
        (overall, subset) when level == "both"; the subset result is
        a DataFrame if return_dataframe is True.
    scalar, ndarray, or DataFrame
        The result for the single requested level otherwise.
    """
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
    parallelize=True,
    return_dataframe=False,
    level="overall",
    eff_no=True,
    backend="numpy",
    device="cpu",
):
    """Calculate similarity-sensitive diversity measures.

    Main public entry point of the package; dispatches on the number
    of inputs.

    - One input (counts_a only): calculates one-community measures
      (default "alpha") via LCR(). The measure "vendi" (Vendi score)
      is routed to vendi_score instead and cannot be combined with
      other measures in the same call, as it has different result
      semantics.
    - Two inputs (counts_a and counts_b): calculates an inter-community
      measure — "sre" (similarity-sensitive relative entropy, the
      default) or "sce" (similarity-sensitive cross-entropy) — via
      sentropy_two_abundances(). Exactly one measure, and a single
      viewpoint q, is accepted.

    Parameters
    ----------
    counts_a : pandas.DataFrame, numpy.ndarray, dict, or torch.Tensor
        Counts with one row per species and one column per subset. See
        LCR() for the accepted shapes.
    counts_b : same types as counts_a, optional
        Counts of the second community, over the same species as
        counts_a. Providing it selects the two-input path (sre/sce).
    similarity : None, ndarray, DataFrame, scipy sparse matrix, str, or callable, optional
        Pairwise species similarity. None uses the identity
        (frequency-only diversity). A str is a path to a csv/tsv
        similarity matrix; a callable must be paired with `sfargs`.
        See LCR() for details. (Vendi scores require a materialized
        matrix: array, DataFrame, or sparse.)
    q : float or iterable of float, default=1
        Viewpoint parameter(s). Viewpoints > 100 (and inf) are
        treated as infinity. Reduced to its first element in the
        two-input path.
    measure : str or iterable of str, optional
        Measure(s) to compute. Defaults to "alpha" (one input) or
        "sre" (two inputs). One input: any of the LCR measures, or
        "vendi" alone. Two inputs: exactly one of "sre", "sce".
    symmetric : bool, default=False
        Whether a callable similarity is symmetric. Only relevant when
        `similarity` is a callable.
    sfargs : array-like, optional
        Feature array X (one row per species) on which a callable
        similarity operates.
    chunk_size : int, default=10
        Rows of the similarity matrix to read/generate at once. Only
        relevant when `similarity` is a file or a callable.
    parallelize : bool, default=True
        Whether to parallelize a callable similarity with Ray.
    return_dataframe : bool, default=False
        Whether to return a tidy pandas.DataFrame (one-input path) or
        DataFrames for subset-level results (two-input path) instead of
        the raw containers.
    level : {'overall', 'subset', 'both'}, default="overall"
        Which level(s) to compute. The value "class" is accepted as
        an alias of "subset" and remapped accordingly.
    eff_no : bool, default=True
        If True, return effective numbers; if False, entropies
        (natural logarithms).
    backend : {'numpy', 'torch'}, default="numpy"
        Computational backend.
    device : {'cpu', 'cuda', 'mps'}, default="cpu"
        Device for the torch backend.

    Returns
    -------
    Depends on the path taken:

    - One input, vendi: vendi_score()'s return (scalar, array, dict,
      or DataFrame depending on level/return_dataframe).
    - One input, LCR: pandas.DataFrame if return_dataframe; a bare
      scalar if a single viewpoint, measure, and subset were
      requested; otherwise a SentropyResult.
    - Two inputs: (overall, subset) tuple if level == "both",
      otherwise the single-level result; subset results become
      DataFrames when return_dataframe is True.
    """
    if level == "class":
        level = "subset"

    if counts_b is None:
        if measure is None:
            measure = 'alpha'
        ms = atleast_1d(measure)
        if "vendi" in ms:
            if len(ms) > 1:
                raise InvalidArgumentError("Measure 'vendi' cannot be combined with LCR measures \
                    in a single call; it has different result semantics.")
            else:
                return vendi_score(
                counts=counts_a,
                similarity=similarity,
                q=q if isinstance(q, (int, float)) else q[0],
                level=level,
                eff_no=eff_no,
                return_dataframe=return_dataframe,
                backend=backend,
                device=device,
            )
        else:
            return LCR(
            counts=counts_a,
            similarity=similarity,
            qs=q,
            ms=measure,
            symmetric=symmetric,
            sfargs=sfargs,
            chunk_size=chunk_size,
            parallelize=parallelize,
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
        return_dataframe=return_dataframe,
        level=level,
        eff_no=eff_no,
        backend=backend,
        device=device,
    )

def interset_ordinariness(
    X,
    Y,
    Y_abundance,
    similarity,
    backend="numpy",
    device="cpu",
):
    """
    Calculate the ordinariness of elements of X with respect to Y.

    Y_abundance is interpreted as counts and normalized so that each
    abundance distribution sums to one.

    For each x_i in X, computes

        ordinariness[i] = sum_j Z(x_i, y_j) * p_j

    where p_j is the normalized abundance of y_j in Y.

    All array operations — including conversion of X, Y, and
    Y_abundance to the backend's framework — go through the backend
    object. Consequence: with backend="torch", non-tensor inputs are
    deduplicated as tensors, and the rows passed to `similarity`
    are tensor rows; with backend="numpy" (default) everything is
    numpy, as before.

    Parameters
    ----------
    X : numpy.ndarray, pandas.DataFrame, or torch.Tensor
        Feature vectors for the elements whose ordinariness is being
        calculated. One row per element.

    Y : numpy.ndarray, pandas.DataFrame, or torch.Tensor
        Feature vectors for the reference set. One row per element.

    Y_abundance : array-like
        Counts for the elements in Y. May be one-dimensional with
        shape (n_Y,) or two-dimensional with shape
        (n_Y, n_distributions).

        For a two-dimensional array, each column is normalized
        independently to sum to one.

    similarity : callable
        Function taking one row from X and one row from Y and
        returning their similarity. Rows are passed in the backend's
        array type.

    backend : str, default="numpy"
        Computational backend.

    device : str, default="cpu"
        Device used by the computational backend.

    Returns
    -------
    ndarray or torch.Tensor
        If Y_abundance is one-dimensional, returns an array with
        shape (n_X,). If Y_abundance is two-dimensional, returns an
        array with shape (n_X, n_distributions). The result type
        matches the backend.
    """
    backend = get_backend(backend, device=device)

    # Convert abundance counts to the backend's array type.
    Y_abundance = backend.asarray(Y_abundance)

    # Fail loudly on misaligned inputs instead of scattering counts
    # against the wrong rows.
    n_Y = Y.shape[0]
    if Y_abundance.shape[0] != n_Y:
        raise InvalidArgumentError(
            f"Y_abundance has {Y_abundance.shape[0]} entries but Y has "
            f"{n_Y} rows; they must correspond one-to-one."
        )

    # Keep track of whether the user supplied a single distribution.
    one_dimensional = Y_abundance.ndim == 1

    if one_dimensional:
        Y_abundance = Y_abundance.reshape(-1, 1)

    # Deduplicate rows via the backend (previously np.unique on both,
    # which copied CPU tensors to host memory and raised for cuda/mps
    # tensors).
    X_unique, X_inverse = backend.unique_rows(X, return_inverse=True)
    Y_unique, Y_inverse = backend.unique_rows(Y, return_inverse=True)

    # Sum counts for duplicate elements. Stays on the backend's dtype
    # and device (previously numpy float64 regardless of backend).
    Y_abundance_unique = backend.index_add(
        index=Y_inverse,
        source=Y_abundance,
        out_shape=(Y_unique.shape[0], Y_abundance.shape[1]),
    )

    # Normalize counts independently for each abundance distribution.
    abundance_totals = backend.sum(Y_abundance_unique, axis=0)
    Y_abundance_unique = Y_abundance_unique / abundance_totals

    try:
        from sentropy.ray import (
            _interset_weighted_abundances_ray,
            _recommend_chunk_params,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "interset_ordinariness() parallelizes with Ray, but ray "
            "could not be imported. Install it with `pip install ray` "
            "(or `pip install sentropy[ray]` if you declare it as an "
            "optional extra)."
        ) from exc

    chunk_size, max_inflight_tasks = _recommend_chunk_params(
        X.shape[0], Y.shape[0]
    )

    result_unique = _interset_weighted_abundances_ray(
        similarity=similarity,
        X=X_unique,
        Y=Y_unique,
        relative_abundance=Y_abundance_unique,
        chunk_size=chunk_size,
        max_inflight_tasks=max_inflight_tasks,
        backend=backend,
    )

    # Gather back per-element results (previously result_unique[
    # X_inverse], which relied on implicit numpy-index-on-tensor
    # coercion).
    result = backend.take(result_unique, X_inverse)
    if one_dimensional:
        result = result[:, 0]

    return result
