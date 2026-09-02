"""Spectral similarity-sensitive diversity measures.

Currently implements the Vendi score: the Rényi entropy of the
eigenvalue spectrum of the abundance-weighted similarity matrix
D^(1/2) Z D^(1/2), optionally expressed as an effective number.

Unlike the LCR measures in sentropy.set, spectral measures require a
materialized similarity matrix, since an eigendecomposition cannot be
performed row-by-row or in distributed chunks. They are therefore kept
out of the LCR core and consume only Abundance and Similarity.
"""

from typing import Optional, Union

from numpy import ndarray, isclose, isinf
from pandas import DataFrame
from torch import Tensor

from sentropy.abundance import Abundance, normalize_counts
from sentropy.backend import get_backend
from sentropy.exceptions import InvalidArgumentError
from sentropy.set import build_similarity
from sentropy.similarity import SimilarityFromArray, SimilarityIdentity


class SpectralError(InvalidArgumentError):
    """Raised when a spectral measure cannot be computed."""


def _dense_similarity(similarity, n: int, backend) -> ndarray:
    """Materialize the similarity as a dense backend array.

    Raises SpectralError if the similarity is defined by a function or
    a file, since the eigendecomposition needs the full matrix.
    """
    if isinstance(similarity, SimilarityFromArray):
        sim = similarity.similarity
        if hasattr(sim, "toarray"):  # scipy sparse
            sim = backend.asarray(sim.toarray())
        return sim
    if isinstance(similarity, SimilarityIdentity):
        return backend.identity(n)
    raise SpectralError(
        "Vendi scores require a materialized similarity matrix "
        "(numpy array, pandas DataFrame, or scipy sparse matrix). "
        "Function- and file-based similarities are not supported "
        "because the eigendecomposition needs the full matrix. "
        "If your similarity is defined by a function over features, "
        "evaluate it into an array first and pass that array as "
        "`similarity`."
    )


def _spectral_entropy(Z, p, q: float, backend) -> float:
    """(Possibly Rényi-ordered) entropy of the spectrum of
    D^(1/2) Z D^(1/2).

    q = 1 gives the Shannon-spectrum entropy of the Vendi score;
    other finite q give the Rényi generalization. Following the
    package-wide convention (cf. parameters.ValidateViewpoint),
    q > 100 is treated as q = infinity, which selects only the
    largest eigenvalue: H_inf = -log(max prob).
    """
    sqrt_p = backend.sqrt(p)
    Z_p = backend.multiply(Z, backend.outer(sqrt_p, sqrt_p))
    eigenvalues = backend.eigvalsh(Z_p)

    probs = eigenvalues[eigenvalues > 0]
    if len(probs) == 0:  # backend-neutral (numpy: size attr; torch: numel)
        raise SpectralError(
            "No positive eigenvalues in the abundance-weighted similarity "
            "matrix; cannot compute a Vendi score."
        )

    if q > 100 or isinf(q):
        # Rényi entropy of order ∞ = -log(max_i p_i)
        return -backend.log(backend.amax(probs))
    if isclose(q, 1.0):
        return -backend.sum(backend.multiply(probs, backend.log(probs)))
    sum_pow = backend.sum(backend.power(probs, q))
    return backend.log(sum_pow) / (1.0 - q)


def vendi_score(
    counts: Union[ndarray, DataFrame, dict],
    similarity=None,
    q: float = 1,
    level: str = "both",
    eff_no: bool = True,
    return_dataframe: bool = False,
    backend: str = "numpy",
    device: Optional[str] = None,
    subsets_names=None,
):
    """Compute Vendi score(s) for a metacommunity and/or each subset.

    Parameters
    ----------
    counts : array-like
        One column per subset, one row per species.
    similarity : None, ndarray, DataFrame, or sparse matrix
        Pairwise species similarity. Must be materializable to a dense
        matrix. None uses the identity (Vendi then reduces to a
        frequency-only Hill number).
    q : float
        Rényi order. q=1 gives the standard (Shannon-spectrum) Vendi
        score; other values generalize it. Values > 100 (and inf)
        are computed analytically as the order-inf limit.
    level : {'both', 'overall', 'subset'}
    eff_no : bool
        True returns effective numbers (exp of the entropy); False
        returns the raw entropy.
    """
    backend_obj = get_backend(backend, device)
    counts_arr, names = normalize_counts(counts)
    if subsets_names is None:
        subsets_names = names

    # Reject the representations we can never support, with a clear message.
    if callable(similarity) or isinstance(similarity, str):
        raise SpectralError(
            "Vendi scores require a materialized similarity matrix "
            "(numpy array, pandas DataFrame, or scipy sparse matrix). "
            "Function- and file-based similarities are not supported "
            "because the eigendecomposition needs the full matrix."
        )

    sim_obj = build_similarity(
        similarity=similarity,
        symmetric=False,
        X=None,
        chunk_size=10,
        parallelize=False,
        backend=backend_obj,
    )
    Z = _dense_similarity(sim_obj, counts_arr.shape[0], backend_obj)

    abundance = Abundance(
        counts=counts_arr, subsets_names=subsets_names, backend=backend_obj
    )

    def _effective(p):
        entropy = _spectral_entropy(Z, p, q, backend_obj)
        return backend_obj.exp(entropy) if eff_no else entropy

    results = {}
    if level in ("both", "overall"):
        # NOTE: this is the Vendi score of the pooled metacommunity.
        results["overall"] = _effective(abundance.set_abundance[:, 0])
    if level in ("both", "subset"):
        vals = [
            _effective(abundance.normalized_subset_abundance[:, i])
            for i in range(abundance.num_subsets)
        ]
        results["subset"] = backend_obj.array(vals)

    if return_dataframe:
        return _to_dataframe(results, subsets_names, q)
    if level == "overall":
        return results["overall"]
    if level == "subset":
        return results["subset"]
    return results


def _to_dataframe(results, subsets_names, q) -> DataFrame:
    rows = []
    if "overall" in results:
        rows.append(("overall", _scalar(results["overall"])))
    if "subset" in results:
        for name, val in zip(subsets_names, results["subset"]):
            rows.append((name, _scalar(val)))
    return DataFrame(
        [(lvl, q, val) for lvl, val in rows],
        columns=["level", "viewpoint", "vendi"],
    )


def _scalar(x):
    if isinstance(x, Tensor):  # pragma: no cover
        return x.item()
    return float(x)