"""Module for set and subset diversity measures."""

from typing import Callable, Iterable, Optional, Union, List

from pandas import DataFrame, Index, Series, concat
from numpy import array, atleast_1d, broadcast_to, zeros as np_zeros, ndarray
from sentropy.exceptions import InvalidArgumentError

from sentropy.abundance import make_abundance
from sentropy.similarity import (
    Similarity,
    SimilarityFromArray,
    SimilarityFromDataFrame,
    SimilarityIdentity,
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
    SimilarityFromFile,
)
from sentropy.ray import SimilarityFromRayFunction, SimilarityFromSymmetricRayFunction
from sentropy.components import Components
from sentropy.powermean import power_mean
from sentropy.backend import get_backend

from torch import Tensor


class Set:
    similarity: Similarity
    """Creates diversity components and calculates diversity measures."""

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
        "vendi",
    )

    def __init__(
        self,
        counts: Union[DataFrame, ndarray],
        similarity: Union[ndarray, Similarity, None] = None,
        symmetric: Optional[bool] = False,
        X: Optional[Union[ndarray, DataFrame]] = None,
        chunk_size: Optional[int] = 10,
        parallelize: Optional[bool] = False,
        max_inflight_tasks: Optional[int] = 64,
        backend: str = "numpy",
        device: Optional[str] = None,
        subsets_names: Optional[Iterable[Union[str,int]]] = None,
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subset, one row per
            species, containing the count of each species in the
            corresponding subsets.
        similarity:
            Optional. Can be:
            - None → use identity (frequency-only)
            - NumPy ndarray → similarity matrix
            - pandas DataFrame → converted to NumPy array
            - Callable[[int, int], float] → similarity function
        symmetric:
            Only relevant if similarity is callable. Indicates whether
            similarity(i,j) == similarity(j,i). Default True.
        X:
            Array of features. Only relevant if similarity is callable.
        chunk_size:
            How many rows in the similarity matrix to generate at once.
            Only relevant if similarity is callable or from file.
        parallelize:
            Whether or not to parallelize with ray.
            Only relevant when similarity is callable.
        max_inflight_tasks:
            How many inflight tasks to submit to ray at a time.
            Only relevant when similarity is callable and parallelize is True.
        backend:
            whether to use numpy or torch
        device:
            if backend is torch, whether to use cpu or gpu
        """
        # store backend instance
        self.backend = get_backend(backend, device)
        self.counts = counts
        self.abundance = make_abundance(
            counts=counts, subsets_names=subsets_names, backend=self.backend
        )
        if similarity is None:
            self.similarity = SimilarityIdentity(backend=self.backend)
        elif isinstance(similarity, ndarray):
            self.similarity = SimilarityFromArray(
                similarity=similarity, backend=self.backend
            )
        elif isinstance(similarity, DataFrame):
            self.similarity = SimilarityFromArray(
                similarity=similarity.values, backend=self.backend
            )
        elif isinstance(similarity, str):
            if chunk_size is None: # pragma: no cover
                raise ValueError("chunk_size cannot be None when similarity is a file.")
            self.similarity = SimilarityFromFile(
                similarity, chunk_size=chunk_size, backend=self.backend
            )
        elif callable(similarity):
            if X is None: # pragma: no cover
                raise ValueError("X cannot be None when similarity is a callable.")
            if chunk_size is None: # pragma: no cover
                raise ValueError(
                    "chunk_size cannot be None when similarity is a callable."
                )
            if symmetric:
                if parallelize:
                    if max_inflight_tasks is None: # pragma: no cover
                        raise ValueError(
                            "max_inflight_task cannot be None when parallelizing."
                        )
                    self.similarity = SimilarityFromSymmetricRayFunction(
                        func=similarity,
                        X=X,
                        chunk_size=chunk_size,
                        max_inflight_tasks=max_inflight_tasks,
                        backend=self.backend,
                    )
                else:
                    self.similarity = SimilarityFromSymmetricFunction(
                        func=similarity,
                        X=X,
                        chunk_size=chunk_size,
                        backend=self.backend,
                    )
            else:
                if parallelize:
                    if max_inflight_tasks is None: # pragma: no cover
                        raise ValueError(
                            "max_inflight_task cannot be None when parallelizing."
                        )
                    self.similarity = SimilarityFromRayFunction(
                        func=similarity,
                        X=X,
                        chunk_size=chunk_size,
                        max_inflight_tasks=max_inflight_tasks,
                        backend=self.backend,
                    )
                else:
                    self.similarity = SimilarityFromFunction(
                        func=similarity,
                        X=X,
                        chunk_size=chunk_size,
                        backend=self.backend,
                    )
        else:
            self.similarity = similarity

        self._components = None
        self.subset_diversity_hash: dict = {}

    @property
    def components(self):
        """Ordinariness vectors — computed on first access, then cached."""
        if self._components is None:
            self._components = Components(
                abundance=self.abundance, similarity=self.similarity
            )
        return self._components

    def _spectral_diversity(self, Z, p, q, eff_no, backend):
        """
        Internal helper to compute Vendi/Renyi spectral diversity.
        Z: Similarity matrix (n x n)
        p: Abundance vector (n,)
        q: Viewpoint parameter
        """
        # Check if we have a materialized similarity matrix
        if isinstance(self.similarity, (SimilarityFromArray, SimilarityFromDataFrame)):
            Z = self.similarity.similarity  # Get the raw matrix
        elif isinstance(self.similarity, SimilarityIdentity):
            Z = backend.identity(p.shape[0])
        else:
            raise InvalidArgumentError(
                "Vendi score requires a pre-computed similarity matrix. "
                "Please pass a numpy array or pandas DataFrame as the 'similarity' argument. "
                "Function-based or file-based similarities are not supported for Vendi scores "
                "because eigenvalue decomposition requires the full similarity matrix."
            )

        # 1. Construct Z_p = D^{1/2} Z D^{1/2}
        
        sqrt_p = backend.sqrt(p)
        outer_product = backend.outer(sqrt_p, sqrt_p)
        Z_p = backend.multiply(Z, outer_product)
        
        # 2. Compute Trace(Z_p^q) if q is integer > 1
        # Check if q is effectively an integer
        if q > 1 and backend.isclose(q, backend.round(q), atol=1e-9):
            k = int(round(q))
            # Compute Z_p^k
            # For large k, repeated squaring is better, but matrix_power is standard
            Z_p_k = backend.matrix_power(Z_p, k)
            trace_val = backend.trace(Z_p_k)
            
            # Renyi Entropy H_q = 1/(1-q) * ln(trace)
            # Vendi Score = exp(H_q) = trace^(1/(1-q))
            if eff_no:
                # Return effective number: trace^(1/(1-q))
                return backend.power(trace_val, 1.0 / (1.0 - q))
            else:
                # Return entropy: ln(trace) / (1-q)
                return backend.log(trace_val) / (1.0 - q)
                
        else:
            # Fallback to eigendecomposition for non-integers or q <= 1
            # Eigenvalues of Z_p are real and non-negative (PSD)
            eigenvalues = backend.eigvalsh(Z_p)
            
            # Filter non-zero eigenvalues for numerical stability
            # Use a small tolerance
            tol = 1e-12
            nonzero_eigs = eigenvalues[eigenvalues > tol]
            
            if len(nonzero_eigs) == 0:
                return 0.0 if eff_no else -float('inf')
                
            # Normalize to sum to 1 (though Tr(Z_p) should be 1 if p sums to 1)
            # Just in case of floating point drift
            s = backend.sum(nonzero_eigs)
            probs = nonzero_eigs / s
            
            if q == 1:
                # Shannon Entropy
                # H = - sum(p * ln(p))
                entropy = -backend.sum(backend.multiply(probs, backend.log(probs)))
                if eff_no:
                    return backend.exp(entropy)
                return entropy
            else:
                # Renyi Entropy
                # H_q = 1/(1-q) * ln(sum(p^q))
                sum_pow = backend.sum(backend.power(probs, q))
                entropy = backend.log(sum_pow) / (1.0 - q)
                if eff_no:
                    return backend.exp(entropy)
                return entropy

    def subset_diversity(
        self, q: float, m: str, eff_no: bool = True
    ) -> Union[ndarray, Tensor]:
        """Calculates subset diversity measures.

        Parameters
        ----------
        viewpoint:
            Viewpoint parameter for diversity measure.
        measure:
            Name of the diversity measure. Valid measures include:
            "alpha", "rho", "beta", "gamma", "normalized_alpha",
            "normalized_rho", and "normalized_beta"

        Returns
        -------
        A numpy.ndarray with a diversity measure for each subset.
        """
        if m not in self.MEASURES:
            raise (
                InvalidArgumentError(
                    f"Invalid measure '{m}'. "
                    "Argument 'measure' must be one of: "
                    f"{', '.join(self.MEASURES)}"
                )
            )

        if m == 'vendi':
            results = []
            for i in range(self.abundance.num_subsets):
                p = self.abundance.normalized_subset_abundance[:,i]
                val = self._spectral_diversity(self.similarity.similarity, p, q, eff_no, self.backend)
                results.append(val)
            return self.backend.array(results)

        if f"subset_{m}_q={q}" in self.subset_diversity_hash.keys():
            diversity_measure = self.subset_diversity_hash[f"subset_{m}_q={q}"]
            if eff_no == False:
                return self.backend.log(diversity_measure)
            else:
                return diversity_measure

        numerator = self.components.numerators[m]
        denominator = self.components.denominators[m]

        if m == "gamma":
            denominator = self.backend.broadcast_to(
                denominator,
                self.abundance.normalized_subset_abundance.shape,
            )

        # divide with safe handling
        ratio = self.backend.divide(numerator, denominator)

        diversity_measure = power_mean(
            order=1 - q,
            weights=self.abundance.normalized_subset_abundance,
            items=ratio,
            atol=self.abundance.min_count,
            backend=self.backend,
        )
        if m in {"beta", "normalized_beta"}:
            return 1 / diversity_measure

        if m in {"rho_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return (diversity_measure - 1) / (N - 1)

        if m in {"beta_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return ((N / diversity_measure) - 1) / (N - 1)

        self.subset_diversity_hash[f"subset_{m}_q={q}"] = diversity_measure

        if eff_no == False:
            return self.backend.log(diversity_measure)
        else:
            return diversity_measure

    def set_diversity(
        self, q: float, m: str, eff_no: bool = True
    ) -> Union[ndarray, Tensor]:
        """Calculates set diversity measures.

        Parameters
        ----------
        viewpoint:
            Viewpoint parameter for diversity measure.
        measure:
            Name of the diversity measure. Valid measures include:
            "alpha", "rho", "beta", "gamma", "normalized_alpha",
            "normalized_rho", and "normalized_beta"

        Returns
        -------
        A numpy.ndarray containing the set diversity measure.
        """
        subset_diversity = self.subset_diversity(
            q, m, eff_no=True
        )  # note: eff_no must be True here !
        diversity_measure = power_mean(
            1 - q,
            self.abundance.subset_normalizing_constants,
            subset_diversity,
            backend=self.backend,
        )

        if eff_no == False:
            return self.backend.log(diversity_measure).item()
        else:
            return diversity_measure.item()

    def subsets_to_dataframe(self, q: float, ms=MEASURES, eff_no: bool = True):
        """Table containing all subset diversity values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contribute.

        Returns
        -------
        A pandas.DataFrame containing all subset diversity
        measures for a given viewpoint
        """
        data = {}

        for m in ms:
            val = self.subset_diversity(q, m, eff_no)

            if isinstance(val, Tensor): #pragma: no cover
                val = val.cpu()

            data[m] = val

        df = DataFrame(data)
        df.insert(0, "viewpoint", q)
        df.insert(0, "level", Series(self.abundance.subsets_names))
        return df

    def set_to_dataframe(self, q: float, ms=MEASURES, eff_no: bool = True):
        """Table containing all set diversity values.
        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contributes.

        Returns
        -------
        A pandas.DataFrame containing all set diversity
        measures for a given viewpoint
        """

        data = {}

        for m in ms:
            val = self.set_diversity(q, m, eff_no)

            if isinstance(val, Tensor): #pragma: no cover
                val = val.cpu()

            data[m] = val

        df = DataFrame(
            data,
            index=Index(["overall"], name="level"),
        )

        df.insert(0, "viewpoint", q)
        df.reset_index(inplace=True)

        return df

    def to_dataframe(
        self, qs: Iterable[float], ms=MEASURES, level: str = "both", eff_no: bool = True
    ):
        """Table containing all set and subset diversity
        values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contributes.

        Returns
        -------
        A pandas.DataFrame containing all set and subset
        diversity measures for a given viewpoint
        """
        dataframes = []
        for q in qs:
            if level in ["both", "overall"]:
                dataframes.append(self.set_to_dataframe(q=q, ms=ms, eff_no=eff_no))
            if level in ["both", "subset"]:
                dataframes.append(self.subsets_to_dataframe(q=q, ms=ms, eff_no=eff_no))
        return concat(dataframes).reset_index(drop=True)
