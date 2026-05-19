"""Module for calculating relative sub- and metacomunity abundances.

Classes
-------
Abundance
    Relative (normalized) species abundances in (meta-/sub-) communities.
    All derived quantities are computed lazily on first access and cached.
"""

from typing import Iterable, Union, Tuple, Optional

from numpy import ndarray
from torch import Tensor
from pandas import DataFrame

from sentropy.backend import get_backend, BaseBackend
from sentropy.similarity import Similarity


class Abundance:
    """Relative species abundances in (meta-/sub-) communities,
    with all components needed for diversity calculations.

    Derived quantities (subset_abundance, set_abundance, etc.) are
    computed lazily on first access and cached. This avoids unnecessary
    work when only a subset of the quantities are needed — for example,
    a Vendi score only requires normalized_subset_abundance, not
    set_abundance or subset_normalizing_constants.

    Dependency chain:
        total ← min_count
        total ← subset_abundance ← set_abundance
        subset_abundance ← subset_normalizing_constants ← normalized_subset_abundance
    """

    def __init__(
        self,
        counts: Union[ndarray, DataFrame, dict],
        subsets_names: Optional[Iterable[Union[str, int]]] = None,
        backend: Union[BaseBackend, None] = None,
    ) -> None:
        self.backend = backend if backend is not None else get_backend("numpy")
        self.counts = (
            self.backend.asarray(counts)
            if hasattr(self.backend, "asarray")
            else self.backend.array(counts)
        )
        self.subsets_names = subsets_names
        self.num_subsets = self.counts.shape[1]

        # Lazy caches — populated on first access
        self._total = None
        self._min_count = None
        self._subset_abundance = None
        self._set_abundance = None
        self._subset_normalizing_constants = None
        self._normalized_subset_abundance = None

    # --- Lazy properties ---

    @property
    def total(self):
        """Total count across all species and subsets."""
        if self._total is None:
            self._total = self.backend.sum(self.counts)
        return self._total

    @property
    def min_count(self):
        """Small nonzero value for numerical stability in power_mean."""
        if self._min_count is None:
            total_scalar = float(self.total)
            self._min_count = min(
                1.0 / (total_scalar if total_scalar != 0 else 1.0), 1e-9
            )
        return self._min_count

    @property
    def subset_abundance(self):
        """Relative abundances: counts / total (columns sum to subset weights)."""
        if self._subset_abundance is None:
            self._subset_abundance = self.counts / self.total
        return self._subset_abundance

    @property
    def set_abundance(self):
        """Metacommunity abundance: row-wise sum of subset_abundance."""
        if self._set_abundance is None:
            self._set_abundance = self.backend.sum(
                self.subset_abundance, axis=1, keepdims=True
            )
        return self._set_abundance

    @property
    def subset_normalizing_constants(self):
        """Column sums of subset_abundance (weights for aggregating subsets)."""
        if self._subset_normalizing_constants is None:
            self._subset_normalizing_constants = self.backend.sum(
                self.subset_abundance, axis=0
            )
        return self._subset_normalizing_constants

    @property
    def normalized_subset_abundance(self):
        """Subset-normalized abundances: each column sums to 1."""
        if self._normalized_subset_abundance is None:
            self._normalized_subset_abundance = (
                self.subset_abundance / self.subset_normalizing_constants
            )
        return self._normalized_subset_abundance

    # --- Similarity interaction ---

    def premultiply_by(
        self, similarity: Similarity
    ) -> Tuple[Union[ndarray, Tensor], Union[ndarray, Tensor], Union[ndarray, Tensor]]:
        """Multiply similarity matrix with all abundance vectors.

        Returns (set_ordinariness, subset_ordinariness, normalized_subset_ordinariness).
        For expensive similarities, batches the three matmuls into one.
        """
        if similarity.is_expensive():
            return self._premultiply_batched(similarity)
        return self._premultiply_separate(similarity)

    def _premultiply_separate(self, similarity: Similarity):
        """Three independent matmuls — cheap similarity (e.g. identity, array)."""
        set_ord = similarity.self_similar_weighted_abundances(self.set_abundance)
        subset_ord = similarity.self_similar_weighted_abundances(
            self.subset_abundance
        )
        norm_ord = similarity.self_similar_weighted_abundances(
            self.normalized_subset_abundance
        )
        return set_ord, subset_ord, norm_ord

    def _premultiply_batched(self, similarity: Similarity):
        """Single matmul on concatenated vectors — expensive similarity (e.g. file, function)."""
        unified = self.backend.concatenate(
            (
                self.set_abundance,
                self.subset_abundance,
                self.normalized_subset_abundance,
            ),
            axis=1,
        )
        all_ord = similarity.self_similar_weighted_abundances(unified)
        n = self.num_subsets
        return (
            all_ord[:, [0]],
            all_ord[:, 1 : (1 + n)],
            all_ord[:, (1 + n) :],
        )
