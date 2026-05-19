"""Module for calculating relative sub- and metacomunity abundances."""

from typing import Iterable, Union, Tuple, Optional

from numpy import ndarray
from torch import Tensor
from pandas import DataFrame

from sentropy.backend import get_backend, BaseBackend
from sentropy.similarity import Similarity


class Abundance:
    """Relative species abundances in (meta-/sub-) communities,
    with all components needed for diversity calculations."""

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

        total = self.backend.sum(self.counts)
        total_scalar = float(total)
        self.min_count = min(
            1.0 / (total_scalar if total_scalar != 0 else 1.0), 1e-9
        )

        # Core abundance quantities
        self.subset_abundance = self._make_subset_abundance()
        self.set_abundance = self._make_set_abundance()
        self.normalized_subset_abundance = self._make_normalized_subset_abundance()

    # --- Construction helpers ---

    def _make_subset_abundance(self) -> Union[ndarray, Tensor]:
        """Relative abundances: counts / total."""
        return self.counts / self.backend.sum(self.counts)

    def _make_set_abundance(self) -> Union[ndarray, Tensor]:
        """Metacommunity abundance: row-wise sum of subset_abundance."""
        return self.backend.sum(self.subset_abundance, axis=1, keepdims=True)

    def _make_normalized_subset_abundance(self) -> Union[ndarray, Tensor]:
        """Subset-normalized abundances: each column sums to 1."""
        self.subset_normalizing_constants = self.backend.sum(
            self.subset_abundance, axis=0
        )
        return self.subset_abundance / self.subset_normalizing_constants

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
