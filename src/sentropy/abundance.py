# abundance.py

"""Module for calculating relative sub- and metacomunity abundances.

Classes
-------
Abundance
    Relative (normalized) species abundances in (meta-/sub-) communities
AbundanceForDiversity
    Species abundances-- normalized over set, normalized over each subset,
    and totalled across set-- as is required for diversity calculations

"""

from functools import cached_property
from typing import Iterable, Union

from numpy import arange
from pandas import DataFrame
from scipy.sparse import issparse  # type: ignore[import]

from sentropy.backend import get_backend, NumpyBackend
from sentropy.exceptions import InvalidArgumentError

# We'll avoid importing numpy or torch directly here. Use backend ops when available.


class Abundance:
    def __init__(
        self,
        counts,
        subset_names: Iterable[Union[str, int]],
        backend=None,
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subset, one row per
            species, containing the count of each species in the
            corresponding subsets.
        backend:
            An instance of a backend (from sentropy.backend.get_backend or backend class).
            If None, numpy backend is used.
        """
        self.backend = backend if backend is not None else get_backend("numpy")
        # convert counts to backend array
        self.counts = self.backend.asarray(counts) if hasattr(self.backend, "asarray") else self.backend.array(counts)
        self.subsets_names = subset_names
        self.num_subsets = self.counts.shape[1]
        # min_count : small nonzero for numerical stability
        total = self.backend.sum(self.counts)
        # guard: total could be scalar tensor -> convert to python float if needed
        try:
            total_scalar = float(total)
        except Exception:
            total_scalar = total
        # compute min_count using backend's array semantics
        # fallback to numpy small value
        self.min_count = min(1.0 / (total_scalar if total_scalar != 0 else 1.0), 1e-9)

        self.subset_abundance = self.make_subset_abundance(counts=self.counts)
        self.normalized_subset_abundance = (
            self.make_normalized_subset_abundance()
        )

    def make_subset_abundance(self, counts):
        """Calculates the relative abundances in subsets."""
        # counts / counts.sum()
        total = self.backend.sum(counts)
        # broadcasting semantics should match numpy/torch
        return counts / total

    def make_subset_normalizing_constants(self):
        """Calculates subset normalizing constants."""
        return self.backend.sum(self.subset_abundance, axis=0)

    def make_normalized_subset_abundance(self):
        """Calculates normalized relative abundances in subsets."""
        self.subset_normalizing_constants = (
            self.make_subset_normalizing_constants()
        )
        # divide by constants: ensure broadcasting
        return self.subset_abundance / self.subset_normalizing_constants

    def premultiply_by(self, similarity):
        return similarity.weighted_abundances(self.normalized_subset_abundance)


class AbundanceForDiversity(Abundance):
    """Calculates metacommuntiy and subset relative abundance
    components from a numpy.ndarray containing species counts
    """

    def __init__(
        self, counts, subset_names: Iterable[Union[str, int]], backend=None
    ) -> None:
        super().__init__(counts, subset_names, backend=backend)
        self.set_abundance = self.make_set_abundance()
        self.unified_abundance_array = None

    def unify_abundance_array(self) -> None:
        """Creates one matrix containing all the abundance matrices:
        set, subset, and normalized subset.
        """
        self.unified_abundance_array = self.backend.concatenate(
            (
                self.set_abundance,
                self.subset_abundance,
                self.normalized_subset_abundance,
            ),
            axis=1,
        )

    def get_unified_abundance_array(self):
        if self.unified_abundance_array is None:
            self.unify_abundance_array()
            # update views (these are backend arrays)
            self.set_abundance = self.unified_abundance_array[:, [0]]
            self.subset_abundance = self.unified_abundance_array[
                :, 1 : (1 + self.num_subsets)
            ]
            self.normalized_subset_abundance = self.unified_abundance_array[
                :, (1 + self.num_subsets) :
            ]
        return self.unified_abundance_array

    def premultiply_by(self, similarity):
        if similarity.is_expensive():
            print("type(similarity):", type(similarity))
            all_ordinariness = similarity.self_similar_weighted_abundances(
                self.get_unified_abundance_array()
            )
            
            set_ordinariness = all_ordinariness[:, [0]]
            subset_ordinariness = all_ordinariness[
                :, 1 : (1 + self.num_subsets)
            ]
            normalized_subset_ordinariness = all_ordinariness[
                :, (1 + self.num_subsets) :
            ]
        else:
            set_ordinariness = similarity.self_similar_weighted_abundances(
                self.set_abundance
            )
            subset_ordinariness = similarity.self_similar_weighted_abundances(
                self.subset_abundance
            )
            normalized_subset_ordinariness = (
                similarity.self_similar_weighted_abundances(
                    self.normalized_subset_abundance
                )
            )
        return (
            set_ordinariness,
            subset_ordinariness,
            normalized_subset_ordinariness,
        )

    def make_set_abundance(self):
        """Calculates the relative abundances in set."""
        return self.backend.sum(self.subset_abundance, axis=1, keepdims=True)


def make_abundance(counts, for_diversity=True, backend=None):
    """Initializes a concrete subclass of Abundance."""
    if not for_diversity:
        specific_class = Abundance
    else:
        specific_class = AbundanceForDiversity
    if isinstance(counts, DataFrame):
        arr = counts.to_numpy()
        return specific_class(
            counts=arr, subset_names=counts.columns.to_list(), backend=backend
        )
    elif hasattr(counts, "shape"):
        if issparse(counts):
            raise TypeError("sparse abundance matrix not yet implemented")
        else:
            return specific_class(
                counts=counts, subset_names=arange(counts.shape[1]), backend=backend
            )
    else:
        raise NotImplementedError(
            f"Type {type(counts)} is not supported for argument "
            "'counts'. Valid types include pandas.DataFrame or"
            "numpy.ndarray"
        )
