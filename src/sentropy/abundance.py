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

from numpy import arange, ndarray, concatenate, minimum
from pandas import DataFrame, RangeIndex
from scipy.sparse import issparse  # type: ignore[import]


class Abundance:
    def __init__(
        self, counts: ndarray, subset_names: Iterable[Union[str, int]]
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subset, one row per
            species, containing the count of each species in the
            corresponding subsets.
        """
        self.subsets_names = subset_names
        self.num_subsets = counts.shape[1]
        self.min_count = minimum(1 / counts.sum(), 1e-9)

        self.subset_abundance = self.make_subset_abundance(counts=counts)
        self.normalized_subset_abundance = (
            self.make_normalized_subset_abundance()
        )

    def make_subset_abundance(self, counts: ndarray) -> ndarray:
        """Calculates the relative abundances in subsets.

        Parameters
        ----------
        counts
            2-d array with one column per subset, one row per
            species, containing the count of each species in the
            corresponding subsets.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subsets), where
        rows correspond to unique species, columns correspond to
        subsets and each element is the abundance of the species
        in the subset relative to the total set size.
        """
        return counts / counts.sum()

    def make_subset_normalizing_constants(self) -> ndarray:
        """Calculates subset normalizing constants.

        Returns
        -------
        A numpy.ndarray of shape (n_subsets,), with the fraction
        of each subset's size of the set.
        """
        return self.subset_abundance.sum(axis=0)

    def make_normalized_subset_abundance(self) -> ndarray:
        """Calculates normalized relative abundances in subsets.

        Returns
        -------
        A numpy.ndarray of shape (n_species, n_subsets), where
        rows correspond to unique species, columns correspond to
        subsets and each element is the abundance of the species
        in the subset relative to the subset size.
        """
        self.subset_normalizing_constants = (
            self.make_subset_normalizing_constants()
        )
        return self.subset_abundance / self.subset_normalizing_constants

    def premultiply_by(self, similarity):
        return similarity.weighted_abundances(self.normalized_subset_abundance)


class AbundanceForDiversity(Abundance):
    """Calculates metacommuntiy and subset relative abundance
    components from a numpy.ndarray containing species counts
    """

    def __init__(
        self, counts: ndarray, subset_names: Iterable[Union[str, int]]
    ) -> None:
        super().__init__(counts, subset_names)
        self.set_abundance = self.make_set_abundance()
        self.unified_abundance_array = None

    def unify_abundance_array(self) -> None:
        """Creates one matrix containing all the abundance matrices:
        set, subset, and normalized subset.
        These matrices are still available as views on the unified
        data structure. (Because we are using basic slicing here, only
        one copy of the data will exist after garbage collection.)

        This allows for a major computational improvement in efficiency:
        The similarity matrix only has to be generated and used
        once (in the case where a pre-computed similarity matrix is not
        in RAM). That is, we make only one call to
        similarity.weighted_abundances(), in cases where generation of the
        similarity matrix is expensive.
        """
        self.unified_abundance_array = concatenate(
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

    def make_set_abundance(self) -> ndarray:
        """Calculates the relative abundances in set.

        Returns
        -------
        A numpy.ndarray of shape (n_species, 1), where rows correspond
        to unique species and each row contains the relative abundance
        of the species in the set.
        """
        return self.subset_abundance.sum(axis=1, keepdims=True)


def make_abundance(counts: Union[DataFrame, ndarray], for_diversity=True) -> Abundance:
    """Initializes a concrete subclass of Abundance.

    Parameters
    ----------
    counts:
        2-d array with one column per subset, one row per species,
        where the elements are the species counts

    Returns
    -------
    An instance of a concrete subclass of Abundance.
    """
    if not for_diversity:
        specific_class = Abundance
    else:
        specific_class = AbundanceForDiversity
    if isinstance(counts, DataFrame):
        return specific_class(
            counts=counts.to_numpy(), subset_names=counts.columns.to_list()
        )
    elif hasattr(counts, "shape"):
        if issparse(counts):
            raise TypeError("sparse abundance matrix not yet implemented")
        else:
            return specific_class(
                counts=counts, subset_names=arange(counts.shape[1])
            )
    else:
        raise NotImplementedError(
            f"Type {type(counts)} is not supported for argument "
            "'counts'. Valid types include pandas.DataFrame or"
            "numpy.ndarray"
        )
