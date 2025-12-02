"""Module for set and subset diversity measures.

Classes
-------
Set
    Represents a set made up of subsets and computes
    set and subset diversity measures.
"""

from typing import Callable, Iterable, Optional, Union

from pandas import DataFrame, Index, Series, concat
from numpy import array, atleast_1d, broadcast_to, divide, zeros, ndarray, power, prod, sum as np_sum, \
identity as np_identity, inf as np_inf, log as np_log
from sentropy.exceptions import InvalidArgumentError

from sentropy.abundance import make_abundance
from sentropy.similarity import Similarity, SimilarityFromArray, SimilarityIdentity, SimilarityFromFunction, \
SimilarityFromSymmetricFunction, SimilarityFromFile
from sentropy.ray import SimilarityFromRayFunction, SimilarityFromSymmetricRayFunction
from sentropy.components import Components
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


class Set:
    similarity: Similarity
    """Creates diversity components and calculates diversity measures.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A set consists of one
    or more subsets and can be represented by the number of
    appearances of each species in each of the subsets that the
    species appears in.
    """
    MEASURES = (
    "alpha",
    "rho",
    "beta",
    "gamma",
    "normalized_alpha",
    "normalized_rho",
    "normalized_beta",
    "rho_hat",
    "beta_hat")

    def __init__(
        self,
        counts: Union[DataFrame, ndarray],
        similarity: Optional[Union[ndarray, DataFrame, str, Callable]] = None,
        symmetric: Optional[bool] = False,
        X: Optional[Union[ndarray, DataFrame]] = None,
        chunk_size: Optional[int] = 10,
        parallelize: Optional[bool] = False,
        max_inflight_tasks: Optional[int] = 64,
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
        """
        self.counts = counts
        self.abundance = make_abundance(counts=counts)
        if similarity is None:
            self.similarity = SimilarityIdentity()
        elif isinstance(similarity, ndarray):
            self.similarity = SimilarityFromArray(similarity=similarity)
        elif isinstance(similarity, DataFrame):
            self.similarity = SimilarityFromArray(similarity=similarity.values)
        elif isinstance(similarity, str):
            self.similarity = SimilarityFromFile(similarity, chunk_size=chunk_size)
        elif callable(similarity):
            if symmetric:
                if parallelize:
                    self.similarity = SimilarityFromSymmetricRayFunction(func=similarity,X=X, chunk_size=chunk_size, max_inflight_tasks=max_inflight_tasks)
                else:
                    self.similarity = SimilarityFromSymmetricFunction(func=similarity,X=X, chunk_size=chunk_size)
            else:
                if parallelize:
                    self.similarity = SimilarityFromRayFunction(func=similarity, X=X, chunk_size=chunk_size, max_inflight_tasks=max_inflight_tasks)
                else:
                    self.similarity = SimilarityFromFunction(func=similarity, X=X, chunk_size=chunk_size)
        elif isinstance(similarity, Similarity):
            # allow passing an already-constructed Similarity object
            self.similarity = similarity
        self.components = Components(
            abundance=self.abundance, similarity=self.similarity
        )

    def subset_diversity(self, viewpoint: float, measure: str, eff_no: bool) -> ndarray:
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
        if measure not in self.MEASURES:
            raise (
                InvalidArgumentError(
                    f"Invalid measure '{measure}'. "
                    "Argument 'measure' must be one of: "
                    f"{', '.join(self.MEASURES)}"
                )
            )
        numerator = self.components.numerators[measure]
        denominator = self.components.denominators[measure]
        if measure == "gamma":
            denominator = broadcast_to(
                denominator,
                self.abundance.normalized_subset_abundance.shape,
            )
        community_ratio = divide(
            numerator,
            denominator,
            out=zeros(denominator.shape),
            where=denominator != 0,
        )
        diversity_measure = power_mean(
            order=1 - viewpoint,
            weights=self.abundance.normalized_subset_abundance,
            items=community_ratio,
            atol=self.abundance.min_count,
        )
        if measure in {"beta", "normalized_beta"}:
            return 1 / diversity_measure

        if measure in {"rho_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return (diversity_measure - 1) / (N - 1)

        if measure in {"beta_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return ((N / diversity_measure) - 1) / (N - 1)

        if eff_no==False:
            diversity_measure = np_log(diversity_measure)

        return diversity_measure

    def set_diversity(self, viewpoint: float, measure: str, eff_no: bool) -> ndarray:
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
        subset_diversity = self.subset_diversity(viewpoint, measure, eff_no=eff_no)
        diversity_measure = power_mean(
            1 - viewpoint,
            self.abundance.subset_normalizing_constants,
            subset_diversity,
        )

        if eff_no==False:
            diversity_measure = np_log(diversity_measure)

        return diversity_measure

    def subsets_to_dataframe(self, viewpoint: float, measures: Iterable[str], eff_no: bool):
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
        df = DataFrame(
            {
                measure: self.subset_diversity(viewpoint, measure, eff_no)
                for measure in measures
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "set/subset", Series(self.abundance.subsets_names))
        return df

    def set_to_dataframe(self, viewpoint: float, measures: Iterable[str], eff_no: bool):
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
        df = DataFrame(
            {
                measure: self.set_diversity(viewpoint, measure, eff_no)
                for measure in measures
            },
            index=Index(["set"], name="set/subset"),
        )
        df.insert(0, "viewpoint", viewpoint)
        df.reset_index(inplace=True)
        return df

    def to_dataframe(self, viewpoint: Union[float, Iterable[float]], measures: Iterable[str] = MEASURES, which: str = "both", eff_no: bool = True):
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
        for q in atleast_1d(array(viewpoint)):
            if which in ["both", "set"]:
                dataframes.append(
                self.set_to_dataframe(viewpoint=q, measures=measures, eff_no=eff_no))
            if which in ["both", "subset"]:
                dataframes.append(
                self.subsets_to_dataframe(viewpoint=q, measures=measures, eff_no=eff_no))
        return concat(dataframes).reset_index(drop=True)






