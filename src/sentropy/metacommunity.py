"""Module for metacommunity and subcommunity diversity measures.

Classes
-------
Metacommunity
    Represents a metacommunity made up of subcommunities and computes
    metacommunity and subcommunity diversity measures.
"""

from typing import Callable, Iterable, Optional, Union

from pandas import DataFrame, Index, Series, concat
from numpy import array, atleast_1d, broadcast_to, divide, zeros, ndarray, power, prod, sum as np_sum, \
identity as np_identity
from sentropy.exceptions import InvalidArgumentError

from sentropy.abundance import make_abundance
from sentropy.similarity import Similarity, SimilarityFromArray, SimilarityIdentity, SimilarityFromFunction, \
SimilarityFromSymmetricFunction, SimilarityFromFile
from sentropy.components import Components
from sentropy.powermean import power_mean


class Metacommunity:
    similarity: Similarity
    """Creates diversity components and calculates diversity measures.

    A community consists of a set of species, each of which may appear
    any (non-negative) number of times. A metacommunity consists of one
    or more subcommunities and can be represented by the number of
    appearances of each species in each of the subcommunities that the
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
        "beta_hat",
    )

    def __init__(
        self,
        counts: Union[DataFrame, ndarray],
        similarity: Optional[Union[ndarray, DataFrame, str, Callable]] = None,
        symmetric: Optional[bool] = False,
        X: Optional[Union[ndarray, DataFrame]] = None,
        chunk_size: Optional[int] = 10,
    ) -> None:
        """
        Parameters
        ----------
        counts:
            2-d array with one column per subcommunity, one row per
            species, containing the count of each species in the
            corresponding subcommunities.
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
        elif callable(similarity) and symmetric==True:
            self.similarity = SimilarityFromSymmetricFunction(func=similarity,X=X, chunk_size=chunk_size)
        elif callable(similarity) and symmetric==False:
            self.similarity = SimilarityFromFunction(func=similarity, X=X, chunk_size=chunk_size)
        elif isinstance(similarity, Similarity):
            # allow passing an already-constructed Similarity object
            self.similarity = similarity
        self.components = Components(
            abundance=self.abundance, similarity=self.similarity
        )

    def subcommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates subcommunity diversity measures.

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
        A numpy.ndarray with a diversity measure for each subcommunity.
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
                self.abundance.normalized_subcommunity_abundance.shape,
            )
        community_ratio = divide(
            numerator,
            denominator,
            out=zeros(denominator.shape),
            where=denominator != 0,
        )
        diversity_measure = power_mean(
            order=1 - viewpoint,
            weights=self.abundance.normalized_subcommunity_abundance,
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

        return diversity_measure

    def metacommunity_diversity(self, viewpoint: float, measure: str) -> ndarray:
        """Calculates metacommunity diversity measures.

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
        A numpy.ndarray containing the metacommunity diversity measure.
        """
        subcommunity_diversity = self.subcommunity_diversity(viewpoint, measure)
        return power_mean(
            1 - viewpoint,
            self.abundance.subcommunity_normalizing_constants,
            subcommunity_diversity,
        )

    def subcommunities_to_dataframe(self, viewpoint: float, measures=MEASURES):
        """Table containing all subcommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contribute.

        Returns
        -------
        A pandas.DataFrame containing all subcommunity diversity
        measures for a given viewpoint
        """
        df = DataFrame(
            {
                measure: self.subcommunity_diversity(viewpoint, measure)
                for measure in measures
            }
        )
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "community", Series(self.abundance.subcommunities_names))
        return df

    def metacommunity_to_dataframe(self, viewpoint: float, measures=MEASURES):
        """Table containing all metacommunity diversity values.

        Parameters
        ----------
        viewpoint:
            Affects the contribution of rare species to the diversity
            measure. When viewpoint = 0 all species (rare or frequent)
            contribute equally. When viewpoint = infinity, only the
            single most frequent species contributes.

        Returns
        -------
        A pandas.DataFrame containing all metacommunity diversity
        measures for a given viewpoint
        """
        df = DataFrame(
            {
                measure: self.metacommunity_diversity(viewpoint, measure)
                for measure in measures
            },
            index=Index(["metacommunity"], name="community"),
        )
        df.insert(0, "viewpoint", viewpoint)
        df.reset_index(inplace=True)
        return df

    def to_dataframe(self, viewpoint: Union[float, Iterable[float]], measures=MEASURES):
        """Table containing all metacommunity and subcommunity diversity
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
        A pandas.DataFrame containing all metacommunity and subcommunity
        diversity measures for a given viewpoint
        """
        dataframes = []
        for q in atleast_1d(array(viewpoint)):
            dataframes.append(
                self.metacommunity_to_dataframe(viewpoint=q, measures=measures)
            )
            dataframes.append(
                self.subcommunities_to_dataframe(viewpoint=q, measures=measures)
            )
        return concat(dataframes).reset_index(drop=True)
