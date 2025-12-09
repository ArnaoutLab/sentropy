"""Module for set and subset diversity measures."""

from typing import Callable, Iterable, Optional, Union

from pandas import DataFrame, Index, Series, concat
from numpy import array, atleast_1d, broadcast_to, zeros as np_zeros, ndarray
from sentropy.exceptions import InvalidArgumentError

from sentropy.abundance import make_abundance
from sentropy.similarity import Similarity, SimilarityFromArray, SimilarityIdentity, SimilarityFromFunction, \
SimilarityFromSymmetricFunction, SimilarityFromFile
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
        self.abundance = make_abundance(counts=counts, backend=self.backend)
        if similarity is None:
            self.similarity = SimilarityIdentity(backend=self.backend)
        elif isinstance(similarity, ndarray):
            self.similarity = SimilarityFromArray(similarity=similarity, backend=self.backend)
        elif isinstance(similarity, DataFrame):
            self.similarity = SimilarityFromArray(similarity=similarity.values, backend=self.backend)
        elif isinstance(similarity, str):
            self.similarity = SimilarityFromFile(similarity, chunk_size=chunk_size, backend=self.backend)
        elif callable(similarity):
            if symmetric:
                if parallelize:
                    self.similarity = SimilarityFromSymmetricRayFunction(func=similarity,X=X, chunk_size=chunk_size, \
                        max_inflight_tasks=max_inflight_tasks, backend=self.backend)
                else:
                    self.similarity = SimilarityFromSymmetricFunction(func=similarity,X=X, chunk_size=chunk_size, backend=self.backend)
            else:
                if parallelize:
                    self.similarity = SimilarityFromRayFunction(func=similarity, X=X, chunk_size=chunk_size, \
                        max_inflight_tasks=max_inflight_tasks, backend=self.backend)
                else:
                    self.similarity = SimilarityFromFunction(func=similarity, X=X, chunk_size=chunk_size, backend=self.backend)
        else:
            self.similarity = similarity

        self.components = Components(
            abundance=self.abundance, similarity=self.similarity
        )
        self.subset_diversity_hash = {}

    def subset_diversity(self, viewpoint: float, measure: str, eff_no: bool=True) -> ndarray:
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

        if f'subset_{measure}_q={viewpoint}' in self.subset_diversity_hash.keys():
            diversity_measure = self.subset_diversity_hash[f'subset_{measure}_q={viewpoint}']
            if eff_no == False:
                return self.backend.log(diversity_measure)
            else:
                return diversity_measure

        numerator = self.components.numerators[measure]
        denominator = self.components.denominators[measure]

        if measure == "gamma":
            denominator = self.backend.broadcast_to(
                denominator,
                self.abundance.normalized_subset_abundance.shape,
            )

        # divide with safe handling
        ratio = self.backend.divide(numerator, denominator)

        diversity_measure = power_mean(
            order=1 - viewpoint,
            weights=self.abundance.normalized_subset_abundance,
            items=ratio,
            atol=self.abundance.min_count,
            backend=self.backend,
        )
        if measure in {"beta", "normalized_beta"}:
            return 1 / diversity_measure

        if measure in {"rho_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return (diversity_measure - 1) / (N - 1)

        if measure in {"beta_hat"} and self.counts.shape[1] > 1:
            N = self.counts.shape[1]
            return ((N / diversity_measure) - 1) / (N - 1)

        self.subset_diversity_hash[f'subset_{measure}_q={viewpoint}'] = diversity_measure

        if eff_no==False:
            return self.backend.log(diversity_measure)
        else:
            return diversity_measure

    def set_diversity(self, viewpoint: float, measure: str, eff_no: bool=True) -> ndarray:
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
        subset_diversity = self.subset_diversity(viewpoint, measure, eff_no = True) #note: eff_no must be True here !
        diversity_measure = power_mean(
            1 - viewpoint,
            self.abundance.subset_normalizing_constants,
            subset_diversity,
            backend=self.backend,
        )

        if eff_no==False:
            return self.backend.log(diversity_measure).item()
        else:
            return diversity_measure.item()

    def subsets_to_dataframe(self, viewpoint: float, measures=MEASURES, eff_no: bool=True):
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
            measure: (self.subset_diversity(viewpoint, measure, eff_no).cpu() if isinstance(self.subset_diversity(viewpoint, measure, eff_no),Tensor) else \
                self.subset_diversity(viewpoint, measure, eff_no)) for measure in measures
        })
        df.insert(0, "viewpoint", viewpoint)
        df.insert(0, "set/subset", Series(self.abundance.subsets_names))
        return df

    def set_to_dataframe(self, viewpoint: float, measures=MEASURES, eff_no: bool=True):
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
            measure: (self.set_diversity(viewpoint, measure, eff_no).cpu() if isinstance(self.set_diversity(viewpoint, measure, eff_no),Tensor) else \
                self.set_diversity(viewpoint, measure, eff_no)) for measure in measures
        },
        index=Index(["set"], name="set/subset"))

        df.insert(0, "viewpoint", viewpoint)
        df.reset_index(inplace=True)
        return df

    def to_dataframe(self, viewpoint: Union[float, Iterable[float]], measures=MEASURES, which: str = "both", eff_no: bool = True):
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
        for q in viewpoint:
            if which in ["both", "set"]:
                dataframes.append(
                self.set_to_dataframe(viewpoint=q, measures=measures, eff_no=eff_no))
            if which in ["both", "subset"]:
                dataframes.append(
                self.subsets_to_dataframe(viewpoint=q, measures=measures, eff_no=eff_no))
        return concat(dataframes).reset_index(drop=True)




