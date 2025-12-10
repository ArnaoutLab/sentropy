# ray.py

from typing import List, Any, Callable, Union, Tuple
import numpy as _np
from numpy import ndarray, concatenate
from pandas import DataFrame
from sentropy.exceptions import InvalidArgumentError

# avoid mypy error: see https://github.com/jorenham/scipy-stubs/issues/100
from scipy.sparse import spmatrix  # type: ignore
import ray  # type: ignore

from sentropy.similarity import (
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
)
from sentropy.backend import get_backend

def weighted_similarity_chunk_nonsymmetric(similarity: Callable,
    X: Union[_np.ndarray, DataFrame],
    Y: Union[_np.ndarray, DataFrame, None],
    relative_abundance,
    backend,
    chunk_size: int,
    chunk_index: int,
    return_Z: bool = True,
    ) -> Tuple[int, _np.ndarray, Union[_np.ndarray, None]]:
    """
    Calculates some rows of the matrix product of Z @ p,
    where Z is not given explicitly but rather each entry
    Z[i,j] is calculated by a function.
    """

    def enum_helper(X):
        if type(X) == DataFrame:
            return X.itertuples()
        return X

    if Y is None:
        Y = X
    chunk = X[chunk_index : chunk_index + chunk_size]
    similarities_chunk = backend.empty(shape=(chunk.shape[0], Y.shape[0]))
    for i, row_i in enumerate(enum_helper(chunk)):
        for j, row_j in enumerate(enum_helper(Y)):
            similarities_chunk[i, j] = similarity(row_i, row_j)

    result = backend.matmul(similarities_chunk, relative_abundance)
    if return_Z:
        return chunk_index, result, similarities_chunk
    else:
        return chunk_index, result, None

def weighted_similarity_chunk_symmetric(similarity: Callable,
        X: Union[_np.ndarray, DataFrame],
        relative_abundance,
        backend,
        chunk_size: int,
        chunk_index: int,
        return_Z: bool = True,
    ):
    def enum_helper(X, start_index=0):
        if type(X) == DataFrame:
            return X.iloc[start_index:].itertuples()
        return X[start_index:]

    chunk = X[chunk_index : chunk_index + chunk_size]
    similarities_chunk = backend.zeros(shape=(chunk.shape[0], X.shape[0]))
    for i, row_i in enumerate(enum_helper(chunk)):
        for j, row_j in enumerate(enum_helper(X, chunk_index + i + 1)):
            similarities_chunk[i, i + j + chunk_index + 1] = similarity(row_i, row_j)
    rows_result = backend.matmul(similarities_chunk, relative_abundance)
    rows_after_count = max(0, relative_abundance.shape[0] - (chunk_index + chunk_size))
    from numpy import vstack, zeros as _zeros
    rows_result = backend.vstack(
        (
            backend.zeros(shape=(chunk_index, relative_abundance.shape[1])),
            rows_result,
            backend.zeros(
                shape=(
                    rows_after_count,
                    relative_abundance.shape[1],
                )
            ),
        )
    )
    relative_abundance_slice = relative_abundance[chunk_index : chunk_index + chunk_size]
    cols_result = backend.matmul(similarities_chunk.T, relative_abundance_slice)
    result = rows_result + cols_result
    if return_Z:
        return chunk_index, result, similarities_chunk
    else:
        return chunk_index, result, None


class SimilarityFromRayFunction(SimilarityFromFunction):
    """Implements Similarity by calculating similarities with a callable
    function using Ray for parallelism."""

    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        max_inflight_tasks: int = 64,
        similarities_out: Union[ndarray, None] = None,
        backend=None,
    ) -> None:
        super().__init__(func, X, chunk_size, similarities_out)
        self.max_inflight_tasks = max_inflight_tasks
        self.backend = backend or get_backend("numpy")

    def get_Y(self):
        return None

    def weighted_abundances(
        self,
        relative_abundance: Union[ndarray, spmatrix],
    ):
        weighted_similarity_chunk = ray.remote(weighted_similarity_chunk_nonsymmetric)
        X_ref = ray.put(self.X)
        Y_ref = ray.put(self.get_Y())
        abundance_ref = ray.put(relative_abundance)
        futures: List[Any] = []
        results = []

        def process_refs(refs):
            nonlocal results
            for chunk_index, abundance_chunk, similarity_chunk in ray.get(refs):
                results.append((chunk_index, abundance_chunk))
                if self.similarities_out is not None:
                    self.similarities_out[
                        chunk_index : chunk_index + similarity_chunk.shape[0], :
                    ] = similarity_chunk

        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            if len(futures) >= self.max_inflight_tasks:
                ready_refs, futures = ray.wait(futures)
                process_refs(ready_refs)
            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.func,
                X=X_ref,
                Y=Y_ref,
                relative_abundance=abundance_ref,
                backend=self.backend,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                return_Z=(self.similarities_out is not None),
            )
            futures.append(chunk_future)
        process_refs(futures)
        results.sort()
        weighted_similarity_chunks = [r[1] for r in results]
        return self.backend.concatenate(weighted_similarity_chunks)


class IntersetSimilarityFromRayFunction(SimilarityFromRayFunction):
    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        Y: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        max_inflight_tasks=64,
        similarities_out: Union[ndarray, None] = None,
        backend=None,
    ):
        super().__init__(func, X, chunk_size, max_inflight_tasks, similarities_out, backend)
        self.Y = Y

    def get_Y(self):
        return self.Y

    def self_similar_weighted_abundances(
        self, relative_abundance: Union[ndarray, spmatrix]
    ):
        raise InvalidArgumentError(
            "Inappropriate similarity class for diversity measures"
        )


class SimilarityFromSymmetricRayFunction(SimilarityFromSymmetricFunction):
    """Parallelized symmetric-function similarity via Ray."""

    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        max_inflight_tasks: int = 64,
        similarities_out: Union[ndarray, None] = None,
        backend=None,
    ) -> None:
        super().__init__(func, X, chunk_size, similarities_out)
        self.max_inflight_tasks = max_inflight_tasks
        self.backend = backend or get_backend("numpy")

    def weighted_abundances(
        self,
        relative_abundance: Union[ndarray, spmatrix],
    ) -> ndarray:
        weighted_similarity_chunk = ray.remote(weighted_similarity_chunk_symmetric)
        X_ref = ray.put(self.X)
        abundance_ref = ray.put(relative_abundance)
        futures: List[Any] = []
        result = relative_abundance
        if self.similarities_out is not None:
            self.similarities_out.fill(0.0)

        def process_refs(refs):
            nonlocal result
            for chunk_index, addend, similarities in ray.get(refs):
                result = result + addend
                if self.similarities_out is not None:
                    self.similarities_out[
                        chunk_index : chunk_index + similarities.shape[0], :
                    ] = similarities

        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            if len(futures) >= self.max_inflight_tasks:
                (ready_refs, futures) = ray.wait(futures)
                process_refs(ready_refs)

            chunk_future = weighted_similarity_chunk.remote(
                similarity=self.func,
                X=X_ref,
                relative_abundance=abundance_ref,
                backend=self.backend,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                return_Z=(self.similarities_out is not None),
            )
            futures.append(chunk_future)
        process_refs(futures)
        if self.similarities_out is not None:
            self.similarities_out += self.similarities_out.T
            for i in range(self.X.shape[0]):
                self.similarities_out[i, i] = 1.0
        return result

