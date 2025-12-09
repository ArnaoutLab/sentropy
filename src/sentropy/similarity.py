"""Module for calculating weighted subset and set
similarities.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple
from pathlib import Path
from pandas import DataFrame, read_csv
from scipy.sparse import spmatrix, issparse  # type: ignore[import]
from sentropy.exceptions import InvalidArgumentError
from sentropy.backend import get_backend

# We'll try to import typing for numpy typing if necessary
import numpy as _np

# Note: We keep many implementation details but route array ops via backend.


class Similarity(ABC):
    """Root superclass for classes computing weighted similarities."""

    def __init__(self, similarities_out: Union[_np.ndarray, None] = None, backend=None):
        """
        similarities_out: optional numpy array into which the similarity matrix will be written.
        backend: instance of backend (NumpyBackend or TorchBackend). If None, default numpy backend.
        """
        self.similarities_out = similarities_out
        self.backend = backend or get_backend("numpy")

    @abstractmethod
    def weighted_abundances(
        self,
        relative_abundances: Union[_np.ndarray, spmatrix],
    ):
        pass

    def self_similar_weighted_abundances(self, relative_abundances: Union[_np.ndarray, spmatrix]):
        return self.weighted_abundances(relative_abundances)

    def is_expensive(self):
        return False

    def __matmul__(self, abundance):
        return abundance.premultiply_by(self)


class SimilarityIdentity(Similarity):
    def __init__(self, similarities_out=None, backend=None):
        super().__init__(similarities_out=similarities_out, backend=backend)

    def weighted_abundances(self, relative_abundance):
        if self.similarities_out is not None:
            self.similarities_out.fill(0.0)
            for i in range(relative_abundance.shape[0]):
                self.similarities_out[i, i] = 1.0
        return relative_abundance


class SimilarityFromArray(Similarity):
    """Implements Similarity using similarities stored in a numpy
    ndarray or backend-supported array."""

    def __init__(
        self,
        similarity: Union[_np.ndarray, None],
        similarities_out: Union[_np.ndarray, None] = None,
        backend=None,
    ):
        super().__init__(similarities_out=similarities_out, backend=backend)
        self.similarity_raw = similarity
        if similarity is not None:
            #convert to backend array if backend supports it
            try:
                self.similarity = self.backend.asarray(similarity)
            except Exception:
                self.similarity = similarity
        else:
            self.similarity = similarity

    def weighted_abundances(self, relative_abundance):
        if self.similarities_out is not None:
            # write to the numpy buffer if provided
            # similarity may be backend tensor — convert to numpy if needed
            try:
                self.similarities_out[:, :] = self.backend.to_numpy(self.similarity)
            except Exception:
                # fallback: try copying via numpy
                self.similarities_out[:, :] = _np.asarray(self.similarity)
        return self.backend.matmul(self.similarity, relative_abundance)

    def self_similar_weighted_abundances(self, relative_abundances):
        if self.similarity.shape[0] == self.similarity.shape[1]:
            return self.weighted_abundances(relative_abundances)
        else:
            raise InvalidArgumentError("Similarity matrix must be square")


class SimilarityFromDataFrame(SimilarityFromArray):
    def __init__(self, similarity: DataFrame, similarities_out=None, backend=None):
        super().__init__(similarity=similarity.to_numpy(), similarities_out=similarities_out, backend=backend)


class SimilarityFromFile(Similarity):
    """Implements Similarity by using similarities stored in file."""

    def __init__(
        self,
        similarity_file_path: Union[str, Path],
        chunk_size: int = 100,
        similarities_out: Union[_np.ndarray, None] = None,
        backend=None,
    ) -> None:
        super().__init__(similarities_out=similarities_out, backend=backend)
        self.path = Path(similarity_file_path)
        self.chunk_size = chunk_size

    def weighted_abundances(self, relative_abundance):
        weighted_abundances = self.backend.zeros(relative_abundance.shape)
        i = 0
        with read_csv(
            self.path,
            chunksize=self.chunk_size,
            sep=None,
            engine="python",
            dtype=float,
        ) as similarity_matrix_chunks:
            for chunk in similarity_matrix_chunks:
                chunk_as_numpy = chunk.to_numpy()
                # convert chunk to backend array
                chunk_backend = self.backend.asarray(chunk_as_numpy)
                weighted_abundances[i : i + self.chunk_size, :] = (
                    self.backend.matmul(chunk_backend, relative_abundance)
                )
                if self.similarities_out is not None:
                    self.similarities_out[i : i + self.chunk_size, :] = chunk_as_numpy
                i += self.chunk_size
        return weighted_abundances

    def is_expensive(self):
        return True


class IntersetSimilarityFromFile(SimilarityFromFile):
    def weighted_abundances(self, relative_abundance):
        weighted_abundance_chunks = []
        with read_csv(
            self.path,
            chunksize=self.chunk_size,
            sep=None,
            engine="python",
            dtype=float,
        ) as similarity_matrix_chunks:
            for j, chunk in enumerate(similarity_matrix_chunks):
                chunk_as_numpy = chunk.to_numpy()
                if self.similarities_out is not None:
                    self.similarities_out[
                        j * self.chunk_size : (j + 1) * self.chunk_size, :
                    ] = chunk_as_numpy
                chunk_backend = self.backend.asarray(chunk_as_numpy)
                weighted_abundance_chunks.append(self.backend.matmul(chunk_backend, relative_abundance))
        # concatenate using backend if possible
        return self.backend.concatenate(weighted_abundance_chunks, axis=0) if hasattr(self.backend, "concatenate") else _np.concatenate(weighted_abundance_chunks, axis=0)

    def self_similar_weighted_abundances(self, relative_abundance):
        raise InvalidArgumentError(
            "Inappropriate similarity class for diversity measures"
        )


class SimilarityFromSymmetricFunction(Similarity):
    def __init__(
        self,
        func: Callable,
        X: Union[_np.ndarray, DataFrame],
        chunk_size: int = 100,
        similarities_out: Union[_np.ndarray, None] = None,
        backend=None,
    ):
        super().__init__(similarities_out=similarities_out, backend=backend)
        self.func = func
        self.X = X
        self.chunk_size = chunk_size

    def is_expensive(self):
        return True

    def weighted_abundances(self, abundance):
        # result should be backend array; start as copy
        result = self.backend.copy(abundance)
        if self.similarities_out is not None:
            self.similarities_out.fill(0.0)
        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            _, chunk, similarities = self.weighted_similarity_chunk_symmetric(
                similarity=self.func,
                X=self.X,
                relative_abundance=abundance,
                chunk_size=self.chunk_size,
                chunk_index=chunk_index,
                return_Z=(self.similarities_out is not None),
            )
            result = result + chunk
            if self.similarities_out is not None:
                self.similarities_out[
                    chunk_index : chunk_index + self.chunk_size, :
                ] = similarities
        if self.similarities_out is not None:
            self.similarities_out += self.similarities_out.T
            # set diagonal to 1.0
            for i in range(self.X.shape[0]):
                self.similarities_out[i, i] = 1.0
        return result


    def weighted_similarity_chunk_symmetric(self, similarity: Callable,
        X: Union[_np.ndarray, DataFrame],
        relative_abundance,
        chunk_size: int,
        chunk_index: int,
        return_Z: bool = True,
    ):
        def enum_helper(X, start_index=0):
            if type(X) == DataFrame:
                return X.iloc[start_index:].itertuples()
            return X[start_index:]

        chunk = X[chunk_index : chunk_index + chunk_size]
        similarities_chunk = self.backend.zeros(shape=(chunk.shape[0], X.shape[0]))
        for i, row_i in enumerate(enum_helper(chunk)):
            for j, row_j in enumerate(enum_helper(X, chunk_index + i + 1)):
                similarities_chunk[i, i + j + chunk_index + 1] = similarity(row_i, row_j)
        rows_result = self.backend.matmul(similarities_chunk, relative_abundance)
        rows_after_count = max(0, relative_abundance.shape[0] - (chunk_index + chunk_size))
        from numpy import vstack, zeros as _zeros
        rows_result = vstack(
            (
                _zeros(shape=(chunk_index, relative_abundance.shape[1])),
                rows_result,
                _zeros(
                    shape=(
                        rows_after_count,
                        relative_abundance.shape[1],
                    )
                ),
            )
        )
        relative_abundance_slice = relative_abundance[chunk_index : chunk_index + chunk_size]
        cols_result = similarities_chunk.T @ relative_abundance_slice
        result = rows_result + cols_result
        if return_Z:
            return chunk_index, result, similarities_chunk
        else:
            return chunk_index, result, None


class SimilarityFromFunction(SimilarityFromSymmetricFunction):
    def weighted_abundances(self, relative_abundance):
        weighted_similarity_chunks = []
        for chunk_index in range(0, self.X.shape[0], self.chunk_size):
            _, result, similarities = self.weighted_similarity_chunk_nonsymmetric(
                self.func,
                self.X,
                self.get_Y(),
                relative_abundance,
                self.chunk_size,
                chunk_index,
                (self.similarities_out is not None),
            )
            weighted_similarity_chunks.append(result)
            if self.similarities_out is not None:
                self.similarities_out[
                    chunk_index : chunk_index + result.shape[0], :
                ] = similarities
        return self.backend.concatenate(weighted_similarity_chunks)

    def get_Y(self):
        return None

    def weighted_similarity_chunk_nonsymmetric(self, similarity: Callable,
    X: Union[_np.ndarray, DataFrame],
    Y: Union[_np.ndarray, DataFrame, None],
    relative_abundance,
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
        similarities_chunk = self.backend.empty(shape=(chunk.shape[0], Y.shape[0]))
        for i, row_i in enumerate(enum_helper(chunk)):
            for j, row_j in enumerate(enum_helper(Y)):
                similarities_chunk[i, j] = similarity(row_i, row_j)

        result = self.backend.matmul(similarities_chunk, relative_abundance)
        if return_Z:
            return chunk_index, result, similarities_chunk
        else:
            return chunk_index, result, None


class IntersetSimilarityFromFunction(SimilarityFromFunction):
    def __init__(
        self,
        func: Callable,
        X: Union[_np.ndarray, DataFrame],
        Y: Union[_np.ndarray, DataFrame],
        chunk_size: int = 100,
        similarities_out: Union[_np.ndarray, None] = None,
        backend=None,
    ):
        super().__init__(func, X, chunk_size, similarities_out, backend)
        self.Y = Y

    def get_Y(self):
        return self.Y

    def self_similar_weighted_abundances(self, relative_abundance):
        raise InvalidArgumentError(
            "Inappropriate similarity class for diversity measures"
        )

