"""Module for calculating weighted subset and set
similarities.
"""

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple
from pathlib import Path
from pandas import DataFrame, read_csv

# from scipy.sparse import spmatrix, issparse  # type: ignore[import]
from sentropy.backend import get_backend

from numpy import ndarray, tril_indices
from torch import Tensor

# Note: We keep many implementation details but route array ops via backend.

def batch_similarity(func: Callable) -> Callable:
    """Tag `func` as vectorizable over pairs of rows.

    A batch similarity function must accept two 2-D arrays
    A of shape (n_A, d) and B of shape (n_B, d) and return an
    (n_A, n_B) array S such that S[p, q] == func(A[p], B[q]).

    Chunks of the similarity matrix are then computed with one
    vectorized call per chunk instead of an element-wise double loop.
    """
    func.is_batch_similarity = True
    return func


def _is_batch_similarity(similarity: Callable) -> bool:
    return bool(getattr(similarity, "is_batch_similarity", False))

def weighted_similarity_chunk_nonsymmetric(
    similarity: Callable,
    X: Union[ndarray, DataFrame],
    Y: Union[ndarray, DataFrame, None],
    relative_abundance,
    backend,
    chunk_size: int,
    chunk_index: int,
    return_Z: bool = True,
) -> Tuple[int, ndarray, Union[ndarray, None]]:
    """
    Calculates some rows of the matrix product of Z @ p,
    where Z is not given explicitly but rather each entry
    Z[i,j] is calculated by a function.
    """
    def enum_helper(A):
        if isinstance(A, DataFrame):
            return A.itertuples()
        return A

    if Y is None:
        Y = X
    chunk = X[chunk_index : chunk_index + chunk_size]

    if _is_batch_similarity(similarity):
        similarities_chunk = backend.asarray(similarity(chunk, Y))
    else:
        similarities_chunk = backend.empty(shape=(chunk.shape[0], Y.shape[0]))
        for i, row_i in enumerate(enum_helper(chunk)):
            for j, row_j in enumerate(enum_helper(Y)):
                similarities_chunk[i, j] = similarity(row_i, row_j)

    result = backend.matmul(similarities_chunk, relative_abundance)
    if return_Z:
        return chunk_index, result, similarities_chunk
    return chunk_index, result, None


def weighted_similarity_chunk_symmetric(
    similarity: Callable,
    X: Union[ndarray, DataFrame],
    relative_abundance,
    backend,
    chunk_size: int,
    chunk_index: int,
    return_Z: bool = True,
) -> Tuple[int, ndarray, Union[ndarray, None]]:
    """
    Calculates some rows of the matrix product of Z @ p for a symmetric
    Z whose entries are given by a pairwise similarity function. Only
    strictly-above-diagonal entries are ever computed; each chunk's
    contribution is assembled from the in-chunk "corner" block and the
    "rest" block between the chunk and the rows after it.
    """
    def enum_helper(A, start_index=0):
        if isinstance(A, DataFrame):
            return A.iloc[start_index:].itertuples()
        return A[start_index:]

    chunk = X[chunk_index : chunk_index + chunk_size]
    after = X[chunk_index + chunk_size :]

    if _is_batch_similarity(similarity):
        corner = backend.copy(backend.asarray(similarity(chunk, chunk)))
        rest = (
            backend.asarray(similarity(chunk, after))
            if len(after)
            else backend.zeros(shape=(len(chunk), 0))
        )
    else:
        similarities_chunk = backend.zeros(shape=(chunk.shape[0], X.shape[0]))
        for i, row_i in enumerate(enum_helper(chunk)):
            for j, row_j in enumerate(enum_helper(X, chunk_index + i + 1)):
                similarities_chunk[i, i + j + chunk_index + 1] = similarity(
                    row_i, row_j
                )
        corner = similarities_chunk[:, chunk_index : chunk_index + chunk_size]
        rest = similarities_chunk[:, chunk_index + chunk_size :]

    # Keep only pairs above the diagonal; assigning (not a 0/1 multiply)
    # also drops NaN/inf on and below the in-chunk diagonal.
    corner[tril_indices(len(chunk))] = 0

    relative_abundance_slice = relative_abundance[
        chunk_index : chunk_index + chunk_size
    ]
    result = backend.zeros(shape=relative_abundance.shape)
    result[chunk_index : chunk_index + chunk_size] = backend.matmul(
        corner + corner.T, relative_abundance_slice
    ) + backend.matmul(rest, relative_abundance[chunk_index + chunk_size :])
    result[chunk_index + chunk_size :] = backend.matmul(
        rest.T, relative_abundance_slice
    )

    if return_Z:
        if _is_batch_similarity(similarity):
            # Rebuild the full-width chunk row from corner + rest.
            similarities_chunk = backend.zeros(shape=(len(chunk), X.shape[0]))
            similarities_chunk[:, chunk_index : chunk_index + chunk_size] = corner
            similarities_chunk[:, chunk_index + chunk_size :] = rest
        # (non-batch path: similarities_chunk already holds the full-width,
        #  strictly-upper-triangular chunk rows)
        return chunk_index, result, similarities_chunk
    return chunk_index, result, None


class Similarity(ABC):
    """Root superclass for all similarity classes.

    A Similarity object represents the pairwise species-similarity
    matrix Z implicitly or explicitly, and knows how to compute
    ordinariness vectors from abundance distributions: the products
    Z @ abundance. Concrete subclasses implement `weighted_abundances`
    for their source of similarities (identity, in-memory array, file,
    or a callable).

    Parameters
    ----------
    similarities_out : ndarray, optional
        Preallocated array into which the full similarity matrix is
        written whenever `weighted_abundances` is called. Pass None
        (default) to skip materializing the matrix.
    backend : BaseBackend, optional
        Backend instance (NumpyBackend or TorchBackend) used for all
        array operations. Defaults to the numpy backend.

    Notes
    -----
    Supporting `similarity @ abundance` via __matmul__, which calls
    `abundance.premultiply_by(self)`, lets callers compute all needed
    ordinariness vectors at once. Expensive subclasses (callable- or
    file-based; see `is_expensive`) batch the set-, subset-, and
    normalized-subset abundance vectors into a single matmul, so the
    similarity is evaluated only once.

    See Also
    --------
    Abundance.premultiply_by : consumes these objects.
    batch_similarity : decorator marking vectorizable callables.
    """

    def __init__(self, similarities_out: Union[ndarray, None] = None, backend=None):
        """
        similarities_out: optional numpy array into which the similarity matrix will be written.
        backend: instance of backend (NumpyBackend or TorchBackend). If None, default numpy backend.
        """
        self.similarities_out = similarities_out
        self.backend = backend or get_backend("numpy")

    @abstractmethod  # pragma: no cover
    def weighted_abundances(
        self,
        relative_abundances: Union[ndarray, Tensor],
    ):
        pass

    def self_similar_weighted_abundances(
        self, relative_abundances: Union[ndarray, Tensor]
    ):
        return self.weighted_abundances(relative_abundances)

    def is_expensive(self):
        return False

    def __matmul__(self, abundance):
        return abundance.premultiply_by(self)


class SimilarityIdentity(Similarity):
    """Similarity matrix given by the identity: each species is similar
    only to itself.

    Selecting this class reduces similarity-sensitive diversity to
    ordinary, frequency-only diversity. `weighted_abundances` returns
    the abundance distribution unchanged, and if `similarities_out`
    was provided it is filled with the identity matrix.

    This is the default when `similarity=None` is passed to LCR(),
    vendi_score(), or build_similarity().
    """
    def __init__(self, similarities_out=None, backend=None):
        super().__init__(similarities_out=similarities_out, backend=backend)

    def weighted_abundances(self, relative_abundance):
        if self.similarities_out is not None:
            self.similarities_out.fill(0.0)
            for i in range(relative_abundance.shape[0]):
                self.similarities_out[i, i] = 1.0
        return relative_abundance


class SimilarityFromArray(Similarity):
    """Similarity defined by an in-memory matrix of pairwise similarities.

    Parameters
    ----------
    similarity : ndarray or scipy sparse matrix
        Square matrix whose entry [i, j] is the similarity between
        species i and species j. Row and column order must match the
        order of species (rows) in the counts.
    similarities_out : ndarray, optional
        As in Similarity. On weighted_abundances, receives the
        similarity converted to a numpy array.
    backend : BaseBackend, optional
        As in Similarity.

    Notes
    -----
    Sparse matrices are accepted and kept sparse: the numpy backend's
    `asarray` passes scipy sparse objects through unchanged, so the
    matmul in `weighted_abundances` exploits sparsity.
    """

    def __init__(
        self,
        similarity: Union[ndarray, None],
        similarities_out: Union[ndarray, None] = None,
        backend=None,
    ):
        super().__init__(similarities_out=similarities_out, backend=backend)
        self.similarity_raw = similarity
        self.similarity = self.backend.asarray(similarity)

    def weighted_abundances(self, relative_abundance):
        if self.similarities_out is not None:
            # write to the numpy buffer if provided
            # similarity may be backend tensor — convert to numpy if needed
            self.similarities_out[:, :] = self.backend.to_numpy(self.similarity)
        return self.backend.matmul(self.similarity, relative_abundance)

    def self_similar_weighted_abundances(self, relative_abundances):
        return self.weighted_abundances(relative_abundances)


class SimilarityFromDataFrame(SimilarityFromArray):
    """Similarity given by a pandas DataFrame of pairwise similarities.

    Simply converts the DataFrame's values to a numpy array and
    delegates to SimilarityFromArray; the row/column order must match
    the species order in the counts. Handy when a similarity matrix
    was loaded with species names as labels.
    """
    def __init__(self, similarity: DataFrame, similarities_out=None, backend=None):
        super().__init__(
            similarity=similarity.to_numpy(),
            similarities_out=similarities_out,
            backend=backend,
        )


class SimilarityFromFile(Similarity):
    """Similarity streamed row-chunk-wise from a csv/tsv file.

    For large similarity matrices that should not be held in memory,
    the file is read with pandas in chunks of `chunk_size` rows; each
    chunk is multiplied with the abundance vector immediately, so only
    one chunk plus the result is resident at a time. The file must be
    square, with a header listing the species names, in the same
    column and row order as the counts.

    Parameters
    ----------
    similarity_file_path : str or Path
        Path to the csv/tsv similarity matrix.
    chunk_size : int, default=100
        Number of rows to read per chunk.
    similarities_out, backend :
        As in Similarity.

    Notes
    -----
    `is_expensive()` returns True: abundance.premultiply_by batches
    all abundance vectors into one pass over the file.
    """

    def __init__(
        self,
        similarity_file_path: Union[str, Path],
        chunk_size: int = 100,
        similarities_out: Union[ndarray, None] = None,
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
                weighted_abundances[i : i + self.chunk_size, :] = self.backend.matmul(
                    chunk_backend, relative_abundance
                )
                if self.similarities_out is not None:
                    self.similarities_out[i : i + self.chunk_size, :] = chunk_as_numpy
                i += self.chunk_size
        return weighted_abundances

    def is_expensive(self):
        return True


class SimilarityFromSymmetricFunction(Similarity):
    """Similarity computed on the fly from a symmetric pairwise function.

    Wraps a callable similarity(a, b) satisfying
    similarity(a, b) == similarity(b, a). Because of the symmetry,
    only strictly-above-diagonal pairs are evaluated (roughly half the
    work); each chunk's result is assembled from its in-chunk "corner"
    block plus the block against all later rows. If
    `similarities_out` was provided, the full symmetric matrix is
    materialized at the end (transposed copy added, diagonal set to
    1.0).

    Parameters
    ----------
    func : callable
        Function taking two feature rows and returning their
        similarity. Decorate with `batch_similarity` to accept whole
        chunks (two 2-D arrays, returning their pairwise matrix) in
        one vectorized call.
    X : ndarray or DataFrame
        Feature array with one row per species; func is applied to
        rows of X.
    chunk_size : int, default=100
        Number of rows of the similarity matrix to generate at once.
    similarities_out, backend :
        As in Similarity.

    See Also
    --------
    SimilarityFromFunction : for non-symmetric callables.
    """
    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        chunk_size: int = 100,
        similarities_out: Union[ndarray, None] = None,
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

    def weighted_similarity_chunk_symmetric(
        self,
        similarity: Callable,
        X: Union[ndarray, DataFrame],
        relative_abundance,
        chunk_size: int,
        chunk_index: int,
        return_Z: bool = True,
    ) -> Tuple[int, ndarray, Union[ndarray, None]]:
        """Delegate to the shared module-level implementation."""
        return weighted_similarity_chunk_symmetric(
            similarity=similarity,
            X=X,
            relative_abundance=relative_abundance,
            backend=self.backend,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            return_Z=return_Z,
        )


class SimilarityFromFunction(SimilarityFromSymmetricFunction):
    """Similarity computed on the fly from a (not necessarily symmetric)
    pairwise function.

    Like its parent, but evaluates func(a, b) for all ordered pairs,
    computing every chunk row against the full row set. Use this when
    the callable is asymmetric; if it is symmetric, prefer
    SimilarityFromSymmetricFunction to halve the evaluations.

    Ray-parallelized variant: SimilarityFromRayFunction in
    sentropy.ray (selected automatically when parallelize=True is
    passed to build_similarity/LCR).
    """
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

    def weighted_similarity_chunk_nonsymmetric(
        self,
        similarity: Callable,
        X: Union[ndarray, DataFrame],
        Y: Union[ndarray, DataFrame, None],
        relative_abundance,
        chunk_size: int,
        chunk_index: int,
        return_Z: bool = True,
    ) -> Tuple[int, ndarray, Union[ndarray, None]]:
        """Delegate to the shared module-level implementation."""
        return weighted_similarity_chunk_nonsymmetric(
            similarity=similarity,
            X=X,
            Y=Y,
            relative_abundance=relative_abundance,
            backend=self.backend,
            chunk_size=chunk_size,
            chunk_index=chunk_index,
            return_Z=return_Z,
        )
