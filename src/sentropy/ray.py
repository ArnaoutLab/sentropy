# ray.py

from typing import List, Any, Callable, Union
from numpy import ndarray
from os import cpu_count
from pandas import DataFrame

# avoid mypy error: see https://github.com/jorenham/scipy-stubs/issues/100
import ray  # type: ignore
from torch import Tensor

from sentropy.similarity import (
    SimilarityFromFunction,
    SimilarityFromSymmetricFunction,
    weighted_similarity_chunk_nonsymmetric,
    weighted_similarity_chunk_symmetric,
)
from sentropy.backend import get_backend

def _recommend_chunk_params(n_X, n_Y):
    num_cpus = cpu_count() or 4
    
    # Fall back: target enough tasks for 4× CPU parallelism,
    # but at least 256 pairs per chunk
    chunk_size = max(
        max(1, n_X // (4 * num_cpus)),
        max(1, 1024 // n_Y) if n_Y > 0 else 1
    )
    
    chunk_size = min(chunk_size, n_X)  # can't exceed n_X
    max_inflight = min(4 * num_cpus, 256)  # cap at 256 to limit memory
    
    return chunk_size, max_inflight

def _interset_weighted_abundances_ray(
    similarity,
    X,
    Y,
    relative_abundance,
    chunk_size,
    max_inflight_tasks,
    backend,
):
    weighted_similarity_chunk = ray.remote(
        weighted_similarity_chunk_nonsymmetric
    )

    X_ref = ray.put(X)
    Y_ref = ray.put(Y)
    abundance_ref = ray.put(relative_abundance)

    futures = []
    results = []

    def process_refs(refs):
        for chunk_index, abundance_chunk, _ in ray.get(refs):
            results.append((chunk_index, abundance_chunk))

    for chunk_index in range(0, X.shape[0], chunk_size):
        if len(futures) >= max_inflight_tasks:
            ready_refs, futures = ray.wait(futures)
            process_refs(ready_refs)

        futures.append(
            weighted_similarity_chunk.remote(
                similarity=similarity,
                X=X_ref,
                Y=Y_ref,
                relative_abundance=abundance_ref,
                backend=backend,
                chunk_size=chunk_size,
                chunk_index=chunk_index,
                return_Z=False,
            )
        )

    process_refs(futures)

    results.sort(key=lambda x: x[0])

    return backend.concatenate([result for _, result in results])


class SimilarityFromRayFunction(SimilarityFromFunction):
    """Similarity from a callable pairwise function, parallelized
    with Ray.

    Same semantics as its parent SimilarityFromFunction — func(a, b)
    is evaluated for all ordered pairs against the full row set — but
    the per-chunk computations are dispatched as Ray remote tasks and
    gathered asynchronously, so chunks are computed concurrently across
    cores. Selected automatically by build_similarity()/LCR() when a
    callable similarity and parallelize=True are passed.

    Parameters
    ----------
    func : callable
        Pairwise similarity function; see
        SimilarityFromSymmetricFunction. Must be picklable to be sent
        to Ray workers.
    X : ndarray or DataFrame
        Feature array with one row per species; func is applied to
        rows of X. Placed in the Ray object store once (ray.put) and
        shared by all workers rather than shipped per task.
    similarities_out : ndarray, optional
        As in Similarity. Each worker writes its chunk's rows into
        this buffer on completion. Pass None to skip materializing
        the matrix.
    backend : BaseBackend, optional
        As in Similarity. Note each worker executes the matmul with
        the same backend.

    Notes
    -----
    `chunk_size` and the maximum number of in-flight tasks are chosen
    automatically by _recommend_chunk_params(): targeting roughly
    4x the number of CPUs in concurrent tasks (capped at 256 to bound
    memory), with chunks sized so that each task handles a reasonable
    number of row pairs. When the in-flight limit is reached, further
    chunks are queued until earlier ones finish (ray.wait), then
    results are reassembled in chunk order — so the output is
    identical to the sequential version, just faster.
    """

    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        similarities_out: Union[ndarray, None] = None,
        backend=None,
    ) -> None:
        chunk_size, max_inflight_tasks = _recommend_chunk_params(X.shape[0], X.shape[0])
        super().__init__(func, X, chunk_size, similarities_out)
        self.max_inflight_tasks = max_inflight_tasks
        self.backend = backend or get_backend("numpy")

    def get_Y(self):
        return None

    def weighted_abundances(
        self,
        relative_abundance: Union[ndarray, Tensor],
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


class SimilarityFromSymmetricRayFunction(SimilarityFromSymmetricFunction):
    """Similarity from a symmetric callable pairwise function,
    parallelized with Ray.

    Same semantics as its parent
    SimilarityFromSymmetricFunction — only strictly-above-diagonal
    pairs are evaluated, and each chunk's result is assembled from
    its in-chunk "corner" block plus the block against all later rows
    — but the per-chunk computations are dispatched as Ray remote
    tasks and run concurrently. Selected automatically by
    build_similarity()/LCR() when a symmetric callable similarity and
    parallelize=True are passed.

    Parameters
    ----------
    func, X, similarities_out, backend :
        As in SimilarityFromRayFunction.

    Notes
    -----
    Chunk parameters are chosen by _recommend_chunk_params() as in
    SimilarityFromRayFunction. Unlike the non-symmetric variant, each
    worker returns only its chunk's additive contribution to the
    ordinariness vector; the driver accumulates them as tasks
    complete (in arbitrary completion order), so the per-task memory
    footprint stays small. If `similarities_out` was provided, the
    strictly-upper-triangular rows are filled in by the workers and
    the driver completes the matrix at the end (adding its transpose
    and setting the diagonal to 1.0).
    """

    def __init__(
        self,
        func: Callable,
        X: Union[ndarray, DataFrame],
        similarities_out: Union[ndarray, None] = None,
        backend=None,
    ) -> None:
        chunk_size, max_inflight_tasks = _recommend_chunk_params(X.shape[0], X.shape[0])
        super().__init__(func, X, chunk_size, similarities_out)
        self.max_inflight_tasks = max_inflight_tasks
        self.backend = backend or get_backend("numpy")

    def weighted_abundances(
        self,
        relative_abundance: Union[ndarray, Tensor],
    ) -> Union[ndarray, Tensor]:
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
                ready_refs, futures = ray.wait(futures)
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
