# How do I ... ?

This page is meant to be a handy reference for how to compute any of the 
various quantities in the sentropy package. The table shows the Python call and the
equivalent command-line invocation. Everything goes through one function,
{func}`~sentropy.sentropy.sentropy`, apart from interset ordinariness
({func}`~sentropy.sentropy.interset_ordinariness`).

**Conventions used below**

- `P` — counts (or relative frequencies) of each element, as a 1-D array: a
  dataset with a single class.
- `P_cls` — counts for several classes, as a `dict` of 1-D arrays or a
  `DataFrame` (one row per element, one column per class).
- `S` — similarity matrix (elements × elements). Omit `similarity=` to use the
  identity matrix, i.e. ordinary similarity-*insensitive* entropy.
- `P1`, `P2` — two single-class count vectors over the **same** elements.
- On the command line, `counts.csv` has one column per class and one row per
  element, and `similarity.csv` is the matching similarity matrix
  (see {doc}`cli`).

## Ordinary entropy (no similarity)

| I want… | Python | Command line |
|---|---|---|
| Shannon entropy, as an effective number | `sentropy(P)` | `sentropy -i counts.csv` |
| Shannon entropy in nats | `sentropy(P, eff_no=False)` | `sentropy -i counts.csv -eff_no 0` |
| Rényi entropy / Hill number of order *q* (0 = counting, 2 = Simpson-type, `np.inf` = Berger–Parker-type) | `sentropy(P, q=2)` | `sentropy -i counts.csv -q 2` |

## Similarity-sensitive diversity of one dataset

| I want… | Python | Command line |
|---|---|---|
| Sentropy of the whole dataset | `sentropy(P, similarity=S)` | `sentropy -i counts.csv -similarity similarity.csv` |
| …at several viewpoints *q* | `res = sentropy(P, similarity=S, q=[0, 1, 2, np.inf])`, then `res(q=2)` | `… -q 0 1 2 inf` |
| …for several measures | `res = sentropy(P, similarity=S, measure=["alpha", "beta", "gamma"])`, then `res(measure="beta")` | `… -measure alpha beta gamma` |
| A value for each class | `sentropy(P_cls, similarity=S, level="subset")` (`"class"` is an alias) | `… -level subset` |
| Class-level and dataset-level values together | `sentropy(P_cls, similarity=S, level="both")` | `… -level both` (the CLI default) |
| Results as a tidy table | add `return_dataframe=True` | the CLI always prints a table |

## How each class relates to the whole dataset

These need `P_cls` (several classes) and `level="subset"`. All are measures
passed as `measure=`; see {doc}`concepts` for what they mean.

| I want… | `measure=` |
|---|---|
| Diversity of each class on its own | `"alpha"` (or `"normalized_alpha"`) |
| How **representative** each class is of the dataset | `"rho"` or `"normalized_rho"` (`"rho_hat"` for the rescaled version) |
| How **distinct** each class is from the rest | `"beta"` or `"normalized_beta"` (`"beta_hat"` rescaled) |
| Each class's contribution to the dataset's diversity | `"gamma"` |

Example: `sentropy(P_cls, similarity=S, level="subset", measure="normalized_rho")`.

## Comparing two sets of counts

Pass **two** count arguments. The first is compared *to* the second, and both
must be indexed by the same elements, so the same `S` applies.

| I want… | Python | Command line |
|---|---|---|
| Similarity-sensitive KL divergence (relative sentropy) from `P1` to `P2` | `sentropy(P1, P2, similarity=S)` (`measure="sre"` is the default) | `sentropy -i a.csv b.csv -similarity similarity.csv -measure sre` |
| Similarity-sensitive cross-entropy | `sentropy(P1, P2, similarity=S, measure="sce")` | `… -measure sce` |
| A matrix of values for every pair of classes | `sentropy(P_cls, P_cls, similarity=S, level="subset", return_dataframe=True)` | — |
| The same, as entropies (not exponentiated) | add `eff_no=False` | `… -eff_no 0` |

The KL divergence is not symmetric: `sentropy(P1, P2, …)` and
`sentropy(P2, P1, …)` differ.

## Other measures

| I want… | Python | Command line |
|---|---|---|
| Vendi score | `sentropy(P, similarity=S, measure="vendi")` | `sentropy -i counts.csv -similarity similarity.csv -measure vendi` |
| Ordinariness of new elements `X` with respect to a reference set `Y` | `interset_ordinariness(X, Y, Y_abundance, similarity=f)` | — |

The Vendi score needs an explicit similarity matrix that is symmetric and
positive semi-definite (this is not checked), and cannot be combined with other
measures in one call.

## Large datasets and hardware

| I want… | Python |
|---|---|
| Compute similarities on the fly instead of storing a matrix | `sentropy(P, similarity=f, sfargs=X)`, where `f(x_i, x_j)` returns a similarity and `X` holds the elements |
| Halve the work when `f` is symmetric with ones on the diagonal | add `symmetric=True` |
| Vectorize `f` over blocks of rows | decorate it with `@batch_similarity` (from `sentropy.similarity`) |
| Turn off Ray parallelism | `parallelize=False` |
| Read the similarity matrix from a CSV/TSV file in chunks | `similarity="similarity.csv"`, tuned with `chunk_size=` |
| Use a sparse similarity matrix | pass any `scipy.sparse` matrix as `similarity=` |
| Run on PyTorch / a GPU | `backend="torch"`, and `device="cuda"` or `device="mps"` |

## Accepted input types

| Argument | Accepted types |
|---|---|
| counts (`P`, `P_cls`) | 1-D `ndarray` or tensor (one class); 2-D `ndarray`, tensor, or `DataFrame` (rows = elements, columns = classes); `dict` mapping class name to a 1-D array of equal length. Counts are normalized internally, so frequencies work too. |
| `similarity` | `None` (identity); `ndarray` or tensor; `DataFrame`; `scipy.sparse` matrix; path to a CSV/TSV file (`str`); a callable (requires `sfargs`) |
| `q` | a number ≥ 0 (including `np.inf`), or a list of them. Values above 100 are treated as ∞. |
| `measure` | a name from {doc}`concepts`, or a list of names |
| `level` | `"overall"` (default), `"subset"` (or `"class"`), `"both"` |

## What comes back

| Call | Returns |
|---|---|
| One count argument, one class, one `q`, one measure | a single number |
| One count argument, anything more | a {class}`~sentropy.sentropy.SentropyResult` — call it with `which=`, `q=`, `measure=` to pick a value |
| …with `return_dataframe=True` | a `pandas.DataFrame` with columns `level`, `viewpoint`, and one per measure |
| `measure="vendi"` | a number for `level="overall"`, an array for `"subset"`, a dict for `"both"` (a DataFrame with `return_dataframe=True`) |
| Two count arguments, `level="overall"` | a single number |
| Two count arguments, `level="subset"` | a matrix: rows are classes of the first argument, columns classes of the second (a DataFrame with `return_dataframe=True`) |
| Two count arguments, `level="both"` | a tuple `(overall, subset)` |

All values are **effective numbers** by default; pass `eff_no=False` for
traditional entropies in nats.
