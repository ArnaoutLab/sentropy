# Tutorial

Every example in this tutorial that is marked with an output is run as a test
whenever the documentation is built (`sphinx-build -b doctest`), so the code and
numbers here are guaranteed to match the released package. Examples that need
optional packages, Ray, or a GPU are shown without being run.

## Shannon entropy

With no similarity matrix, `sentropy` returns ordinary Shannon entropy
($q=1$) in effective-number form:

```{testcode}
P = np.array([0.7, 0.3])        # two elements: 70% and 30% of one dataset
D1 = sentropy(P)                # effective-number form (preferred)
H1 = sentropy(P, eff_no=False)  # traditional form, in nats

print(f"D1 = {D1:.2f} elements, which corresponds to H1 = {H1:.2f} nats (= {np.log2(D1):.2f} bits)")
```

```{testoutput}
D1 = 1.84 elements, which corresponds to H1 = 0.61 nats (= 0.88 bits)
```

**Why effective numbers?** If the frequencies were $(0.999999, 0.000001)$ the
dataset would be almost entirely one element, and the effective number is
$\approx 1.00001$ (the traditional entropy is $\approx 10^{-5}$ nats). At
$(0.7, 0.3)$ it is 1.84; at $(0.5, 0.5)$ it would be exactly 2.

## Adding similarity

Pass a similarity matrix to get similarity-sensitive entropy (here
Shannon-type, $q=1$):

```{testcode}
S = np.array([[1.0, 0.2],       # the two elements are 20% similar
              [0.2, 1.0]])
D1Z = sentropy(P, similarity=S)
H1Z = sentropy(P, similarity=S, eff_no=False)

print(f"D1Z = {D1Z:.2f} elements, which corresponds to H1Z = {H1Z:.2f} nats")
```

```{testoutput}
D1Z = 1.55 elements, which corresponds to H1Z = 0.44 nats
```

Because the two elements are partly similar, the dataset is less diverse than
when they were completely distinct (1.84). The "Z" is a convention meaning
"non-trivial similarity". The matrix may be a NumPy array, a pandas
DataFrame, or a SciPy sparse matrix.

## Several viewpoints and measures at once

Pass lists to `q` and/or `measure`. The call then returns a
{class}`~sentropy.sentropy.SentropyResult`, which you call to pick out a value:

```python
qs = [0.0, 1.0, 2.0, np.inf]
measures = ["alpha", "beta", "gamma"]
DZ = sentropy(P, similarity=S, q=qs, measure=measures)

for measure in measures:
    for q in qs:
        print(f"D{q:.0f}Z {measure}: {DZ(q=q, measure=measure, which='overall'):.2f}")
```

Because this dataset has a single class, $\alpha = \gamma$ and $\beta = 1$.

## Representativeness of classes

Give `sentropy` several classes (a dict or DataFrame) and ask for
class-level (`level="subset"`) values. Here we measure how
*representative* ($\rho$) apples and oranges are of the whole fruit dataset:

```{testcode}
P_apples  = np.array([12, 3, 0, 0])   # 12 Granny Smith, 3 McIntosh
P_oranges = np.array([0, 0, 4, 4])    # 4 navel, 4 cara cara
P_cls = {"apples": P_apples, "oranges": P_oranges}

S_fruit = np.array([[1.0, 0.7, 0.0, 0.0],
                    [0.7, 1.0, 0.1, 0.3],
                    [0.0, 0.1, 1.0, 0.9],
                    [0.0, 0.3, 0.9, 1.0]])

res = sentropy(P_cls, similarity=S_fruit, level="subset", measure="normalized_rho")
print(f"Normalized rho of Class 1 (apples):  {res(which='apples'):.2f}")
print(f"Normalized rho of Class 2 (oranges): {res(which='oranges'):.2f}")
```

```{testoutput}
Normalized rho of Class 1 (apples):  0.67
Normalized rho of Class 2 (oranges): 0.38
```

The dataset contains more apples, so apples are more representative. Add
`return_dataframe=True` to get the same numbers as a table.

## Relative sentropy between two classes

Passing **two** count arguments compares them. This computes the
similarity-sensitive KL divergence (relative sentropy), in each direction:

```{testcode}
S_kl = np.array([[1.0, 0.7, 0.1, 0.1],
                 [0.7, 1.0, 0.1, 0.3],
                 [0.1, 0.1, 1.0, 0.9],
                 [0.1, 0.3, 0.9, 1.0]])

KL_12 = sentropy(P_apples, P_oranges, similarity=S_kl)   # apples -> oranges
KL_21 = sentropy(P_oranges, P_apples, similarity=S_kl)   # oranges -> apples

print("Effective-number form of similarity-sensitive KL divergence (KLZ) of:")
print(f"  apples to oranges: {KL_12:.2f}")
print(f"  oranges to apples: {KL_21:.2f}")
```

```{testoutput}
Effective-number form of similarity-sensitive KL divergence (KLZ) of:
  apples to oranges: 7.84
  oranges to apples: 8.03
```

KL divergence is not symmetric, so the two values differ. Use
`measure="sce"` for cross-sentropy. To get all pairs of classes at once pass the
same multi-class object twice with `level="subset"`:

```python
import pandas as pd

df = pd.DataFrame(P_cls)
sentropy(df, df, similarity=S_kl, level="subset", return_dataframe=True)
```

## Similarity from a function

When the similarity matrix is too large to hold in memory, pass a function and
the data it needs (`sfargs`). Here similarity falls off with the edit distance
between amino-acid sequences (requires `pip install polyleven`):

```python
from polyleven import levenshtein

elements = np.array(["CARDYW", "CTRDYW", "CAKDYW"])
counts = np.array([20, 1, 1])

def similarity_function(i, j):
    return 0.3 ** levenshtein(i, j)

D1Z = sentropy(counts, similarity=similarity_function, sfargs=elements)
# D1Z ≈ 1.18
```

By default the computation is parallelized with Ray; pass `parallelize=False`
to turn that off, and `symmetric=True` if your function is symmetric with ones
on the diagonal (about a 2× speed-up).

To use Ray's resources deliberately, initialize it yourself before calling
`sentropy`, and shut it down afterwards to free memory:

```python
import ray
ray.init(num_cpus=4)
# ... call sentropy ...
ray.shutdown()
```

### Batch similarity functions

A function that computes a whole block of the similarity matrix at once is much
faster. Decorate it with {func}`~sentropy.similarity.batch_similarity`; it must
take two 2-D arrays `A` (n×d) and `B` (m×d) and return an n×m array
(requires `pip install rapidfuzz`):

```python
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from sentropy.similarity import batch_similarity

@batch_similarity
def similarity_function(A, B):
    distances = process.cdist(list(A), list(B), scorer=Levenshtein.distance, workers=-1)
    return 0.3 ** distances

D1Z = sentropy(counts, similarity=similarity_function, symmetric=True, sfargs=elements)
# D1Z ≈ 1.18, as with the scalar version
```

## Interset ordinariness

How "ordinary" would each element of one set be in a different reference set?
(Requires Ray.)

```python
from sentropy import interset_ordinariness

X = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])             # elements to score
Y = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])  # reference set
Y_counts = np.array([100, 50, 25, 25])

interset_ordinariness(X=X, Y=Y, Y_abundance=Y_counts,
                      similarity=lambda x, y: np.dot(x, y))
# array([0.7375, 0.2625, 0.7   ])
```

## Vendi score

```python
sentropy(P, similarity=S, measure="vendi")
```

The Vendi score requires a symmetric, positive semi-definite similarity matrix;
this is not checked. It cannot be combined with other measures in one call.

## PyTorch and GPUs

Pass `backend="torch"`, and `device="cuda"` or `device="mps"` to run on a GPU:

```python
res = sentropy(P_cls, similarity=S_fruit, level="subset",
               measure="normalized_rho", backend="torch", device="mps")
res(which="apples")   # a torch tensor
```
