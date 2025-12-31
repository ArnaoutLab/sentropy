![alt text](images/diversity_logo.png)

# <h1> <i>sentropy</i>: A Python package for measuring the composition of complex datasets</h1>

[![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/python-380/)
[![Tests](https://github.com/ArnaoutLab/sentropy/actions/workflows/tests.yml/badge.svg)](https://github.com/ArnaoutLab/sentropy/actions/workflows/tests.yml)

# About

`sentropy` calculates similarity-sensitive entropies (S-entropy), plus traditional Shannon entropy and the other Rényi entropies (of which Shannon entropy is the best known) as special cases.

- **Shannon entropy** is a weighted sum of the relative probabilities of unique elements in a system (e.g. a dataset).
- **Rényi entropies** generalize Shannon entropy by allowing for different weightings (viewpoint parameter *q*).
- **S-entropy** generalizes Rényi entropies by incorporating elements' similarities and differences via a **similarity matrix** (often constructed using a **similarity function**).
- Exponentiating entropy yields **effective-number/D-number forms**, which put entropies in the same, natural units—**effective numbers**—among other advantages.
- `sentropy` calculates multiple S-entropic **measures**, including $\alpha, \beta/\rho, \gamma$ at both the subset (classes) **level** and for the overall (data)set

For more background, see [Leinster 2020](https://arxiv.org/abs/2012.02113) and references therein.

[Installation](#installation) | [Basic usage](#basic-usage) | 

# Installation

```
pip install sentropy
```

# Basic usage

The workhorse function is `sentropy.sentropy`:

```
from sentropy import sentropy
```

`sentropy`'s only required argument is a list-like object (e.g. a list, a numpy array) of relative frequencies `P`. 

The most important optional arguments are:

- `similarity`, which can be passed as a matrix or a function; the default is the identity matrix $I$
- `q`, the viewpoint parameter; default is `q=1.`
- `measure`, which can be `alpha`, `beta`, `gamma`, or others in the Leinster-Cobbold-Reeve (LCR) framework; the default is `alpha`
- `level`, which can be `overall` (a.k.a. `dataset`) or `subset` (a.k.a. `class`); the default is `overall`

## Vanilla Shannon entropy

When the similarity matrix is the identity matrix---`sentropy`'s default for `similarity`---there is no similarity between elements *i*≠*j* and S-entropy reduces to traditional (Rényi) entropy. At the default `q=1`, this is Shannon entropy. Therefore passing `sentropy` only a `P` yields Shannon entropy, in effective-number form.

```
from sentropy import sentropy
import numpy as np

P = np.array([0.7, 0.3])        # two unique elements comprising 70% and 30% of the dataset, respectively
D1 = sentropy(P)                # S-entropy *without* similarity at default q (q=1) = Shannon entropy,
                                # returned in default effective-number (D-number) form (preferred)
print(f"D1: {D1:.1f}")          # Note defaults: level="both", measure="alpha", q=1.

H1 = sentropy(P, eff_no=False)  # traditional, non-effective-number form (eff_no=False)
print(f"H1: {H1:.1f}")
```

## Shannon-type (i.e. *q*=1) S-entropy

Passing a non-*I* similarity results in S-entropy.
```
from sentropy import sentropy
import numpy as np

P = np.array([0.7, 0.3])                      # same dataset as above
S = np.array([                                # similarity matrix
  [1. , 0.2],                                 # 20% similar to each other
  [0.2, 1. ],
  ])
D1Z = sentropy(P, similarity=S)               # D-number form (preferred). Note defaults: level="both", measure="alpha", q=1.
print(f"D1Z: {D1Z:.1f}")              

H1Z = sentropy(P, similarity=S, eff_no=False) # traditional form
print(f"H1Z: {H1Z:.1f}")
```

## S-entropy with multiple measures and viewpoint parameters

To get results for multiple `q` (e.g. 0, 1, and ∞), multiple measures (e.g. alpha and beta), and/or both levels (overall and subset), pass a list-like object to the relevant argument; `sentropy()` returns an object with relevant values.
```
from sentropy import sentropy
import numpy as np

P = np.array([0.7, 0.3])                      # same dataset as above
S = np.array([                                # same similarity matrix as above
  [1. , 0.2],
  [0.2, 1. ],
  ])
qs = [0., 1., 2., np.inf]                     # multiple viewpoint parameters
ms = ["alpha", "beta", "gamma"]               # multiple measures
DZ = sentropy(P, similarity=S,                # S-entropy...
              qs=qs,                          #   ...at multple qs...
              ms=ms)                          #   ...for multiple measures
for q in qs:
  for m in ms:
    DqZ = DZ(q=q, m=m, which='overall')       # D-number form (preferred)
    HqZ = np.log(DqZ)                         # traditional form
    print(f"D{q}Z {m}: {DqZ:.1f}")
    print(f"H{q}Z {m}: {HqZ:.1f}")
```

## Similarity on the fly

When the similarity matrix would be too large to hold in memory, a function can be passed to `similarity`.
```
from sentropy import sentropy
import numpy as np

# define a dataset consisting of two amino-acid sequences
elements = np.array(['CARDYW', 'CTRDYW'])
P = np.array([10, 1])                                   # the first is present 10 times; the second is present once

# define a similarity function where similarity decreases with edit distance between the sequences
from polyleven import levenshtein as edit_distance
def similarity_function(i, j):                          # i, j members of elements
    return 0.3**edit_distance(i, j)

# calculate datset sentropy (at the defaults meausure="alpha" and q=1.)
D1Z = sentropy(P, similarity=similarity_function,
               sfargs=elements)                         # sfargs contains arguments needed by the similarity_function
H1Z = np.log(D1Z)                                       # traditional form
print(f"D1Z: {D1Z:.1f}")
print(f"H1Z: {H1Z:.1f}")
```

## How well each of two classes represents the whole dataset

Suppose you have a dataset of fruits that has two classes, apples and oranges, and you want to know how representative each class is of the whole dataset. Representativeness (𝜌) is the reciprocal of beta diversity, which measures distinctiveness. 
```
from sentropy import sentropy
import numpy as np

# a dataset with two classes, "apples" and "oranges"
C1 = np.array([5, 3, 0, 0])                   # apples; e.g. 5 McIntosh and 3 gala
C2 = np.array([0, 0, 6, 2])                   # oranges; e.g. 6 navel and 2 cara cara
P  = {"apples": C1, "oranges": C2}            # package the classes as P
S = np.array([                                # similarities of all elements, including between classes
  [1.,  0.8, 0.2, 0.1],                       #    note here the non-zero similarity between apples and oranges
  [0.8, 1.,  0.1, 0.3],
  [0.2, 0.1, 1.,  0.9],
  [0.1, 0.3, 0.9, 1. ],
  ])

D1Z = sentropy(P, similarity=S, level="subset",            # level="subset" is identical; an alias/synonym
               ms="normalized_rho")
R1 = D1Z(which="apples")                                   # note, no need to pass a measure to "m" or a viewpoint to "q"
R2 = D1Z(which="oranges")                                  # because D1Z only computed 1 measure and 1 viewpoint anyway
print(f"Normalized rho of class 1: {R1:.2f}")
print(f"Normalized rho of class 2: {R2:.2f}")
```

## Relative S-entropies between two classes as a pandas DataFrame

Same dataset as above, except now results are returned as a dataframe. The below calculates the similarity-sensitive version of traditional relative entropy at q=1 (a.k.a. Kullback-Leibler divergence, information divergence, etc.).
```
from sentropy import sentropy
import numpy as np

# a dataset with two classes, "apples" and "oranges"
C1 = np.array([5, 3, 0, 0])                   # apples; e.g. 5 McIntosh and 3 gala
C2 = np.array([0, 0, 6, 2])                   # oranges; e.g. 6 navel and 2 cara cara
P  = {"apples": C1, "oranges": C2}            # package the classes as P
S = np.array([                                # similarities of all elements, including between classes
  [1.,  0.8, 0.2, 0.1],                       #    note here the non-zero similarity between apples and oranges
  [0.8, 1.,  0.1, 0.3],
  [0.2, 0.1, 1.,  0.9],
  [0.1, 0.3, 0.9, 1. ],
  ])

D1Z = sentropy(P, similarity=S,
               return_dataframe=True)

display(D1Z)                              # S-entropies on the diagonals; relative S-entropies on the off-diagonals
```

## Ordinariness

Suppose you have two datasets of animals. The first dataset consists of fish (a vertebrate) and ladybugs (an invertebrate). The second dataset consists of bees, butterflies, and lobsters—all invertebrates. The two datasets are disjoint—there are no fish or ladybugs in the second dataset—but  genetically speaking, there are similarities. Suppose you want some measure of how similar each element of the first dataset is, to the second dataset: how much would each element "belong" in the second dataset. This is measured by *ordinariness*: ladybugs would be more "ordinary" in the second dataset, since it is an invertebrate. Strictly speaking this can be calculated without `sentropy`, but `sentropy` provides speedups (see documentation).

```
import numpy as np
P = np.array([5000, 2000, 3000])             # frequencies of a dataset of bees, butterflies, and lobsters, respectively
S_fish    = np.array([0.22, 0.27, 0.28])     # fish's genetic similarities to bee, butterfly, and lobster
S_ladybug = np.array([0.60, 0.55, 0.45])     # ladybug's genetic similarities to each of these
S = np.stack([S_fish, S_ladybug])
S @ (P/P.sum())                              # ordinariness of fish and ladybugs in the bees/butterflies/lobsters dataset
```


## Availability and installation
`sentropy` is available on GitHub at https://github.com/ArnaoutLab/sentropy. It can be installed by running

`pip install sentropy`

from the command-line interface. The test suite runs successfully on Macintosh, Windows, and Unix systems. The unit tests (including a coverage report) can be run after installation by

```
pip install 'sentropy[tests]'
pytest --pyargs sentropy --cov sentropy
```

## How to cite this work

If you use this package, please cite it as:

Nguyen et al., <i>sentropy: A Python Package for Measuring The Composition of Complex Datasets</i>. <https://github.com/ArnaoutLab/diversity>


# Applications

For applications of the `sentropy` package to various fields (immunomics, metagenomics, medical imaging and pathology), we refer to the Jupyter notebooks below:

- [Immunomics](https://github.com/ArnaoutLab/diversity_notebooks_and_data/blob/main/immunomics/immunomics_fig3.ipynb)
- [Metagenomics](https://github.com/ArnaoutLab/diversity_notebooks_and_data/blob/main/metagenomics/metagenomics_figs4-5.ipynb)
- [Medical imaging](https://github.com/ArnaoutLab/diversity_notebooks_and_data/blob/main/medical_imaging/medical_imaging_fig6-7.ipynb)
- [Pathology](https://github.com/ArnaoutLab/diversity_notebooks_and_data/blob/main/pathology/pathology_fig8.ipynb)

The examples in the Basic usage section are also made available as a notebook [here](https://github.com/ArnaoutLab/diversity_notebooks_and_data/blob/main/fruits_and_animals/fruits_and_animals_fig1_2.ipynb). For more information, please see our [preprint](https://arxiv.org/abs/2401.00102).

# Alternatives

To date, we know of no other python package that implements the partitioned frequency- and similarity-sensitive diversity measures defined by [Reeve at al.](https://arxiv.org/abs/1404.6520). However, there is a [R package](https://github.com/boydorr/rdiversity) and a [Julia package](https://github.com/EcoJulia/Diversity.jl).


