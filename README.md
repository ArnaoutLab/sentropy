![alt text](images/diversity_logo.png)

# <h1> <i>sentropy</i>: A Python package for measuring the composition of complex datasets</h1>

[![Python version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/release/python-380/)
[![Tests](https://github.com/ArnaoutLab/sentropy/actions/workflows/tests.yml/badge.svg)](https://github.com/ArnaoutLab/sentropy/actions/workflows/tests.yml)

# About

`sentropy` calculates similarity-sensitive entropies (S-entropy), plus Shannon entropy and the other traditional Rényi entropies (of which Shannon entropy is the best known).

- **Shannon entropy** is a weighted sum of the relative probabilities of unique elements in a system (e.g. a dataset)
- **Rényi entropies** generalize Shannon entropy by allowing for different weightings (viewpoint parameter *q*)
- **S-entropy** generalizes Rényi entropies by incorporating elements' similarities and differences via a **similarity matrix** (often constructed using a **similarity function**)
- Exponentiating entropy yields **effective-number/D-number forms**, which put entropies in the same, natural units—**effective numbers**—among other advantages
- `sentropy` calculates multiple S-entropic **measures**, including *ɑ*, *β/𝜌*, *ɣ*, at both the subset (classes) **level** and for the overall (data)set

For more background, see [Leinster 2020](https://arxiv.org/abs/2012.02113) and references therein.

[Installation](#installation) | [Basic usage](#basic-usage) | [Shannon entropy](#shannon-entropy) | [Shannon-type S-entropy](#s-entropy) | [ɑ, β, ɣ, and multiple *q*](ɑ-β-ɣ-and-multiple-q) | [Passing similarity functions](#passing-a-similarity-function) | [Representativeness](#representativeness) | [Relative S-entropy and pandas](#relative-s-entropies-between-two-classes-as-a-pandas-dataframe) | [Ordinariness](#ordinariness) | [Cite this work](#how-to-cite-this-work)

# Installation

`sentropy` is available via the Python Package Index (PyPI) and on GitHub at [https://github.com/ArnaoutLab/sentropy](https://github.com/ArnaoutLab/sentropy)

To install:
```
pip install sentropy
```

For unit tests (including a coverage report), run:
```
pip install 'sentropy[tests]'
pytest --pyargs sentropy --cov sentropy
```
The test suite runs successfully on Macintosh, Windows, and Unix systems. If you get errors, you may be missing requirements that you need to install.

# Basic usage

The workhorse function is `sentropy.sentropy`:

```
from sentropy import sentropy
```

Main arguments: 

- `P`, a list-like object (e.g. a list, a numpy array) of relative frequencies (or a dictionary of these, one per subset or class; see [below](#how-well-each-of-two-classes-represents-the-whole-dataset))
- `similarity`, which can be passed as a matrix or a function; the default is the identity matrix $I$
- `q`, the viewpoint parameter; default is `q=1.`
- `measure`, which can be `alpha`, `beta`, `gamma`, or others in the Leinster-Cobbold-Reeve (LCR) framework; the default is `alpha`
- `level`, which can be `overall` (a.k.a. `dataset`) or `subset` (a.k.a. `class`); the default is `overall`

We recommend running the below in [ipython](https://ipython.org/).

## Shannon entropy

When the similarity matrix is the identity matrix—`sentropy`'s default for `similarity`—there is no similarity between elements *i*≠*j* and S-entropy reduces to traditional (Rényi) entropy. At the default `q=1`, this is Shannon entropy. Therefore passing `sentropy` only a `P` yields Shannon entropy, in effective-number form:

```
from sentropy import sentropy
import numpy as np

P = np.array([0.7, 0.3])          # two unique elements comprising 70% and 30% of this single-class dataset, respectively
D1 = sentropy(P)                  # S-entropy *without* similarity at default q (q=1) = Shannon entropy,
                                  #   returned in default effective-number (D-number) form (preferred)
                                  # Equivalent to also including the arguments similarity=np.eye(2), measure="alpha", level="both"
H1 = sentropy(P, eff_no=False)    # traditional, non-effective-number form (eff_no=False)

print(f"D1: {D1:.2f} elements")
print(f"which corresponds to H1 = {H1:.2f} nats (= {np.log2(D1):.2f} bits)")
```
Expected output:
```
D1: 1.84 elements
which corresponds H1 = 0.61 nats (= 0.88 bits)
```
**Intuition behind effective numbers.** Suppose instead of the frequencies being (0.7, 0.3) they were (0.999999, 0.000001). The population would consist almost completely of element 1. In an intuitive sense, element 2 "shouldn't count" as much. In this case, the effective-number form would be `sentropy(np.array([0.999999, 0.000001]))` = 1.00001. (The traditional form would be ~0, at 1e-5 nats or 2e-5 bits.) In contrast, (0.7, 0.3) is far less skewed, and so the effective number is far closer to 2, at 1.84. The effective number would be *equal* to 2 if the frequencies were equal (0.5, 0.5).

## S-entropy

Passing a non-identity similarity matrix (similarity≠*I*) results in S-entropy–here, Shannon-type (i.e. *q*=1):
```
from sentropy import sentropy
import numpy as np

P = np.array([0.7, 0.3])                      # same dataset as above (two elements, one class)
S = np.array([                                # similarity matrix
  [1. , 0.2],                                 # 20% similar to each other
  [0.2, 1. ],
  ])
D1Z = sentropy(P, similarity=S)               # D-number form (preferred). Note defaults: level="both", measure="alpha", q=1.
H1Z = sentropy(P, similarity=S, eff_no=False) # traditional form

print(f"D1Z: {D1Z:.2f} elements")              
print(f"which corresponds H1Z = {H1Z:.2f} nats")
```
Expected output:
```
D1Z: 1.55 elements
which corresponds to H1Z = 0.44 nats
```
Non-zero similarity between elements 1 and 2 reduces the overall entropy of the system relative to the [first example above](#vanilla-shannon-entropy), in which there was zero similarity between the two elements. This can be thought of as a reduction in diversity (whence the "D" in D-number). The "Z" is a convention that means "non-trivial similarity."

## ɑ, β, ɣ, and multiple *q*

To get results for multiple `q` (e.g. 0, 1, and ∞), multiple measures (e.g. alpha and beta), and/or both levels (overall and subset), pass a list-like object to the relevant argument; `sentropy()` returns an object with relevant values:
```
from sentropy import sentropy
import numpy as np

P = np.array([0.7, 0.3])                      # same dataset as above (two elements, one class)
S = np.array([                                # same similarity matrix as above
  [1. , 0.2],
  [0.2, 1. ],
  ])
qs = [0., 1., 2., np.inf]                     # multiple viewpoint parameters
measures = ["alpha", "beta", "gamma"]         # multiple measures
DZ = sentropy(P, similarity=S,                # S-entropy...
              qs=qs,                          #   ...at multple qs...
              ms=measures)                    #   ...for multiple measures
                                              # note when the result of the sentropy() call contains
                                              # multiple values, it returns a function
for measure in measures:
  for q in qs:
    DqZ = DZ(q=q, m=measure, which='overall') # D-number form (preferred)
    print(f"D{q:.0f}Z {measure}:\t{DqZ:.2f} elements")
  print()
```
Expected output:
```
D0Z alpha:	1.60 elements
D1Z alpha:	1.55 elements
D2Z alpha:	1.51 elements
DinfZ alpha:	1.32 elements

D0Z beta:	1.00 elements
D1Z beta:	1.00 elements
D2Z beta:	1.00 elements
DinfZ beta:	1.00 elements

D0Z gamma:	1.60 elements
D1Z gamma:	1.55 elements
D2Z gamma:	1.51 elements
DinfZ gamma:	1.32 elements
```
Values never rise, and almost always fall, with increasing *q*. *q*=0, 1, 2, and ∞ can be thought of as "counting-", "Shannon-", "Simpson-", and "Berger-Parker-type" S-entropy, respectively. *ɑ*=*ɣ* in this example because there is only one class (see [Leinster 2020](https://arxiv.org/abs/2012.02113)).

## Passing a similarity function

When the similarity matrix would be too large to hold in memory, a function can be passed to `similarity`, allowing similarilty to be calculated on the fly:
```
from sentropy import sentropy
import numpy as np

# define a dataset consisting of two amino-acid sequences
elements = np.array(['CARDYW', 'CTRDYW', 'CAKDYW'])     # amino-acid sequences (reminiscent of human IGH CDR3s)
P = np.array([20, 1, 1])                                # the first is present 20 times; the second two are each present once

# define a similarity function where similarity decreases with edit distance between the sequences
from polyleven import levenshtein as edit_distance
def similarity_function(i, j):                          # i, j members of elements
    return 0.3**edit_distance(i, j)

# calculate datset sentropy (at the defaults meausure="alpha" and q=1.)
D1Z = sentropy(P, similarity=similarity_function,
               sfargs=elements)                         # sfargs contains arguments needed by the similarity_function
H1Z = np.log(D1Z)                                       # traditional form
print(f"D1Z: {D1Z:.2f} elements")
print(f"H1Z: {H1Z:.2f} nats")
```
Expected output:
```
D1Z: 1.18 elements
H1Z: 0.16 nats
```
The strings in this example are amino acid sequences, such as might exist in a next-generation sequencing dataset. CARDYW outnumbers the other two 10:1; CTRDYW and CAKDYW might be sequencing errors or mutations. The three sequences are very similar. The combination of these two factors—a big difference in relative frequencies and small differences in sequence—results in this three-element dataset having an effective number of only 1.18 elements. 

## Representativeness

Suppose you have a dataset of fruits that has two classes, apples and oranges, and you want to know how representative each class is of the whole dataset. `sentropy` lets you do this by measuring the representativeness of each class (*𝜌*, "rho"), which is the reciprocal of *β* diversity (which measures distinctiveness):
```
from sentropy import sentropy
import numpy as np

# a dataset with two classes, "apples" and "oranges"
C1 = np.array([12, 3, 0, 0])                  # apples; e.g. 12 granny smith and 3 gala
C2 = np.array([0,  0, 4, 4])                  # oranges; e.g. 4 navel and 4 cara cara
P  = {"apples": C1, "oranges": C2}            # package the classes as P
S = np.array([                                # similarities of all elements, regardless of class
  [1.,  0.7, 0.0, 0.0],                       #    note here the non-zero similarity between apples and oranges
  [0.7, 1.,  0.1, 0.3],
  [0.0, 0.1, 1.,  0.9],
  [0.0, 0.3, 0.9, 1. ],
  ])

D1Z = sentropy(P, similarity=S,
               level="subset",                # level="class" is identical; an alias/synonym
               ms="normalized_rho")
R1 = D1Z(which="apples")                      # note, no need to pass a measure to "m" or a viewpoint to "q"
R2 = D1Z(which="oranges")                     # because D1Z only computed 1 measure and 1 viewpoint anyway
print(f"Normalized rho of Class 1 (apples):\t{R1:.2f}")
print(f"Normalized rho of Class 2 (oranges):\t{R2:.2f}")
```
Expected output:
```
Normalized rho of Class 1 (apples):	0.67
Normalized rho of Class 2 (oranges):	0.38
```
The dataset has more apples, and so the "apples" class (Class 1) is more representative of the dataset.

## Relative S-entropy and pandas dataframes

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

display(D1Z)                              # (ipython) S-entropies on the diagonals; relative S-entropies on the off-diagonals
```
Expected output:
```
     level  viewpoint     alpha
0  overall          1  2.142100
1   apples          1  2.207692
2  oranges          1  2.078457
```

## Ordinariness

Suppose you have two datasets of animals. The first dataset consists of fish (a vertebrate) and ladybugs (an invertebrate). The second dataset consists of bees, butterflies, and lobsters—all invertebrates. The two datasets are disjoint—there are no fish or ladybugs in the second dataset—but  genetically speaking, there are similarities between the elements of the two datasets. Suppose you want some measure of how similar each element of the first dataset is, to the second dataset, i.e. how much each element would "belong" in the second dataset. This is measured by *ordinariness*: ladybugs would be more "ordinary" in the second dataset, since it ladybugs are invertebrates. Strictly speaking this can be calculated without `sentropy`, but `sentropy` provides speedups (see documentation).

```
import numpy as np
P = np.array([5000, 2000, 3000])             # frequencies of a dataset of bees, butterflies, and lobsters, respectively
S_fish    = np.array([0.22, 0.27, 0.28])     # fish's genetic similarities to bee, butterfly, and lobster
S_ladybug = np.array([0.60, 0.55, 0.45])     # ladybug's genetic similarities to each of these
S = np.stack([S_fish, S_ladybug])
S @ (P/P.sum())                              # ordinariness of fish and ladybugs in the bees/butterflies/lobsters dataset
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


