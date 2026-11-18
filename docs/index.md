# sentropy

*A Python package for revealing hidden differences in complex datasets.*

`sentropy` calculates **similarity-sensitive entropies** ("sentropy"), plus
Shannon entropy and the other Rényi entropies, in a way that accounts for how
similar the elements of a dataset are to one another. It implements the
Leinster–Cobbold–Reeve (LCR) family of measures (α, β, γ, ρ and their
normalized variants), similarity-sensitive relative entropy and cross-entropy,
ordinariness, and the Vendi score. It runs on NumPy or PyTorch (including GPUs),
can compute similarities on the fly for datasets too large for a matrix, and has
a command-line interface.

```{testcode}
import numpy as np
from sentropy import sentropy

P = np.array([0.7, 0.3])            # two elements, 70% / 30%
S = np.array([[1.0, 0.2],           # ...that are 20% similar to each other
              [0.2, 1.0]])

print(f"{sentropy(P):.2f}")                 # ordinary Shannon-type diversity
print(f"{sentropy(P, similarity=S):.2f}")   # lower once similarity is accounted for
```

```{testoutput}
1.84
1.55
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Installation
:link: installation
:link-type: doc

`pip install sentropy`, optional extras, and a development setup.
:::

:::{grid-item-card} Which call gives me X?
:link: which_call
:link-type: doc

A lookup table from "what I want to compute" to the exact function call
and command-line invocation.
:::

:::{grid-item-card} Tutorial
:link: tutorial
:link-type: doc

Worked examples: Shannon entropy, similarity, multiple measures, class
representativeness, relative sentropy, GPUs.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc

Every public function and class, with parameters and return types.
:::
::::

```{toctree}
:caption: Getting started
:maxdepth: 2

installation
tutorial
which_call
concepts
```

```{toctree}
:caption: Reference
:maxdepth: 2

api/index
cli
```

```{toctree}
:caption: Project
:maxdepth: 2

community
changelog
```
