# Command-line interface

Installing `sentropy` also installs a `sentropy` command (equivalently,
`python -m sentropy`). It reads counts and a precomputed similarity matrix from
CSV/TSV files and prints a table. For the arbitrary-similarity-function
interface, use the Python API.

## Input files

**Counts** (`-i`): a CSV or TSV with one column per class and one row per
element, saved without the index:

```python
import pandas as pd

pd.DataFrame(
    {"apples": [2, 4, 0], "oranges": [4, 0, 6]},
    index=["ladybug", "bee", "fish"],
).to_csv("counts.csv", index=False)
```

**Similarity** (`-similarity`): a CSV/TSV with a header row of element names;
row and column order must match the rows of the counts file.

```python
pd.DataFrame(
    [[1.0, 0.60, 0.22], [0.60, 1.0, 0.27], [0.22, 0.27, 1.0]],
    index=["ladybug", "bee", "fish"],
    columns=["ladybug", "bee", "fish"],
).to_csv("similarity.csv", index=False)
```

Without `-similarity`, the identity matrix is used (ordinary entropy).

## Examples

One input file — LCR measures at several viewpoints:

```bash
sentropy -i counts.csv -similarity similarity.csv -q 0 1 2 inf -measure alpha beta normalized_rho
```

The Vendi score (cannot be combined with other measures):

```bash
sentropy -i counts.csv -similarity similarity.csv -measure vendi -q 1
```

Save the (tab-delimited) table to a file:

```bash
sentropy -i counts.csv -similarity similarity.csv -q 0 1 inf -o results.tsv
```

Two input files — relative sentropy (`sre`, the default) or cross-sentropy
(`sce`) between classes; both files must list elements in the same order as the
similarity matrix:

```bash
sentropy -i counts_a.csv counts_b.csv -similarity similarity.csv -q 1 -measure sre
```

## All options

```{argparse}
:module: sentropy.parameters
:func: configure_arguments
:prog: sentropy
```
