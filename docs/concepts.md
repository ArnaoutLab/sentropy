# Understanding sentropy

This page is a short glossary. For the full theory see
[Leinster & Cobbold (2012)](https://doi.org/10.1890/10-2402.1),
[Reeve et al.](https://arxiv.org/abs/1404.6520) and
[Leinster (2020)](https://arxiv.org/abs/2012.02113).

## From Shannon entropy to sentropy

**Shannon entropy** is a weighted sum of the log relative frequencies of the
unique elements of a dataset. **Rényi entropies** generalize it with a
*viewpoint parameter* $q$ ($q=1$ is Shannon). **Sentropy** generalizes Rényi
entropy further by using a **similarity matrix** $Z$, whose entry
$Z_{ij} \in [0, 1]$ says how similar elements $i$ and $j$ are
($Z_{ii}=1$). With the identity matrix, sentropy reduces to ordinary Rényi
entropy.

**Effective numbers.** Exponentiating an entropy gives an *effective number*
(also called a D-number or Hill number): the number of equally abundant,
completely dissimilar elements that would give the same entropy. `sentropy`
returns effective numbers by default; use `eff_no=False` for entropies in nats.
Higher means more diverse.

## The viewpoint parameter $q$

$q$ controls how much rare elements count. At $q=0$ every element counts
equally; as $q$ grows, common elements dominate; at $q=\infty$ only the most
common element matters. Informally, $q = 0, 1, 2, \infty$ give "counting-",
"Shannon-", "Simpson-" and "Berger–Parker-type" sentropy. Values never rise as
$q$ increases. Values of $q$ above 100 are treated as $\infty$.

## Levels: dataset and class

A dataset can be split into **classes** (also called subsets). Measures can be
computed for the **overall** dataset, for each **subset**/class, or both
(`level="overall"`, `"subset"`/`"class"`, `"both"`).

## Measures

Pass these names as `measure=`. "Raw" measures weight each class by its share
of the whole dataset; "normalized" measures first normalize each class to sum
to one, so classes of different sizes are directly comparable.

| `measure=` | What it quantifies |
|---|---|
| `alpha`, `normalized_alpha` | Diversity of a class on its own: the effective number of distinct elements in it. |
| `beta`, `normalized_beta` | **Distinctiveness** of a class relative to the rest of the dataset. |
| `rho`, `normalized_rho` | **Representativeness** of a class of the whole dataset; the reciprocal of $\beta$. |
| `beta_hat`, `rho_hat` | $\beta$ and $\rho$ rescaled using the number of classes. |
| `gamma` | Diversity of the whole dataset as seen from a class; the exponentiated cross-entropy between the class and the dataset. |
| `sre` | **Relative sentropy** (similarity-sensitive KL divergence). Needs two count arguments. |
| `sce` | **Cross-sentropy** (similarity-sensitive cross-entropy). Needs two count arguments. |
| `vendi` | The [Vendi score](https://arxiv.org/abs/2210.02410), an alternative similarity-sensitive diversity based on the eigenvalues of the similarity matrix. |

For a dataset with a single class, $\alpha = \gamma$ and $\beta = 1$.

## Ordinariness

The **ordinariness** of an element is the abundance-weighted average of its
similarity to every element of a dataset: how much it "belongs". It is the
building block of all the measures above, and
{func}`~sentropy.sentropy.interset_ordinariness` computes it for elements of one set with
respect to a *different* reference set.
