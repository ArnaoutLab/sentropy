"""Package for calculating Similarity-sensitive diversities.

The diversity measures that can be calculated are described rigorously
in https://arxiv.org/abs/1404.6520.

See sentropy.set.Set
"""

from sentropy.set import Set
from sentropy.set import sentropy
from sentropy.divergences import kl_div_effno, rel_ent