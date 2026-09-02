from sentropy.spectral import vendi_score
from numpy import array, allclose, sum as np_sum, log as np_log

def test_spectral_diversity():
    """Entropy forms of the Vendi score with identity similarity:
    at q=1 it is Shannon entropy; at q=-1, (1/2)·log Σ 1/p."""
    p = array([[1], [2], [3]]) / 6

    VE_1 = vendi_score(p, q=1, eff_no=False, level="overall")
    assert allclose(VE_1, -np_sum(p * np_log(p)))

    VE_2 = vendi_score(p, q=-1, eff_no=False, level="overall")
    assert allclose(VE_2, 0.5 * np_log(np_sum(1 / p)))