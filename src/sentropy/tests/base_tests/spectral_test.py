from scipy.sparse import csr_matrix
from sentropy.exceptions import SpectralError
from sentropy.spectral import vendi_score
from sentropy.sentropy import sentropy
from numpy import array, allclose, zeros, sum as np_sum, log as np_log, sqrt as np_sqrt, exp as np_exp, max as np_max
from numpy.linalg import eigvals
from pytest import raises

def test_spectral_diversity():
    """Entropy forms of the Vendi score with identity similarity:
    at q=1 it is Shannon entropy; at q=-1, (1/2)·log Σ 1/p."""
    p = array([[1], [2], [3]]) / 6

    VE_1 = vendi_score(p, q=1, eff_no=False, level="overall")
    assert allclose(VE_1, -np_sum(p * np_log(p)))

    VE_2 = vendi_score(p, q=-1, eff_no=False, level="overall")
    assert allclose(VE_2, 0.5 * np_log(np_sum(1 / p)))

def test_vendi_sparse_sim():
    points = array([[0, 0], [1, 1], [-1, 2]])
    diff = points[:, None, :] - points[None, :, :]   # shape (3, 3, 2)
    dist = np_sqrt((diff ** 2).sum(axis=-1))         # shape (3, 3)
    sim = np_exp(-dist)
    sim_sparse = csr_matrix(sim)
    p = array([[1], [1], [1]]) / 3
    VE_dense = sentropy(p, similarity=sim, q=1, eff_no=False, measure='vendi')
    VE_sparse = sentropy(p, similarity=sim_sparse, q=1, eff_no=False, measure='vendi')
    assert allclose(VE_dense, VE_sparse)

def test_vendi_spectralerror():
    # define a dataset consisting of two amino-acid sequences
    elements = array(['CARDYW', 'CTRDYW', 'CAKDYW'])     # amino-acid sequences (reminiscent of human IGH CDR3s)
    P = array([20, 1, 1])                                # the first is present 20 times; the second two are each present once
    with raises(SpectralError):
        sentropy(P, similarity=lambda i,j: len(i)-len(j), sfargs=elements, measure='vendi')

def test_vendi_no_positive_eigenvalues():
    P = array([20, 1, 1])
    Z = zeros((3, 3))
    with raises(SpectralError):
        sentropy(P, similarity=Z, measure='vendi')

def test_vendi_infinite_viewpoint():
    sim = array([[1.        , 0.24311673, 0.10687793],
       [0.24311673, 1.        , 0.10687793],
       [0.10687793, 0.10687793, 1.        ]])
    p = array([[1], [1], [1]]) / 3
    VE = sentropy(p, similarity=sim, q=1000, eff_no=False, measure='vendi')
    expected = -np_log(np_max(eigvals(sim)/np_sum(eigvals(sim))))
    assert allclose(VE, expected)

