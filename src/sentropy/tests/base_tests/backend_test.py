from sentropy.backend import *
from pytest import fixture, raises, mark
import pandas as pd
import torch
import numpy as np

numpy_bkd = NumpyBackend()
torch_bkd = TorchBackend()

def test_backend_equivalence_of_array_and_asarray_and_to_numpy():
	x = [[1,2,3],[4,5,6],[7,8,9]]
	assert (numpy_bkd.array(x) == torch_bkd.array(x)).all()
	assert (numpy_bkd.asarray(x) == torch_bkd.asarray(x)).all()
	assert (numpy_bkd.to_numpy(pd.DataFrame(x)) == torch_bkd.to_numpy(torch.tensor(x))).all()

def test_backend_equivalence_of_matmul():
	x = np.array([[1,2,3],[4,5,6],[7,8,9]])
	y = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
	assert (numpy_bkd.matmul(x, x)==torch_bkd.matmul(y, y)).all()

def test_backend_equivalence_of_sum():
	x = np.array([[1,2,3],[4,5,6],[7,8,9]])
	where = [[True,True,True],[False,False,False],[True,True,True]]
	numpy_bkd_answer = numpy_bkd.sum(x, axis=0, keepdims=True, where=where)
	torch_bkd_answer = torch_bkd.sum(torch.tensor(x), axis=0, keepdims=True, where=torch.tensor(where))
	assert (numpy_bkd_answer == torch_bkd_answer).all()

def test_backend_equivalence_of_ones():
	assert (numpy_bkd.ones(shape=(1,5)) == torch_bkd.ones(shape=(1,5))).all()

def test_backend_equivalence_of_concatenate_and_vstack():
	tuple_of_arrays = ([[1,2,3]],[[4,5,6]])
	assert (numpy_bkd.concatenate(tuple_of_arrays, axis=1) == torch_bkd.concatenate(tuple_of_arrays, axis=1)).all()
	assert (numpy_bkd.vstack(tuple_of_arrays) == torch_bkd.vstack(tuple_of_arrays)).all()

def test_backend_equivalence_of_identity():
	assert (numpy_bkd.identity(3) == torch_bkd.identity(3)).all()

def test_backend_equivalence_of_power():
	x = [1,2,3]
	exponent = [2,3,2]
	where = [True, True, False]

	np_result = numpy_bkd.power(x, exponent)
	torch_result = torch_bkd.power(torch.Tensor(x), torch.Tensor(exponent))

	np_result_with_where = numpy_bkd.power(x, exponent, where=where)
	torch_result_with_where = torch_bkd.power(torch.tensor(x), torch.tensor(exponent), \
		where=torch.tensor(where, dtype=torch.bool))

	assert (np_result == torch_result).all()
	assert (np_result_with_where[:2] == torch_result_with_where[:2]).all()

def test_backend_equivalence_of_prod():
	x = [[1,2,3],[4,5,6]]
	where = [[True, True, False], [False, False, True]]
	assert (numpy_bkd.prod(x, axis=1) == torch_bkd.prod(torch.Tensor(x), axis=1)).all()
	assert (numpy_bkd.prod(x, where=where) == torch_bkd.prod(torch.tensor(x), where=torch.tensor(where)))
	assert (numpy_bkd.prod(x, axis=0, where=where) == torch_bkd.prod(torch.tensor(x), axis=0, where=torch.tensor(where))).all()

def test_backend_equivalence_of_amin_and_amax():
	x = np.array([[1,2,3],[4,5,6]])
	where = np.array([[False, True, False], [True, False, False]])
	assert (numpy_bkd.amin(x) == torch_bkd.amin(torch.tensor(x))).all()
	assert (numpy_bkd.amin(x, where=where, initial=100) == torch_bkd.amin(torch.tensor(x), where=torch.tensor(where), initial=100)).all()
	assert (numpy_bkd.amin(x, where=where, initial=100, axis=0) == torch_bkd.amin(torch.tensor(x), where=torch.tensor(where), initial=100, axis=0)).all()

	assert (numpy_bkd.amax(x) == torch_bkd.amax(torch.tensor(x))).all()
	assert (numpy_bkd.amax(x, where=where, initial=-100) == torch_bkd.amax(torch.tensor(x), where=torch.tensor(where), initial=-100)).all()
	assert (numpy_bkd.amax(x, where=where, initial=-100, axis=0) == torch_bkd.amax(torch.tensor(x), where=torch.tensor(where), initial=-100, axis=0)).all()

def test_backend_equivalence_of_isclose():
	x = np.array([[1,2,3],[4,5,6]])
	y = np.array([[1.01,2.005,2.99],[4.001,4.995,5.999]])

	assert (numpy_bkd.isclose(x,y, rtol=1e-2, atol=1e-2) == torch_bkd.isclose(torch.tensor(x), torch.tensor(y), rtol=1e-2, atol=1e-2)).all()

def test_backend_equivalence_of_multiply():
	x = np.array([[1,2,3],[4,5,6]])
	y = np.array([[1.01,2.005,2.99],[4.001,4.995,5.999]])
	where = np.array([[False, False, True],[False, False, False]])

	np_result = numpy_bkd.multiply(x,y)
	torch_result = torch_bkd.multiply(torch.tensor(x),torch.tensor(y))
	np_result_with_where = numpy_bkd.multiply(x,y,where=where, out=np.ones_like(y))
	torch_result_with_where = torch_bkd.multiply(torch.tensor(x),torch.tensor(y),where=torch.tensor(where), out=torch.ones(y.shape))

	assert np.allclose(np_result, torch_result)
	assert np.allclose(np_result_with_where, torch_result_with_where)

def test_backend_equivalence_of_abs():
	x = np.array([[1,-2,3],[0,5,-6]])
	assert np.allclose(numpy_bkd.abs(x), torch_bkd.abs(torch.tensor(x)))

def test_backend_equivalence_of_all_and_any():
	x = np.array([[True, False, True], [False, False, True]])
	assert np.allclose(numpy_bkd.all(x), torch_bkd.all(torch.tensor(x)))
	assert np.allclose(numpy_bkd.all(x, axis=0), torch_bkd.all(torch.tensor(x), axis=0))
	assert np.allclose(numpy_bkd.any(x), torch_bkd.any(torch.tensor(x)))
	assert np.allclose(numpy_bkd.any(x, axis=0), torch_bkd.any(torch.tensor(x), axis=0))

def test_backend_equivalence_of_log():
	x = np.array([[1, 2],[3, 4]])
	assert np.allclose(numpy_bkd.log(x), torch_bkd.log(torch.tensor(x)))

def test_backend_equivalence_of_broadcast_to():
	x = np.array([1,2,3])
	assert np.allclose(numpy_bkd.broadcast_to(x, shape=(2,3)), torch_bkd.broadcast_to(torch.tensor(x), shape=(2,3)))

def test_backend_equivalence_of_zeros_and_empty():
	assert np.allclose(numpy_bkd.zeros(shape=(2,2)),torch_bkd.zeros(shape=(2,2)))
	assert np.allclose(numpy_bkd.empty(shape=(2,2)),torch_bkd.empty(shape=(2,2)))

def test_backend_equivalence_of_copy():
	x = np.array([1,2,3])
	assert np.allclose(numpy_bkd.copy(x),torch_bkd.copy(torch.tensor(x)))

def test_backend_equivalence_of_divide():
	x = np.array([1,2,3,4,5])
	y = np.array([3,0,2,0,1])	
	assert np.allclose(numpy_bkd.divide(x,y), torch_bkd.divide(x,y))
