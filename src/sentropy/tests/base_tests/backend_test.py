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