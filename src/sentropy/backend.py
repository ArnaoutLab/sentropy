# backend.py
"""Backend abstraction to allow NumPy (default) or PyTorch computation (optional).

Usage:
    from sentropy.backend import get_backend
    bk = get_backend("torch", device="cuda")
    x = bk.array([1,2,3])
    y = bk.matmul(A, B)
"""

from typing import Optional
import numpy as _np
import torch as _torch
import scipy.sparse


sparse_classes = [
        scipy.sparse.bsr_array,
        scipy.sparse.coo_array,
        scipy.sparse.csc_array,
        scipy.sparse.csr_array,
        scipy.sparse.bsr_matrix,
        scipy.sparse.coo_matrix,
        scipy.sparse.csc_matrix,
        scipy.sparse.csr_matrix,
        scipy.sparse.dia_array,
        scipy.sparse.dia_matrix,
        scipy.sparse.lil_array,
        scipy.sparse.lil_matrix,
        scipy.sparse.dok_array,
        scipy.sparse.dok_matrix,
    ]

class BackendError(RuntimeError):
    pass


class BaseBackend: # pragma: no cover
    name = "base"

    def __init__(self, device: Optional[str] = None):
        self.device = device

    # fundamental wrappers used across package
    def array(self, x, dtype=None):
        raise NotImplementedError

    def asarray(self, x):
        raise NotImplementedError

    def to_numpy(self, x):
        raise NotImplementedError

    def to_device(self, x):
        """Ensure x is on the backend's device / type."""
        raise NotImplementedError

    def matmul(self, A, B):
        raise NotImplementedError

    def sum(self, x, axis=None, keepdims=False, where=None):
        raise NotImplementedError

    def ones(self, shape, dtype=None):
        raise NotImplementedError

    def concatenate(self, xs, axis=0):
        raise NotImplementedError

    def vstack(self, xs):
        raise NotImplementedError

    def identity(self, n):
        raise NotImplementedError

    def power(self, x, exponent, out=None, where=None):
        raise NotImplementedError

    def prod(self, x, axis=None, where=None):
        raise NotImplementedError

    def amin(self, x, axis=None, where=None, initial=None):
        raise NotImplementedError

    def amax(self, x, axis=None, where=None, initial=None):
        raise NotImplementedError

    def isclose(self, a, b, atol=1e-9):
        raise NotImplementedError

    def multiply(self, a, b, out=None, where=None):
        raise NotImplementedError

    def abs(self, x):
        raise NotImplementedError

    def all(self, x, axis=None):
        raise NotImplementedError

    def any(self, x, axis=None):
        raise NotImplementedError

    def where(self, cond, x, y):
        raise NotImplementedError

    def log(self, x):
        raise NotImplementedError

    def broadcast_to(self, x, shape):
        raise NotImplementedError

    def zeros(self, shape):
        raise NotImplementedError

    def empty(self, shape):
        raise NotImplementedError

    def copy(self, x):
        raise NotImplementedError

    def divide(self, x, y):
        raise NotImplementedError

class NumpyBackend(BaseBackend):
    name = "numpy"

    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    def array(self, x, dtype=None):
        return _np.array(x, dtype=dtype)

    def asarray(self, x):
        if type(x) not in sparse_classes:
            return _np.asarray(x)
        else:
            return x

    def to_numpy(self, x):
        # x already numpy-like
        return _np.asarray(x)

    # def to_device(self, x):
    #     return _np.asarray(x)

    def matmul(self, A, B):
        return A @ B

    def sum(self, x, axis=None, keepdims=False, where=None):
        if where is not None:
            return _np.sum(x, axis=axis, keepdims=keepdims, where=where)
        else:
            return _np.sum(x, axis=axis, keepdims=keepdims)

    def ones(self, shape, dtype=None):
        return _np.ones(shape, dtype=dtype)

    def concatenate(self, xs, axis=0):
        return _np.concatenate(xs, axis=axis)

    def vstack(self, xs):
        return _np.vstack(xs)

    def identity(self, n):
        return _np.identity(n)

    def power(self, x, exponent, out=None, where=None):
        if where is not None:
            return _np.power(x, exponent, out=out, where=where)
        else:
            return _np.power(x, exponent, out=out)

    def prod(self, x, axis=None, where=None):
        # numpy prod doesn't accept where prior to newer numpy; use fallback
        if where is not None:
            return _np.prod(x, axis=axis, where=where)
        else:
            return _np.prod(x, axis=axis)

    def amin(self, x, axis=None, where=None, initial=None):
        return _np.amin(x, axis=axis, where=where, initial=initial)

    def amax(self, x, axis=None, where=None, initial=None):
        return _np.amax(x, axis=axis, where=where, initial=initial)

    def isclose(self, a, b, atol=1e-9):
        return _np.isclose(a, b, atol=atol)

    def multiply(self, a, b, out=None, where=None):
        if where is not None:
            return _np.multiply(a, b, out=out, where=where)
        else:
            return _np.multiply(a, b, out=out)

    def abs(self, x):
        return _np.abs(x)

    def all(self, x, axis=None):
        return _np.all(x, axis=axis)

    def any(self, x, axis=None):
        return _np.any(x, axis=axis)

    # def where(self, cond, x, y):
    #     return _np.where(cond, x, y)

    def log(self, x):
        return _np.log(x)

    def broadcast_to(self, x, shape):
        return _np.broadcast_to(x, shape)

    def zeros(self, shape):
        return _np.zeros(shape)

    def empty(self, shape):
        return _np.empty(shape)

    def copy(self, x):
        return x.copy()

    def divide(self, x, y):
        return _np.divide(x, y, out=_np.zeros(y.shape), where=y !=0)


class TorchBackend(BaseBackend):
    name = "torch"
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device or ("cuda" if _torch.cuda.is_available() else "cpu"))
        self.torch = _torch
        # default dtype to float64 to preserve numeric behavior
        if device == 'mps':
            self.dtype = self.torch.float32
        else:
            self.dtype = self.torch.float64

    def array(self, x, dtype=None):
        # if x already tensor, cast
        if isinstance(x, self.torch.Tensor):
            return x.to(device=self.device, dtype=(dtype or self.dtype))
        return self.torch.as_tensor(x, dtype=(dtype or self.dtype), device=self.device)

    def asarray(self, x):
        return self.array(x)

    def to_numpy(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.detach().cpu().numpy()
        return _np.asarray(x)

    def to_device(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return self.torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def matmul(self, A, B):
        B = B.to(A.dtype)
        return A @ B

    def sum(self, x, axis=None, keepdims=False, where=None):
        if where is not None:
            x = x*where
        return self.torch.sum(x, dim=axis, keepdim=keepdims)

    def ones(self, shape, dtype=None):
        return self.torch.ones(shape, dtype=(dtype or self.dtype), device=self.device)

    def concatenate(self, xs, axis=0):
        # xs: list of tensors or arrays
        xs_t = [self.asarray_if_needed(x) for x in xs]
        return self.torch.cat(xs_t, dim=axis)

    def vstack(self, xs):
        xs_t = [self.asarray_if_needed(x) for x in xs]
        return self.torch.vstack(xs_t)

    def identity(self, n):
        return self.torch.eye(n, dtype=self.dtype, device=self.device)

    def power(self, x, exponent, out=None, where=None):
        if where is None:
            if out is not None:
                return out.copy_(self.torch.pow(x, exponent))
            else:
                return self.torch.pow(x, exponent)
        else:
            result = self.torch.pow(x, exponent)
            if out is None:
                return self.torch.where(where, result, x)
            else:
                out.copy_(x)
                result = result.to(out.dtype)
                out[where] = result[where]
                return out

    def prod(self, x, axis=None, where=None):
        return self.torch.prod(x, dim=axis)

    def amin(self, x, axis=None, where=None, initial=None):
        # Convert axis naming
        dim = axis

        # Case 1: No `where` mask — simplest path
        if where is None:
            if initial is None:
                return self.torch.amin(x, dim=dim)
            else:
                # Apply initial: min(initial, torch.min(x))
                min_x = self.torch.amin(x, dim=dim)
                return self.torch.minimum(min_x, self.torch.tensor(initial))

        # Case 2: `where` mask exists
        # PyTorch has no where-min, so emulate:
        # Mask out invalid entries by replacing them with +inf so they don't affect min

        # Get +inf of correct dtype
        inf = self.torch.tensor(float("inf"))

        # Apply mask: where False → replace x with +inf
        masked = self.torch.where(where, x, inf)

        # Compute min of masked entries
        result = self.torch.amin(masked, dim=dim)

        # Apply `initial` if needed
        if initial is not None:
            result = self.torch.minimum(result, self.torch.tensor(initial))

        return result


    def amax(self, x, axis=None, where=None, initial=None):
        dim = axis

        # Case 1: No `where` mask
        if where is None:
            if initial is None:
                return self.torch.amax(x, dim=dim)
            else:
                # min_x = torch.max(x)
                max_x = self.torch.amax(x, dim=dim)
                initial_tensor = self.torch.tensor(initial)
                return self.torch.maximum(max_x, initial_tensor)

        # Case 2: where mask exists
        # Replace masked-out entries with -inf so they can't affect max
        ninf = self.torch.tensor(float("-inf"))

        masked = self.torch.where(where, x, ninf)

        result = self.torch.amax(masked, dim=dim)

        # Apply initial if provided
        if initial is not None:
            initial_tensor = self.torch.tensor(initial)
            result = self.torch.maximum(result, initial_tensor)

        return result


    def isclose(self, a, b, atol=1e-9):
        a = self.torch.tensor(a)
        b = self.torch.tensor(b).to(a.dtype)
        return self.torch.isclose(a, b, atol=atol)

    def multiply(self, a, b, out=None, where=None):
        if where is None:
            return self.torch.multiply(a,b, out=out)
        else:
            result = a*b
            if result.dtype != out.dtype:
                result = result.to(out.dtype)
            if out is None:
                return self.torch.where(where, result, a)
            else:
                out.copy_(a)
                out[where] = result[where]
                return out

    def abs(self, x):
        return self.torch.abs(x)

    def all(self, x, axis=None):
        if axis is None:
            return self.torch.all(x)
        return self.torch.all(x, dim=axis)

    def any(self, x, axis=None):
        if axis is None:
            return self.torch.any(x)
        return self.torch.any(x, dim=axis)

    def where(self, cond, x, y):
        return self.torch.where(cond, x, y)

    def asarray_if_needed(self, x):
        if isinstance(x, self.torch.Tensor):
            return x
        return self.torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def log(self, x):
        return self.torch.log(x)

    def broadcast_to(self, x, shape):
        return self.torch.broadcast_to(x, shape)

    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.device)

    def empty(self, shape):
        return self.torch.empty(shape, device=self.device)

    def copy(self, x):
        return x.clone()

    def divide(self, x, y):
        # Ensure x and y are tensors
        x = self.torch.as_tensor(x)
        y = self.torch.as_tensor(y)
        out = self.torch.where(y!=0, x/y, self.torch.zeros_like(x))
        return out


def get_backend(name: str = "numpy", device: Optional[str] = None) -> BaseBackend:
    name = (name or "numpy").lower()
    if name in ("numpy", "np"):
        return NumpyBackend(device=device)
    if name in ("torch", "pytorch"):
        return TorchBackend(device=device)

