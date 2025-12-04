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

# Try to import torch (optional)
try:
    import torch as _torch  # type: ignore
    _has_torch = True
except Exception:
    _torch = None
    _has_torch = False


class BackendError(RuntimeError):
    pass


class BaseBackend:
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

    def sum(self, x, axis=None):
        raise NotImplementedError

    def ones(self, shape, dtype=None):
        raise NotImplementedError

    def concatenate(self, xs, axis=0):
        raise NotImplementedError

    def vstack(self, xs):
        raise NotImplementedError

    def identity(self, n):
        raise NotImplementedError

    def power(self, x, exponent):
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

class NumpyBackend(BaseBackend):
    name = "numpy"

    def __init__(self, device: Optional[str] = None):
        super().__init__(device)

    def array(self, x, dtype=None):
        return _np.array(x, dtype=dtype)

    def asarray(self, x):
        return _np.asarray(x)

    def to_numpy(self, x):
        # x already numpy-like
        return _np.asarray(x)

    def to_device(self, x):
        return _np.asarray(x)

    def matmul(self, A, B):
        return A @ B

    def sum(self, x, axis=None, keepdims=False):
        return _np.sum(x, axis=axis, keepdims=keepdims)

    def ones(self, shape, dtype=None):
        return _np.ones(shape, dtype=dtype)

    def concatenate(self, xs, axis=0):
        return _np.concatenate(xs, axis=axis)

    def vstack(self, xs):
        return _np.vstack(xs)

    def identity(self, n):
        return _np.identity(n)

    def power(self, x, exponent):
        return _np.power(x, exponent)

    def prod(self, x, axis=None, where=None):
        # numpy prod doesn't accept where prior to newer numpy; use fallback
        return _np.prod(x, axis=axis)

    def amin(self, x, axis=None, where=None, initial=None):
        if where is None:
            return _np.amin(x, axis=axis)
        return _np.amin(x, axis=axis)

    def amax(self, x, axis=None, where=None, initial=None):
        if where is None:
            return _np.amax(x, axis=axis)
        return _np.amax(x, axis=axis)

    def isclose(self, a, b, atol=1e-9):
        return _np.isclose(a, b, atol=atol)

    def multiply(self, a, b, out=None, where=None):
        return _np.multiply(a, b, out=out)

    def abs(self, x):
        return _np.abs(x)

    def all(self, x, axis=None):
        return _np.all(x, axis=axis)

    def any(self, x, axis=None):
        return _np.any(x, axis=axis)

    def where(self, cond, x, y):
        return _np.where(cond, x, y)

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


class TorchBackend(BaseBackend):
    name = "torch"

    if not _has_torch:
        raise BackendError("PyTorch is not available. Install torch to use 'torch' backend.")

    def __init__(self, device: Optional[str] = None):
        super().__init__(device or ("cuda" if _torch.cuda.is_available() else "cpu"))
        self.torch = _torch
        # default dtype to float64 to preserve numeric behavior
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
        import numpy as np

        return np.asarray(x)

    def to_device(self, x):
        if isinstance(x, self.torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return self.torch.as_tensor(x, dtype=self.dtype, device=self.device)

    def matmul(self, A, B):
        B = B.to(A.dtype)
        return A @ B

    def sum(self, x, axis=None, keepdims=False):
        if axis is None:
            return self.torch.sum(x, dim=axis, keepdim=keepdims)
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

    def power(self, x, exponent):
        return self.torch.pow(x, exponent)

    def prod(self, x, axis=None, where=None):
        if axis is None:
            return self.torch.prod(x)
        return self.torch.prod(x, dim=axis)

    def amin(self, x, axis=None, where=None, initial=None):
        if axis is None:
            return self.torch.amin(x)
        return self.torch.amin(x, dim=axis)

    def amax(self, x, axis=None, where=None, initial=None):
        if axis is None:
            return self.torch.amax(x)
        return self.torch.amax(x, dim=axis)

    def isclose(self, a, b, atol=1e-9):
        a = self.torch.tensor(a)
        b = self.torch.tensor(b).to(a.dtype)
        return self.torch.isclose(a, b, atol=atol)

    def multiply(self, a, b, out=None, where=None):
        return a * b

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
        return self.torch.zeros(shape)

    def empty(self, shape):
        return self.torch.empty(shape)

    def copy(self, x):
        return x.clone()


def get_backend(name: str = "numpy", device: Optional[str] = None) -> BaseBackend:
    name = (name or "numpy").lower()
    if name in ("numpy", "np"):
        return NumpyBackend(device=device)
    if name in ("torch", "pytorch"):
        if not _has_torch:
            raise BackendError("PyTorch is not installed but 'torch' backend was requested.")
        return TorchBackend(device=device)
    raise BackendError(f"Unknown backend '{name}'. Valid names: 'numpy', 'torch'.")

