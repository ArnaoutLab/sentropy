"""Miscellaneous helper module for the set package.

Functions
---------
power_mean
    Calculates weighted power means.
"""

from numpy import inf as _np_inf

from sentropy.exceptions import InvalidArgumentError
from sentropy.backend import get_backend


def __validate_power_mean_args(
    weights, items, atol: float, weight_is_nonzero, backend
) -> None:
    """Validates arguments for power_mean."""
    # use backend.shape if available
    shape = getattr(weights, "shape", None)
    if shape is None:
        raise InvalidArgumentError("'weights' must have a shape attribute")
    if len(shape) > 2:
        raise InvalidArgumentError(
            f"'weights' shape must have 1 or 2 dimensions, but 'weights' had shape {shape}."
        )
    if getattr(weights, "shape", None) != getattr(items, "shape", None):
        raise InvalidArgumentError(
            f"Shape of 'weights' ({weights.shape}) must be the same as"
            f" shape of 'items' ({items.shape})."
        )
    # all_0_column = all(~weight_is_nonzero, axis=0)
    all_0_column = backend.all(~weight_is_nonzero, axis=0)
    if backend.any(all_0_column):
        raise InvalidArgumentError(
            "Argument 'weights' must have at least one nonzero weight in each column. A weight is"
            " considered 0 if its absolute value is greater than or equal to"
            f" configurable minimum threshold: {atol:.2e}."
        )


def power_mean(
    order: float,
    weights,
    items,
    atol: float = 1e-9,
    backend=None,
):
    """Calculates weighted power means.

    If backend is None, will select torch backend if inputs are torch tensors,
    otherwise numpy backend is used.
    """

    # Choose backend
    if type(backend)==str:
        backend=get_backend(backend) 

    abs_weights = backend.abs(weights)
    weight_is_nonzero = abs_weights >= atol
    __validate_power_mean_args(weights, items, atol, weight_is_nonzero, backend)

    # Analytical limits
    if order < -100:
        return backend.amin(items, axis=0, where=weight_is_nonzero, initial=_np_inf)
    elif order > 100:
        return backend.amax(items, axis=0, where=weight_is_nonzero, initial=-_np_inf)
    elif backend.isclose(order, 0, atol):
        # product of power(items, weights) across axis 0 where weight_is_nonzero
        powered = backend.power(items, weights)
        return backend.prod(powered, axis=0)

    else:
        result = backend.zeros(shape=items.shape)
        backend.power(items, order, where=weight_is_nonzero, out=result)
        backend.multiply(result, weights, where=weight_is_nonzero, out=result)
        items_sum = backend.sum(result, axis=0, where=weight_is_nonzero)
        return backend.power(items_sum, 1.0 / order)

