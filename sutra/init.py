"""Weight initialisation schemes. Each returns a NumPy array drawn from Sutra's global RNG."""
from __future__ import annotations

import numpy as np

from .tensor import get_default_dtype, rng


def _fans(shape):
    if len(shape) < 2:
        return shape[0], shape[0]
    receptive = int(np.prod(shape[2:])) if len(shape) > 2 else 1
    return shape[1] * receptive, shape[0] * receptive  # (fan_in, fan_out) for (out, in, ...) layout


def uniform(shape, low=0.0, high=1.0):
    return rng().uniform(low, high, size=shape).astype(get_default_dtype())


def normal(shape, mean=0.0, std=1.0):
    return (mean + std * rng().standard_normal(shape)).astype(get_default_dtype())


def zeros(shape):
    return np.zeros(shape, dtype=get_default_dtype())


def xavier_uniform(shape, gain=1.0):
    fan_in, fan_out = _fans(shape)
    bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return uniform(shape, -bound, bound)


def xavier_normal(shape, gain=1.0):
    fan_in, fan_out = _fans(shape)
    return normal(shape, 0.0, gain * np.sqrt(2.0 / (fan_in + fan_out)))


def kaiming_uniform(shape, gain=np.sqrt(2.0)):
    fan_in, _ = _fans(shape)
    bound = gain * np.sqrt(3.0 / fan_in)
    return uniform(shape, -bound, bound)


def kaiming_normal(shape, gain=np.sqrt(2.0)):
    fan_in, _ = _fans(shape)
    return normal(shape, 0.0, gain / np.sqrt(fan_in))
