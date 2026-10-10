"""
Gradient checkpointing (Chen et al., 2016): trade compute for memory.

`checkpoint(fn, *inputs)` runs `fn` without recording its internal graph, so none of its intermediate
activations are kept. During the backward pass it replays `fn` with the graph on and backpropagates
through the replay. Parameters used inside `fn` receive their gradients during that replay.

The random-number state is captured before the first run and restored for the replay, so stochastic
layers (dropout) draw exactly the same masks both times and the gradients are exact.
"""
from __future__ import annotations

from .tensor import Tensor, enable_grad, is_grad_enabled, no_grad, rng


def checkpoint(fn, *inputs: Tensor) -> Tensor:
    state = rng().bit_generator.state
    with no_grad():
        out = fn(*inputs)
    result = Tensor(out.data)
    if not is_grad_enabled():
        return result

    def backward(g):
        saved = rng().bit_generator.state
        rng().bit_generator.state = state
        replay_inputs = [Tensor(x.data, requires_grad=x.requires_grad) for x in inputs]
        with enable_grad():
            replay = fn(*replay_inputs)
        rng().bit_generator.state = saved
        replay.backward(g)
        return tuple(r.grad if x.requires_grad else None for r, x in zip(replay_inputs, inputs))

    result.requires_grad = True
    result._parents = tuple(inputs)
    result._backward = backward
    result._op = "checkpoint"
    return result
