"""
Optimizers and learning-rate schedules. Update rules match torch.optim exactly
(verified step-for-step in tests/test_optim.py).
"""
from __future__ import annotations

import math

import numpy as np


class Optimizer:
    def __init__(self, params, defaults: dict):
        params = list(params)
        if params and isinstance(params[0], dict):
            groups = params
        else:
            groups = [{"params": params}]
        self.param_groups = []
        seen = set()
        for g in groups:
            group = {**defaults, **g}
            ps = []
            for p in group["params"]:
                if id(p) not in seen and p.requires_grad:
                    seen.add(id(p))
                    ps.append(p)
            group["params"] = ps
            group.setdefault("initial_lr", group["lr"])
            self.param_groups.append(group)
        self.state: dict[int, dict] = {}

    @property
    def params(self):
        return [p for g in self.param_groups for p in g["params"]]

    @property
    def lr(self):
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value):
        for g in self.param_groups:
            g["lr"] = value

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, p.grad, self.state.setdefault(id(p), {}), group)

    def _update(self, p, g, state, group):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, dampening=dampening,
                                      weight_decay=weight_decay, nesterov=nesterov))

    def _update(self, p, g, state, group):
        if group["weight_decay"]:
            g = g + group["weight_decay"] * p.data
        mu = group["momentum"]
        if mu:
            buf = state.get("momentum")
            if buf is None:
                buf = state["momentum"] = g.copy()
            else:
                buf *= mu
                buf += (1 - group["dampening"]) * g
            g = g + mu * buf if group["nesterov"] else buf
        p.data -= group["lr"] * g


class Adam(Optimizer):
    """Adam with L2-style (coupled) weight decay, as in torch.optim.Adam."""

    decoupled = False

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def _update(self, p, g, state, group):
        lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
        if wd:
            if self.decoupled:
                p.data *= 1 - lr * wd
            else:
                g = g + wd * p.data
        if not state:
            state["t"] = 0
            state["m"] = np.zeros_like(p.data)
            state["v"] = np.zeros_like(p.data)
        state["t"] += 1
        t, m, v = state["t"], state["m"], state["v"]
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * g * g
        bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
        p.data -= (lr / bc1) * m / (np.sqrt(v) / math.sqrt(bc2) + eps)


class AdamW(Adam):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

    decoupled = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, weight_decay)


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay))

    def _update(self, p, g, state, group):
        if group["weight_decay"]:
            g = g + group["weight_decay"] * p.data
        sq = state.setdefault("sq", np.zeros_like(p.data))
        sq *= group["alpha"]
        sq += (1 - group["alpha"]) * g * g
        p.data -= group["lr"] * g / (np.sqrt(sq) + group["eps"])


def clip_grad_norm_(params, max_norm: float) -> float:
    """Scale gradients in place so their global L2 norm is at most `max_norm`. Returns the pre-clip norm."""
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    coef = max_norm / (total + 1e-6)
    if coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * coef
    return total


# ------------------------------------------------------------------ schedules

class LRScheduler:
    """Sets each group's lr to initial_lr * factor(step). Call `.step()` once per optimizer step."""

    def __init__(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.last_step = 0
        self._apply()

    def factor(self, step: int) -> float:
        return 1.0

    def _apply(self):
        f = self.factor(self.last_step)
        for g in self.optimizer.param_groups:
            g["lr"] = g["initial_lr"] * f

    def step(self):
        self.last_step += 1
        self._apply()

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


class LambdaLR(LRScheduler):
    def __init__(self, optimizer, fn):
        self.fn = fn
        super().__init__(optimizer)

    def factor(self, step):
        return self.fn(step)


class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        self.step_size, self.gamma = step_size, gamma
        super().__init__(optimizer)

    def factor(self, step):
        return self.gamma ** (step // self.step_size)


class CosineWarmup(LRScheduler):
    """Linear warmup to the base lr, then cosine decay to `min_ratio * base lr`."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.1):
        self.warmup, self.total, self.min_ratio = warmup_steps, total_steps, min_ratio
        super().__init__(optimizer)

    def factor(self, step):
        if step < self.warmup:
            return (step + 1) / self.warmup
        progress = min(1.0, (step - self.warmup) / max(1, self.total - self.warmup))
        return self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
