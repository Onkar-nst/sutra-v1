"""Modules, optimizers and schedules vs torch.nn / torch.optim."""
import math

import numpy as np
import pytest
import torch

import sutra as st
from sutra import nn, optim
from sutra.models import MLP, DigitCNN


def to_torch_state(sd):
    return {k: torch.tensor(v) for k, v in sd.items()}


def test_module_registration_and_state_dict_roundtrip(tmp_path):
    from sutra import serialize
    m = DigitCNN()
    names = [n for n, _ in m.named_parameters()]
    assert "features.0.weight" in names and "classifier.4.bias" in names
    sd = m.state_dict()
    assert "features.1.running_mean" in sd  # buffers are saved too
    serialize.save(m, tmp_path / "m.npz", note="hi")
    m2 = DigitCNN()
    assert serialize.load(m2, tmp_path / "m.npz") == {"note": "hi"}
    for k, v in m2.state_dict().items():
        np.testing.assert_array_equal(v, sd[k])
    with pytest.raises(KeyError):
        m2.load_state_dict({"nope": np.zeros(1)})


def test_digit_cnn_matches_torch_forward_and_backward():
    ours = DigitCNN()
    ref = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.BatchNorm2d(16), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(1568, 128), torch.nn.ReLU(), torch.nn.Dropout(0.25), torch.nn.Linear(128, 10))
    ref_sd = {}
    for k, v in ours.state_dict().items():  # features.i -> i, classifier.i -> 8 + i
        part, idx, rest = k.split(".", 2)
        ref_sd[f"{int(idx) + (8 if part == 'classifier' else 0)}.{rest}"] = torch.tensor(v)
    missing, unexpected = ref.load_state_dict(ref_sd, strict=False)
    assert not unexpected and all(k.endswith("num_batches_tracked") for k in missing)
    ours.eval(), ref.eval()  # deterministic (dropout off, BN running stats)
    x = np.random.default_rng(0).standard_normal((3, 1, 28, 28))
    y = np.array([1, 7, 3])
    xs, xt = st.Tensor(x.copy(), requires_grad=True), torch.tensor(x, requires_grad=True)
    ls = st.F.cross_entropy(ours(xs), y)
    lt = torch.nn.functional.cross_entropy(ref(xt), torch.tensor(y))
    assert ls.item() == pytest.approx(lt.item(), rel=1e-9)
    ls.backward(), lt.backward()
    np.testing.assert_allclose(xs.grad, xt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ours.features[0].weight.grad, ref[0].weight.grad.numpy(), atol=1e-10)


def test_train_eval_switches_every_submodule():
    m = DigitCNN().eval()
    assert all(not mod.training for mod in m.modules())
    m.train()
    assert all(mod.training for mod in m.modules())


def _torch_twin(model):
    params = [torch.tensor(p.data.copy(), requires_grad=True) for p in model.parameters()]
    return params


def _run(opt_st_fn, opt_th_fn, steps=25, sched=None):
    """Train the same tiny regression problem in both frameworks; parameters must stay identical."""
    rng = np.random.default_rng(0)
    X, Y = rng.standard_normal((16, 5)), rng.standard_normal((16, 2))
    m = MLP([5, 8, 2])
    ps_th = _torch_twin(m)
    opt_s, opt_t = opt_st_fn(m.parameters()), opt_th_fn(ps_th)
    for _ in range(steps):
        opt_s.zero_grad()
        st.F.mse_loss(m(st.Tensor(X)), Y).backward()
        opt_s.step()
        opt_t.zero_grad()
        h = torch.relu(torch.tensor(X) @ ps_th[0].T + ps_th[1])
        torch.nn.functional.mse_loss(h @ ps_th[2].T + ps_th[3], torch.tensor(Y)).backward()
        opt_t.step()
    for a, b in zip(m.parameters(), ps_th):
        np.testing.assert_allclose(a.data, b.detach().numpy(), atol=1e-10, rtol=1e-8)


@pytest.mark.parametrize("kw", [dict(lr=0.1), dict(lr=0.1, momentum=0.9), dict(lr=0.05, momentum=0.9, nesterov=True),
                                dict(lr=0.1, momentum=0.8, dampening=0.1, weight_decay=0.01)])
def test_sgd_matches_torch(kw):
    _run(lambda p: optim.SGD(p, **kw), lambda p: torch.optim.SGD(p, **kw))


@pytest.mark.parametrize("cls,kw", [("Adam", dict(lr=0.01)), ("Adam", dict(lr=0.01, weight_decay=0.1)),
                                    ("AdamW", dict(lr=0.01, weight_decay=0.1)),
                                    ("AdamW", dict(lr=0.02, betas=(0.8, 0.95), eps=1e-6)),
                                    ("RMSprop", dict(lr=0.01)), ("RMSprop", dict(lr=0.01, weight_decay=0.05))])
def test_adaptive_optimizers_match_torch(cls, kw):
    _run(lambda p: getattr(optim, cls)(p, **kw), lambda p: getattr(torch.optim, cls)(p, **kw))


def test_param_groups_with_different_weight_decay():
    m = MLP([5, 8, 2])
    groups = [{"params": [m.parameters()[0]], "weight_decay": 0.5}, {"params": m.parameters()[1:]}]
    opt = optim.AdamW(groups, lr=0.01, weight_decay=0.0)
    assert opt.param_groups[0]["weight_decay"] == 0.5 and opt.param_groups[1]["weight_decay"] == 0.0
    assert len(opt.params) == 4


def test_clip_grad_norm_matches_torch():
    rng = np.random.default_rng(0)
    gs = [rng.standard_normal((3, 4)) * 5, rng.standard_normal(7)]
    ps = [st.Tensor(np.zeros_like(g), requires_grad=True) for g in gs]
    pt = [torch.zeros(g.shape, requires_grad=True) for g in gs]
    for p, q, g in zip(ps, pt, gs):
        p.grad, q.grad = g.copy(), torch.tensor(g)
    n1 = optim.clip_grad_norm_(ps, 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(pt, 1.0)
    assert n1 == pytest.approx(n2.item())
    for p, q in zip(ps, pt):
        np.testing.assert_allclose(p.grad, q.grad.numpy(), rtol=1e-6)


def test_schedules():
    m = MLP([2, 2])
    opt = optim.SGD(m.parameters(), lr=1.0)
    s = optim.CosineWarmup(opt, warmup_steps=10, total_steps=110, min_ratio=0.1)
    lrs = [opt.lr]
    for _ in range(120):
        s.step()
        lrs.append(opt.lr)
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert lrs[60] == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.5)))
    assert lrs[110] == pytest.approx(0.1) and lrs[119] == pytest.approx(0.1)
    opt2 = optim.SGD(m.parameters(), lr=1.0)
    s2 = optim.StepLR(opt2, step_size=3, gamma=0.5)
    for _ in range(7):
        s2.step()
    assert opt2.lr == pytest.approx(0.25)


def test_linear_default_init_matches_torch_distribution():
    lin = nn.Linear(400, 300)
    bound = 1 / math.sqrt(400)
    assert np.abs(lin.weight.data).max() <= bound
    assert np.std(lin.weight.data) == pytest.approx(bound / math.sqrt(3), rel=0.02)
