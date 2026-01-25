import torch

from sso import SSO
from dion import NorMuon


def _make_params(seed: int = 1234):
    g = torch.Generator().manual_seed(seed)
    p1 = torch.nn.Parameter(torch.randn(4, 3, generator=g))
    p2 = torch.nn.Parameter(torch.randn(7, generator=g))
    p3 = torch.nn.Parameter(torch.randn(2, 5, generator=g))
    return [p1, p2, p3]


def _assign_grads(params, seed: int):
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn_like(p, generator=g)


def _build_group(params):
    return [
        dict(
            params=params,
            algorithm="adamw",
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            weight_decay=0.01,
            cautious_wd=True,
        )
    ]


def _assert_state_close(opt_a, opt_b, params_a, params_b):
    for pa, pb in zip(params_a, params_b):
        state_a = opt_a.state[pa]
        state_b = opt_b.state[pb]
        torch.testing.assert_close(state_a["momentum"], state_b["momentum"], rtol=0, atol=1e-6)
        torch.testing.assert_close(state_a["variance"], state_b["variance"], rtol=0, atol=1e-6)


def _assert_params_close(params_a, params_b):
    for pa, pb in zip(params_a, params_b):
        torch.testing.assert_close(pa, pb, rtol=0, atol=1e-6)


def test_sso_adamw_matches_normuon():
    params_sso = _make_params(seed=1234)
    params_nor = _make_params(seed=1234)

    opt_sso = SSO(_build_group(params_sso), weight_decay=0.01, cautious_wd=True)
    opt_nor = NorMuon(_build_group(params_nor), weight_decay=0.01, cautious_wd=True)

    for step in range(1, 4):
        _assign_grads(params_sso, seed=1000 + step)
        _assign_grads(params_nor, seed=1000 + step)

        opt_sso.step()
        opt_nor.step()

        _assert_params_close(params_sso, params_nor)
        _assert_state_close(opt_sso, opt_nor, params_sso, params_nor)


if __name__ == "__main__":
    test_sso_adamw_matches_normuon()
    print("SSO AdamW matches NorMuon AdamW for tested steps.")
