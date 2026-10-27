import math

import torch

from mamex_robustness.domain.gating import vgr_reweight


def emb_with_cosine(c):
    """Returns (image_emb, text_emb) rows whose cosine similarity is c."""
    s = math.sqrt(1 - c * c)
    return torch.tensor([[1.0, 0.0]]), torch.tensor([[c, s]])


def test_rows_sum_to_one():
    torch.manual_seed(0)
    w = torch.softmax(torch.randn(6, 2), dim=1)
    out = vgr_reweight(w, torch.randn(6, 4), torch.randn(6, 4), gamma=8.0)
    assert torch.allclose(out.sum(1), torch.ones(6))


def test_low_cosine_item_loses_text_weight():
    w = torch.full((2, 2), 0.5)
    (v1, t1), (v2, t2) = emb_with_cosine(0.9), emb_with_cosine(-0.9)
    out = vgr_reweight(w, torch.cat([v1, v2]), torch.cat([t1, t2]), gamma=8.0)
    assert out[1, 1] < out[0, 1] < 0.5


def test_matches_formula():
    w = torch.tensor([[0.3, 0.7]])
    v, t = emb_with_cosine(0.5)
    s = torch.sigmoid(torch.tensor(8.0 * 0.5))
    expected_wt = 0.7 * s / (0.3 + 0.7 * s)
    out = vgr_reweight(w, v, t, gamma=8.0)
    assert torch.allclose(out[0, 1], expected_wt, atol=1e-6)
    assert torch.allclose(out[0, 0], 1 - expected_wt, atol=1e-6)


def test_no_gradient_through_the_cosine():
    v = torch.randn(3, 4, requires_grad=True)
    t = torch.randn(3, 4, requires_grad=True)
    w = torch.softmax(torch.randn(3, 2), dim=1).requires_grad_()
    vgr_reweight(w, v, t, gamma=8.0)[:, 1].sum().backward()
    assert v.grad is None and t.grad is None
    assert w.grad is not None


def test_zero_embedding_gives_finite_weights():
    w = torch.full((2, 2), 0.5)
    out = vgr_reweight(w, torch.zeros(2, 4), torch.zeros(2, 4), gamma=8.0)
    assert torch.isfinite(out).all()


def _mixed_cosine_batch():
    cosines = [0.9, 0.5, -0.2, -0.9]
    v = torch.cat([emb_with_cosine(c)[0] for c in cosines])
    t = torch.cat([emb_with_cosine(c)[1] for c in cosines])
    return torch.full((4, 2), 0.5), v, t


def test_shuffled_control_keeps_penalty_values_but_moves_them_between_items():
    w, v, t = _mixed_cosine_batch()
    plain = vgr_reweight(w, v, t, gamma=8.0)
    shuffled = vgr_reweight(w, v, t, gamma=8.0, control='shuffled')
    assert torch.allclose(plain[:, 1].sort().values, shuffled[:, 1].sort().values, atol=1e-6)
    assert not torch.allclose(plain[:, 1], shuffled[:, 1])
    assert torch.equal(shuffled, vgr_reweight(w, v, t, gamma=8.0, control='shuffled'))


def test_constant_control_applies_the_mean_factor_to_every_item():
    w, v, t = _mixed_cosine_batch()
    mean_factor = torch.sigmoid(8.0 * torch.tensor([0.9, 0.5, -0.2, -0.9])).mean()
    expected_wt = 0.5 * mean_factor / (0.5 + 0.5 * mean_factor)
    out = vgr_reweight(w, v, t, gamma=8.0, control='constant')
    assert torch.allclose(out[:, 1], expected_wt.expand(4), atol=1e-6)
    assert torch.allclose(out.sum(1), torch.ones(4))


def test_unknown_control_is_rejected():
    w, v, t = _mixed_cosine_batch()
    import pytest
    with pytest.raises(ValueError):
        vgr_reweight(w, v, t, gamma=8.0, control='bogus')
