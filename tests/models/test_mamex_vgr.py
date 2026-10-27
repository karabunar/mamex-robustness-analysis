import pytest
import torch

from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model


def build(env, use_vgr=None, gamma=None):
    if use_vgr is not None:
        env.args.use_vgr = use_vgr
    if gamma is not None:
        env.args.vgr_gamma = gamma
    torch.manual_seed(0)
    return MOME_model(env, Loader4MM(env)).eval()


def test_default_is_original_behaviour(env):
    model = build(env)
    assert model.use_vgr is False
    _, item_emb, image_emb, text_emb, _, _ = model()
    w = model.last_gate_weights
    assert torch.allclose(item_emb, w[:, :1] * image_emb + w[:, 1:] * text_emb, atol=1e-6)
    assert item_emb.sum().item() == pytest.approx(8.45648, abs=1e-3)


def test_vgr_lowers_text_weight_and_rows_sum_to_one(env):
    model = build(env, use_vgr=1, gamma=8.0)
    model.use_vgr = False
    model()
    plain = model.last_gate_weights.clone()
    model.use_vgr = True
    model()
    vgr = model.last_gate_weights
    assert torch.allclose(vgr.sum(1), torch.ones(40), atol=1e-6)
    assert (vgr[:, 1] < plain[:, 1]).all()
    assert model.last_cos.shape == (40,)


def test_state_dict_keys_identical_with_and_without_vgr(env):
    plain = build(env, use_vgr=0)
    vgr = build(env, use_vgr=1)
    assert list(plain.state_dict()) == list(vgr.state_dict())
    vgr.load_state_dict(plain.state_dict())


def test_control_name_is_read_from_args_and_changes_the_weights(env):
    model = build(env, use_vgr=1, gamma=8.0)
    assert model.vgr_control == 'cosine'
    model()
    cosine_w = model.last_gate_weights.clone()
    model.vgr_control = 'constant'
    model()
    assert not torch.allclose(model.last_gate_weights, cosine_w)
