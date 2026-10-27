import numpy as np
import torch

from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model


def test_eval_text_override_changes_only_corrupted_rows_and_can_be_cleared(env):
    ds = Loader4MM(env)
    torch.manual_seed(0)
    model = MOME_model(env, ds).eval()
    clean = model()[1].detach().clone()
    original_text = model.text_feat.clone()

    noisy = ds.noisy_text_feat(1.0, 'shuffle', seed=3)
    model.set_eval_text(noisy)
    with_noise = model()[1].detach().clone()
    assert not torch.allclose(with_noise[30:], clean[30:])
    assert torch.allclose(with_noise[:30], clean[:30])
    assert torch.equal(model.text_feat, original_text)

    model.set_eval_text(None)
    assert torch.allclose(model()[1], clean)
