import numpy as np

from mamex_robustness.data.loader import Loader4MM


def test_noisy_text_feat_corrupts_only_cold_items_and_keeps_original(env):
    ds = Loader4MM(env)
    original = ds.text_feat.copy()
    noisy = ds.noisy_text_feat(noise_rate=0.4, noise_type='hashtag', seed=7)
    changed = set(np.where((noisy != original).any(axis=1))[0].tolist())
    assert len(changed) == 4
    assert changed <= set(ds.cold_item_index.tolist())
    assert np.array_equal(ds.text_feat, original)
