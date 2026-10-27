from types import SimpleNamespace

import numpy as np
import torch
import pytest

N_USER, N_WARM, N_COLD = 30, 30, 10
IMG_DIM, TXT_DIM = 12, 8


@pytest.fixture
def data_dir(tmp_path):
    """Tiny synthetic Baby-like dataset: items 0-29 warm, 30-39 cold."""
    rng = np.random.RandomState(0)
    with open(tmp_path / 'train.txt', 'w') as f:
        for u in range(N_USER):
            items = rng.choice(N_WARM, 3, replace=False)
            f.write(f'{u} ' + ' '.join(map(str, items)) + '\n')
    for name, offset in (('val', 0), ('test', 5)):
        with open(tmp_path / f'{name}.txt', 'w') as f:
            for u in range(N_USER):
                cold = N_WARM + (u + offset) % N_COLD
                f.write(f'{u} {cold}\n')
    np.save(tmp_path / 'cold_item_index.npy', np.arange(N_WARM, N_WARM + N_COLD))
    np.save(tmp_path / 'image_feat.npy', rng.randn(N_WARM + N_COLD, IMG_DIM).astype(np.float32))
    np.save(tmp_path / 'text_feat.npy', rng.randn(N_WARM + N_COLD, TXT_DIM).astype(np.float32))
    return tmp_path


@pytest.fixture
def env(data_dir):
    args = SimpleNamespace(dataset='baby', free_emb_dimension=16)
    return SimpleNamespace(DATA_PATH=str(data_dir), args=args, device='cpu')


@pytest.fixture
def session(env):
    from mamex_robustness.application.session import Session
    from mamex_robustness.data.loader import Loader4MM
    from mamex_robustness.models.mamex import MOME_model

    env.args.__dict__.update(
        lr=0.01, topk=[3, 5], batch_size=64, eva_interval=1, early_stop=100, ckpt_start_epoch=0,
        tensorboard=0, log=0, save=0, align_coeff=1, reg_coeff=1e-6, moe_balance_coeff=1,
        fusion_balance_coeff=1)
    dataset = Loader4MM(env)
    torch.manual_seed(0)
    return Session(env, MOME_model(env, dataset), dataset)
