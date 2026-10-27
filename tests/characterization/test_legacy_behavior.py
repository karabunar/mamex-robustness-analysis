"""Pin the behaviour of the original MAMEX code before it is restructured."""
import numpy as np
import pytest
import torch

from mamex_robustness.models.criterion import BPR
from mamex_robustness.application.evaluation import num_faiss_evaluate
from mamex_robustness.data.loader import Loader4MM
from mamex_robustness.models.mamex import MOME_model, MoE_Layer
GOLDEN_ITEM_EMB_SUM = 8.45648


def test_loader_counts(env):
    ds = Loader4MM(env)
    assert ds.n_user == 30
    assert ds.m_item == 40
    assert sorted(ds.cold_item_index.tolist()) == list(range(30, 40))
    assert sorted(ds.warm_item_index.tolist()) == list(range(30))
    assert ds.image_feat.shape == (40, 12) and ds.text_feat.shape == (40, 8)


def test_moe_layer_output_shape_and_aux_loss_nonneg():
    torch.manual_seed(0)
    layer = MoE_Layer(8, 16, num_experts=4)
    out, aux = layer(torch.randn(5, 8))
    assert out.shape == (5, 16)
    assert aux.item() >= 0


def test_model_forward_shapes_and_gate_rows_sum_to_one(env):
    torch.manual_seed(0)
    ds = Loader4MM(env)
    model = MOME_model(env, ds).eval()
    user_emb, item_emb, image_emb, text_emb, moe_aux, balance = model()
    assert user_emb.shape == (30, 16)
    assert item_emb.shape == image_emb.shape == text_emb.shape == (40, 16)
    # item_emb is a convex combination of the two modality embeddings
    lo = torch.minimum(image_emb, text_emb) - 1e-5
    hi = torch.maximum(image_emb, text_emb) + 1e-5
    assert bool(((item_emb >= lo) & (item_emb <= hi)).all())
    assert item_emb.sum().item() == pytest.approx(GOLDEN_ITEM_EMB_SUM, abs=1e-3)


def test_bpr_loss_golden():
    user = torch.tensor([[1.0, 0.0]])
    items = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    loss, reg = BPR()(user, items, torch.tensor([0]), torch.tensor([1]), torch.tensor([2]))
    # pos score 1, neg score 0 -> softplus(0 - 1)
    assert loss.item() == np.float32(np.log1p(np.exp(-1.0)))
    assert reg.item() == 1.0 + 1.0 + 1.0


def test_num_faiss_evaluate_golden():
    items = np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32)
    users = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)
    test = {0: [1], 1: [2], 2: [0]}
    train = {0: [3], 1: [3], 2: [3]}
    hr, recall, ndcg = num_faiss_evaluate(test, [0, 1, 2], [0, 1, 2, 3], train, [1, 2], users, items)
    # user0: ranks 0,1 -> hit at rank 2; user1: ranks 2,.. -> hit at rank 1; user2: hit at rank 1
    assert hr[1] == np.float64(2 / 3)
    assert hr[2] == 1.0
    assert abs(ndcg[2] - (np.log(2) / np.log(3) + 1 + 1) / 3) < 1e-9
