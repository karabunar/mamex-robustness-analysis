import numpy as np
import pytest

from mamex_robustness.domain import repair


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def feats():
    rng = np.random.default_rng(0)
    image = rng.normal(size=(40, 6))
    text = rng.normal(size=(40, 5))
    return image, text, np.arange(30), np.arange(30, 40)


def test_neighbor_text_uses_the_image_neighbours(feats):
    image, text, warm, cold = feats
    image[30] = image[7]                                   # cold item 30 looks exactly like warm item 7
    out = repair.neighbor_text(image, text, warm, cold, k=1)
    assert np.allclose(out[0], unit(text[7]))


def test_neighbor_text_rows_are_unit_length(feats):
    image, text, warm, cold = feats
    assert np.allclose(np.linalg.norm(repair.neighbor_text(image, text, warm, cold, k=5), axis=1), 1.0)


def test_random_control_ignores_the_images_and_is_repeatable(feats):
    image, text, warm, cold = feats
    a = repair.neighbor_text(image, text, warm, cold, k=5, random_seed=0)
    b = repair.neighbor_text(image * 0 + 1, text, warm, cold, k=5, random_seed=0)
    assert np.allclose(a, b)
    assert not np.allclose(a, repair.neighbor_text(image, text, warm, cold, k=5))


def test_agreement_is_the_row_cosine():
    a = np.array([[1.0, 0.0], [0.0, 2.0]])
    b = np.array([[3.0, 0.0], [2.0, 0.0]])
    assert repair.agreement(a, b) == pytest.approx([1.0, 0.0])


def test_gated_lambda_is_high_for_low_agreement_and_half_at_the_median():
    agree = np.array([-1.0, 0.0, 1.0])
    lam = repair.gated_lambda(agree, 1.0, 20.0)
    assert lam[0] > 0.99 and lam[2] < 0.01 and lam[1] == pytest.approx(0.5)
    assert repair.gated_lambda(agree, 0.5, 20.0)[1] == pytest.approx(0.25)


def test_blend_endpoints_and_unit_length():
    text = np.array([[3.0, 0.0], [0.0, 2.0]])
    neighbor = unit(np.array([[1.0, 1.0], [1.0, -1.0]]))
    assert np.allclose(repair.blend(text, neighbor, 0.0), unit(text))
    assert np.allclose(repair.blend(text, neighbor, 1.0), neighbor)
    mixed = repair.blend(text, neighbor, np.array([0.2, 0.7]))
    assert np.allclose(np.linalg.norm(mixed, axis=1), 1.0)


def test_with_cold_rows_touches_only_cold_rows():
    text = np.arange(12.0).reshape(6, 2)
    out = repair.with_cold_rows(text, [4, 5], np.zeros((2, 2)))
    assert np.array_equal(out[:4], text[:4]) and not out[4:].any() and text[4:].any()


def test_auroc():
    pos = np.array([True, True, False, False])
    assert repair.auroc(np.array([4, 3, 2, 1.0]), pos) == 1.0
    assert repair.auroc(np.array([1, 2, 3, 4.0]), pos) == 0.0
    assert repair.auroc(np.ones(4), pos) == 0.5
    assert np.isnan(repair.auroc(np.ones(3), np.array([True, True, True])))
