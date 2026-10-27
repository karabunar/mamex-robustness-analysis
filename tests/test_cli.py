import pytest

from mamex_robustness.cli import parse_args
from mamex_robustness.tool import apply_mode_defaults


def parsed(*argv):
    return apply_mode_defaults(parse_args(list(argv)))


def test_baseline_is_clean_without_vgr():
    args = parsed('--mode', 'baseline', '--noise_rate', '0.4')
    assert args.use_vgr == 0
    assert args.noise_rate == 0.0


def test_stress_test_keeps_noise_and_no_vgr():
    args = parsed('--mode', 'stress_test', '--noise_rate', '0.4')
    assert args.use_vgr == 0
    assert args.noise_rate == 0.4


def test_improved_enables_vgr():
    args = parsed('--mode', 'improved', '--noise_rate', '0.4')
    assert args.use_vgr == 1
    assert args.noise_rate == 0.4


def test_explicit_use_vgr_overrides_mode_default():
    assert parsed('--mode', 'stress_test', '--use_vgr', '1').use_vgr == 1


def test_noise_seed_defaults_to_training_seed():
    assert parsed('--mode', 'baseline', '--seed', '2026').noise_seed == 2026
    assert parsed('--mode', 'baseline', '--seed', '2026', '--noise_seed', '5').noise_seed == 5


def test_topk_accepts_ints_from_the_command_line():
    assert parsed('--topk', '3', '5').topk == [3, 5]


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(['--mode', 'bogus'])


def test_gamma_grid_is_parsed_as_floats_and_off_by_default():
    assert parsed('--gamma_grid', '1', '8').gamma_grid == [1.0, 8.0]
    assert parsed().gamma_grid is None


def test_train_with_vgr_defaults_to_off_even_in_improved_mode():
    assert parsed('--mode', 'improved').train_with_vgr == 0
    assert parsed('--mode', 'improved', '--train_with_vgr', '1').train_with_vgr == 1
