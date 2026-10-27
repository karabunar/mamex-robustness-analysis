import json

from mamex_robustness.cli import main


def run_cli(monkeypatch, tmp_path, data_dir, *extra):
    monkeypatch.chdir(tmp_path)
    main(['--data_path', str(data_dir), '--use_gpu', '0', '--topk', '3', '5', '--epoch', '2',
          '--eva_interval', '1', '--batch_size', '64', '--free_emb_dimension', '16',
          '--results_dir', str(tmp_path / 'results'), *extra])
    return sorted((tmp_path / 'results').glob('*.json'))


def test_improved_mode_writes_result_json(monkeypatch, tmp_path, data_dir):
    files = run_cli(monkeypatch, tmp_path, data_dir, '--mode', 'improved', '--noise_rate', '0.4')
    assert [f.name for f in files] == ['improved_hashtag_r0.40_g8.0_seed2025.json']
    result = json.loads(files[0].read_text())
    assert result['args']['use_vgr'] == 1 and result['args']['mode'] == 'improved'
    assert set(result['report']) >= {'clean', 'noisy', 'noisy_vgr', 'gate', 'cost'}
    assert result['best_epoch'] in (0, 1) and result['train_seconds'] >= 0


def test_baseline_mode_has_zero_noise_in_name(monkeypatch, tmp_path, data_dir):
    files = run_cli(monkeypatch, tmp_path, data_dir, '--mode', 'baseline')
    assert [f.name for f in files] == ['baseline_hashtag_r0.00_g8.0_seed2025.json']


def test_improved_with_gamma_grid_records_the_selected_gamma(monkeypatch, tmp_path, data_dir):
    files = run_cli(monkeypatch, tmp_path, data_dir, '--mode', 'improved', '--noise_rate', '0.4',
                    '--gamma_grid', '1', '8')
    assert [f.name for f in files] == ['improved_hashtag_r0.40_gsel_seed2025.json']
    result = json.loads(files[0].read_text())
    assert result['report']['gamma_selection']['selected'] == result['args']['vgr_gamma']
    assert result['args']['vgr_gamma'] in (1.0, 8.0)


def test_improved_and_stress_test_share_the_same_clean_checkpoint(monkeypatch, tmp_path, data_dir):
    run_cli(monkeypatch, tmp_path, data_dir, '--mode', 'stress_test', '--noise_rate', '0.4')
    run_cli(monkeypatch, tmp_path, data_dir, '--mode', 'improved', '--noise_rate', '0.4')
    stress, improved = (json.loads(f.read_text()) for f in sorted((tmp_path / 'results').glob('*.json'))[::-1])
    assert stress['args']['mode'] == 'stress_test'
    assert stress['best_epoch'] == improved['best_epoch']
    assert stress['report']['clean'] == improved['report']['clean']
    assert stress['report']['noisy_vgr'] == improved['report']['noisy_vgr']


def test_run_writes_per_pair_diagnostics_next_to_the_json(monkeypatch, tmp_path, data_dir):
    import numpy as np
    files = run_cli(monkeypatch, tmp_path, data_dir, '--mode', 'improved', '--noise_rate', '0.4')
    npz = tmp_path / 'results' / 'diagnostics' / (files[0].stem + '.npz')
    assert npz.exists()
    d = np.load(npz)
    assert {'users', 'items', 'corrupted_items', 'ranks_clean', 'ranks_clean_vgr', 'ranks_noisy', 'ranks_noisy_vgr'} <= set(d.files)
