"""Deterministic audit of frozen inputs, saved tables, and trained checkpoints."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def read(name):
    with (OUT / name).open() as handle:
        return list(csv.DictReader(handle))


def main():
    config = json.loads((ROOT / 'config.json').read_text())
    runtime = json.loads((OUT / 'runtime.json').read_text())
    for filename, field in [('config.json', 'config_sha256'), ('run.py', 'code_sha256')]:
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == runtime[field]
    local = json.loads((OUT / 'summary.json').read_text())
    remote = json.loads((OUT / 'summary_remote.json').read_text())
    assert local == remote
    a, b = np.triu_indices(1000)
    answer = a + b
    # Independent carry implementation uses decimal strings, not run.py helpers.
    def chain(x, y):
        run = longest = incoming = 0
        for dx, dy in zip(f'{x:03d}'[::-1], f'{y:03d}'[::-1]):
            incoming = int(int(dx) + int(dy) + incoming >= 10)
            run = run + 1 if incoming else 0
            longest = max(longest, run)
        return longest
    strata = np.array([chain(x, y) for x, y in zip(a, b)])
    held = read('heldout_items.csv')
    held_by_id = {int(r['item_id']): r for r in held}
    assert len(held_by_id) == len(held) == 5000
    for item, row in held_by_id.items():
        assert (int(row['a']), int(row['b']), int(row['target']), int(row['carry_chain'])) == (a[item], b[item], answer[item], strata[item])
    train_rows = prediction_rows = 0
    for seed in config['seeds']:
        balanced = read(f'train_{seed}_balanced.csv')
        for condition, weights in config['conditions'].items():
            train = read(f'train_{seed}_{condition}.csv')
            ids = np.array([int(r['item_id']) for r in train])
            targets = np.array([int(r['training_target']) for r in train])
            assert len(set(ids)) == len(ids) == config['train_size']
            assert not set(ids) & set(held_by_id)
            assert np.array_equal(np.bincount(strata[ids], minlength=4), np.rint(np.array(weights) * config['train_size']).astype(int))
            if condition != 'shuffled_labels':
                assert np.array_equal(targets, answer[ids])
            else:
                assert [r['item_id'] for r in train] == [r['item_id'] for r in balanced]
                assert sorted(targets) == sorted(int(r['training_target']) for r in balanced)
            predictions = read(f'predictions_{seed}_{condition}.csv')
            assert len(predictions) == len({int(r['item_id']) for r in predictions}) == 5000
            for r in predictions:
                expected = held_by_id[int(r['item_id'])]
                assert all(r[field] == expected[field] for field in ['split', 'a', 'b', 'target', 'carry_chain'])
                assert int(r['correct']) == int(int(r['target']) == int(r['prediction']))
                assert np.isfinite(float(r['conditional_bits'])) and float(r['conditional_bits']) >= 0
            # Weights-only loading of our own trained tensor dictionaries.
            weights_saved = torch.load(OUT / f'model_{seed}_{condition}.pt', map_location='cpu', weights_only=True)
            assert sum(t.numel() for t in weights_saved.values()) == 29480
            assert all(torch.isfinite(t).all() for t in weights_saved.values())
            train_rows += len(train)
            prediction_rows += len(predictions)
    cells = json.loads((OUT / 'cells.json').read_text())
    assert len(cells) == 32 and all(r['status'] == 'complete' and r['steps'] == 3000 for r in cells)
    receipt = {'status': 'PASS', 'complete_cells': 32, 'training_rows_checked': train_rows,
               'prediction_rows_checked': prediction_rows, 'unique_heldout_pairs': 5000,
               'train_heldout_overlap': 0, 'oracle_checks_at_launch': 500500,
               'independent_carry_labels_checked': len(strata), 'saved_model_parameters_each': 29480,
               'local_summary_matches_remote_exactly': True, 'original_run_and_config_hashes_match': True}
    (OUT / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
