"""Export existing addition data for exposure-aware predictive-law development.

No training, fitting, provider calls, or modification of frozen artifacts.
Features and outcomes are deliberately separate; metadata are not predictors.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = REPO / 'experiments/archive/01_fixed_width_addition_exposure/expt_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def carry_chain(a, b):
    carry = run = longest = 0
    for _ in range(3):
        carry = (a % 10 + b % 10 + carry) // 10
        run = run + 1 if carry else 0
        longest = max(longest, run)
        a //= 10
        b //= 10
    return longest


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export(output):
    config = json.loads((EXPERIMENT / 'config.json').read_text())
    if config['digits'] != 3 or config['hidden_width'] != 128:
        raise ValueError('This export describes the frozen three-digit, width-128 run.')
    pairs = [(a, b) for a in range(1000) for b in range(a, 1000)]
    strata = [carry_chain(a, b) for a, b in pairs]
    inputs = {str(EXPERIMENT.relative_to(REPO) / 'config.json'): sha(EXPERIMENT / 'config.json')}
    features, outcomes = [], []
    test_identity = None
    for seed in config['seeds']:
        for condition, weights in config['conditions'].items():
            train_path = EXPERIMENT / f'results/train_{seed}_{condition}.csv'
            pred_path = EXPERIMENT / f'results/predictions_{seed}_{condition}.csv'
            for path in (train_path, pred_path):
                inputs[str(path.relative_to(REPO))] = sha(path)
            training = read_csv(train_path)
            ids = [int(row['item_id']) for row in training]
            assert len(ids) == len(set(ids)) == config['train_size']
            counts = Counter(strata[i] for i in ids)
            assert [counts[k] for k in range(4)] == [round(w * len(ids)) for w in weights]
            if condition != 'shuffled_labels':
                assert all(int(row['training_target']) == sum(pairs[int(row['item_id'])]) for row in training)
            groups = defaultdict(list)
            test = [row for row in read_csv(pred_path) if row['split'] == 'test']
            identity = {(int(row['item_id']), int(row['a']), int(row['b']), int(row['target'])) for row in test}
            assert len(test) == len(identity) == 4000
            assert not set(ids) & {row[0] for row in identity}
            if test_identity is None:
                test_identity = identity
            assert identity == test_identity
            for row in test:
                i, k = int(row['item_id']), int(row['carry_chain'])
                assert k == strata[i]
                assert (int(row['a']), int(row['b'])) == pairs[i]
                assert int(row['target']) == sum(pairs[i])
                assert int(row['correct']) == int(row['prediction'] == row['target'])
                groups[k].append(row)
            for k in range(4):
                key = {'seed': seed, 'condition': condition, 'carry_chain': k}
                # Computed entirely from protocol, input structure and training manifest.
                features.append({**key, 'parameter_count': 29480, 'optimizer_updates': config['steps'],
                                 'training_pairs': len(ids), 'stratum_training_pairs': counts[k],
                                 'stratum_training_fraction': counts[k] / len(ids),
                                 'log1p_stratum_training_pairs': math.log1p(counts[k]),
                                 'labels_are_valid': int(condition != 'shuffled_labels')})
                rows = groups[k]
                assert len(rows) == config['test_per_stratum']
                outcomes.append({**key, 'test_items': len(rows),
                                 'correct': sum(int(r['correct']) for r in rows),
                                 'accuracy': sum(int(r['correct']) for r in rows) / len(rows),
                                 'mean_conditional_bits': math.fsum(float(r['conditional_bits']) for r in rows) / len(rows)})
    assert len(features) == len(outcomes) == 128
    assert len({tuple(row[k] for k in ['seed', 'condition', 'carry_chain']) for row in features}) == 128
    assert not {'accuracy', 'correct', 'mean_conditional_bits', 'prediction'} & features[0].keys()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'features.csv', features)
    write_csv(output / 'outcomes.csv', outcomes)
    manifest = {
        'purpose': 'Retrospective feature export; no new fitted law or validation claim.',
        'rows': 128, 'seeds': 8, 'conditions': 4, 'carry_strata': 4,
        'test_items_per_cell': 1000, 'distinct_test_items': 4000,
        'metadata_not_predictors': ['seed', 'condition'],
        'constant_coordinates_not_identifiable': ['parameter_count', 'optimizer_updates', 'training_pairs'],
        'exclude_from_valid_label_law_fit': 'labels_are_valid == 0 (32 shuffled-label control rows)',
        'candidate_predictors': ['carry_chain', 'log1p_stratum_training_pairs', 'carry_chain × log1p_stratum_training_pairs'],
        'redundant_exposure_encodings_choose_one': ['stratum_training_pairs', 'stratum_training_fraction', 'log1p_stratum_training_pairs'],
        'warning': 'All four exposure counts sum to a fixed total. This is a distribution intervention, not an isolated count effect. Test outcomes were already inspected; new model selection here is exploratory.',
        'source_sha256': inputs,
        'export_sha256': {name: sha(output / name) for name in ['features.csv', 'outcomes.csv']},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO / 'experiments/ideas/predictability_bridge/data')
    args = parser.parse_args()
    manifest = export(args.output)
    print(json.dumps({k: manifest[k] for k in ['rows', 'seeds', 'conditions', 'carry_strata', 'distinct_test_items']}))
