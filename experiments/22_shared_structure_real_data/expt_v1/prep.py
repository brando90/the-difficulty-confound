"""Raw files -> prepared data in $EXP22_DATA, then check every hash against data_sha256.json (exit 1 on a mismatch).

  python prep.py --raw <raw folder>            # build + check (what the cluster prep job runs)
  python prep.py --raw <raw folder> --write    # build + record the hashes (done once, before the freeze)"""
import argparse, json, os, sys
import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
PIN = os.path.join(HERE, 'data_sha256.json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()
    os.makedirs(D.DATA, exist_ok=True)
    raw = {f: D.sha_file(os.path.join(a.raw, f)) for f in sorted(D.RAW)}
    want = None if a.write else json.load(open(PIN))
    if want:
        bad = [f for f in raw if raw[f] != want['raw'][f]]
        if bad:
            print('raw MISMATCH', bad)
            sys.exit(1)
        print('raw files OK')
    for name, fn in (('cifar100', D.prep_cifar100), ('cifar10', D.prep_cifar10)):
        if not os.path.exists(os.path.join(D.DATA, name + '.npz')):  # a stale file still fails the hash check
            fn(a.raw, D.DATA)
    D.prep_20ng(a.raw, D.DATA)
    prepared = D.prepared_hashes()
    ng = D.load_20ng()
    counts = {'20ng_train': len(ng['train']), '20ng_test': len(ng['test']), '20ng_dropped': ng['dropped']}
    if a.write:
        json.dump({'raw': raw, 'prepared': prepared, 'counts': counts}, open(PIN, 'w'), indent=1, sort_keys=True)
        print('wrote', PIN, counts)
        return
    bad = [k for k in want['prepared'] if prepared.get(k) != want['prepared'][k]]
    print('data check:', 'OK' if not bad else 'MISMATCH ' + ' '.join(bad), counts)
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
