#!/bin/bash
# Build the FarmShare package (repo side, not on the cluster): the committed source experiments 13 and 17 and this
# folder at HEAD (git archive), plus prep/ (the frozen references, not in git; checked against the committed
# prep/prep_manifest.json). Usage: bash farmshare/make_package.sh OUTDIR    -> OUTDIR/e23_pkg_<commit>.tgz(.sha256)
# On FarmShare: mkdir -p ~/e23 && tar xzf e23_pkg_<commit>.tgz -C ~/e23   (gives ~/e23/code/experiments/...)
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
NAME=$(basename "$HERE")
OUT=${1:?usage: make_package.sh OUTDIR}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
cd "$REPO"
git diff --quiet HEAD -- "experiments/$NAME" || { echo "uncommitted changes in experiments/$NAME: commit first"; exit 1; }
SHA=$(git rev-parse --short=8 HEAD)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git archive --format=tar --prefix=code/ HEAD experiments/13_ginc_hmm_complexity_vs_alignment \
  experiments/17_ginc_disjoint_support_and_hard_to_easy "experiments/$NAME" | tar -x -C "$TMP"
cp -r "$HERE/prep/e13" "$HERE/prep/e17b" "$TMP/code/experiments/$NAME/prep/"
python3 - "$TMP/code/experiments/$NAME/prep" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
man = json.loads((root / 'prep_manifest.json').read_text())
bad = [f'{e}/{k}' for e, files in man.items() for k, h in files.items()
       if hashlib.sha256((root / e / k).read_bytes()).hexdigest() != h]
assert not bad, f'prep differs from the committed manifest: {bad[:5]}'
print('prep matches prep_manifest.json:', sum(len(v) for v in man.values()), 'files')
PY
echo "$SHA" > "$TMP/code/PACKAGE_COMMIT"
tar -C "$TMP" -czf "$OUT/e23_pkg_$SHA.tgz" code
(cd "$OUT" && sha256sum "e23_pkg_$SHA.tgz" > "e23_pkg_$SHA.tgz.sha256" && cat "e23_pkg_$SHA.tgz.sha256" && du -h "e23_pkg_$SHA.tgz")
