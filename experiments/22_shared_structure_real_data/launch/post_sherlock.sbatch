#!/bin/bash
#SBATCH -J exp22-post
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/exp22_post_%j.out
# After every array has ended: verify.py --stage remote (data hashes, runs present, two GPU replays, structure +
# analysis, manifest), then pack results/ into $SCRATCH/exp22/exp22_results.tgz for transfer.
set -uo pipefail
cd "${EXP22_DIR:?set EXP22_DIR}/expt_v1"
export EXP22_DATA=$SCRATCH/exp22_data OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY=${EXP22_VENV:?set EXP22_VENV}/bin/python
[ -d results ] || { echo "EXP22 POST FAILED: no results/ (nothing ran)"; exit 1; }
cp ../runs_*.txt results/ 2>/dev/null
# verify writes results/MANIFEST.sha256 last; everything added after it (logs, report) is outside the manifest
$PY verify.py --stage remote 2>&1 | tee ../logs/verify_remote.log | tail -40
$PY report.py > ../logs/report_remote.md 2>&1; head -30 ../logs/report_remote.md
mkdir -p results/logs && cp ../logs/* results/logs/ 2>/dev/null
tar czf "$SCRATCH/exp22/exp22_results.tgz" results
sha256sum "$SCRATCH/exp22/exp22_results.tgz" | tee "$SCRATCH/exp22/exp22_results.tgz.sha256"
ls -la "$SCRATCH/exp22/exp22_results.tgz"
echo "EXP22 POST DONE"
