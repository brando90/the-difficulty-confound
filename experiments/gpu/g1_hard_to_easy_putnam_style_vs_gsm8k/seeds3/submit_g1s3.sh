#!/bin/bash
# g1 seeds3 on Sherlock. From a login shell, once:
#   mkdir -p $SCRATCH/tdc_g1s3 && cd $SCRATCH/tdc_g1s3 && tar xzf g1s3_sherlock.tgz && bash submit_g1s3.sh 2>&1 | tee submit_g1s3.log
# Builds the venv (login node, wheels only, ~4 min), fetches Qwen2.5-0.5B into its own cache, checks the code against
# CODE_SHA256 and the pilot's data against its manifest hash, then submits the 16-run H100 array and a report job.
# Ordinary batch jobs only; nothing persistent. Results: $SCRATCH/tdc_g1s3/g1s3_results.tgz
set -uo pipefail
cd "$(dirname "$0")" || exit 2
source the-difficulty-confound/experiments/gpu/g1_hard_to_easy_putnam_style_vs_gsm8k/seeds3/env.sh || exit 2
[[ -f $VENV/.ready ]] || { echo "venv build failed: see the pip output above"; exit 2; }
(cd "$REPO" && sha256sum -c --quiet CODE_SHA256) || { echo "code differs from CODE_SHA256"; exit 2; }
echo "code: $(cat "$REPO/CODE_COMMIT")"
[[ -f $DATA/v1/manifest.json ]] || { echo "no pilot data at $DATA/v1 (set DATA=... to the folder holding v1/)"; exit 2; }
[[ $(sha256sum "$DATA/v1/manifest.json" | cut -d' ' -f1) == "$DATA_MANIFEST_SHA256" ]] || { echo "data manifest hash differs from the pilot's"; exit 2; }
echo "data: $DATA/v1 (pilot manifest hash OK)"
"$PY" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen2.5-0.5B', allow_patterns=['*.json','*.safetensors','*.txt','merges.txt','vocab.json']))" || { echo "model download failed"; exit 2; }
HF_HUB_OFFLINE=1 "$PY" -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B'); print('offline tokenizer OK')" || exit 2
N=$(cd "$GPU_DIR" && "$PY" common/runs.py count --tier seeds3 --plans g1)
[[ $N == 16 ]] || { echo "expected 16 runs in tier seeds3, got $N"; exit 2; }
(cd "$GPU_DIR" && "$PY" common/runs.py list --tier seeds3 --plans g1)
mkdir -p slurm_logs
{ echo "== $(date) $(hostname)"; echo "== H100 nodes in gpu"; sinfo -p gpu -h -N -o "%N %f %T %G" 2>&1 | grep -i H100
  echo "== jobs waiting in gpu"; squeue -p gpu -h -t PD | wc -l; echo "== sshare"; sshare -U 2>&1; } >inventory_g1s3.txt 2>&1
sbatch --test-only --array=0-$((N - 1)) "$S3/g1s3.sbatch" 2>&1 | tail -1
A=$(sbatch --parsable --array=0-$((N - 1)) "$S3/g1s3.sbatch") || { echo "array submit failed"; exit 2; }
R=$(sbatch --parsable --dependency=afterany:$A -p normal -c 1 --mem=4G -t 00:20:00 -J g1s3_report -o slurm_logs/g1s3_report_%j.out \
  --mail-type=END,FAIL --mail-user=brandojazz@gmail.com,brando.science@gmail.com "$S3/pack_results.sh") || echo "report submit failed (each array task packs the results anyway)"
echo "ARRAY=$A REPORT=${R:-none} submitted $(date '+%m-%d-%Y %H:%M:%S')" | tee jobs_g1s3.txt
squeue -u "$USER" -o "%.18i %.12j %.9P %.8T %.10M %.20R" | head -20
