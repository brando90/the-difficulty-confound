#!/bin/bash
cd /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/matched-10-16-2026 || exit 1
export CUDA_VISIBLE_DEVICES=''
export CODEX_HOME=/lfs/skampere2/0/brando9/.codex-vals
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
out=experiments/03_answer_matched_alignment/expt_v1/results
codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-5.6-terra -c 'model_reasoning_effort="high"' 'Read experiments/03_answer_matched_alignment/RUNBOOK.md and execute it end to end. This is the named SNAP execution owner, recovery attempt 3 in session codex-difficulty-e03-matched-vals. The personal and Stanford profiles both failed token refresh before any useful action or training; use the existing Vals CODEX_HOME. Record the failure and actual profile; no scientific budget was spent. The master froze the protocol and owns only the Mac; you own this isolated remote checkout. Start by acknowledging source hashes and effective execution settings, then implement, freeze source, run and verify the complete study and publish. Do not stop after planning or launching.' > "$out/executor_transcript.txt" 2>&1
code=$?
printf '%s\n' "$code" > /dfs/scratch0/brando9/.worktrees/the-difficulty-confound/matched-executor-exit.txt
exit "$code"
