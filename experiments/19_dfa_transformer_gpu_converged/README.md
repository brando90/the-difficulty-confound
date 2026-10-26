# Experiment 19: Experiment 11's transformer DFA grid trained to convergence on a GPU

**Doc link:** <https://github.com/brando90/the-difficulty-confound/blob/main/experiments/19_dfa_transformer_gpu_converged/README.md>

Experiment 11's transformer barely learned within its CPU budget, so its elasticity ratio E (how much alignment moves the data bill relative to exact DFA complexity) could have been an artefact of under-training. This experiment reruns the same 512-run grid (same model, initialization, optimizer, tasks, test set and seeds) on one Sherlock GPU for up to 64 times longer, with a convergence criterion and a yes/no rule frozen before the run.

Status: RUN, 10-19-2026. Protocol frozen at commit 5b00f295. Trained to 2^20 examples (16x) in 1.06 GPU-hours; verify.py PASS. See [RESULTS_dfa_transformer_gpu.md](RESULTS_dfa_transformer_gpu.md) and [PROTOCOL.md](PROTOCOL.md).

**TLDR-end:** [tdc: Expt 19 README] Experiment 11's DFA grid trained 16x longer on a GPU: E rose from 0.083 to 0.165, still far below 1; the grid did not converge because the 16- and 32-state automata stalled near chance, so the transformer, not the data budget, is what holds E down.
