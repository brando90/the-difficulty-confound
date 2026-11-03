# Experiment 23 on FarmShare: paths and environment. `source farmshare/env.sh` from anywhere.
# Layout: $E23_HOME/code = the unpacked package (experiments/...), $E23_HOME/out/{e13,e17b} = outputs,
# $E23_HOME/logs = Slurm logs, $E23_VENV = Python environment (built by prepare.sh).
export E23_HOME=${E23_HOME:-$HOME/e23}
export E23_CODE=$E23_HOME/code/experiments/23_exp13_full_training_8seeds
export E23_OUT=$E23_HOME/out
export E23_LOGS=$E23_HOME/logs
export E23_VENV=${E23_VENV:-$HOME/venvs/tdc23}
export E23_PY=$E23_VENV/bin/python
export E23_MAIL=brandojazz@gmail.com,brando.science@gmail.com
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$E23_OUT" "$E23_LOGS"
