"""Persistent ordinary-code execution owner; bounded child process group.

No provider calls or scientific retries. Survives SSH disconnect in tmux;
does not claim automatic node-reboot recovery.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def supervise(command, timeout, status_file, name):
    started = time.monotonic()
    child = subprocess.Popen(command, start_new_session=True)
    while True:
        code = child.poll()
        elapsed = time.monotonic() - started
        status = {'phase': name, 'supervisor_pid': os.getpid(), 'child_pid': child.pid,
                  'elapsed_seconds': elapsed, 'exit_code': code,
                  'state': 'running' if code is None else ('complete' if code == 0 else 'failed')}
        tmp = status_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(status, indent=2) + '\n')
        tmp.replace(status_file)
        if code is not None:
            return code
        if elapsed > timeout:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            status.update(state='timeout', exit_code=124)
            status_file.write_text(json.dumps(status, indent=2) + '\n')
            return 124
        time.sleep(5)


def main():
    if '--self-test' in sys.argv:
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / 'status.json'
            code = supervise([sys.executable, '-c', 'raise SystemExit(7)'], 20, receipt, 'failure_test')
            assert code == 7 and json.loads(receipt.read_text())['state'] == 'failed'
        print('Supervisor detected synthetic child failure: PASS', flush=True)
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    if (out / 'job_status.json').exists():
        raise RuntimeError('Refusing to overwrite an earlier supervisor receipt.')
    config = json.loads((ROOT / 'config.json').read_text())
    status_file = out / 'job_status.json'
    code = supervise([sys.executable, str(ROOT / 'run.py')], config['max_wall_seconds'] + 30, status_file, 'training')
    if code == 0:
        code = supervise([sys.executable, str(ROOT / 'analyze.py')], 300, status_file, 'analysis')
    if code == 0:
        code = supervise([sys.executable, str(ROOT / 'verify.py')], 600, status_file, 'verification')
    print(f'FINAL_EXIT_CODE={code}', flush=True)
    sys.exit(code)


if __name__ == '__main__':
    main()
