"""GPU hours used by Slurm jobs, from sacct. Two uses (PROTOCOL Section 3.3):

  budget.py --at441-jobs jobs_at441.txt --config config.json   -> JSON with AT-441's GPU hours and this run's cap
  budget.py --job <id>                                          -> JSON with that job's GPU hours (gpu_usage.json)
Running or pending jobs count at their full time limit (the worst case).
"""
import argparse
import json
import math
import re
import subprocess


def gpu_hours(job):
    out = subprocess.run(['sacct', '-j', str(job), '-X', '-n', '-P', '-o',
                          'JobID,State,ElapsedRaw,TimelimitRaw,AllocTRES,NodeList'],
                         capture_output=True, text=True).stdout
    rows, total = [], 0.0
    for line in out.strip().splitlines():
        jid, state, el, tl, tres, nodes = (line.split('|') + [''] * 6)[:6]
        m = re.search(r'gres/gpu(?::[^=,]*)?=(\d+)', tres)
        g = int(m.group(1)) if m else 0
        secs = int(el or 0)
        if state.split()[0] in ('RUNNING', 'PENDING', 'REQUEUED', 'CONFIGURING'):
            secs = max(secs, 60 * int(tl or 0))
            if not g:
                g = 1
        total += g * secs / 3600
        rows.append(dict(job=jid, state=state, elapsed_s=int(el or 0), gpus=g, node=nodes))
    return total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--at441-jobs')
    ap.add_argument('--config')
    ap.add_argument('--job')
    a = ap.parse_args()
    if a.job:
        h, rows = gpu_hours(a.job)
        print(json.dumps(dict(job=a.job, gpu_hours=round(h, 4), rows=rows)))
        return
    x = json.load(open(a.config))['exp19']
    used, rows, note = 0.0, [], 'no AT-441 job file: AT-441 counted as 0'
    try:
        txt = open(a.at441_jobs).read()
        m = re.search(r'pilot=(\d+)', txt)
        if m:
            used, rows = gpu_hours(m.group(1))
            note = f'AT-441 pilot array {m.group(1)}'
    except OSError:
        pass
    left_min = (x['budget_total_gpu_hours'] - used) * 60 - x['budget_margin_minutes']
    cap = max(0, min(x['gpu_cap_minutes'], math.floor(left_min)))
    print(json.dumps(dict(at441_gpu_hours=round(used, 4), at441_note=note, at441_rows=rows,
                          budget_total_gpu_hours=x['budget_total_gpu_hours'], cap_minutes=cap)))


if __name__ == '__main__':
    main()
