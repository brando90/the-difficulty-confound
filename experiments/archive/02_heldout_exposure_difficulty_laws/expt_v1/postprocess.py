"""Bounded, death-aware verification stage of the existing run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results'


def receipt(state,**extra):
    tmp=OUT/'postprocess_status.tmp'
    tmp.write_text(json.dumps(dict(state=state,pid=os.getpid(),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra),indent=2)+'\n')
    tmp.replace(OUT/'postprocess_status.json')


def main():
    supervisor=int(sys.argv[1]);deadline=time.monotonic()+6000
    receipt('waiting_for_existing_supervisor',supervisor_pid=supervisor)
    while True:
        try:
            os.kill(supervisor,0)
        except ProcessLookupError:
            break
        if time.monotonic()>deadline:
            raise TimeoutError('Existing supervisor wait exceeded fixed deadline')
        time.sleep(5)
    status=json.loads((OUT/'job_status.json').read_text())
    assert status['phase']=='analysis' and status['state']=='complete' and status['exit_code']==0,status
    receipt('verifying')
    subprocess.run([sys.executable,str(ROOT/'verify_artifacts.py')],check=True,timeout=300)
    receipt('rendering')
    subprocess.run([sys.executable,str(ROOT/'report.py')],check=True,timeout=180)
    receipt('complete')
    print('POSTPROCESS_COMPLETE',flush=True)

if __name__=='__main__':
    try:
        main()
    except BaseException as error:
        receipt('failed',error=repr(error))
        raise
