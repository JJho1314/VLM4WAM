"""Independent deadline/owner-loss enforcement for one registered process group."""
import argparse
import json
from pathlib import Path
import time
from autoresearch.registry import process_identity,owned_group_identities,cleanup_owned_group,Registry,atomic_json

def enforce(record_path,owner,root,maximum):
    path=Path(record_path)
    while True:
        record=json.loads(path.read_text())
        if record['status']!='running':return
        elapsed=max(0.,time.monotonic()-record['started_monotonic'])
        if process_identity(owner['pid'])!=owner or elapsed>=record['timeout_seconds']:
            target=record['process']
            cleanup_owned_group(target['pid'],record.get('owned_processes',[])+owned_group_identities(target))
            registry=Registry(root,maximum)
            while True:
                try:
                    with registry.lock():
                        latest=json.loads(path.read_text())
                        if latest['status']=='running':
                            registry.charge(str(path.parent),elapsed,record['gpus'])
                            latest.update(status='interrupted',ended_at=time.time(),wall_seconds=elapsed,gpu_hours=elapsed*record['gpus']/3600,watchdog_reason='owner lost or phase deadline')
                            atomic_json(path,latest)
                    return
                except RuntimeError:time.sleep(.2)
        time.sleep(.2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',type=Path,required=True);p.add_argument('--owner',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--maximum',type=float,required=True);a=p.parse_args()
    enforce(a.record,json.loads(a.owner),a.root,a.maximum)
if __name__=='__main__':main()
