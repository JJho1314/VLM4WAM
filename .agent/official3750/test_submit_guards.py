"""Offline checks: unavailable resources must never reach real submission."""
import ast
from pathlib import Path
import shlex
import subprocess

script = Path(__file__).with_name('submit_when_ready.py')
loop = next(n for n in ast.parse(script.read_text()).body if isinstance(n, ast.While))


def reply(code=0, out='', err=''):
    return subprocess.CompletedProcess([], code, out, err)


def check(replies, expected_command):
    calls = []
    class StopLoop(Exception):
        pass
    def run_command(command, **kwargs):
        calls.append(command)
        return replies.pop(0)
    def stop(seconds):
        raise StopLoop
    scope = dict(run_command=run_command, STATE='/state', META='/meta', ROOT='/root',
                 preflight_done=True, subprocess=subprocess, shlex=shlex,
                 time=type('Clock', (), {'sleep': staticmethod(stop)})())
    try:
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(script), 'exec'), scope)
    except StopLoop:
        pass
    assert not replies
    assert any(expected_command in c for c in calls), calls
    assert not any('flock ' in c for c in calls), calls


check([reply(1), reply(1)], 'DATA_READY')
check([reply(1), reply(), reply(1, out='Slurmctld is DOWN')], 'scontrol ping')
check([reply(1), reply(), reply(out='Slurmctld is UP'), reply(1)], 'show partition')
check([reply(1), reply(), reply(out='Slurmctld is UP'),
       reply(out='PartitionName=acd_u State=UP'), reply(1)], 'sacctmgr')
check([reply(1), reply(), reply(out='Slurmctld is UP'),
       reply(out='PartitionName=acd_u State=UP'), reply(out='jhe724\n'),
       reply(1, err='Resource validation failed')], '--test-only')
check([reply(out='123456\n')], 'jobid')
print('Six guarded submission paths passed; no mutation attempted')
