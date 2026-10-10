"""Wait for verified data and Slurm, then submit one 8-GPU baseline as jhe724."""
import getpass
import shlex
import subprocess
import time

ROOT = '/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750'
META = '/data/user/jhe724/outputs/robofollow_q35/data_official3750'
STATE = ROOT + '/.agent/official3750'


def run_command(command, timeout=90):
    return subprocess.run(['bash', '-c', command], capture_output=True, text=True, timeout=timeout)


assert getpass.getuser() == 'jhe724', 'Run this watcher on HPC3 as jhe724'
assert ROOT.startswith('/data/user/jhe724/') and META.startswith('/data/user/jhe724/')
preflight_done = False
while True:
    try:
        existing = run_command('test -s ' + STATE + '/jobid && cat ' + STATE + '/jobid')
        if existing.returncode == 0:
            print('Training already submitted: ' + existing.stdout.strip(), flush=True)
            break
        ready = run_command('test -f ' + META + '/DATA_READY && test -f ' + STATE + '/CODE_READY')
        if ready.returncode:
            print('Waiting for complete, verified data and code on HPC3', flush=True)
            time.sleep(60)
            continue
        if not preflight_done:
            command = ('cd ' + ROOT + ' && sha256sum --status -c .agent/official3750/code.sha256 && '
                       '/data/user/jhe724/.pixi/bin/pixi run --manifest-path autoresearch/runtime/pixi.toml '
                       'research .agent/official3750/check_data.py')
            checked = run_command(command, timeout=900)
            print(checked.stdout + checked.stderr, flush=True)
            if checked.returncode:
                raise RuntimeError('HPC3 data/config preflight failed; no training submitted')
            preflight_done = True
        ping = run_command('timeout 20s scontrol ping', timeout=40)
        if ping.returncode or ' is UP' not in ping.stdout:
            print('Waiting for HPC3 Slurm recovery: ' + (ping.stdout + ping.stderr).strip(), flush=True)
            time.sleep(300)
            continue
        partition = run_command('timeout 20s scontrol show partition acd_u -o', timeout=40)
        if partition.returncode or 'PartitionName=acd_u ' not in partition.stdout or 'State=UP' not in partition.stdout:
            print('Waiting for HPC3 GPU partition acd_u to return', flush=True)
            time.sleep(300)
            continue
        accounts = run_command('timeout 20s sacctmgr -nP show user jhe724 format=DefaultAccount', timeout=40)
        names = {line.strip().strip('|') for line in accounts.stdout.splitlines() if line.strip().strip('|')}
        if accounts.returncode or len(names) != 1:
            print('Waiting for an unambiguous jhe724 default Slurm account', flush=True)
            time.sleep(300)
            continue
        account = names.pop()
        command = shlex.join(['sbatch', '--account=' + account, '--test-only', STATE + '/train.sbatch'])
        tested = run_command('timeout 30s ' + command, timeout=50)
        if tested.returncode:
            print('Slurm submission validation failed: ' + tested.stdout + tested.stderr, flush=True)
            time.sleep(300)
            continue
        # A durable intent prevents duplicate jobs if the process exits after Slurm accepts the job.
        body = ('set -euo pipefail\ncd ' + STATE + '\n'
                'if test -s jobid; then cat jobid; exit 0; fi\n'
                'if test -e SUBMITTING; then echo "Ambiguous prior submission; inspect Slurm before retrying" >&2; exit 88; fi\n'
                'touch SUBMITTING\n' +
                shlex.join(['sbatch', '--parsable', '--account=' + account, STATE + '/train.sbatch']) +
                ' > jobid.tmp\nmv jobid.tmp jobid\nrm SUBMITTING\ncat jobid\n')
        submitted = run_command('flock ' + STATE + '/submit.lock bash -c ' + shlex.quote(body), timeout=90)
        if submitted.returncode:
            raise RuntimeError('Submission outcome needs inspection; durable SUBMITTING guard prevents duplicates: ' + submitted.stderr)
        jobid = submitted.stdout.strip().split(';')[0]
        assert jobid.isdigit(), submitted.stdout
        print('SUBMITTED: HPC3 jhe724, job ' + jobid + ', 1 node / 8 GPUs, action chunk 32, 53000 steps', flush=True)
        break
    except (subprocess.TimeoutExpired, OSError) as error:
        print('Transient connection failure: ' + str(error), flush=True)
        time.sleep(60)
