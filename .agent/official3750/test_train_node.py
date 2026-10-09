"""Check Slurm ranks/endpoints without launching Torch or accessing GPUs."""
import os
from pathlib import Path
import subprocess

launcher = Path(__file__).with_name('train_node.sh')
for rank in ('0', '1'):
    env = dict(os.environ, SLURM_NNODES='2', SLURM_NODEID=rank,
               MASTER_ADDR='ACD-Test-1', MASTER_PORT='23456')
    result = subprocess.run(['bash', '-c',
        'exec() { printf "%s\\n" "$@"; }; source "$1"', '_', str(launcher)],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert '--nnodes=2' in args and '--nproc_per_node=8' in args, args
    assert '--node_rank=' + rank in args, args
    assert '--master_addr=ACD-Test-1' in args and '--master_port=23456' in args, args
    assert '--standalone' not in args
for nodes, rank in [('1', '0'), ('3', '0'), ('2', '2')]:
    env = dict(os.environ, SLURM_NNODES=nodes, SLURM_NODEID=rank,
               MASTER_ADDR='ACD-Test-1', MASTER_PORT='23456')
    result = subprocess.run(['bash', '-c',
        'exec() { printf "UNEXPECTED_LAUNCH\\n"; }; source "$1"', '_', str(launcher)],
        env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'UNEXPECTED_LAUNCH' not in result.stdout
print('Two node ranks and three invalid-allocation guards passed')
