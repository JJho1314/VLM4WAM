"""Resolve pinned official URLs on Mac; download HDF5 directly on HPC3."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from urllib.parse import quote, urlsplit

HERE = Path(__file__).resolve().parent
OLA = ['ssh', '-S', '/tmp/agibot-ola2208-resolver.sock', '-o', 'ControlMaster=no',
       '-o', 'ConnectTimeout=30', 'Ola_2208']
HPC = ['ssh', '-S', '/tmp/robofollow-hpc3-jhe724.sock',
       '-o', 'ControlMaster=auto', '-o', 'ControlPersist=86400',
       '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=30',
       '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4',
       '-o', 'ProxyCommand=nc -X 5 -x 127.0.0.1:1080 %h %p', 'HPC3_jhe724']
SOURCE = '/data/users/junjie/workspace/hpc3_jhe724/outputs/robofollow_q35/data_official3750'
TARGET = '/data/user/jhe724/datasets/RoboFollow-official3750'
META = '/data/user/jhe724/outputs/robofollow_q35/data_official3750'


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


REVISION='bbb1e266ed585f1557773de5a3f1e3b4cd944f97'
WORKER='/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750/.agent/official3750/download_task.py'
PIXI='/data/user/jhe724/.pixi/bin/pixi'
REPO='/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750'


def resolve_cdn(row):
    relative,digest=row
    url='https://huggingface.co/datasets/AutoLab-SJTU/robofollow-data/resolve/'+REVISION+'/'+quote(str(relative))+'?download=true&fresh='+str(time.time_ns())
    result=run(['curl','-fsSI','--retry','3','--connect-timeout','10','--max-time','30',url],capture_output=True,text=True)
    location=next(line.split(':',1)[1].strip() for line in result.stdout.splitlines() if line.lower().startswith('location:'))
    if urlsplit(location).scheme!='https' or urlsplit(location).hostname!='us.aws.cdn.hf.co':raise ValueError('unexpected official CDN redirect')
    return dict(path=str(relative),sha256=digest,url=location)


while True:
    ready = subprocess.run(OLA + ['test -f ' + SOURCE + '/coverage.json'])
    if ready.returncode == 0:
        break
    print('Waiting for complete source statistics and coverage report', flush=True)
    time.sleep(30)

payloads = {name: json.loads(run(OLA + ['cat ' + SOURCE + '/' + name],
                               capture_output=True, text=True).stdout)
            for name in ('manifest.json', 'stats.json', 'coverage.json')}
manifest = payloads['manifest.json']
assert len(manifest['episodes']) == 3750
assert {r['split'] for r in manifest['episodes']} == {'train'}
groups = {}
source_root = Path(manifest['root'])
for row in manifest['episodes']:
    relative = Path(row['path']).relative_to(source_root)
    groups.setdefault(str(relative.parent.parent), []).append((relative, row['file_hash']))
assert len(groups) == 75 and all(len(rows) == 50 for rows in groups.values())
run(HPC + [shlex.join(['mkdir', '-p', TARGET + '/.transfer-checksums', META])])
for index, (task, rows) in enumerate(sorted(groups.items()), 1):
    label = task.replace('/', '_')
    checks = ''.join(f'{digest}  {relative}\n' for relative, digest in rows)
    instructions = [Path(task) / 'instructions' / (relative.stem + '.json')
                    for relative, _ in rows]
    checks += run(OLA + ['cd ' + shlex.quote(str(source_root)) + ' && ' +
                        shlex.join(['sha256sum'] + [str(p) for p in instructions])],
                  capture_output=True, text=True).stdout
    checksum = TARGET + '/.transfer-checksums/' + label + '.sha256'
    run(HPC + ['cat > ' + shlex.quote(checksum)], input=checks.encode())
    verify = 'cd ' + shlex.quote(TARGET) + ' && sha256sum --status -c ' + shlex.quote(checksum)
    if subprocess.run(HPC + [verify], stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL).returncode:
        for attempt in range(1, 4):
            print(f'Transfer {index}/75 {task}, attempt {attempt}', flush=True)
            # Only tiny instruction JSON files traverse the Mac/VPN.
            producer = subprocess.Popen(OLA + [shlex.join(['tar','-C',str(source_root),'-cf','-']+[str(p) for p in instructions])],stdout=subprocess.PIPE)
            consumer = subprocess.Popen(HPC + [shlex.join(['tar','-xf','-','-C',TARGET])],stdin=producer.stdout)
            producer.stdout.close()
            if consumer.wait() or producer.wait():raise RuntimeError('instruction transfer failed')
            with ThreadPoolExecutor(max_workers=8) as pool:
                links=list(pool.map(resolve_cdn,rows))
            command='cd '+shlex.quote(REPO)+' && '+shlex.join([PIXI,'run','--manifest-path','autoresearch/runtime/pixi.toml','research',WORKER])
            consumer=subprocess.run(HPC+[command],input=json.dumps(links).encode())
            received,sent=consumer.returncode,0
            if received==0:received=subprocess.run(HPC+[verify]).returncode
            if received == sent == 0:
                break
            if attempt == 3:
                raise RuntimeError(f'Transfer/checksum failed: {task}')
            time.sleep(15)
    print(f'Verified {index}/75 {task}: 50 episodes', flush=True)

for row in manifest['episodes']:
    row['path'] = str(Path(TARGET) / Path(row['path']).relative_to(source_root))
manifest['root'] = TARGET
manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode()
target_hash = hashlib.sha256(manifest_bytes).hexdigest()
payloads['stats.json']['manifest_hash'] = target_hash
payloads['coverage.json'].update(manifest_sha256=target_hash, transfer_status='complete',
                                transfer_method='HPC3 direct official CDN; Mac resolves links and transfers instruction JSON',
                                official_revision=REVISION,
                                transferred_episodes=3750,
                                file_hash_verification='3750 HDF5 files and 3750 instruction JSON files')
payloads['coverage.json']['stats_sha256'] = hashlib.sha256(
    (json.dumps(payloads['stats.json'], indent=2, allow_nan=False) + '\n').encode()).hexdigest()
for name, payload in payloads.items():
    data = (json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()
    destination = META + '/' + name
    run(HPC + ['cat > ' + shlex.quote(destination + '.tmp') + ' && mv ' +
               shlex.quote(destination + '.tmp') + ' ' + shlex.quote(destination)], input=data)
run(HPC + ['touch ' + META + '/DATA_READY'])
print('COMPLETE: 3750 verified official episodes and matching statistics; direct CDN on HPC3', flush=True)
subprocess.run(['launchctl', 'remove', 'com.junjie.robofollow.official3750.transfer'])
