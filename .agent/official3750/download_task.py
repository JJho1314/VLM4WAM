"""Download one pinned official task on HPC3; publish files only after SHA256 matches."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

ROOT=Path('/data/user/jhe724/datasets/RoboFollow-official3750')
WORKERS=24


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download(row,root):
    relative=Path(row['path']);target=root/relative
    if relative.is_absolute() or '..' in relative.parts or not target.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe dataset path')
    url=urlsplit(row['url'])
    if url.scheme!='https' or url.hostname!='us.aws.cdn.hf.co':raise ValueError('unexpected official CDN URL')
    if len(row['sha256'])!=64 or any(c not in '0123456789abcdef' for c in row['sha256']):raise ValueError('invalid SHA256')
    if target.is_file() and digest(target)==row['sha256']:return 0
    target.parent.mkdir(parents=True,exist_ok=True)
    part=target.with_name(target.name+'.direct.part')
    if not part.resolve().is_relative_to(root.resolve()):raise ValueError('unsafe partial path')
    # Restart curl so each retry recalculates the resume offset from the partial file.
    args=['curl','-fLsS','--connect-timeout','15','--max-time','60','--speed-limit','16384','--speed-time','20','-C','-','-o',str(part),row['url']]
    for attempt in range(5):
        result=subprocess.run(args,capture_output=True,text=True)
        if not result.returncode:break
        if result.returncode in (33,36):part.unlink(missing_ok=True)
        if result.returncode not in (5,6,7,18,28,33,35,36,52,55,56):break
        if attempt<4:time.sleep(min(attempt+1,3))
    if result.returncode:raise RuntimeError(f'curl exit {result.returncode}')
    if digest(part)!=row['sha256']:
        part.unlink()
        raise ValueError('downloaded file SHA256 mismatch')
    size=part.stat().st_size
    part.replace(target)
    return size


def main():
    rows=json.load(sys.stdin);started=time.monotonic();total=0
    print(f'CDN START workers={WORKERS} files={len(rows)}',flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(download,row,ROOT):row for row in rows}
        for count,future in enumerate(as_completed(futures),1):
            try:total+=future.result()
            except Exception as error:
                print(f"FAILED {futures[future]['path']}: {type(error).__name__}: {error}",flush=True)
                raise SystemExit(1)
            print(f'CDN verified {count}/{len(rows)}',flush=True)
    seconds=time.monotonic()-started
    print(f'CDN COMPLETE bytes={total} seconds={seconds:.1f} MB/s={total/max(seconds,0.001)/1e6:.2f}',flush=True)

if __name__=='__main__':main()
