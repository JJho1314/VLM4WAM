"""Atomic records, exclusive experiment lock, and Linux process identity."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import time


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def process_identity(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=int(raw[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (FileNotFoundError,ProcessLookupError):return None


def safe_signal(identity,sig=signal.SIGTERM,group=False):
    if not identity or process_identity(identity['pid'])!=identity:return False
    try:
        if group:
            if os.getpgid(identity['pid'])!=identity['pid']:return False
            os.killpg(identity['pid'],sig)
        else:os.kill(identity['pid'],sig)
        return True
    except ProcessLookupError:return False


class Registry:
    def __init__(self,root,max_gpu_hours=8):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.max_gpu_hours=max_gpu_hours
    @contextmanager
    def lock(self):
        with (self.root/'controller.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('another controller owns the experiment lock')
            try:yield
            finally:fcntl.flock(f,fcntl.LOCK_UN)
    def charges(self):
        p=self.root/'budget.json';return json.loads(p.read_text()) if p.exists() else {}
    def used_gpu_hours(self):return sum(self.charges().values())
    def charge(self,identity,seconds,gpus):
        if seconds<0 or gpus not in (0,1,2):raise ValueError('invalid charge')
        charges=self.charges();charges[identity]=max(charges.get(identity,0),seconds*gpus/3600)
        atomic_json(self.root/'budget.json',charges)
    def check_budget(self,requested_gpu_hours):
        if self.used_gpu_hours()+requested_gpu_hours>self.max_gpu_hours:raise RuntimeError('GPU-hour budget exhausted')


def descendant_identities(pid):
    parents={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            raw=(entry/'stat').read_text().rsplit(')',1)[1].split();parents[int(entry.name)]=int(raw[1])
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    owned={pid}
    while True:
        expanded=owned|{p for p,parent in parents.items() if parent in owned}
        if expanded==owned:break
        owned=expanded
    return [identity for p in sorted(owned,reverse=True) if (identity:=process_identity(p)) is not None]


def stop_process_tree(proc,timeout=15):
    identities=descendant_identities(proc.pid)
    for identity in identities:safe_signal(identity,signal.SIGTERM)
    try:proc.wait(timeout=timeout)
    except __import__('subprocess').TimeoutExpired:
        for identity in identities:safe_signal(identity,signal.SIGKILL)
        proc.wait()
    # A launcher may have exited before a child finished; retain the captured identity.
    for identity in identities:
        if identity['pid']!=proc.pid:safe_signal(identity,signal.SIGKILL)

def cleanup_owned_group(pgid,identities,timeout=2):
    """A surviving captured member proves ownership even after launcher exit."""
    def alive_member():
        for identity in identities:
            if not identity or process_identity(identity['pid'])!=identity:continue
            try:
                state=Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')',1)[1].split()[0]
                if state!='Z' and os.getpgid(identity['pid'])==pgid:return True
            except (FileNotFoundError,ProcessLookupError):continue
        return False
    if not alive_member():return
    try:os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError:return
    deadline=time.monotonic()+timeout
    while alive_member() and time.monotonic()<deadline:time.sleep(.05)
    if alive_member():
        try:os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:pass

def owned_group_identities(owner):
    if not owner or owner['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return []
    live=process_identity(owner['pid'])
    if live is not None and live!=owner:return []
    identities=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            raw=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(raw[2])==owner['pid'] and int(raw[3])==owner['pid'] and int(raw[19])>=owner['start_ticks']:
                identity=process_identity(int(entry.name))
                if identity:identities.append(identity)
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    return identities
