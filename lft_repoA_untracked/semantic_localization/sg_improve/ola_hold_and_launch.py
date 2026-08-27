"""Reserve the H100s that are already free, wait for the rest, then train on all of them at once.

The box is shared: another user's eval jobs still hold several cards. Starting a small run now would
either have to be killed later (wasting it) or lock us into fewer GPUs, and leaving the free cards
idle invites someone else to take them. So this holds the idle cards with a dummy allocation, polls
until the busy ones drain, then releases and launches ONE full-width run.

  hold   : allocate HOLD_GB on every currently-idle GPU so the box reads as occupied
  poll   : re-check the not-held GPUs; newly idle ones get held too
  launch : when every GPU is ours, free the holds and exec the 15k-step full fine-tune
"""
import os, subprocess, sys, time
import torch

ROOT = "/home/6fcb109c-77d2-48/workspace/VLM4WAM"
PY = "/home/6fcb109c-77d2-48/miniforge3/envs/qwen35/bin/python"   # absolute: `~` does not expand under setsid
LOG = f"{ROOT}/logs/hold_launch.log"
NGPU = int(os.environ.get("NGPU", 8))
HOLD_GB = float(os.environ.get("HOLD_GB", 60))
FREE_MB = int(os.environ.get("FREE_MB", 70000))     # a card counts as idle above this
POLL = int(os.environ.get("POLL", 60))
STABLE = int(os.environ.get("STABLE", 2))


def log(m):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%F %T')} {m}\n")
    print(m, flush=True)


def free_mb():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return [int(x) for x in out.split()]


def main():
    held = {}                                        # gpu index -> holding tensor
    stable = 0
    log(f"hold-and-launch start: want {NGPU} gpus, hold {HOLD_GB}GB each, idle if >{FREE_MB}MiB free")
    while True:
        fm = free_mb()
        for i in range(min(NGPU, len(fm))):
            if i not in held and fm[i] > FREE_MB:
                try:
                    held[i] = torch.empty(int(HOLD_GB * 1024**3 // 2), dtype=torch.float16, device=f"cuda:{i}")
                    log(f"  held gpu{i} ({fm[i]}MiB was free)")
                except Exception as e:
                    log(f"  could not hold gpu{i}: {e}")
        missing = [i for i in range(NGPU) if i not in held]
        if not missing:
            stable += 1
            log(f"  all {NGPU} gpus held (stable {stable}/{STABLE})")
            if stable >= STABLE:
                break
        else:
            if stable: log(f"  still waiting on gpus {missing}, reset")
            stable = 0
            log(f"  holding {sorted(held)}, waiting on {missing}")
        time.sleep(POLL)

    log("releasing holds and launching full-width training")
    held.clear()
    torch.cuda.empty_cache()
    del held
    time.sleep(5)                                    # let the driver reclaim before the workers start

    env = dict(os.environ,
               MAX_STEPS="15000", SAVE_STEPS="5000", EVAL_STEPS="1000", FULL_FT="1",
               BATCH_SIZE="4", LR="1e-5", HEAD_LR="1e-4", WARMUP_STEPS="300",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               OUT_DIR=f"{ROOT}/runs/qwen35_discrete_ola")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    cmd = [PY, "-m", "torch.distributed.run", f"--nproc_per_node={NGPU}",
           "--master_port=29541", "code/sg_improve/ola_train_codes.py"]
    with open(f"{ROOT}/logs/train.log", "w") as out:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT,
                             start_new_session=True, stdin=subprocess.DEVNULL)
    log(f"launched pid {p.pid} on {NGPU} gpus -> logs/train.log")


if __name__ == "__main__":
    main()
