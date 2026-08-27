"""Precompute the training set for the Qwen3.5-2B discrete-SigLIP planner (Option 6, real training).

Per LIBERO episode (main camera): the CURRENT frame as raw RGB (fed to Qwen3.5-2B's vision) + the
instruction + the K future keyframes' frozen SigLIP2 tokens (the discrete-plan target). Sharded per
suite. Env-configurable paths so it runs unchanged on the local box or HPC3.

  cur_rgb [N,224,224,3] uint8   -> Qwen sees this
  kf      [N,K,256,1024] f16    -> GT SigLIP2 plan (VQ-quantized to code targets at train time)
  prompt  [N] str
"""
import os, json, av, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModel
from PIL import Image

dev = "cuda"
ROOT = os.environ.get("LIBERO_ROOT", "/data/LFT-W02_data/junjie/data/LIBERO-fastwam")
SIG = os.environ.get("SIGLIP2_DIR", "/data/LFT-W02_data/junjie/weights/siglip2-large-patch16-256")
OUT = os.environ.get("OUT_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "qwen35_train"))
CAM = "observation.images.image"
NPREV, KFI, GRID, RES = 4, [0, 3, 5, 8], 16, 256
PER_SUITE = int(os.environ.get("PER_SUITE", 0))    # 0 = all episodes
SUITES = os.environ.get("SUITES", "object,spatial,goal,10").split(",")
STRIDE = int(os.environ.get("WINDOW_STRIDE", 10))          # slide the current frame by this many frames
MAXW = int(os.environ.get("MAX_WINDOWS_PER_EP", 16))       # cap windows per episode (long episodes)
def log(m): print(m, flush=True)


def read_all_frames(suite, ei):
    path = f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000/{CAM}/episode_{ei:06d}.mp4"
    c = av.open(path); frs = [np.asarray(fr.to_ndarray(format="rgb24")) for fr in c.decode(video=0)]
    c.close(); return frs


@torch.no_grad()
def siglip_tokens(sig, imgs):
    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    out = sig.vision_model(pixel_values=x).last_hidden_state
    if out.shape[1] == GRID * GRID + 1: out = out[:, 1:]
    return out.float().cpu().numpy().astype(np.float16)


def read_frames(suite, ei, idxs):
    path = f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000/{CAM}/episode_{ei:06d}.mp4"
    c = av.open(path); want, frs = set(idxs), {}
    for j, fr in enumerate(c.decode(video=0)):
        if j in want: frs[j] = np.asarray(fr.to_ndarray(format="rgb24"))
        if len(frs) == len(want): break
    c.close(); return frs


def main():
    sig = AutoModel.from_pretrained(SIG, dtype=torch.float32).eval().to(dev)
    for p in sig.parameters(): p.requires_grad_(False)
    log("siglip2 loaded")
    os.makedirs(OUT, exist_ok=True)
    kmax = max(KFI)
    for suite in SUITES:
        eps = [json.loads(l) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]
        rgb_l, kf_l, pr_l = [], [], []
        n_ep = 0
        for e in eps:
            if PER_SUITE and n_ep >= PER_SUITE: break
            ei, task = e["episode_index"], e["tasks"][0]
            try: frames = read_all_frames(suite, ei)
            except Exception: continue
            T = len(frames)
            # slide current frame c; window = current c + future keyframes (c+1)+KFI, last must be in range
            currents = list(range(NPREV - 1, T - 1 - kmax, STRIDE))[:MAXW]
            if not currents: continue
            for c in currents:
                kf_abs = [c + 1 + k for k in KFI]
                cur = np.asarray(Image.fromarray(frames[c]).resize((224, 224))).astype(np.uint8)
                kf = siglip_tokens(sig, [frames[i] for i in kf_abs])   # [K,256,1024]
                rgb_l.append(cur); kf_l.append(kf); pr_l.append(task)
            n_ep += 1
            if n_ep % 100 == 0: log(f"  {suite}: {n_ep} eps -> {len(rgb_l)} windows")
        np.save(f"{OUT}/{suite}_rgb.npy", np.stack(rgb_l))
        np.save(f"{OUT}/{suite}_kf.npy", np.stack(kf_l))
        np.save(f"{OUT}/{suite}_prompt.npy", np.array(pr_l, dtype=object), allow_pickle=True)
        log(f"{suite}: saved {n_ep} eps -> {len(rgb_l)} windows  kf={np.stack(kf_l).shape}")
    log("ALLDONE")


if __name__ == "__main__":
    main()
