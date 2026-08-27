"""Ola data prep for the Qwen3.5-2B discrete-plan trainer: build the VQ codebook and store CODES, not
features.

The training target is a code index over a 2048-entry codebook, so the raw SigLIP2 plan features are
only ever needed to *derive* those codes. Storing codes instead of features shrinks the dataset from
~47 GB to ~45 MB, which is what makes this fit on Ola (84 GB free). Features are quantized on the fly
and never written to disk.

  pass 1: SigLIP2 over a sample of episodes -> k-means codebook (2048 x 1024)
  pass 2: every window -> SigLIP2 -> nearest code -> int16 codes

Outputs (per suite): {suite}_rgb.npy (uint8, fed to Qwen), {suite}_codes.npy (int16), {suite}_prompt.npy
plus a shared codebook.npy. Multi-window sampling matches the HPC3 set (stride 10, <=16 windows/ep).
"""
import os, json, av, numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModel
from PIL import Image

dev = "cuda"
ROOT = os.environ.get("LIBERO_ROOT", os.path.expanduser("~/workspace/VLM4WAM/data/LIBERO-fastwam"))
SIG = os.environ.get("SIGLIP2_DIR", os.path.expanduser("~/workspace/VLM4WAM/weights/siglip2-large-patch16-256"))
OUT = os.environ.get("OUT_DIR", os.path.expanduser("~/workspace/VLM4WAM/data/qwen35_codes"))
CAM = "observation.images.image"
NPREV, KFI, GRID, RES = 4, [0, 3, 5, 8], 16, 256
STRIDE = int(os.environ.get("WINDOW_STRIDE", 10))
MAXW = int(os.environ.get("MAX_WINDOWS_PER_EP", 16))
SUITES = os.environ.get("SUITES", "object,spatial,goal,10").split(",")
NUM_CODES = int(os.environ.get("NUM_CODES", 2048))
KM_SAMPLE = int(os.environ.get("KMEANS_SAMPLE", 400000))
KM_EP = int(os.environ.get("KMEANS_EPISODES", 60))     # episodes per suite used to fit the codebook
D = 1024
def log(m): print(m, flush=True)


@torch.no_grad()
def siglip_tokens(sig, imgs):
    x = torch.from_numpy(np.stack(imgs)).to(dev).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5
    o = sig.vision_model(pixel_values=x).last_hidden_state
    if o.shape[1] == GRID * GRID + 1: o = o[:, 1:]
    return o.float()                                    # (N,256,1024) on GPU


def read_all(suite, ei):
    p = f"{ROOT}/libero_{suite}_no_noops_lerobot/videos/chunk-000/{CAM}/episode_{ei:06d}.mp4"
    c = av.open(p); fr = [np.asarray(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]; c.close()
    return fr


def windows_of(T):
    kmax = max(KFI)
    return list(range(NPREV - 1, T - 1 - kmax, STRIDE))[:MAXW]


def episodes(suite):
    return [json.loads(l) for l in open(f"{ROOT}/libero_{suite}_no_noops_lerobot/meta/episodes.jsonl")]


@torch.no_grad()
def kmeans(x, k, iters=25):
    """x (M,D) L2-normalized on GPU -> codebook (k,D). Fixed seed for reproducibility."""
    g = torch.Generator(device=x.device).manual_seed(0)
    c = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()
    for it in range(iters):
        a = torch.cat([(x[i:i + 65536] @ c.t()).argmax(1) for i in range(0, x.shape[0], 65536)])
        oh = F.one_hot(a, k).float()
        cnt = oh.sum(0).clamp(min=1)
        c = F.normalize((oh.t() @ x) / cnt[:, None], dim=-1)
        if it % 8 == 0: log(f"  kmeans iter {it}: used {int((cnt > 1).sum())}/{k} codes")
    return c


def main():
    os.makedirs(OUT, exist_ok=True)
    sig = AutoModel.from_pretrained(SIG, dtype=torch.float32).eval().to(dev)
    for p in sig.parameters(): p.requires_grad_(False)
    log("siglip2 loaded")

    # ---- pass 1: codebook ----
    cb_path = os.path.join(OUT, "codebook.npy")
    if os.path.exists(cb_path):
        codebook = torch.from_numpy(np.load(cb_path)).to(dev); log("codebook: reusing existing")
    else:
        buf, need = [], KM_SAMPLE
        for suite in SUITES:
            for e in episodes(suite)[:KM_EP]:
                if need <= 0: break
                try: fr = read_all(suite, e["episode_index"])
                except Exception: continue
                w = windows_of(len(fr))
                if not w: continue
                c = w[len(w) // 2]
                f = siglip_tokens(sig, [fr[c + 1 + k] for k in KFI]).reshape(-1, D)
                buf.append(F.normalize(f, dim=-1).cpu()); need -= f.shape[0]
            log(f"  codebook sample: {suite} done, remaining {max(0,need)}")
            if need <= 0: break
        x = torch.cat(buf)[:KM_SAMPLE].to(dev)
        log(f"kmeans on {tuple(x.shape)}")
        codebook = kmeans(x, NUM_CODES)
        np.save(cb_path, codebook.cpu().numpy()); del x, buf
        torch.cuda.empty_cache(); log(f"saved {cb_path}")

    # ---- pass 2: quantize every window ----
    for suite in SUITES:
        rgb_l, code_l, pr_l, n_ep = [], [], [], 0
        for e in episodes(suite):
            ei, task = e["episode_index"], e["tasks"][0]
            try: fr = read_all(suite, ei)
            except Exception: continue
            for c in windows_of(len(fr)):
                f = siglip_tokens(sig, [fr[c + 1 + k] for k in KFI])          # (K,256,1024)
                codes = (F.normalize(f.reshape(-1, D), dim=-1) @ codebook.t()).argmax(-1)
                code_l.append(codes.reshape(len(KFI), GRID * GRID).cpu().numpy().astype(np.int16))
                rgb_l.append(np.asarray(Image.fromarray(fr[c]).resize((224, 224))).astype(np.uint8))
                pr_l.append(task)
            n_ep += 1
            if n_ep % 100 == 0: log(f"  {suite}: {n_ep} eps -> {len(rgb_l)} windows")
        np.save(f"{OUT}/{suite}_rgb.npy", np.stack(rgb_l))
        np.save(f"{OUT}/{suite}_codes.npy", np.stack(code_l))
        np.save(f"{OUT}/{suite}_prompt.npy", np.array(pr_l, dtype=object), allow_pickle=True)
        log(f"{suite}: {n_ep} eps -> {len(rgb_l)} windows, codes={np.stack(code_l).shape}")
    log("PRECOMPUTE-CODES-DONE")


if __name__ == "__main__":
    main()
