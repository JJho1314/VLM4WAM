"""One figure per keyframe showing BOTH probes on the SAME planner forward pass.

The repo's depth visualiser and this folder's localisation probe each sampled their own episodes, so
their outputs could not be placed side by side. Everything here comes from one dump
(planner_feats_dualcam_k4_big.npz), which stores the semantic plan AND the depth plan produced by the
same forward pass, so every panel in a row describes the same RGB frame and the same instruction.

Columns: real keyframe | target localisation from the predicted plan | predicted depth |
         teacher SigLIP2 localisation (reference) | teacher DA3 depth is not needed here, the WSA
         probe decodes both target and prediction from stored features.

Depth is decoded with the same WSA 4-layer probe the repo script uses; no DA3 model is required
because the dump already holds the features.
"""
import os, json
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import sys

dev = "cuda"
ROOT = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM"
HERE = f"{ROOT}/semantic_localization"
sys.path.insert(0, f"{ROOT}/qwen3_vl_semantic_planner/dinov3_da3_2b")
NPZ = os.environ.get("NPZ", f"{HERE}/data/planner_feats_dualcam_k4_big.npz")
PROBE = os.environ.get("WSA_PROBE", "/data/LFT-W02_data/junjie/probes_2b/da3_depth_wsa_probe.pt")
OUT = f"{HERE}/figs/combined_probe"; os.makedirs(OUT, exist_ok=True)
HEADS = f"{HERE}/figs/dualcam_probe"
SIG = f"{ROOT}/third_party/siglip2-large-patch16-256"
GRID, DIM, RES = 16, 1024, 256
CAM = int(os.environ.get("CAM_IDX", 0))          # 0 = main
MAXN = int(os.environ.get("MAXN", 10))
from transformers import AutoModel
from transformers.models.clipseg.modeling_clipseg import CLIPSegForImageSegmentation
from transformers.models.clipseg.processing_clipseg import CLIPSegProcessor


class LocHead(nn.Module):
    def __init__(self, d=DIM, hid=256, tdim=512):
        super().__init__()
        self.film = nn.Linear(tdim, 2 * hid); self.inp = nn.Conv2d(d, hid, 1)
        self.net = nn.Sequential(nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU(),
                                 nn.Conv2d(hid, hid, 3, padding=1), nn.GroupNorm(8, hid), nn.GELU())
        self.out = nn.Conv2d(hid, 1, 1)

    def forward(self, f, t):
        x = self.inp(f.permute(0, 2, 1).reshape(f.shape[0], -1, GRID, GRID))
        g, b = self.film(t).chunk(2, -1)
        x = x * (1 + g[..., None, None]) + b[..., None, None]
        return self.out(self.net(x)).squeeze(1)


def main():
    from wsa_depth_probe import WSAMultiLayerDPTProbe
    pay = torch.load(PROBE, map_location="cpu", weights_only=False)
    depth_probe = WSAMultiLayerDPTProbe.from_config(pay["config"])
    depth_probe.load_state_dict(pay["state_dict"], strict=True)
    depth_probe.to(dev).eval().requires_grad_(False)
    print(f"WSA depth probe loaded ({sum(p.numel() for p in depth_probe.parameters())/1e6:.1f}M params)", flush=True)

    sig = AutoModel.from_pretrained(SIG).eval().to(dev)
    cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
    for m in (sig, cseg):
        for p in m.parameters(): p.requires_grad_(False)

    def load_head(name):
        h = LocHead().to(dev)
        h.load_state_dict(torch.load(f"{HEADS}/{name}", map_location=dev)["state"]); h.eval(); return h
    head_p, head_t = load_head("head_predicted.pt"), load_head("head_teacher.pt")

    @torch.no_grad()
    def temb(w):
        i = cproc(text=[f"a photo of a {w}"], return_tensors="pt", padding="max_length", max_length=77)
        i = {k: v.to(dev) for k, v in i.items() if k in ("input_ids", "attention_mask")}
        o = cseg.clip.text_model(**i)
        t = o.pooler_output if getattr(o, "pooler_output", None) is not None else o[1]
        return F.normalize(t.float(), dim=-1)

    @torch.no_grad()
    def teacher(img):
        x = torch.from_numpy(np.ascontiguousarray(img)).to(dev).permute(2, 0, 1)[None].float() / 255.
        x = (F.interpolate(x, (RES, RES), mode="bilinear", align_corners=False) - 0.5) / 0.5
        o = sig.vision_model(pixel_values=x).last_hidden_state
        return o[:, 1:] if o.shape[1] == GRID * GRID + 1 else o

    z = np.load(NPZ, allow_pickle=True)
    FUT, FP, DP = z["fut"], z["fp"], z["dp"]
    PROMPTS = [str(x) for x in z["prompts"]]; NOUNS = [str(x).split("|")[0] for x in z["nouns"]]
    SUITES = [str(x) for x in z["suites"]]
    meta = json.loads(str(z["meta"])); K = int(meta["num_keyframes"]); OFFS = meta["offsets"]
    split = f"{HEADS}/split.json"
    HELD = set(json.load(open(split))["test_samples"]) if os.path.exists(split) else None
    order = [i for i in range(len(FUT)) if HELD is None or i in HELD][:MAXN]
    print(f"combining {len(order)} held-out samples, camera={CAM}, depth plan {DP.shape}", flush=True)

    TURBO, SPEC = cm.get_cmap("turbo"), cm.get_cmap("Spectral")
    def ov(img, hm, lo=55.0, hi=99.5, gamma=1.9):
        h = F.interpolate(torch.from_numpy(hm)[None, None].float(), img.shape[:2],
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        plo, phi = np.percentile(h, [lo, hi])
        h = np.clip((h - plo) / (phi - plo + 1e-6), 0, 1) ** gamma
        return img.astype(float) / 255. * 0.5 + TURBO(h)[..., :3] * 0.5

    for i in order:
        noun = NOUNS[i]; t = temb(noun)
        rows = []
        for k in range(K):
            frame = FUT[i][k][CAM]
            sl = slice(k * GRID * GRID, (k + 1) * GRID * GRID)
            pf = torch.from_numpy(FP[i][CAM].astype(np.float32))[sl][None].to(dev)
            # depth plan is [V, N_tokens, L_layers, D]; the probe wants [B, L, N, D]
            dfeat = torch.from_numpy(DP[i][CAM].astype(np.float32))[sl][None].to(dev).transpose(1, 2).contiguous()
            with torch.no_grad():
                loc_p = head_p(pf, t)[0].cpu().numpy()
                loc_t = head_t(teacher(frame).float(), t)[0].cpu().numpy()
                dep = depth_probe(dfeat)
                dep = dep[0] if isinstance(dep, (tuple, list)) else dep
                dep = dep.squeeze().float().cpu().numpy()
            dep = (dep - dep.min()) / (dep.max() - dep.min() + 1e-6)
            rows.append((frame, loc_t, loc_p, dep))

        fig, ax = plt.subplots(4, K, figsize=(3.1 * K, 12.4))
        for k, (frame, loc_t, loc_p, dep) in enumerate(rows):
            ax[0, k].imshow(frame); ax[0, k].set_title(f"keyframe {k}  (t+{OFFS[k]})", fontsize=9)
            ax[1, k].imshow(ov(frame, loc_t))
            ax[2, k].imshow(ov(frame, loc_p))
            d = F.interpolate(torch.from_numpy(dep)[None, None].float(), frame.shape[:2], mode="bilinear")[0, 0].numpy()
            ax[3, k].imshow(SPEC(d)[..., :3])
            for r in range(4): ax[r, k].axis("off")
        for r, lab in enumerate(("real keyframe",
                                 "teacher SigLIP2 -> localisation",
                                 "PREDICTED plan -> localisation",
                                 "PREDICTED plan -> depth (WSA probe)")):
            ax[r, 0].axis("on"); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([])
            ax[r, 0].set_ylabel(lab, fontsize=9, fontweight="bold")
        fig.suptitle(f"[{SUITES[i]}] {PROMPTS[i][:84]}   |  target='{noun}'  "
                     f"(same RGB + instruction for every row)", fontsize=11)
        fig.tight_layout()
        out = f"{OUT}/{i:02d}_{SUITES[i]}_{noun}.png"
        fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
        print(f"[sample {i}] {SUITES[i]} '{noun}' saved", flush=True)
    print(f"SAVED -> {OUT}", flush=True); print("COMBINED-DONE", flush=True)


if __name__ == "__main__":
    main()
