"""Write every panel as its own image, all from ONE planner forward pass per sample.

Nothing is stitched: each (sample, keyframe) gets a folder holding the RGB frame and the four maps,
so any comparison figure can be assembled downstream without cropping.

Layout: one folder per task, every keyframe's images flat inside it as k<i>_off<t>_<what>.png

  frame.png       the real keyframe
  loc_target.png  teacher SigLIP2 features -> localisation head   (semantic target)
  loc_pred.png    the planner's predicted plan -> same head       (semantic prediction)
  depth_target.png      DA3 teacher features -> WSA probe        (what the plan is trained on)
  depth_da3_full_gt.png DA3's own depth head on the real frame   (reference ground truth)
  depth_pred.png  the planner's predicted depth plan -> same probe (depth prediction)

Semantic maps are decoded here; depth maps come pre-decoded from dump_depth_maps.py, which runs on the
box that holds DA3. Main camera by default.
"""
import os, json, sys
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from matplotlib import cm, colormaps
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_noun import target_noun
dev = "cuda"
ROOT = "/data/LFT-W02_data/junjie/VLA_WM/VLM4WAM"
HERE = f"{ROOT}/semantic_localization"
NPZ = os.environ.get("NPZ", f"{HERE}/data/planner_feats_dualcam_k4_big.npz")
DEPTH = os.environ.get("DEPTH_NPZ", f"{HERE}/data/depth_maps_k4.npz")
HEADS = f"{HERE}/figs/dualcam_probe"
OUT = os.environ.get("PANEL_DIR", f"{HERE}/figs/combined_panels")
SIG = f"{ROOT}/third_party/siglip2-large-patch16-256"
GRID, DIM, RES = 16, 1024, 256
CAM = int(os.environ.get("CAM_IDX", 0))
MAXN = int(os.environ.get("MAXN", 10))
UPSCALE = int(os.environ.get("UPSCALE", 2))
HELD_ONLY = int(os.environ.get("HELD_ONLY", 1))
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
    sig = AutoModel.from_pretrained(SIG).eval().to(dev)
    cproc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    cseg = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").eval().to(dev)
    for m in (sig, cseg):
        for p in m.parameters(): p.requires_grad_(False)

    def load_head(n):
        h = LocHead().to(dev); h.load_state_dict(torch.load(f"{HEADS}/{n}", map_location=dev)["state"])
        h.eval(); return h
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
    FUT, FP = z["fut"], z["fp"]
    PROMPTS = [str(x) for x in z["prompts"]]; # recompute from the prompt: the stored nouns used the old first-hit rule, which
    # returned the destination or a modifier ("basket" for "orange juice", "yellow" for mug)
    NOUNS = [target_noun(str(p), str(n).split("|")[0]) for p, n in zip(z["prompts"], z["nouns"])]
    SUITES = [str(x) for x in z["suites"]]
    meta = json.loads(str(z["meta"])); K = int(meta["num_keyframes"]); OFFS = meta["offsets"]

    D = np.load(DEPTH) if os.path.exists(DEPTH) else None
    DPR = D["depth_pred"] if D is not None else None
    DGT = D["depth_gt"] if (D is not None and "depth_gt" in D.files) else None
    DFULL = D["depth_da3_full"] if (D is not None and "depth_da3_full" in D.files) else None
    print(f"depth: pred={'yes' if DPR is not None else 'no'} "
          f"target(WSA)={'yes' if DGT is not None else 'no'} "
          f"da3_full={'yes' if DFULL is not None else 'no'}", flush=True)

    split = f"{HEADS}/split.json"
    HELD = set(json.load(open(split))["test_samples"]) if (HELD_ONLY and os.path.exists(split)) else None
    order = [i for i in range(len(FUT)) if HELD is None or i in HELD][:MAXN]

    TURBO = cm.get_cmap("turbo")        # localisation overlays
    DEPTH_CMAP = cm.get_cmap("turbo")   # depth, matching the repo visualiser (turbo on disparity)
    def save(arr, path):
        im = Image.fromarray(arr if arr.dtype == np.uint8 else (np.clip(arr, 0, 1) * 255).astype(np.uint8))
        if UPSCALE > 1: im = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.LANCZOS)
        im.save(path)

    def ov(img, hm, lo=55.0, hi=99.5, gamma=1.9):
        h = F.interpolate(torch.from_numpy(hm.astype(np.float32))[None, None], img.shape[:2],
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        plo, phi = np.percentile(h, [lo, hi])
        h = np.clip((h - plo) / (phi - plo + 1e-6), 0, 1) ** gamma
        return img.astype(float) / 255. * 0.5 + TURBO(h)[..., :3] * 0.5

    DEPTH_CMAP = os.environ.get("DEPTH_CMAP", "magma")
    DEPTH_ENH = os.environ.get("DEPTH_ENH", "histeq")     # histeq | pct | none
    DEPTH_LIGHTEN = float(os.environ.get("DEPTH_LIGHTEN", 0.0))

    def _enhance(x):
        """The table's smooth gradient eats most of the colour range, so objects sitting on it end up
        within a few shades of each other. Equalising the histogram gives every depth band the same
        share of the range, which is what makes edges and object bodies readable."""
        if DEPTH_ENH == "histeq":
            h, e = np.histogram(x.ravel(), bins=512, range=(0, 1))
            c = np.cumsum(h) / max(h.sum(), 1)
            return np.interp(x.ravel(), (e[:-1] + e[1:]) / 2, c).reshape(x.shape)
        if DEPTH_ENH == "pct":
            a, b = np.percentile(x, [2, 98])
            return np.clip((x - a) / (b - a + 1e-6), 0, 1)
        return x

    def depth_img(dm, hw, invert):
        d = F.interpolate(torch.from_numpy(dm.astype(np.float32))[None, None], hw,
                          mode="bilinear", align_corners=False)[0, 0].numpy()
        d = (d - d.min()) / (d.max() - d.min() + 1e-6)
        if invert: d = 1.0 - d                       # probe emits depth; figures use disparity
        d = _enhance(d)
        rgb = colormaps[DEPTH_CMAP](d)[..., :3]
        if DEPTH_LIGHTEN > 0:                        # pull the near end toward white
            rgb = rgb * (1 - d[..., None] * DEPTH_LIGHTEN) + d[..., None] * DEPTH_LIGHTEN
        return rgb

    os.makedirs(OUT, exist_ok=True)
    idx = ["sample\tsuite\theld_out\tkeyframe\toffset\ttarget\tprompt"]
    for i in order:
        noun = NOUNS[i]; t = temb(noun)
        held = HELD is None or i in HELD
        base = f"{i:02d}_{SUITES[i]}_{noun}" + ("" if held else "_TRAINSEEN")
        for k in range(K):
            frame = FUT[i][k][CAM]
            d = os.path.join(OUT, base); os.makedirs(d, exist_ok=True)
            pre = f"k{k}_off{OFFS[k]}_"          # one folder per task, keyframe encoded in the name
            pf = torch.from_numpy(FP[i][CAM].astype(np.float32))[k * GRID * GRID:(k + 1) * GRID * GRID][None].to(dev)
            with torch.no_grad():
                lp = head_p(pf, t)[0].cpu().numpy()
                lt = head_t(teacher(frame).float(), t)[0].cpu().numpy()
            save(frame, f"{d}/{pre}frame.png")
            save(ov(frame, lt), f"{d}/{pre}loc_target.png")
            save(ov(frame, lp), f"{d}/{pre}loc_pred.png")
            if DPR is not None: save(depth_img(DPR[i][CAM][k], frame.shape[:2], True), f"{d}/{pre}depth_pred.png")
            if DGT is not None: save(depth_img(DGT[i][CAM][k], frame.shape[:2], True), f"{d}/{pre}depth_target.png")
            if DFULL is not None: save(depth_img(DFULL[i][CAM][k], frame.shape[:2], False), f"{d}/{pre}depth_da3_full_gt.png")
            idx.append(f"{i}\t{SUITES[i]}\t{int(held)}\tk{k}\t+{OFFS[k]}\t{noun}\t{PROMPTS[i]}")
        print(f"[sample {i}] {SUITES[i]} '{noun}' -> {K} keyframes", flush=True)
    open(f"{OUT}/index.tsv", "w").write("\n".join(idx) + "\n")
    n_png = sum(len(f) for _, _, f in os.walk(OUT))
    print(f"SAVED {len(order)} samples x {K} keyframes -> {OUT} ({n_png} files)", flush=True)
    print("EXPORT-COMBINED-DONE", flush=True)


if __name__ == "__main__":
    main()
