"""Score every (sample, camera, keyframe) by how well the planner predicted it, then export the best.

Two independent scores, both computed against the same forward pass:
  depth_r  : Pearson r between the decoded predicted depth and the decoded TEACHER depth. The teacher
             path itself reaches r=0.985 against full DA3, so this isolates the planner's error rather
             than the probe's.
  loc_iou  : agreement of the localisation readouts -- overlap of the top-10% cells of the predicted
             map and the teacher map. Argmax distance is too brittle on a 16x16 grid.
  mask_ok  : whether the CLIPSeg pseudo-ground-truth for this noun can be trusted at all. CLIPSeg does
             not recognise several LIBERO objects (e.g. "butter" resolves to the basket), and the
             localisation head is trained on those masks, so an unreliable mask makes both the head
             and every metric derived from it meaningless. Instances failing this are dropped rather
             than ranked, since a figure built on them shows the head repeating CLIPSeg's mistake.
A combined rank (mean of the two percentile ranks) picks instances that are good on BOTH, which is
what a figure needs; RANK_BY=depth or loc restricts it to one.

Exports the same per-instance folders as export_combined_panels.py, ordered best-first, plus a
scores.tsv holding every instance so the ranking can be audited or re-cut differently.
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
OUT = os.environ.get("PANEL_DIR", f"{HERE}/figs/best_panels")
SIG = f"{ROOT}/third_party/siglip2-large-patch16-256"
GRID, DIM, RES = 16, 1024, 256
CAM_IDX = [int(x) for x in os.environ.get("CAM_IDX", "0").split(",")]
TOPK = int(os.environ.get("TOPK", 24))
PER_SAMPLE = int(os.environ.get("PER_SAMPLE", 2))    # cap keyframes kept per (sample,camera)
RANK_BY = os.environ.get("RANK_BY", "both")          # both | depth | loc
UPSCALE = int(os.environ.get("UPSCALE", 2))
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
    D = np.load(DEPTH)
    DPR, DGT = D["depth_pred"], D["depth_gt"]
    DFULL = D["depth_da3_full"] if "depth_da3_full" in D.files else None
    N = len(FUT)
    print(f"scoring {N} samples x {len(CAM_IDX)} cam x {K} keyframes", flush=True)

    CONTROL_NOUNS = ["wall", "floor", "table"]        # never the manipulated object in LIBERO
    MIN_CONC = float(os.environ.get("MIN_MASK_CONC", 0.42))   # top-10% mass; below this = diffuse
    MAX_CTRL_IOU = float(os.environ.get("MAX_CTRL_IOU", 0.55))  # overlap with a control-noun mask

    @torch.no_grad()
    def clipseg_mask(img, noun):
        x = Image.fromarray(img).resize((352, 352))
        i = cproc(text=[noun], images=[x], return_tensors="pt", padding=True)
        i = {k: v.to(dev) for k, v in i.items()}
        lg = cseg(**i).logits
        if lg.ndim == 2: lg = lg[None]
        m = F.interpolate(torch.sigmoid(lg[0])[None, None], (GRID, GRID), mode="bilinear")[0, 0].cpu().numpy()
        return (m - m.min()) / (m.max() - m.min() + 1e-6)

    def conc(m, frac=0.10):
        v = m.flatten().astype(np.float64); v = v - v.min(); p = v / (v.sum() + 1e-9)
        return float(np.sort(p)[::-1][:max(1, int(frac * len(p)))].sum())

    def mask_quality(img, noun):
        """(peakedness, worst overlap with a control noun) -- both needed for the mask to be usable."""
        m = clipseg_mask(img, noun)
        ctrl = max(top_iou(m, clipseg_mask(img, c)) for c in CONTROL_NOUNS)
        return conc(m), ctrl

    def top_iou(a, b, frac=0.10):
        k = max(1, int(frac * a.size))
        sa = set(np.argsort(-a.flatten())[:k].tolist()); sb = set(np.argsort(-b.flatten())[:k].tolist())
        return len(sa & sb) / len(sa | sb)

    rows = []
    tcache = {}
    for i in range(N):
        noun = NOUNS[i]
        if noun not in tcache: tcache[noun] = temb(noun)
        t = tcache[noun]
        for v in CAM_IDX:
            for k in range(K):
                frame = FUT[i][k][v]
                pf = torch.from_numpy(FP[i][v].astype(np.float32))[k * GRID * GRID:(k + 1) * GRID * GRID][None].to(dev)
                with torch.no_grad():
                    lp = head_p(pf, t)[0].cpu().numpy(); lt = head_t(teacher(frame).float(), t)[0].cpu().numpy()
                dp, dg = DPR[i][v][k].astype(np.float32), DGT[i][v][k].astype(np.float32)
                r = float(np.corrcoef(dp.ravel(), dg.ravel())[0, 1])
                mc, ci = mask_quality(frame, noun)
                rows.append(dict(i=i, v=v, k=k, depth_r=r, loc_iou=float(top_iou(lp, lt)),
                                 mask_conc=mc, ctrl_iou=ci, mask_ok=bool(mc >= MIN_CONC and ci <= MAX_CTRL_IOU),
                                 suite=SUITES[i], noun=noun, prompt=PROMPTS[i]))
        if (i + 1) % 10 == 0: print(f"  {i+1}/{N}", flush=True)

    n_bad = sum(1 for x in rows if not x["mask_ok"])
    print(f"CLIPSeg pseudo-GT unusable for {n_bad}/{len(rows)} instances "
          f"(conc<{MIN_CONC} or control-IoU>{MAX_CTRL_IOU}); they are excluded from the ranking",
          flush=True)
    for x in rows:
        if not x["mask_ok"]:
            print(f"    drop s{x['i']:03d} k{x['k']} '{x['noun']}' conc={x['mask_conc']:.3f} ctrlIoU={x['ctrl_iou']:.3f}", flush=True)
    rows_all = rows
    rows = [x for x in rows if x["mask_ok"]] or rows_all
    dr = np.array([x["depth_r"] for x in rows]); li = np.array([x["loc_iou"] for x in rows])
    pr = lambda a: a.argsort().argsort() / (len(a) - 1)          # percentile rank
    score = {"both": (pr(dr) + pr(li)) / 2, "depth": pr(dr), "loc": pr(li)}[RANK_BY]
    for x, s in zip(rows, score): x["score"] = float(s)
    rows.sort(key=lambda x: -x["score"])

    with open(f"{HERE}/data/instance_scores.tsv", "w") as f:
        f.write("rank\tsample\tcam\tkf\toffset\tdepth_r\tloc_iou\tmask_conc\tctrl_iou\tmask_ok\tscore\tsuite\ttarget\tprompt\n")
        for n, x in enumerate(sorted(rows_all, key=lambda y: -y.get("score", -1)), 1):
            f.write(f"{n}\t{x['i']}\t{x['v']}\t{x['k']}\t+{OFFS[x['k']]}\t{x['depth_r']:.4f}\t"
                    f"{x['loc_iou']:.4f}\t{x['mask_conc']:.4f}\t{x['ctrl_iou']:.4f}\t{int(x['mask_ok'])}\t"
                    f"{x.get('score', -1):.4f}\t{x['suite']}\t{x['noun']}\t{x['prompt']}\n")
    print(f"depth_r  mean={dr.mean():.3f}  top-{TOPK} cut={sorted(dr)[-TOPK]:.3f}", flush=True)
    print(f"loc_iou  mean={li.mean():.3f}  top-{TOPK} cut={sorted(li)[-TOPK]:.3f}", flush=True)

    kept, seen = [], {}
    for x in rows:
        key = (x["i"], x["v"])
        if seen.get(key, 0) >= PER_SAMPLE: continue
        seen[key] = seen.get(key, 0) + 1; kept.append(x)
        if len(kept) >= TOPK: break

    TURBO = cm.get_cmap("turbo")
    def save(a, p):
        im = Image.fromarray(a if a.dtype == np.uint8 else (np.clip(a, 0, 1) * 255).astype(np.uint8))
        if UPSCALE > 1: im = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.LANCZOS)
        im.save(p)
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

    def dimg(dm, hw, invert):
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
    idx = ["rank\tsample\tcam\tkeyframe\toffset\tdepth_r\tloc_iou\ttarget\tprompt"]
    for n, x in enumerate(kept, 1):
        i, v, k = x["i"], x["v"], x["k"]
        frame = FUT[i][k][v]; t = tcache[x["noun"]]
        pf = torch.from_numpy(FP[i][v].astype(np.float32))[k * GRID * GRID:(k + 1) * GRID * GRID][None].to(dev)
        with torch.no_grad():
            lp = head_p(pf, t)[0].cpu().numpy(); lt = head_t(teacher(frame).float(), t)[0].cpu().numpy()
        d = os.path.join(OUT, f"{n:02d}_s{i:03d}_{'main' if v == 0 else 'wrist'}_k{k}_{x['suite']}_{x['noun']}")
        os.makedirs(d, exist_ok=True)
        save(frame, f"{d}/frame.png")
        save(ov(frame, lt), f"{d}/loc_target.png")
        save(ov(frame, lp), f"{d}/loc_pred.png")
        save(dimg(DGT[i][v][k], frame.shape[:2], True), f"{d}/depth_target.png")
        save(dimg(DPR[i][v][k], frame.shape[:2], True), f"{d}/depth_pred.png")
        if DFULL is not None: save(dimg(DFULL[i][v][k], frame.shape[:2], False), f"{d}/depth_da3_full_gt.png")
        idx.append(f"{n}\t{i}\t{v}\t{k}\t+{OFFS[k]}\t{x['depth_r']:.4f}\t{x['loc_iou']:.4f}\t{x['noun']}\t{x['prompt']}")
        print(f"[{n:02d}] depth_r={x['depth_r']:.3f} loc_iou={x['loc_iou']:.3f} mask={x['mask_conc']:.2f} "
              f"{x['suite']}/{x['noun']} k{k} :: {x['prompt'][:44]}", flush=True)
    open(f"{OUT}/index.tsv", "w").write("\n".join(idx) + "\n")
    print(f"SAVED top-{len(kept)} -> {OUT}", flush=True); print("RANK-EXPORT-DONE", flush=True)


if __name__ == "__main__":
    main()
