#!/usr/bin/env python3
"""Train small feature->dense decoder probes for the 2B DINOv3+DA3 planner viz.

Two probes, frozen teacher backbones, trained on LIBERO frame-cache frames:
  --which dino : DINOv3 last-layer patch feats [B,256,1280] -> RGB [B,3,224,224]   (supervise = the frame)
  --which da3  : DA3   last-layer patch feats [B,256,2048] -> disparity [B,1,224,224]
                 (supervise = normalized inverse-depth from the FULL DA3 depth head on the same frame)

Purpose: let the planner visualization decode BOTH the teacher TARGET features and the planner's
PREDICTED features back to a full-res image (RGB recon / depth), instead of 16x16 PCA mosaics.
"""
from __future__ import annotations
import os, sys, json, glob, math, argparse, random
from pathlib import Path
import numpy as np
import torch
try:
    import resource as _res
    _s,_h=_res.getrlimit(_res.RLIMIT_NOFILE)
    _res.setrlimit(_res.RLIMIT_NOFILE,(min(_h,65536),_h))
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dinov3_da3_2b"))

FRAME_CACHE = os.environ.get("FASTWAM_FRAME_CACHE_DIR", "/data/users/junjie/data/frame_cache/libero")


class FrameCacheDataset(torch.utils.data.Dataset):
    """Random (episode, frame) sampler over the pre-decoded uint8 [N,3,224,224] .npy memmaps."""
    def __init__(self, cache_dir, virtual_len=200000, seed=0):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.npy"), recursive=True))
        if not self.files:
            raise RuntimeError(f"no .npy frame-cache files under {cache_dir}")
        self.virtual_len = virtual_len
        self._rng = random.Random(seed)
        # cache memmaps lazily per worker
        from collections import OrderedDict
        self._mm = OrderedDict()

    def __len__(self):
        return self.virtual_len

    def _get_mm(self, path):
        m = self._mm.get(path)
        if m is None:
            m = np.load(path, mmap_mode="r")
            self._mm[path] = m
            if len(self._mm) > 96:              # bounded LRU: close oldest memmap fd
                _op, _om = self._mm.popitem(last=False)
                try: _om._mmap.close()
                except Exception: pass
        return m

    def __getitem__(self, idx):
        # deterministic-ish but varied: derive rng from idx
        rng = random.Random(idx * 2654435761 % (2**31))
        path = self.files[rng.randrange(len(self.files))]
        m = self._get_mm(path)
        n = m.shape[0]
        fi = rng.randrange(n)
        frame = np.ascontiguousarray(m[fi])  # uint8 [3,224,224]
        return torch.from_numpy(frame).float() / 255.0  # [3,224,224] in [0,1]


class ProbeDecoder(nn.Module):
    """[B, grid*grid, in_dim] tokens -> [B, out_ch, 224, 224] dense map."""
    def __init__(self, in_dim, out_ch, grid=16, out_act="sigmoid"):
        super().__init__()
        self.in_dim, self.grid, self.out_ch, self.out_act = in_dim, grid, out_ch, out_act
        chs = [in_dim, 512, 256, 128, 64]  # 16 ->32 ->64 ->128 ->256
        blocks = []
        for i in range(len(chs) - 1):
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1),
                nn.GroupNorm(min(32, chs[i + 1]), chs[i + 1]),
                nn.GELU(),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Conv2d(chs[-1], out_ch, 1)

    def forward(self, tok):
        B = tok.shape[0]
        x = tok.transpose(1, 2).reshape(B, self.in_dim, self.grid, self.grid).contiguous()
        for b in self.blocks:
            x = b(x)                       # -> [B,64,256,256]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.head(x)
        return torch.sigmoid(x) if self.out_act == "sigmoid" else x

    def config(self):
        return {"in_dim": self.in_dim, "out_ch": self.out_ch, "grid": self.grid, "out_act": self.out_act}


class FeatureUpsampler(nn.Module):
    """FeatUp-style FEATURE-ONLY upsampler: [B, in_grid^2, dim] -> [B, out_grid^2, dim].

    Learns a residual over a bilinear baseline; NO image guidance (so it never leaks the
    real frame into a predicted-feature panel). Trained to match a higher-input-res DINOv3
    feature grid on the same frame.
    """
    def __init__(self, dim=1280, in_grid=16, out_grid=32):
        super().__init__()
        self.dim, self.in_grid, self.out_grid = dim, in_grid, out_grid
        self.refine = nn.Sequential(
            nn.Conv2d(dim, 512, 3, padding=1), nn.GroupNorm(32, 512), nn.GELU(),
            nn.Upsample(size=(out_grid, out_grid), mode="bilinear", align_corners=False),
            nn.Conv2d(512, 512, 3, padding=1), nn.GroupNorm(32, 512), nn.GELU(),
            nn.Conv2d(512, dim, 3, padding=1),
        )

    def forward(self, tok):
        B = tok.shape[0]
        x = tok.transpose(1, 2).reshape(B, self.dim, self.in_grid, self.in_grid).contiguous()
        base = F.interpolate(x, size=(self.out_grid, self.out_grid), mode="bilinear", align_corners=False)
        out = base + self.refine(x)
        return out.flatten(2).transpose(1, 2)  # [B, out_grid^2, dim]

    def config(self):
        return {"dim": self.dim, "in_grid": self.in_grid, "out_grid": self.out_grid}


class MiniDPTProbe(nn.Module):
    """DPT-style single-layer depth head: reassemble one 16x16xD token map to 4 scales
    (8/16/32/64), FPN-fuse coarse->fine, then a depth head -> [B,1,224,224] LOG-depth (linear)."""
    def __init__(self, in_dim=2048, feat=256, grid=16, out_ch=1):
        super().__init__()
        self.in_dim, self.feat, self.grid, self.out_ch = in_dim, feat, grid, out_ch
        self.proj = nn.Conv2d(in_dim, feat, 1)
        self.to_l0 = nn.Conv2d(feat, feat, 3, stride=2, padding=1)                 # 16 -> 8
        self.to_l1 = nn.Identity()                                                 # 16
        self.to_l2 = nn.ConvTranspose2d(feat, feat, 2, stride=2)                   # 16 -> 32
        self.to_l3 = nn.Sequential(nn.ConvTranspose2d(feat, feat, 2, stride=2),
                                   nn.GELU(), nn.ConvTranspose2d(feat, feat, 2, stride=2))  # 16 -> 64
        def rf():
            return nn.Sequential(nn.Conv2d(feat, feat, 3, padding=1), nn.GroupNorm(32, feat),
                                 nn.GELU(), nn.Conv2d(feat, feat, 3, padding=1))
        self.rf0, self.rf1, self.rf2, self.rf3 = rf(), rf(), rf(), rf()
        self.head = nn.Sequential(nn.Conv2d(feat, feat // 2, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(feat // 2, out_ch, 1))

    def forward(self, tok):
        B = tok.shape[0]
        x = tok.transpose(1, 2).reshape(B, self.in_dim, self.grid, self.grid).contiguous()
        x = self.proj(x)
        l0, l1, l2, l3 = self.rf0(self.to_l0(x)), self.rf1(self.to_l1(x)), self.rf2(self.to_l2(x)), self.rf3(self.to_l3(x))
        f = F.interpolate(l0, size=l1.shape[-2:], mode="bilinear", align_corners=False) + l1
        f = F.interpolate(f, size=l2.shape[-2:], mode="bilinear", align_corners=False) + l2
        f = F.interpolate(f, size=l3.shape[-2:], mode="bilinear", align_corners=False) + l3
        f = F.interpolate(f, size=(224, 224), mode="bilinear", align_corners=False)
        return self.head(f)  # [B,1,224,224] log-depth

    def config(self):
        return {"in_dim": self.in_dim, "feat": self.feat, "grid": self.grid, "out_ch": self.out_ch}


def depth_to_disp(depth, eps=1e-6):
    """[B,1,H,W] depth -> per-frame min-max normalized disparity (1/depth) in [0,1]."""
    disp = 1.0 / depth.clamp_min(eps)
    B = disp.shape[0]
    flat = disp.view(B, -1)
    lo = flat.min(1, keepdim=True).values.view(B, 1, 1, 1)
    hi = flat.max(1, keepdim=True).values.view(B, 1, 1, 1)
    return ((disp - lo) / (hi - lo + eps)).clamp(0, 1)


def silog_loss(pred_logd, gt_logd, lam=0.85):
    g = pred_logd - gt_logd
    return (g ** 2).mean() - lam * (g.mean() ** 2)


def grad_match(pred, gt, scales=(1, 2, 4)):
    loss = 0.0
    for sc in scales:
        p = F.avg_pool2d(pred, sc) if sc > 1 else pred
        q = F.avg_pool2d(gt, sc) if sc > 1 else gt
        loss = loss + (p[..., :, 1:] - p[..., :, :-1] - (q[..., :, 1:] - q[..., :, :-1])).abs().mean()
        loss = loss + (p[..., 1:, :] - p[..., :-1, :] - (q[..., 1:, :] - q[..., :-1, :])).abs().mean()
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["dino", "dino_up", "da3", "da3_v2", "both"], default="both")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=28)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=Path("/data/users/junjie/probes_2b"))
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    ds = FrameCacheDataset(FRAME_CACHE, virtual_len=args.steps * args.batch_size + 1000)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                         shuffle=False, drop_last=True, pin_memory=True, persistent_workers=True)
    print(f"[data] {len(ds.files)} cache videos; virtual_len={len(ds)}", flush=True)

    which = ["dino", "da3"] if args.which == "both" else [args.which]

    for w in which:
        print(f"\n==== training probe: {w} ====", flush=True)
        if w == "dino":
            from dinov3_target import Dinov3TargetEncoder
            enc = Dinov3TargetEncoder(input_size=256, device=device)
            in_dim = enc.feature_dim  # 1280
            probe = ProbeDecoder(in_dim, 3, out_act="sigmoid").to(device)
            teacher_feats = lambda fr: enc._patch_tokens(enc._prep(fr)).float()          # [B,256,1280]
            def target_of(fr):
                return fr.to(device)                                                     # RGB [B,3,224,224]
            def loss_fn(pred, tgt):
                return F.l1_loss(pred, tgt) + 0.25 * F.mse_loss(pred, tgt)
        elif w == "dino_up":
            from dinov3_target import Dinov3TargetEncoder
            enc = Dinov3TargetEncoder(input_size=256, device=device)
            enc_hi = Dinov3TargetEncoder(input_size=512, device=device)   # 32x32 dense target
            in_dim = enc.feature_dim  # 1280
            probe = FeatureUpsampler(dim=in_dim, in_grid=16, out_grid=32).to(device)
            teacher_feats = lambda fr: enc._patch_tokens(enc._prep(fr)).float()           # [B,256,1280]
            def target_of(fr):
                return enc_hi._patch_tokens(enc_hi._prep(fr)).float()                     # [B,1024,1280]
            def loss_fn(pred, tgt):
                cos = F.cosine_similarity(pred, tgt, dim=-1).mean()
                return F.mse_loss(pred, tgt) + (1.0 - cos)
        elif w == "da3_v2":
            from depth_anything3_target import _import_da3, DepthAnything3TargetEncoder
            enc = DepthAnything3TargetEncoder(process_res=int(os.environ.get("DA3_PROCESS_RES", "224")), device=device)
            in_dim = enc.feature_dim  # 2048
            DA3 = _import_da3(os.environ["DA3_CODE_ROOT"])
            full = DA3.from_pretrained(os.environ["DA3_CKPT_DIR"]).to(device).eval()
            for p in full.parameters():
                p.requires_grad_(False)
            m = full.model
            probe = MiniDPTProbe(in_dim=in_dim, feat=256).to(device)              # -> log-depth
            teacher_feats = lambda fr: enc._patch_tokens(enc._prep(fr)).float()   # [B,256,2048]
            @torch.no_grad()
            def target_of(fr):
                x = enc._prep(fr).to(next(m.parameters()).dtype).unsqueeze(1)
                feats, _aux = m.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
                with torch.autocast(device_type="cuda", enabled=False):
                    out = m._process_depth_head(feats, 224, 224)                  # 4-layer DPT GT
                depth = (out["depth"] if hasattr(out, "keys") else out.depth).float().clamp_min(1e-3)
                return torch.log(depth)                                           # log-depth GT
            def loss_fn(pred, tgt):
                return silog_loss(pred, tgt) + 0.5 * grad_match(pred, tgt)
        else:
            from depth_anything3_target import _import_da3, DepthAnything3TargetEncoder
            enc = DepthAnything3TargetEncoder(process_res=int(os.environ.get("DA3_PROCESS_RES", "224")), device=device)
            in_dim = enc.feature_dim  # 2048
            DA3 = _import_da3(os.environ["DA3_CODE_ROOT"])
            full = DA3.from_pretrained(os.environ["DA3_CKPT_DIR"]).to(device).eval()
            for p in full.parameters():
                p.requires_grad_(False)
            m = full.model
            probe = ProbeDecoder(in_dim, 1, out_act="sigmoid").to(device)
            teacher_feats = lambda fr: enc._patch_tokens(enc._prep(fr)).float()          # [B,256,2048]
            @torch.no_grad()
            def target_of(fr):
                x = enc._prep(fr).to(next(m.parameters()).dtype).unsqueeze(1)             # [B,1,3,224,224]
                feats, _aux = m.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
                with torch.autocast(device_type="cuda", enabled=False):
                    out = m._process_depth_head(feats, 224, 224)
                depth = (out["depth"] if hasattr(out, "keys") else out.depth).float()     # [B,1,224,224]
                return depth_to_disp(depth)
            def loss_fn(pred, tgt):
                return F.l1_loss(pred, tgt)

        opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
        probe.train()
        it = iter(loader)
        losses = []
        for step in range(1, args.steps + 1):
            try:
                fr = next(it)
            except StopIteration:
                it = iter(loader); fr = next(it)
            fr = fr.to(device, non_blocking=True)               # [B,3,224,224] in [0,1]
            with torch.no_grad():
                feats = teacher_feats(fr)
                tgt = target_of(fr)
            pred = probe(feats)
            loss = loss_fn(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            losses.append(loss.item())
            if step % args.log_every == 0 or step == 1:
                avg = sum(losses[-args.log_every:]) / min(len(losses), args.log_every)
                print(f"[{w}] step {step}/{args.steps} loss={loss.item():.5f} avg{args.log_every}={avg:.5f} lr={sched.get_last_lr()[0]:.2e}", flush=True)

        name = {"dino": "dino_rgb", "dino_up": "dino_upsample", "da3": "da3_depth", "da3_v2": "da3_depth_v2"}[w]
        ckpt = args.out_dir / f"{name}_probe.pt"
        torch.save({"state_dict": probe.state_dict(), "config": probe.config(),
                    "which": w, "in_dim": in_dim, "final_loss": float(sum(losses[-100:]) / min(len(losses),100))}, ckpt)
        (args.out_dir / f"{name}_probe_meta.json").write_text(json.dumps(probe.config()))
        print(f"[{w}] SAVED {ckpt} final_avg100={sum(losses[-100:])/min(len(losses),100):.5f}", flush=True)
        del probe, enc
        if w in ("da3", "da3_v2"):
            del full
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
