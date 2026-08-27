"""OLA: decode depth for the SAME samples the semantic probe uses, and dump the maps.

The feature dump stores the planner's predicted depth plan but not the DA3 ground truth, which needs
the DA3 model (1.6 GB, only present on this box). Rather than shipping DA3 to the workstation, this
computes both maps here and returns only the decoded depth images, which are tiny.

For every (sample, camera, keyframe) in planner_feats_dualcam_k4_big.npz:
  depth_gt   : DA3 teacher features of the real keyframe -> WSA probe
  depth_pred : the planner's predicted depth plan        -> same WSA probe
Same probe on both sides, so the two maps are directly comparable.
"""
import os, sys, json
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

R = os.environ.get("REPO", "/data/users/junjie/code/VLM4WAM_dual_camera_k4")
for p in ["qwen3_vl_semantic_planner/dinov3_da3_2b", "qwen3_vl_semantic_planner",
          "qwen3_vl_semantic_planner/lingbot_dino_4b",   # depth_target lives here
          "third_party/FastWAM/src"]:
    d = f"{R}/{p}"
    if os.path.isdir(d): sys.path.insert(0, d)
os.chdir(R)
dev = "cuda"
NPZ = os.environ.get("NPZ", "/data/users/junjie/planner_feats_dualcam_k4_big.npz")
PROBE = os.environ.get("WSA_PROBE", "/data/users/junjie/probes_2b/da3_depth_wsa_probe.pt")
CK = Path(os.environ.get("CKPT",
    f"{R}/outputs/qwen3vl2b_siglip2_da3_libero_dual_camera_k4_wsa_predecoded_b8_restart/step_030000"))
OUT = os.environ.get("OUT", "/data/users/junjie/depth_maps_k4.npz")
GRID = 16
def log(m): print(m, flush=True)


def main():
    from wsa_depth_probe import WSAMultiLayerDPTProbe
    from depth_anything3_target import DepthAnything3TargetEncoder
    meta_ck = json.loads((CK / "planner_meta.json").read_text())

    pay = torch.load(PROBE, map_location="cpu", weights_only=False)
    probe = WSAMultiLayerDPTProbe.from_config(pay["config"])
    probe.load_state_dict(pay["state_dict"], strict=True)
    probe.to(dev).eval().requires_grad_(False)
    log(f"probe loaded, layers={meta_ck.get('da3_teacher_layers')}")

    # the DA3 line uses DepthAnything3TargetEncoder; depth_target.DepthTargetEncoder is the MoGe
    # encoder from the 4B lingbot line and takes completely different arguments
    # must mirror the checkpoint: default align_strategy is last_layer (1 layer), but this run is
    # wsa_multilayer, and the WSA probe rejects anything that is not its 4 teacher layers
    enc = DepthAnything3TargetEncoder(
        ckpt_dir=os.environ["DA3_CKPT_DIR"], code_root=os.environ["DA3_CODE_ROOT"],
        process_res=int(os.environ.get("DA3_PROCESS_RES", "224")),
        align_strategy=meta_ck.get("da3_align_strategy", "wsa_multilayer"),
        teacher_layers=meta_ck.get("da3_teacher_layers"),
        layer_weights=meta_ck.get("da3_layer_weights"),
        device=dev,
    ) if os.environ.get("DA3_CKPT_DIR") else None
    # full DA3 model, for the reference depth the paper figure calls "DA3-full future GT":
    # this is the model's OWN depth head output, not a probe decoding of its features
    full = None
    if enc is not None:
        from depth_anything3_target import _import_da3
        DA3 = _import_da3(os.environ["DA3_CODE_ROOT"])
        full = DA3.from_pretrained(os.environ["DA3_CKPT_DIR"]).to(dev).eval()
        for pm in full.parameters(): pm.requires_grad_(False)
    log(f"DA3 teacher encoder: {'ready' if enc is not None else 'MISSING (pred only)'}"
        + (f" strategy={enc.align_strategy} layers={list(enc.teacher_layers)}" if enc is not None else ""))

    z = np.load(NPZ, allow_pickle=True)
    FUT, DP = z["fut"], z["dp"]
    N, K, V = len(FUT), FUT.shape[1], FUT.shape[2]
    log(f"samples={N} K={K} V={V} depth_plan{DP.shape}")

    @torch.no_grad()
    def decode(feat_LND):
        """feat as [1, L, N, D] -> normalised depth map (H, W)."""
        d = probe(feat_LND)
        d = d[0] if isinstance(d, (tuple, list)) else d
        d = d.squeeze().float().cpu().numpy()
        return ((d - d.min()) / (d.max() - d.min() + 1e-6)).astype(np.float16)

    @torch.no_grad()
    def da3_full_disp(frame_hwc):
        """DA3's own depth head -> normalised disparity, matching the reference figure."""
        fr = torch.from_numpy(frame_hwc.astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
        m = full.model
        x = enc._prep(fr).to(next(m.parameters()).dtype).unsqueeze(1)
        feats, _ = m.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
        with torch.autocast(device_type="cuda", enabled=False):
            out = m._process_depth_head(feats, 224, 224)
        d = (out["depth"] if hasattr(out, "keys") else out.depth).float()[0, 0]
        disp = 1.0 / d.clamp_min(1e-3)
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
        return disp.cpu().numpy().astype(np.float16)

    preds, gts, fulls = [], [], []
    for i in range(N):
        pr_s, gt_s, fl_s = [], [], []
        for v in range(V):
            pr_k, gt_k = [], []
            for k in range(K):
                sl = slice(k * GRID * GRID, (k + 1) * GRID * GRID)
                fp = torch.from_numpy(DP[i][v].astype(np.float32))[sl][None].to(dev)  # [1,N,L,D]
                pr_k.append(decode(fp.transpose(1, 2).contiguous()))                  # -> [1,L,N,D]
                if enc is not None:
                    # encode_future_keyframes takes a sequence of b3hw tensors, not 3hw
                    img = torch.from_numpy(FUT[i][k][v].astype(np.float32) / 255.).permute(2, 0, 1)[None].to(dev)
                    with torch.no_grad():
                        # the encoder exposes encode_future_keyframes(list of 3HW), not forward()
                        tf = enc.encode_future_keyframes([img])
                    if tf.ndim == 5: tf = tf[0]             # (B,K,L,N,D) -> (K,L,N,D)
                    if tf.ndim == 4 and tf.shape[0] == 1: tf = tf[0]
                    gt_k.append(decode(tf[None] if tf.ndim == 3 else tf))
            pr_s.append(np.stack(pr_k))
            if gt_k: gt_s.append(np.stack(gt_k))
            if full is not None:
                fl_s.append(np.stack([da3_full_disp(FUT[i][k][v]) for k in range(K)]))
        preds.append(np.stack(pr_s))
        if gt_s: gts.append(np.stack(gt_s))
        if fl_s: fulls.append(np.stack(fl_s))
        if (i + 1) % 10 == 0: log(f"  {i+1}/{N}")

    out = {"depth_pred": np.stack(preds)}                   # (N,V,K,H,W)
    if gts: out["depth_gt"] = np.stack(gts)                 # WSA probe on teacher features
    if fulls: out["depth_da3_full"] = np.stack(fulls)       # DA3's own depth head
    np.savez_compressed(OUT, **out)
    log(f"SAVED {OUT} pred{out['depth_pred'].shape}"
        + (f" gt{out['depth_gt'].shape}" if "depth_gt" in out else " (no GT)")
        + (f" da3full{out['depth_da3_full'].shape}" if "depth_da3_full" in out else ""))
    log("DEPTH-MAPS-DONE")


if __name__ == "__main__":
    main()
