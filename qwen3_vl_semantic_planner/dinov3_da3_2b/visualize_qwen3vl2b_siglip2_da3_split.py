#!/usr/bin/env python3
"""Per-camera SQUARE viz for the 2B **SigLIP2**+DA3 LIBERO planner (main & wrist shown separately).

Sibling of visualize_qwen3vl2b_dinov3_da3_split.py, adapted for the SigLIP2 video teacher:
  * Semantic row uses the SigLIP2 teacher (1024-d) instead of DINOv3 (1280-d).
  * Because SigLIP2 has no trained upsampling probe (the dino_upsample_probe is 1280-d only),
    the semantic PCA is the plain 16x16 token PCA (joint per target/pred pair) bilinear-upscaled
    for display — the low-res "mosaic" form, not the hi-res FeatUp-style panel. Structure is
    visible; edges are soft (16x16 -> 224).
  * Depth rows are IDENTICAL to the dinov3 script: this line's depth teacher is still DA3 (2048-d),
    so da3_depth_v2_probe.pt and the DA3-full GT path are reused verbatim.

Features/depth are computed on the full [224,448] 2-cam composite; each rendered panel is
un-squished to [224,448] then split at the horizontal midpoint into two 224x224 squares.
Two PNGs per sample: sample_XX_main.png / sample_XX_wrist.png, each a 3x6 grid:
  Row0 SigLIP2 PCA | Row1 depth probe_v2 (turbo) | Row2 DA3-full GT depth (ref under TARGET cols).
"""
from __future__ import annotations
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
HERE = os.path.dirname(os.path.abspath(__file__))          # .../qwen3_vl_semantic_planner/dinov3_da3_2b
ROOT = os.path.dirname(HERE)                               # .../qwen3_vl_semantic_planner
for p in (ROOT, HERE, os.path.join(ROOT, "lingbot_dino_4b")):
    sys.path.insert(0, p)
import train_qwen3vl4b_lingbot_dino_planner as T
from qwen3vl_wrapper import move_qwen_inputs_to_device, configure_qwen3vl_processor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from train_feature_probes import MiniDPTProbe
TURBO = cm.get_cmap("turbo")

def to_disp(img):
    x = img.float()
    if x.max() > 1.5: x = x/255.0
    return x.clamp(0,1).cpu().numpy()

def to448(a):
    """[H,W,3] -> [224,448,3] (un-squish to the native 2-cam composite aspect)."""
    t = torch.from_numpy(np.ascontiguousarray(a)).permute(2,0,1).unsqueeze(0).float()
    t = F.interpolate(t, size=(224,448), mode="bilinear", align_corners=False)
    return t.squeeze(0).permute(1,2,0).clamp(0,1).numpy()

def halves(a448):
    return a448[:, :224], a448[:, 224:]   # main (left 224x224), wrist (right 224x224)

def load(path, cls, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = cls(**ck["config"]).to(device).eval(); m.load_state_dict(ck["state_dict"]); return m

@torch.no_grad()
def lowres_pca(tt, tp, grid):
    """tt,tp: [grid*grid, D] target/pred tokens. Joint PCA(3) per pair -> two [grid,grid,3] maps.
    No probe: colors are comparable only WITHIN a (target,pred) pair."""
    ft = tt.float(); fp = tp.float()
    X = torch.cat([ft, fp], 0); Xc = X - X.mean(0, keepdim=True)
    _,_,V = torch.linalg.svd(Xc, full_matrices=False); proj = Xc @ V[:3].T
    lo, hi = proj.min(0).values, proj.max(0).values; proj = (proj - lo) / (hi - lo + 1e-6)
    n = ft.shape[0]
    return proj[:n].reshape(grid,grid,3).cpu().numpy(), proj[n:].reshape(grid,grid,3).cpu().numpy()

@torch.no_grad()
def depth_v2(probe, tt, tp, device):
    lt = probe(tt.unsqueeze(0).to(device).float())[0,0]; lp = probe(tp.unsqueeze(0).to(device).float())[0,0]
    dispt = 1.0/torch.exp(lt).clamp_min(1e-3); dispp = 1.0/torch.exp(lp).clamp_min(1e-3)
    both = torch.stack([dispt,dispp]); lo,hi=both.min(),both.max(); both=(both-lo)/(hi-lo+1e-6)
    return TURBO(both[0].cpu().numpy())[...,:3], TURBO(both[1].cpu().numpy())[...,:3]

@torch.no_grad()
def gt_turbo(m, da3, fr, device):
    x = da3._prep(fr).to(next(m.parameters()).dtype).unsqueeze(1)
    feats,_ = m.backbone(x, cam_token=None, export_feat_layers=[], ref_view_strategy="saddle_balanced")
    with torch.autocast(device_type="cuda", enabled=False):
        out = m._process_depth_head(feats,224,224)
    d = (out["depth"] if hasattr(out,"keys") else out.depth).float()[0,0]
    disp = 1.0/d.clamp_min(1e-3); disp = (disp-disp.min())/(disp.max()-disp.min()+1e-6)
    return TURBO(disp.cpu().numpy())[...,:3]

def render(path, titles_rows, panels_rows, suptitle):
    fig, ax = plt.subplots(3, 6, figsize=(20, 10.5))
    for r,(titles,imgs) in enumerate(zip(titles_rows, panels_rows)):
        for c in range(6):
            ax[r,c].imshow(imgs[c]); ax[r,c].set_box_aspect(1.0)   # square per-camera panel
            if titles[c]: ax[r,c].set_title(titles[c], fontsize=9)
            ax[r,c].axis("off")
    fig.suptitle(suptitle, fontsize=11); fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--da3-v2-probe", type=Path, default=Path("/data/users/junjie/probes_2b/da3_depth_v2_probe.pt"))
    ap.add_argument("--output-dir", type=Path, default=Path("/data/users/junjie/viz_2b_siglip2_split"))
    ap.add_argument("--num-samples", type=int, default=6); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda"); args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True); device=args.device
    ck=args.checkpoint_dir; meta=json.loads((ck/"planner_meta.json").read_text())
    grid=int(meta["grid_size"])
    processor=configure_qwen3vl_processor(AutoProcessor.from_pretrained(str(ck/"processor"),local_files_only=True))
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(ck/"qwen3vl_lora_or_model"),dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(device).eval()
    if hasattr(model.config,"text_config"): model.config.hidden_size=model.config.text_config.hidden_size
    model.config.use_cache=False
    wrapper=T.PlannerWrapper.from_exported_checkpoint(model=model,checkpoint_dir=ck,metadata=meta)
    wrapper.to(device).eval()
    latent_len=(4 if meta.get("independent_modality_task_tokens") else 2)*int(meta["num_task_tokens"])
    plan_sequence=[f"<|sem_plan_{i}|>" for i in range(latent_len)]
    dataset=T.FastWAMOnlinePlannerDataset.from_config(Path(os.environ["FASTWAM_DATA_CONFIG"]),
        dataset_dirs=os.environ["FASTWAM_DATASET_DIRS"].split(":"),
        text_embedding_cache_dir=Path(os.environ["FASTWAM_TEXT_EMBEDDING_CACHE_DIR"]),
        pretrained_norm_stats=Path(os.environ["FASTWAM_PRETRAINED_NORM_STATS"]),max_samples=0,
        offsets=[int(x) for x in meta["keyframe_offsets"]])
    coll=T.Collator(processor=processor,plan_sequence=plan_sequence)
    from siglip2_target import Siglip2TargetEncoder
    from depth_anything3_target import _import_da3, DepthAnything3TargetEncoder
    _skw={"input_size":int(os.environ.get("SIGLIP2_INPUT_SIZE", meta.get("siglip2_input_size") or 256)),
          "grid_size":int(os.environ.get("SIGLIP2_GRID_SIZE", grid)),"device":device}
    if os.environ.get("SIGLIP2_MODEL_DIR"): _skw["model_dir"]=os.environ["SIGLIP2_MODEL_DIR"]
    siglip2=Siglip2TargetEncoder(**_skw)
    da3=DepthAnything3TargetEncoder(process_res=int(os.environ.get("DA3_PROCESS_RES","224")),device=device)
    DA3=_import_da3(os.environ["DA3_CODE_ROOT"]); full=DA3.from_pretrained(os.environ["DA3_CKPT_DIR"]).to(device).eval()
    for p in full.parameters(): p.requires_grad_(False)
    m=full.model
    dprobe=load(args.da3_v2_probe, MiniDPTProbe, device)
    rng=random.Random(args.seed); idxs=rng.sample(range(len(dataset)),min(args.num_samples*4,len(dataset)))
    cf=lambda x:x[0].float().cpu(); saved=0; seen=set()
    for i in idxs:
        if saved>=args.num_samples: break
        s=dataset[i]; instr=str(s.get("instruction") or s.get("prompt") or f"idx{i}"); key=instr.strip()[:60]
        if key in seen: continue
        batch=coll([s]); batch.pop("stems",None)
        kfs=batch.pop("keyframe_images"); cur_img=batch.pop("current_image"); fps=batch.pop("future_video_effective_fps",None)
        md=next(wrapper.model.parameters()).dtype; binp=move_qwen_inputs_to_device(dict(batch),device,model_dtype=md)
        with torch.no_grad():
            plans=wrapper.predict_current_future_plans(**binp)
            cur=cur_img.permute(0,3,1,2).contiguous().to(device); kf=kfs[:,0].permute(0,3,1,2).contiguous().to(device)
            efps=fps[:,0].to(device) if fps is not None else None
            csem_t,fsem_t=siglip2.encode_current_and_future(cur,kf,effective_fps=efps)
            cdep_t,fdep_t=da3.encode_current_and_future(cur,kf)
            cur_gt=gt_turbo(m,da3,cur,device); fut_gt=gt_turbo(m,da3,kf,device)
        csem_p,fsem_p=cf(plans["current_dino"]),cf(plans["future_dino"])   # "dino" = historical semantic-channel name; holds SigLIP2 here
        cdep_p,fdep_p=cf(plans["current_depth"]),cf(plans["future_depth"])
        csem_tt,fsem_tt=cf(csem_t),cf(fsem_t); cdep_tt,fdep_tt=cf(cdep_t),cf(fdep_t)
        csem_mse=F.mse_loss(csem_p,csem_tt).item(); fsem_mse=F.mse_loss(fsem_p,fsem_tt).item()
        cdep_sl1=F.smooth_l1_loss(cdep_p,cdep_tt).item(); fdep_sl1=F.smooth_l1_loss(fdep_p,fdep_tt).item()
        csem_tr,csem_pr=lowres_pca(csem_tt,csem_p,grid); fsem_tr,fsem_pr=lowres_pca(fsem_tt,fsem_p,grid)
        cdep_tr,cdep_pr=depth_v2(dprobe,cdep_tt,cdep_p,device); fdep_tr,fdep_pr=depth_v2(dprobe,fdep_tt,fdep_p,device)
        BLANK=np.ones((224,448,3))
        full_rows_imgs=[
            [to448(to_disp(cur_img[0])), to448(to_disp(kfs[0,0])), to448(csem_tr), to448(csem_pr), to448(fsem_tr), to448(fsem_pr)],
            [to448(to_disp(cur_img[0])), to448(to_disp(kfs[0,0])), to448(cdep_tr), to448(cdep_pr), to448(fdep_tr), to448(fdep_pr)],
            [BLANK, BLANK, to448(cur_gt), BLANK, to448(fut_gt), BLANK],
        ]
        titles=[
            ["Current RGB","Future RGB","SigLIP2 cur TARGET","SigLIP2 cur PRED","SigLIP2 fut TARGET","SigLIP2 fut PRED"],
            ["Current RGB","Future RGB","Depth cur TARGET(v2)","Depth cur PRED(v2)","Depth fut TARGET(v2)","Depth fut PRED(v2)"],
            ["","","DA3-full cur GT","","DA3-full fut GT",""],
        ]
        for cam, sel in (("main", 0), ("wrist", 1)):
            panels=[[halves(p)[sel] for p in row] for row in full_rows_imgs]
            sup=f"[{cam} cam] {instr[:80]}\nsiglip2_mse cur={csem_mse:.4f} fut={fsem_mse:.4f}  |  depth_sl1 cur={cdep_sl1:.3f} fut={fdep_sl1:.3f}"
            render(args.output_dir/f"sample_{saved:02d}_{cam}.png", titles, panels, sup)
        print(f"[{saved}] idx={i} :: {instr[:50]}", flush=True)
        seen.add(key); saved+=1
    print(f"DONE saved {saved*2} PNGs -> {args.output_dir}", flush=True)

if __name__=="__main__": main()
