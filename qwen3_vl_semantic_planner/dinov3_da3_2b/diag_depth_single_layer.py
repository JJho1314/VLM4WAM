#!/usr/bin/env python3
"""Step-1 diagnostic: quantify the single-last-layer depth-probe bottleneck.

GT            = DA3 FULL model depth (4-layer DPT: backbone all out_layers -> _process_depth_head).
probe_real    = da3_depth_probe fed the REAL frame's LAST-layer DA3 features (aligned space).
probe_pred    = da3_depth_probe fed the planner's PREDICTED da3 features.
All compared as per-frame normalized disparity (the probe's target space).
  probe_real vs GT       -> single-last-layer CEILING (probe/1-of-4-layer cost, planner-independent)
  probe_pred vs probe_real -> planner prediction gap
"""
from __future__ import annotations
import os, sys, json, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE,"dinov3_da3_2b"), os.path.join(HERE,"lingbot_dino_4b")):
    sys.path.insert(0, p)
import train_qwen3vl4b_lingbot_dino_planner as T
from qwen3vl_wrapper import move_qwen_inputs_to_device, configure_qwen3vl_processor
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from train_feature_probes import ProbeDecoder, depth_to_disp

def metrics(pred, gt, eps=1e-3):
    pred, gt = pred.flatten().float(), gt.flatten().float()
    absrel = (torch.abs(pred-gt)/(gt+eps)).mean().item()
    rmse = torch.sqrt(((pred-gt)**2).mean()).item()
    corr = torch.corrcoef(torch.stack([pred,gt]))[0,1].item()
    return absrel, rmse, corr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--da3-probe", type=Path, default=Path("/data/users/junjie/probes_2b/da3_depth_probe.pt"))
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(); device="cuda"
    ck=args.checkpoint_dir; meta=json.loads((ck/"planner_meta.json").read_text())
    processor=configure_qwen3vl_processor(AutoProcessor.from_pretrained(str(ck/"processor"),local_files_only=True))
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(ck/"qwen3vl_lora_or_model"),dtype=torch.bfloat16,attn_implementation="sdpa",local_files_only=True).to(device).eval()
    if hasattr(model.config,"text_config"): model.config.hidden_size=model.config.text_config.hidden_size
    model.config.use_cache=False
    _o=torch.allclose; torch.allclose=lambda *a,**k:True
    try: wrapper=T.PlannerWrapper.from_exported_checkpoint(model=model,checkpoint_dir=ck,metadata=meta)
    finally: torch.allclose=_o
    wrapper.to(device).eval()
    latent_len=(4 if meta.get("independent_modality_task_tokens") else 2)*int(meta["num_task_tokens"])
    plan_sequence=[f"<|sem_plan_{i}|>" for i in range(latent_len)]
    dataset=T.FastWAMOnlinePlannerDataset.from_config(Path(os.environ["FASTWAM_DATA_CONFIG"]),
        dataset_dirs=os.environ["FASTWAM_DATASET_DIRS"].split(":"),
        text_embedding_cache_dir=Path(os.environ["FASTWAM_TEXT_EMBEDDING_CACHE_DIR"]),
        pretrained_norm_stats=Path(os.environ["FASTWAM_PRETRAINED_NORM_STATS"]),max_samples=0,
        offsets=[int(x) for x in meta["keyframe_offsets"]])
    coll=T.Collator(processor=processor,plan_sequence=plan_sequence)
    from depth_anything3_target import _import_da3, DepthAnything3TargetEncoder
    da3=DepthAnything3TargetEncoder(process_res=int(os.environ.get("DA3_PROCESS_RES","224")),device=device)
    DA3=_import_da3(os.environ["DA3_CODE_ROOT"]); full=DA3.from_pretrained(os.environ["DA3_CKPT_DIR"]).to(device).eval()
    for p in full.parameters(): p.requires_grad_(False)
    m=full.model
    pk=torch.load(args.da3_probe,map_location="cpu",weights_only=False)
    probe=ProbeDecoder(**pk["config"]).to(device).eval(); probe.load_state_dict(pk["state_dict"])

    @torch.no_grad()
    def gt_disp(fr_b3hw):
        x=da3._prep(fr_b3hw).to(next(m.parameters()).dtype).unsqueeze(1)
        feats,_=m.backbone(x,cam_token=None,export_feat_layers=[],ref_view_strategy="saddle_balanced")
        with torch.autocast(device_type="cuda",enabled=False):
            out=m._process_depth_head(feats,224,224)
        d=(out["depth"] if hasattr(out,"keys") else out.depth).float()
        return depth_to_disp(d)[0,0]  # [224,224]

    rng=random.Random(args.seed); idxs=rng.sample(range(len(dataset)),min(args.num_samples*4,len(dataset)))
    rows=[]; seen=set()
    for i in idxs:
        if len(rows)>=args.num_samples: break
        s=dataset[i]; instr=str(s.get("instruction") or s.get("prompt") or f"idx{i}")[:40]; k=instr
        if k in seen: continue
        # (dedup by task text; falls back to idx when text is absent)
        batch=coll([s]); batch.pop("stems",None)
        kfs=batch.pop("keyframe_images"); cur_img=batch.pop("current_image"); fps=batch.pop("future_video_effective_fps",None)
        md=next(wrapper.model.parameters()).dtype
        binp=move_qwen_inputs_to_device(dict(batch),device,model_dtype=md)
        with torch.no_grad():
            plans=wrapper.predict_current_future_plans(**binp)
            cur=cur_img.permute(0,3,1,2).contiguous().to(device)
            cdep_t,_=da3.encode_current_and_future(cur, kfs[:,0].permute(0,3,1,2).contiguous().to(device))
            gt=gt_disp(cur)                                   # [224,224]
            pr_real=probe(cdep_t.float())[0,0]                # probe on REAL last-layer feats
            pr_pred=probe(plans["current_depth"].float())[0,0]# probe on PRED feats
        # renormalize each to [0,1] for fair comparison
        def nz(a): 
            return (a-a.min())/(a.max()-a.min()+1e-6)
        gt,pr_real,pr_pred=nz(gt),nz(pr_real),nz(pr_pred)
        m_ceil=metrics(pr_real,gt); m_plan=metrics(pr_pred,pr_real); m_e2e=metrics(pr_pred,gt)
        rows.append((instr,m_ceil,m_plan,m_e2e)); seen.add(k)
    def avg(j,sub):
        import statistics; return statistics.mean(r[j][sub] for r in rows)
    print("\n================ DEPTH DIAGNOSTIC (normalized disparity) ================")
    print(f"{'sample':<42}{'CEIL AbsRel/RMSE/corr':<30}{'PLAN gap':<26}{'E2E vs GT'}")
    for instr,c,p,e in rows:
        print(f"{instr:<42}{c[0]:.3f}/{c[1]:.3f}/{c[2]:.2f}      {p[0]:.3f}/{p[1]:.3f}/{p[2]:.2f}   {e[0]:.3f}/{e[1]:.3f}/{e[2]:.2f}")
    print("-"*100)
    print(f"{'MEAN':<42}{avg(1,0):.3f}/{avg(1,1):.3f}/{avg(1,2):.2f}      {avg(2,0):.3f}/{avg(2,1):.3f}/{avg(2,2):.2f}   {avg(3,0):.3f}/{avg(3,1):.3f}/{avg(3,2):.2f}")
    print("CEIL = probe(real last-layer) vs GT(full 4-layer DA3): the single-layer ceiling.")
    print("PLAN = probe(pred) vs probe(real): planner prediction gap.  E2E = probe(pred) vs GT.")

if __name__=="__main__": main()
