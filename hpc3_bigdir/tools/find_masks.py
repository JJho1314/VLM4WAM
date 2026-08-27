import torch, os, glob
BASE="/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets"
for name,p in [
 ("carrot_raw",f"{BASE}/robointer_74616_yellow_carrot_prompt_targetaware_dataset/target_features_rawseg_ft/74616_exterior_image_1_left.pt"),
 ("banana_raw",f"{BASE}/robointer_74616_banana_prompt_targetaware_dataset/target_features_rawseg_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt"),
]:
    d=torch.load(p,map_location="cpu",weights_only=False)
    mp=d.get("mask_png")
    print(f"\n{name}: mask_png={mp!r}")
    if mp:
        for cand in [mp, os.path.join(os.path.dirname(p),mp), os.path.join("/data/user/jhe724/workspace/cosmos-predict2.5",mp), os.path.join("/data/user/jhe724/workspace/VLM4WAM",mp)]:
            print(f"    exists? {os.path.exists(cand)}  {cand}")
# search for any banana mask png anywhere reasonable
print("\n=== search banana mask pngs ===")
for base in ["/data/user/jhe724/workspace/VLM4WAM","/data/user/jhe724/workspace/cosmos-predict2.5/outputs/tavid_generation_runs"]:
    hits=glob.glob(f"{base}/**/*74616*banana*mask*.png",recursive=True)[:5]+glob.glob(f"{base}/**/*banana*74616*.png",recursive=True)[:5]
    for h in hits: print("  ",h)
