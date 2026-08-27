import torch, os
BASE="/data/user/jhe724/workspace/VLM4WAM/eval_prev_iter2000_full/input_datasets"
files={
 "carrot_raw":  f"{BASE}/robointer_74616_yellow_carrot_prompt_targetaware_dataset/target_features_rawseg_ft/74616_exterior_image_1_left.pt",
 "carrot_dense":f"{BASE}/robointer_74616_yellow_carrot_prompt_targetaware_dataset/target_features_instructsam_decoder_dense_stage2_lora_green_leaf_prompt_s20260613/74616_exterior_image_1_left.pt",
 "banana_dense":f"{BASE}/robointer_74616_banana_prompt_targetaware_dataset/target_features_instructsam_decoder_dense_stage2_lora_banana_prompt_s20260613/74616_exterior_image_1_left.pt",
}
def desc(v):
    if torch.is_tensor(v): return f"Tensor{tuple(v.shape)} {v.dtype}"
    if isinstance(v,str): return f"str={v[:80]!r}"
    return f"{type(v).__name__}={v}"
for name,p in files.items():
    print(f"\n===== {name} =====\n{p}\n exists={os.path.exists(p)}")
    if not os.path.exists(p): continue
    d=torch.load(p, map_location="cpu", weights_only=False)
    if isinstance(d,dict):
        for k,v in d.items(): print(f"   {k}: {desc(v)}")
    else:
        print("   (not dict):", desc(d))
# directory listing for masks / images
print("\n===== carrot dataset dir tree (1 level) =====")
cd=f"{BASE}/robointer_74616_yellow_carrot_prompt_targetaware_dataset"
for r in sorted(os.listdir(cd)):
    full=os.path.join(cd,r)
    if os.path.isdir(full):
        sub=os.listdir(full)[:2]
        print(f"   [d] {r}/  ({len(os.listdir(full))} files, e.g. {sub})")
    else:
        print(f"   [f] {r}")
