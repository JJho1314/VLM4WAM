import torch
from torch.utils.data import default_collate
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2.experiments.base import robointer as R

ds = instantiate(R._video_dataset_droid_success_v21_what_where_softlogit)
print("dataset len:", len(ds))
samples=[ds[i] for i in range(8)]
gen=sum(s["ai_caption"].startswith("A Franka robotic arm with a parallel-jaw gripper grasps the [TGT] object") for s in samples)
tgt_present=sum("tgt_token_text" in s for s in samples)
print(f"generic captions: {gen}/8 ; tgt_token_text present: {tgt_present}/8")
for i,s in enumerate(samples[:4]):
    print(f"[{i}] tgt_token_text={s.get('tgt_token_text')!r} tdf={tuple(s['target_dense_feature'].shape)} caption={s['ai_caption'][:55]!r}")
# the bug was here: collate a mixed batch of 4
for b in range(2):
    batch=default_collate(samples[b*4:(b+1)*4])
    assert "tgt_token_text" in batch, "tgt_token_text missing in collated batch"
    print(f"batch{b} collated OK: tgt_token_text={batch['tgt_token_text']} video={tuple(batch['video'].shape)} tdf={tuple(batch['target_dense_feature'].shape)}")
print("SMOKE+COLLATE OK")
