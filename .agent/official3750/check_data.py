"""Fail before training if the full official dataset or its configuration changes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
import yaml
from ge_act.data.robofollow_hdf5_dataset import RoboFollowHDF5Dataset

root = Path('/data/user/jhe724/workspace/VLM4WAM_robofollow_official3750')
config = yaml.safe_load((root / 'ge_act/configs/ltx_model/robofollow/q35_text_p256_official3750_53k_hpc3.yaml').read_text())
args = config['data']['train']
assert config['train_steps'] == 53000
assert config['batch_size'] * config['gradient_accumulation_steps'] * 16 == 128
assert args['action_chunk'] == args['history_action_stride'] == 32
assert config['robofollow_metadata']['history_action_stride'] == 32
assert set(config['data']) == {'train'}
manifest_path = Path(args['manifest_path'])
manifest = json.loads(manifest_path.read_text())
stats = json.loads(Path(args['stat_file']).read_text())
assert stats['manifest_hash'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
assert len(manifest['episodes']) == 3750
assert {r['split'] for r in manifest['episodes']} == {'train'}
counts = Counter(str(Path(r['path']).parent.parent) for r in manifest['episodes'])
assert len(counts) == 75 and set(counts.values()) == {50}
assert len({r['identity'] for r in manifest['episodes']}) == 3750
for row in manifest['episodes']:
    path = Path(row['path'])
    assert path.is_file(), path
    assert (path.parent.parent / 'instructions' / (path.stem + '.json')).is_file(), path
for path in (config['pretrained_model_name_or_path'], config['diffusion_model']['model_path']):
    assert Path(path).exists(), path
dataset = RoboFollowHDF5Dataset(**args)
assert len(dataset) == 3750
# Include a restored episode and one representative from each scene.
indices = {next(i for i, r in enumerate(dataset.records) if r['path'].endswith('scene4_s42_l_pick_yellow_cylinder/data/episode2.hdf5'))}
indices.update(next(i for i, r in enumerate(dataset.records) if '/' + scene + '/' in r['path']) for scene in ('scene1', 'scene2', 'scene3', 'scene4'))
for i in sorted(indices):
    item = dataset[i]
    assert item['caption'] and tuple(item['video'].shape) == (3, 3, 13, 256, 320)
    assert tuple(item['actions'].shape) == (36, 14)
    assert all(torch.isfinite(item[k]).all() for k in ('video', 'actions', 'state'))
print('Verified official 3750/75 x 50 training set, matching statistics, five real samples and 53k/global128 config', flush=True)
