"""Explicit checkpoint transfer, with no blanket strict=False omissions."""
from pathlib import Path
import hashlib
import json
from safetensors.torch import load_file

def load_robofollow_weights(model,checkpoint: Path,allowed_new_keys: set[str]) -> dict:
    state=load_file(str(checkpoint),device='cpu');target=model.state_dict()
    unexpected=set(state)-set(target)
    if unexpected:raise ValueError(f'unexpected checkpoint keys: {sorted(unexpected)}')
    missing=set(target)-set(state)
    mismatch={k for k in state if state[k].shape!=target[k].shape}
    if mismatch-allowed_new_keys:raise ValueError(f'shape mismatch: {sorted(mismatch-allowed_new_keys)}')
    if missing-allowed_new_keys:raise ValueError(f'missing keys: {sorted(missing-allowed_new_keys)}')
    # Allowed topology changes initialize in the destination; matching source layers still transfer.
    loaded={k:v for k,v in state.items() if k not in mismatch}
    model.load_state_dict(loaded,strict=False)
    topology=[(k,list(v.shape)) for k,v in target.items()]
    return {'checkpoint':str(checkpoint),'loaded':sorted(loaded),'new':sorted(missing|mismatch),'topology_hash':hashlib.sha256(json.dumps(topology).encode()).hexdigest()}

def ensure_vae_channels(vae):
    channels=int(getattr(vae.config,'latent_channels',0))
    if channels<=0:raise ValueError('VAE config lacks valid latent channels')
    if hasattr(vae,'z_dim') and vae.z_dim!=channels:raise ValueError('VAE latent channel aliases disagree')
    vae.z_dim=channels
