"""Official RoboFollow policy contract for trained GE Act absolute joints."""
from collections import deque
from pathlib import Path
import hashlib
import json
import numpy as np
from ge_act.data.robofollow_schema import normalize,denormalize,validate_stats

def pipeline_state(state):
    state=np.asarray(state,dtype=np.float32)
    if state.shape!=(1,14) or not np.isfinite(state).all():raise ValueError('expected one normalized 14D state')
    return state[None]

class RoboFollowPolicy:
    def __init__(self,infer,stats,instruction_mode='correct',instruction_pool=None,actions_per_step=50,action_budget=None):
        validate_stats(stats)
        if not isinstance(actions_per_step,int) or actions_per_step<=0:raise ValueError('actions_per_step must be a positive integer')
        if action_budget is not None and (not isinstance(action_budget,int) or action_budget<=0):raise ValueError('action_budget must be a positive integer')
        self.actions_per_step=actions_per_step;self.action_budget=action_budget
        if instruction_mode not in ('correct','shuffle','empty'):raise ValueError('unknown instruction mode')
        self.infer=infer;self.stats=stats;self.instruction_mode=instruction_mode;self.instruction_pool=sorted(set(instruction_pool or []))
        if instruction_mode=='shuffle' and len(self.instruction_pool)<2:raise ValueError('shuffle requires an independent training instruction pool')
        self.reset()
    def reset(self):
        self.history=deque(maxlen=4);self.instruction=None;self.executed_actions=0
    def set_instruction(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError('instruction must be nonempty')
        if text!=self.instruction:self.history.clear();self.executed_actions=0
        self.instruction=text
    def predict(self,observation):
        if self.instruction is None:raise ValueError('set_instruction required')
        count=self.actions_per_step
        if self.action_budget is not None:
            count=min(count,self.action_budget-self.executed_actions)
            if count<=0:raise ValueError('episode action budget exhausted')
        try:
            rgb=[np.asarray(observation['observation'][cam]['rgb']) for cam in ('head_camera','left_camera','right_camera')]
            state=np.asarray(observation['joint_action']['vector'],dtype=np.float32)
        except (KeyError,TypeError) as e:raise ValueError('missing public RGB/state observation') from e
        if state.shape!=(14,) or not np.isfinite(state).all() or any(x.ndim!=3 or x.shape[-1]!=3 or x.dtype!=np.uint8 for x in rgb) or len({x.shape for x in rgb})!=1:raise ValueError('invalid RGB/state observation')
        self.history.append(np.stack(rgb).copy())
        history=[self.history[0]]*(4-len(self.history))+list(self.history)
        text=self.instruction
        if self.instruction_mode=='empty':text=''
        elif self.instruction_mode=='shuffle':
            pool=[x for x in self.instruction_pool if x!=text]
            text=pool[int(hashlib.sha256(text.encode()).hexdigest()[:8],16)%len(pool)]
        pred=np.asarray(self.infer(np.stack(history),normalize(state[None],self.stats,'state'),text),dtype=np.float32)
        if pred.ndim!=2 or pred.shape[1]!=14 or not len(pred) or not np.isfinite(pred).all():raise ValueError('inference must return finite normalized [T,14]')
        # The final evaluation call is clipped to preserve the 500-action budget.
        result=denormalize(pred[:count],self.stats,'action').astype(np.float32)
        if not np.isfinite(result).all():raise ValueError('nonfinite absolute targets')
        self.executed_actions+=len(result)
        return result
    def close(self):
        self.history.clear()


def make_policy(config_path,checkpoint_path,stats_path,planner_mode='disabled',instruction_mode='correct',manifest_path=None,device='cuda'):
    import torch
    import torch.nn.functional as F
    import yaml
    from transformers import T5Tokenizer,T5EncoderModel
    from diffusers import FlowMatchEulerDiscreteScheduler
    from ge_act.models.ltx_models.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    from ge_act.models.ltx_models.transformer_ltx_multiview import LTXVideoTransformer3DModel
    from ge_act.models.pipeline.custom_pipeline import CustomPipeline
    from ge_act.utils.model_utils import load_condition_models,load_latent_models,resolve_checkpoint_files
    from ge_act.experiments.robofollow_loading import load_robofollow_weights,ensure_vae_channels,semantic_contract,validate_contract
    if planner_mode not in ('disabled','predicted'):raise ValueError('deployment cannot request teacher plans')
    c=yaml.safe_load(Path(config_path).read_text());s=json.loads(Path(stats_path).read_text());validate_stats(s)
    md=c['robofollow_metadata']
    dc=c['data']['train'];action_chunk=dc['action_chunk']
    actions_per_step=md['history_action_stride']
    if actions_per_step!=min(50,action_chunk):raise ValueError('policy history stride differs from action execution')
    action_budget=c.get('robofollow_evaluation',{}).get('action_budget')
    if md['action_type']!='absolute' or md['action_dim']!=14 or md['camera_order']!=s['camera_order']:raise ValueError('policy config metadata mismatch')
    files=resolve_checkpoint_files(checkpoint_path)
    if len(files)!=1:raise ValueError('one explicit checkpoint required')
    validate_contract(files[0],semantic_contract(c,stats_path))
    dtype=torch.bfloat16;weights=c['pretrained_model_name_or_path']
    cond=load_condition_models(T5Tokenizer,T5EncoderModel,weights,load_weights=True)
    textenc=cond['text_encoder'].to(device,dtype=dtype).eval().requires_grad_(False)
    vae=load_latent_models(AutoencoderKLLTXVideo,weights)['vae'].to(device,dtype=dtype).eval().requires_grad_(False);ensure_vae_channels(vae)
    if c.get('enable_slicing'):vae.enable_slicing()
    model=LTXVideoTransformer3DModel(**c['diffusion_model']['config']).to(device,dtype=dtype)
    planner=None
    if planner_mode=='predicted':
        from qwen35_baton.robofollow_provider import RoboFollowResearchPlanner
        pc=c['robofollow_planner'];planner=RoboFollowResearchPlanner.from_checkpoint(pc['checkpoint'],pc['qwen_path'],pc['siglip_path'],device=device)
        if pc.get('joint',False):
            from qwen35_baton.research_provider import ResearchPlannerHead
            model.research_planner_head=ResearchPlannerHead(planner.base.planner).to(device,dtype=dtype)
            planner.head=model.research_planner_head
    elif c.get('robofollow_planner',{}).get('enabled',False):raise ValueError('trained planner policy must use predicted source')
    files=resolve_checkpoint_files(checkpoint_path)
    if len(files)!=1:raise ValueError('RoboFollow first-stage policy requires one explicit safetensors checkpoint')
    # A deployment checkpoint must contain all action layers; no random head allowance here.
    load_robofollow_weights(model,files[0],set());model.eval().requires_grad_(False)
    pipe=CustomPipeline(scheduler=FlowMatchEulerDiscreteScheduler(),vae=vae,text_encoder=textenc,tokenizer=cond['tokenizer'],transformer=model).to(device)
    pipe.set_progress_bar_config(disable=True)
    height,width=c['data']['train']['sample_size']
    @torch.no_grad()
    def infer(history,state,text):
        x=torch.from_numpy(history).to(device).permute(1,4,0,2,3).float()/127.5-1
        flat=x.permute(0,2,1,3,4).reshape(12,3,*x.shape[-2:])
        if c['data']['train'].get('resize_mode','resize')=='pad':
            from ge_act.data.robofollow_schema import pad_frames
            if tuple(flat.shape[-2:])!=tuple(c['data']['train'].get('native_size',(240,320))):raise ValueError('observation size differs from training native_size')
            flat=pad_frames(flat,(height,width))
        else:flat=F.interpolate(flat,size=(height,width),mode='bilinear',align_corners=False)
        x=flat.reshape(3,4,3,height,width).permute(0,2,1,3,4).to(dtype)
        args={}
        if planner is not None:
            current=torch.from_numpy(history[-1:]).to(device)
            tokens=planner.predict_tokens(current,[text])
            from ge_act.models.ltx_models.semantic_conditioning import build_semantic_plan_times
            args={'semantic_plan':tokens,'semantic_plan_times':build_semantic_plan_times(batch_size=1,n_view=3,n_previous=4,num_future_frames=9,num_latent_frames=6,indices=(0,3,5,8),device=device),'semantic_condition_mask':torch.ones(3,device=device,dtype=dtype)}
        model.eval()
        pred=pipe.infer(image=x,n_prev=4,prompt=text,height=height,width=width,chunk=2,n_view=3,return_action=True,return_video=False,action_chunk=action_chunk,action_dim=14,history_action_state=torch.from_numpy(pipeline_state(state)).to(device,dtype=dtype),num_inference_steps=c['num_inference_step'],guidance_scale=1.0,noise_seed=42,n_chunk=1,frame_rate=dc['source_fps']*dc['chunk']/action_chunk,**args)[0]['action']
        return pred[0].float().cpu().numpy()
    pool=[]
    if manifest_path:
        m=json.loads(Path(manifest_path).read_text());pool=[t for r in m['episodes'] if r['split']=='train' for t in r['instructions']]
    return RoboFollowPolicy(infer,s,instruction_mode,pool,actions_per_step,action_budget)
