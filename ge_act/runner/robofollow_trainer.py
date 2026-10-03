"""RoboFollow opt-in runner: preserve generic and strict dual-camera paths."""
from pathlib import Path
import json
from runner.ge_trainer import Trainer
from ge_act.experiments.robofollow_loading import load_robofollow_weights, ensure_vae_channels

class RoboFollowTrainer(Trainer):
    def prepare_models(self):
        original=getattr(self.args,'load_diffusion_model_weights',True)
        self.args.load_diffusion_model_weights=False
        semantic=self.args.semantic_plan
        self.args.semantic_plan=dict(semantic,enabled=False)
        try:super().prepare_models()
        finally:
            self.args.load_diffusion_model_weights=original
            self.args.semantic_plan=semantic
        pc=self.raw_config.get('robofollow_planner',{})
        self.rf_teacher=None
        if pc.get('enabled',False):
            from qwen35_baton.robofollow_provider import RoboFollowResearchPlanner
            self.rf_planner=RoboFollowResearchPlanner.from_checkpoint(pc['checkpoint'],pc['qwen_path'],pc['siglip_path'],device=self.state.accelerator.device)
            if pc['joint']:
                from qwen35_baton.research_provider import ResearchPlannerHead
                self.diffusion_model.research_planner_head=ResearchPlannerHead(self.rf_planner.base.planner).to(self.state.accelerator.device)
                self.rf_planner.head=self.diffusion_model.research_planner_head
            if pc['predicted_probability']<1 or pc['anchor_weight']>0:
                from ge_act.models.ltx_models.semantic_conditioning import OnlineSiglip2SemanticEncoder
                self.rf_teacher=OnlineSiglip2SemanticEncoder(pc['siglip_path'],device=self.state.accelerator.device,frame_microbatch_size=3)
            if self.state.accelerator.is_main_process:
                (Path(self.save_folder)/'planner_metadata.json').write_text(json.dumps(self.rf_planner.metadata,indent=2)+'\n')
        ensure_vae_channels(self.vae)
        if self.args.load_weights and original:
            allowed=set(self.raw_config.get('robofollow_allowed_new_keys',[]))
            if pc.get('joint',False):allowed.update(k for k in self.diffusion_model.state_dict() if k.startswith('research_planner_head.'))
            report=load_robofollow_weights(self.diffusion_model,Path(self.args.diffusion_model['model_path']),allowed)
            if self.state.accelerator.is_main_process:
                (Path(self.save_folder)/'weight_transfer.json').write_text(json.dumps(report,indent=2)+'\n')

    def robofollow_condition(self,video,instructions,n_previous):
        import torch
        from qwen35_baton.robofollow_provider import mix_training_plans
        from ge_act.models.ltx_models.semantic_conditioning import build_semantic_plan_times
        pc=self.raw_config.get('robofollow_planner',{})
        if not pc.get('enabled',False):return None,None,None
        current=((video[:,:,:,n_previous-1].permute(0,2,3,4,1)+1)*127.5).round().clamp(0,255).to(torch.uint8)
        pred=self.rf_planner.predict_tokens(current,instructions)
        teacher=None
        if self.rf_teacher is not None:
            future=video[:,:,:,n_previous:].permute(0,2,3,1,4,5)[:,:,list((0,3,5,8))]
            teacher=self.rf_teacher.encode(future)
        tokens,loss=mix_training_plans(pred,teacher,pc['predicted_probability'],pc['anchor_weight'])
        times=build_semantic_plan_times(batch_size=len(video),n_view=3,n_previous=n_previous,num_future_frames=9,num_latent_frames=6,indices=(0,3,5,8),device=video.device)
        return tokens.to(video.dtype),times,loss

    def prepare_trainable_parameters(self):
        super().prepare_trainable_parameters()
        self.rf_gradient_report={}
        head=getattr(self.diffusion_model,'research_planner_head',None)
        if head is not None:
            for name,param in head.named_parameters():
                if not param.requires_grad:raise ValueError('joint planner head is unexpectedly frozen')
                if name.endswith('weight'):
                    def report(grad,name=name):
                        import torch
                        finite=bool(torch.isfinite(grad).all())
                        if not finite:raise ValueError('nonfinite joint head gradient '+name)
                        self.rf_gradient_report[name]=dict(finite=finite,norm=float(grad.float().norm()))
                        return grad
                    param.register_hook(report)

    def train(self):
        try:return super().train()
        finally:
            if self.raw_config.get('robofollow_planner',{}).get('joint',False) and self.state.accelerator.is_main_process:
                frozen=all(not p.requires_grad for p in self.rf_planner.base.planner.parameters())
                (Path(self.save_folder)/'joint_gradient_report.json').write_text(json.dumps(dict(qwen_frozen=frozen,head_gradients=self.rf_gradient_report),indent=2)+'\n')
