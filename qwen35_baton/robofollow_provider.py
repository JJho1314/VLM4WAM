"""Explicit per-view migration of a row-independent two-view Baton planner."""
import torch
import torch.nn.functional as F
from ge_act.data.robofollow_schema import CAMERA_ORDER

KIND='qwen35_baton_robofollow_research'

def validate_metadata(m):
    if m.get('kind')!=KIND or m.get('version')!=1 or m.get('camera_order')!=CAMERA_ORDER or m.get('keyframes')!=[0,3,5,8] or m.get('feature_dim')!=1024 or not m.get('source_checkpoint'):
        raise ValueError('incompatible RoboFollow planner metadata')

class RoboFollowResearchPlanner:
    def __init__(self,base,*,source_checkpoint,head=None):
        self.base=base;self.wrapper=getattr(base,'planner',None);self.head=head
        self.metadata={'kind':KIND,'version':1,'camera_order':CAMERA_ORDER,'keyframes':[0,3,5,8],'feature_dim':1024,'source_checkpoint':str(source_checkpoint),'migration':'independent rows; duplicate each view into legacy pair, select first output'}
        validate_metadata(self.metadata)
    @classmethod
    def from_checkpoint(cls,checkpoint,qwen_path,siglip_path,device='cuda'):
        from qwen35_baton.provider import FrozenBatonPlanner
        base=FrozenBatonPlanner.from_checkpoint(checkpoint,qwen_model_path=qwen_path,qwen_tokenizer_path=qwen_path,qwen_processor_path=qwen_path,siglip2_model_path=siglip_path,device=device)
        return cls(base,source_checkpoint=checkpoint)
    def predict_tokens(self,images,instructions):
        if images.ndim!=5 or images.shape[1]!=3 or images.shape[-1]!=3 or images.dtype!=torch.uint8 or len(instructions)!=len(images) or not all(isinstance(t,str) and t.strip() for t in instructions):
            raise ValueError('expected [B,3,H,W,3] uint8 with one nonempty instruction per sample')
        b=images.shape[0]
        rgb=images.permute(0,1,4,2,3).reshape(b*3,3,*images.shape[2:4])
        rgb=F.interpolate(rgb.float(),size=(256,256),mode='bilinear',align_corners=False).round().clamp(0,255).to(torch.uint8)
        pair=rgb[:,None].expand(-1,2,-1,-1,-1).contiguous()
        texts=[t for t in instructions for _ in range(3)]
        if self.head is None:
            with torch.no_grad():pred=self.base.predict(pair,texts).tokens
        else:
            from qwen35_baton.research_provider import predict_with_research_head
            pred=predict_with_research_head(self.base,self.head,pair,texts)
        expected=(b*3,2,4,256,1024)
        if tuple(pred.shape)!=expected or not torch.isfinite(pred).all():raise ValueError('invalid migrated planner output')
        return pred[:,0].reshape(b,3,4,256,1024)


def mix_training_plans(predicted,teacher,predicted_probability,anchor_weight):
    if not 0<=predicted_probability<=1 or anchor_weight<0:raise ValueError('invalid mixing/anchor')
    if (predicted_probability<1 or anchor_weight>0) and teacher is None:raise ValueError('training teacher required')
    if teacher is not None and teacher.shape!=predicted.shape:raise ValueError('teacher shape mismatch')
    loss=None
    if anchor_weight>0:loss=anchor_weight*F.mse_loss(predicted.float(),teacher.detach().float())
    if predicted_probability==1:return predicted,loss
    mask=torch.rand((len(predicted),1,1,1,1),device=predicted.device)<predicted_probability
    return torch.where(mask,predicted,teacher.detach().to(predicted)),loss
