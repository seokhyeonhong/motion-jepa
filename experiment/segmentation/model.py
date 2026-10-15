"""Raw-motion frame classification with a Transformer trained from scratch."""

import torch
from torch import nn
from model.motion_transformer_1d import MotionTransformer1D
from model.modules import initialize_transformer
from model.specs import MODEL_SPECS

class MotionSegmentationTransformer(nn.Module):
    def __init__(self, model_size="base", input_dim=366, num_classes=120, num_frames=150):
        super().__init__()
        if model_size not in MODEL_SPECS:
            raise ValueError(f"Unknown model size: {model_size}")
        for name,value in (("input_dim",input_dim),("num_classes",num_classes),("num_frames",num_frames)):
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim,self.num_classes,self.num_frames=input_dim,num_classes,num_frames
        self.config={"model_version":"raw_rope_v1","model_size":model_size,
            **MODEL_SPECS[model_size],"input_dim":input_dim,"num_classes":num_classes,
            "num_frames":num_frames,"position_encoding":"rope","rope_theta":100.,
            "rope_time_scale":1.,"mlp_ratio":4.,"dropout":0.,"pretrained":False}
        self.encoder=MotionTransformer1D(in_chans=input_dim,num_frames=num_frames,
            **MODEL_SPECS[model_size])
        self.head=nn.Linear(MODEL_SPECS[model_size]["embed_dim"],num_classes)
        self.head.apply(initialize_transformer)

    def forward(self,motion,valid_frames,fps):
        if motion.ndim!=3 or motion.shape[1:]!=(self.num_frames,self.input_dim):
            raise ValueError("Motion shape differs from saved architecture")
        if valid_frames.dtype!=torch.bool or valid_frames.shape!=motion.shape[:2]:
            raise ValueError("Expected boolean frame validity mask")
        active=valid_frames.to(motion.device)
        if not active.any(dim=1).all():
            raise ValueError("Each sample must contain a valid frame")
        fps=torch.as_tensor(fps,device=motion.device,dtype=torch.float32)
        if fps.shape!=(motion.shape[0],) or not torch.isfinite(fps).all() or not (fps>0).all():
            raise ValueError("Expected finite positive FPS per sample")
        motion=motion.masked_fill(~active.unsqueeze(-1),0)
        tokens=self.encoder(motion,fps,masks=None,valid_frames=active)
        return self.head(tokens).masked_fill(~active.unsqueeze(-1),0)

    def parameter_counts(self):
        encoder=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head=sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return dict(encoder=encoder,head=head,total=encoder+head)

