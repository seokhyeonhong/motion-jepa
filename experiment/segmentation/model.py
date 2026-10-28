"""Raw-motion frame classification with a Transformer trained from scratch."""

import torch
from torch import nn
from model.motion_transformer_1d import MotionTransformer1D
from model.modules import initialize_transformer
from model.specs import MODEL_SPECS

class MotionSegmentationTransformer(nn.Module):
    def __init__(self, model_size="base", input_dim=366, num_classes=120, num_frames=150, motion_depth=None, patchify=False):
        super().__init__()
        if model_size not in MODEL_SPECS:
            raise ValueError(f"Unknown model size: {model_size}")
        for name,value in (("input_dim",input_dim),("num_classes",num_classes),("num_frames",num_frames)):
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f"{name} must be a positive integer")
        spec = dict(MODEL_SPECS[model_size])
        if motion_depth is not None:
            if isinstance(motion_depth, bool) or not isinstance(motion_depth, int) or motion_depth <= 0:
                raise ValueError("motion_depth must be a positive integer")
            spec["depth"] = motion_depth
        self.input_dim,self.num_classes,self.num_frames=input_dim,num_classes,num_frames
        self.config={"model_version":"raw_rope_v1","model_size":model_size,
            **spec,"input_dim":input_dim,"num_classes":num_classes,
            "num_frames":num_frames,"position_encoding":"rope","rope_theta":100.,
            "rope_time_scale":1.,"mlp_ratio":4.,"dropout":0.,"pretrained":False}
        if not isinstance(patchify, bool):
            raise ValueError("patchify must be a boolean")
        self.patchify = patchify
        encoder_frames = ((num_frames + 2) // 3) * 3 if patchify else num_frames
        if patchify:
            self.config.update(model_version="raw_rope_patch_v1", patchify=True,
                temporal_patch_size=3, encoder_num_frames=encoder_frames,
                patch_validity="any_valid_frame_zero_padded",
                frame_decoder="linear_three_frame_logits")
        self.encoder=MotionTransformer1D(in_chans=input_dim,num_frames=encoder_frames,
            patchify=patchify, **spec)
        self.head=nn.Linear(spec["embed_dim"],num_classes * (3 if patchify else 1))
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
        encoder_active = active
        if self.patchify:
            padding = self.encoder.num_frames - self.num_frames
            motion = torch.nn.functional.pad(motion, (0, 0, 0, padding))
            padded_active = torch.nn.functional.pad(active, (0, padding), value=False)
            # Keep partial patches so every real frame retains supervision.
            # Invalid inputs were zeroed above; fully padded patches stay masked.
            patch_active = padded_active.reshape(len(active), -1, 3).any(dim=-1)
            encoder_active = patch_active.repeat_interleave(3, dim=1)
        tokens=self.encoder(motion,fps,masks=None,valid_frames=encoder_active)
        logits = self.head(tokens)
        if self.patchify:
            logits = logits.reshape(len(active), -1, self.num_classes)[:, :self.num_frames]
        return logits.masked_fill(~active.unsqueeze(-1),0)

    def parameter_counts(self):
        encoder=sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head=sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return dict(encoder=encoder,head=head,total=encoder+head)



class MotionSegmentationProbe(nn.Module):
    """One attention layer and one frame-classification projection."""

    def __init__(self, input_dim, num_classes, num_frames, mean, std, num_heads=6):
        super().__init__()
        from experiment.probing import AttentiveProbe

        self.input_dim, self.num_frames = input_dim, num_frames
        self.probe = AttentiveProbe(input_dim, num_classes, mean, std, num_heads=num_heads, pool=False)
        self.config = {
            "model_version": "attentive_frame_probe", "input_dim": input_dim,
            "num_classes": num_classes, "num_frames": num_frames,
            "num_heads": num_heads, "attention_layers": 1, "output_linear_layers": 1,
            "normalization": "training_feature_mean_std",
        }

    def forward(self, features, valid_frames, fps):
        if features.ndim != 3 or features.shape[1:] != (self.num_frames, self.input_dim):
            raise ValueError("Feature shape differs from saved probe architecture")
        return self.probe(features, valid_frames)

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(encoder=0, probe=total, total=total)


class MotionSegmentationPatchTransformerProbe(nn.Module):
    """Concatenated frozen patch features into one Base-width Transformer."""

    def __init__(self, input_dim, num_classes, num_frames, mean, std):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in (input_dim, num_classes, num_frames)) or num_frames % 3:
            raise ValueError("Positive dimensions and a frame count divisible by three are required")
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.shape != (input_dim,) or std.shape != mean.shape:
            raise ValueError("Feature normalization must match concatenated input width")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Feature normalization must be finite with positive std")
        self.input_dim, self.num_frames, self.num_classes = input_dim, num_frames, num_classes
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        self.encoder = MotionTransformer1D(
            in_chans=input_dim, num_frames=num_frames // 3,
            embed_dim=384, depth=1, num_heads=6, patchify=False,
        )
        self.head = nn.Linear(384, 3 * num_classes)
        self.head.apply(initialize_transformer)
        self.config = {
            "model_version": "jepa_patch_motion_transformer_probe_v1",
            "input_dim": input_dim, "num_frames": num_frames, "num_classes": num_classes,
            "depth": 1, "embed_dim": 384, "num_heads": 6, "mlp_ratio": 4.,
            "dropout": 0., "patchify": False, "feature_patch_size": 3,
            "position_encoding": "rope", "normalization": "training_patch_feature_mean_std",
            "frame_decoder": "linear_three_frame_logits",
            "patch_validity": "any_valid_frame_zero_padded",
        }

    def forward(self, features, valid_frames, fps):
        if features.ndim != 3 or features.shape[1:] != (self.num_frames // 3, self.input_dim):
            raise ValueError("Invalid patch feature shape")
        if valid_frames.dtype != torch.bool or valid_frames.shape != (len(features), self.num_frames):
            raise ValueError("Expected boolean frame validity mask")
        valid_frames = valid_frames.to(features.device)
        active = valid_frames.reshape(len(features), -1, 3).any(-1)
        if not active.any(1).all():
            raise ValueError("Each sample must contain a valid frame")
        fps = torch.as_tensor(fps, device=features.device, dtype=torch.float32)
        if fps.shape != (len(features),) or not torch.isfinite(fps).all() or (fps <= 0).any():
            raise ValueError("Expected finite positive FPS per sample")
        x = features.masked_fill(~active[..., None], 0)
        x = ((x - self.feature_mean) / self.feature_std).masked_fill(~active[..., None], 0)
        # Patch centers are three raw frames apart; a constant RoPE offset cancels.
        x = self.encoder(x, fps / 3, valid_frames=active)
        logits = self.head(x).reshape(len(features), self.num_frames, self.num_classes)
        return logits.masked_fill(~valid_frames[..., None], 0)

    def parameter_counts(self):
        encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return dict(encoder=encoder, head=head, total=encoder + head)


class MotionSegmentationMLPProbe(nn.Module):
    """Token-wise GELU MLP with three-frame patch decoding."""

    def __init__(self, input_dim, num_classes, num_frames, mean, std, model_size="base", motion_depth=None):
        super().__init__()
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
               for v in (input_dim, num_classes, num_frames)) or num_frames % 3:
            raise ValueError("Positive dimensions and a frame count divisible by three are required")
        mean, std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()
        if mean.shape != (input_dim,) or std.shape != mean.shape:
            raise ValueError("Feature normalization must match concatenated input width")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Feature normalization must be finite with positive std")
        self.input_dim, self.num_frames, self.num_classes = input_dim, num_frames, num_classes
        self.register_buffer("feature_mean", mean.clone())
        self.register_buffer("feature_std", std.clone())
        if model_size not in MODEL_SPECS:
            raise ValueError("Invalid model size")
        depth = 2 if motion_depth is None else motion_depth
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
            raise ValueError("motion_depth must be a positive integer")
        width = MODEL_SPECS[model_size]["embed_dim"]
        layers = []
        for index in range(depth):
            layers.extend([nn.Linear(input_dim if index == 0 else width, width), nn.GELU()])
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(width, 3 * num_classes)
        self.apply(initialize_transformer)
        self.config = {
            "model_version": "jepa_patch_mlp_probe_v1",
            "input_dim": input_dim, "num_frames": num_frames, "num_classes": num_classes,
            "depth": depth, "embed_dim": width, "model_size": model_size, "activation": "gelu",
            "dropout": 0., "patchify": False, "feature_patch_size": 3,
            "position_encoding": "none", "normalization": "training_patch_feature_mean_std",
            "frame_decoder": "linear_three_frame_logits",
            "patch_validity": "any_valid_frame_zero_padded",
        }

    def forward(self, features, valid_frames, fps):
        if features.ndim != 3 or features.shape[1:] != (self.num_frames // 3, self.input_dim):
            raise ValueError("Invalid patch feature shape")
        if valid_frames.dtype != torch.bool or valid_frames.shape != (len(features), self.num_frames):
            raise ValueError("Expected boolean frame validity mask")
        valid_frames = valid_frames.to(features.device)
        active = valid_frames.reshape(len(features), -1, 3).any(-1)
        if not active.any(1).all():
            raise ValueError("Each sample must contain a valid frame")
        fps = torch.as_tensor(fps, device=features.device, dtype=torch.float32)
        if fps.shape != (len(features),) or not torch.isfinite(fps).all() or (fps <= 0).any():
            raise ValueError("Expected finite positive FPS per sample")
        x = features.masked_fill(~active[..., None], 0)
        x = ((x - self.feature_mean) / self.feature_std).masked_fill(~active[..., None], 0)
        if not torch.isfinite(x).all():
            raise ValueError("Valid features must be finite")
        x = self.encoder(x)
        logits = self.head(x).reshape(len(features), self.num_frames, self.num_classes)
        return logits.masked_fill(~valid_frames[..., None], 0)

    def parameter_counts(self):
        encoder = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        return dict(encoder=encoder, head=head, total=encoder + head)
