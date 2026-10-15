"""Raw-motion segmentation with the exact JEPA frame encoder, trained from scratch."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn
from model.motion_transformer_1d import MotionTransformer1D
from model.modules import initialize_transformer
from model.specs import MODEL_SPECS

class JEPASegmentationTransformer(nn.Module):
    def __init__(self, model_size="base", input_dim=366, num_classes=120, num_frames=150):
        super().__init__()
        if model_size not in MODEL_SPECS:
            raise ValueError(f"Unknown JEPA size: {model_size}")
        for name,value in (("input_dim",input_dim),("num_classes",num_classes),("num_frames",num_frames)):
            if isinstance(value,bool) or not isinstance(value,int) or value<=0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim,self.num_classes,self.num_frames=input_dim,num_classes,num_frames
        self.config={"model_version":"jepa_rope_v1","model_size":model_size,
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

def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size",choices=list(MODEL_SPECS),default="base")
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--describe",action="store_true",help="Print architecture and counts without loading data or training")
    return parser

def main():
    args=build_parser().parse_args()
    if args.describe:
        model=JEPASegmentationTransformer(args.model_size)
        print(json.dumps({"architecture":model.config,"trainable_parameters":model.parameter_counts()},indent=2))
        return
    root=Path("/source/seokhyeon/motion-jepa/motion-jepa")
    prior=root/"findings/022-segmentation-supervised-comparison"
    sys.path[:0]=[str(root),str(prior),str(root/"findings/019-ema99925-segmentation-probe-sweep")]
    spec=importlib.util.spec_from_file_location("jepa_sized_segmentation_engine",prior/"run.py")
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    output=args.output_root.expanduser().resolve()
    if output==prior.resolve() or prior.resolve() in output.parents:
        raise ValueError("Use a separate output directory; previous findings are immutable")
    output.mkdir(parents=True,exist_ok=True)
    engine.OUT=output
    model_config={"model_size":args.model_size,**MODEL_SPECS[args.model_size],
        "version":"jepa_rope_v1","position_encoding":"rope","rope_theta":100.,
        "rope_time_scale":1.,"mlp_ratio":4.,"dropout":0.}
    engine.MODEL_CONFIGS={"transformer":model_config}
    engine.build_model=lambda name,**dims:JEPASegmentationTransformer(args.model_size,**dims)
    engine.configure()
    engine.dataset_cache.RUN=root/"output/mot_base_1d-bs.512-ep.300-ema99925"
    engine.dataset_cache.PATCH_RUN=root/"output/mot_patch_base_1d-p3-bs.512-ep.300-nframes150-ema99925"
    original_record=engine.dataset_cache._record
    def record_preserving_alias(path):
        record=original_record(path)
        record["path"]=str(Path(path))
        return record
    engine.dataset_cache._record=record_preserving_alias
    manifest=engine.dataset_cache.prepare()
    split=engine.read(engine.PRIOR/"split.json")
    settings={"models":engine.MODEL_CONFIGS,"seed":42,"batch_size":256,
        "deep_learning_rates":engine.DEEP_RATES,"deep_weight_decays":engine.DECAYS,
        "epochs_per_trial":100,"warmup":5,"eval_every":5,
        "objective":"masked_unweighted_frame_BCE","optimizer":"AdamW",
        "selection":"source_disjoint_inner_BABEL120_mAP_LR_WD_epoch_then_full_train_refit",
        "normalization":"original_pretraining_stats_then_partition_fit_channel_zscore_valid_frames",
        "data_manifest_sha256":engine.sha(prior/"data-manifest.json"),
        "split_sha256":engine.sha(engine.PRIOR/"split.json"),
        "source_hashes":{str(x):engine.sha(x) for x in (
            Path(__file__),prior/"run.py",prior/"data.py",root/"model/specs.py",
            root/"model/motion_transformer_1d.py",root/"model/modules.py",root/"model/pos_embs.py")}}
    config_path=output/"experiment.json"
    if config_path.exists() and engine.read(config_path)!=settings:
        raise ValueError("Resume architecture, recipe, or source provenance changed")
    engine.probe.atomic_json(settings,config_path)
    result=engine.fit_one("transformer",settings,split,args.device)
    d=MODEL_SPECS[args.model_size]["embed_dim"]
    head_parameters=(d+1)*120
    result["parameter_counts"]={"encoder":result["parameters"]-head_parameters,
        "head":head_parameters,"total":result["parameters"]}
    engine.probe.atomic_json(result,output/"transformer/summary.json")
    engine.probe.atomic_json({"settings":settings,"models":[result]},output/"summary.json")

if __name__=="__main__":
    main()
