from dataclasses import asdict
import pytest
import torch
from torch import nn
from experiment.tmr import model as mm, train

@pytest.mark.parametrize("size,width", [("tiny",192),("small",256),("base",384)])
def test_width_masking_and_gradients(size,width):
    config=mm.AlignmentConfig(8,1536,5,10,model_size=size,motion_encoder="mlp_probe",motion_depth=2)
    head=mm.MLPProbe(config)
    assert head.layers[0].out_features==width
    assert len([x for x in head.modules() if isinstance(x,nn.GELU)])==2
    x=torch.randn(2,5,1536,requires_grad=True);mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
    y=head(x,mask);dirty=x.detach().clone();dirty[~mask]=float('nan')
    torch.testing.assert_close(y,head(dirty,mask))
    torch.testing.assert_close(y.norm(dim=-1),torch.ones(2))
    y[:,0].sum().backward();assert torch.isfinite(x.grad).all();assert (x.grad[~mask]==0).all()
    with pytest.raises(ValueError):head(x,torch.zeros_like(mask))

def test_normalization_and_checkpoint(monkeypatch):
    monkeypatch.setitem(mm.MODEL_SPECS,'base',dict(embed_dim=12,depth=2,num_heads=3))
    cfg=mm.AlignmentConfig(8,6,5,10,motion_encoder='mlp_probe')
    m=mm.TextMotionAlignment(cfg,motion_mean=torch.ones(6),motion_std=torch.ones(6)*2)
    assert len(m.text_encoder.backbone.blocks)==2
    x=torch.randn(2,5,6);mask=torch.ones(2,5,dtype=torch.bool)
    torch.testing.assert_close(m.encode_motion(x,mask),m.motion_encoder((x-1)/2,mask))
    restored=mm.TextMotionAlignment(mm.AlignmentConfig(**asdict(cfg)));restored.load_state_dict(m.state_dict(),strict=True)
    torch.testing.assert_close(restored.encode_motion(x,mask),m.encode_motion(x,mask))


def test_training_saves_depth_and_resumes(tmp_path, monkeypatch):
    from experiment.tmr.test_probing import fixtures
    monkeypatch.setitem(mm.MODEL_SPECS, 'base', dict(embed_dim=12, depth=2, num_heads=3))
    args=train.build_parser().parse_args(['--mode','probe','--feature-source','raw','--probe-architecture','mlp','--motion-depth','3','--output-root',str(tmp_path),'--device','cpu','--iterations','1','--warmup-iterations','0','--eval-interval','1','--checkpoint-interval','1','--batch-size','3','--eval-batch-size','3','--num-workers','0','--no-tensorboard'])
    assert 'motion_depth' not in train.data_configuration(args)
    monkeypatch.setattr(train, 'load_prepared_datasets', lambda **kwargs: fixtures())
    result=train.run(args)
    checkpoint=train.load_alignment_checkpoint(tmp_path/'best.pth.tar')
    assert checkpoint['config']['model']['motion_depth']==3
    assert checkpoint['config']['model']['depth']==2
    restored=mm.TextMotionAlignment(mm.AlignmentConfig(**checkpoint['config']['model']))
    restored.load_state_dict(checkpoint['model'],strict=True)
    assert len(restored.motion_encoder.layers)==6
    assert len(restored.text_encoder.backbone.blocks)==2
    args.resume=True
    assert train.run(args)['test']==result['test']
