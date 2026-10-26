"""Motion-only depth preserves text architecture and checkpoint compatibility."""
from dataclasses import asdict
import pytest
import torch
from experiment.tmr import model as mm, train

@pytest.mark.parametrize('depth', [3, 6, 9, 12])
def test_motion_depth_and_checkpoint_roundtrip(depth, monkeypatch):
    monkeypatch.setitem(mm.MODEL_SPECS, 'base', dict(embed_dim=12, depth=12, num_heads=3))
    torch.manual_seed(42)
    baseline = mm.TextMotionAlignment(mm.AlignmentConfig(8, 6, 6, 30))
    torch.manual_seed(42)
    config = mm.AlignmentConfig(8, 6, 6, 30, motion_encoder='transformer_probe', motion_depth=depth)
    model = mm.TextMotionAlignment(config, motion_mean=torch.ones(6), motion_std=torch.ones(6)*2)
    assert len(model.motion_encoder.backbone.blocks) == depth
    assert len(model.text_encoder.backbone.blocks) == 12
    for name, value in baseline.text_encoder.state_dict().items():
        torch.testing.assert_close(value, model.text_encoder.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(2,6,6)
    valid = torch.tensor([[1,1,1,1,0,0],[1,1,1,1,1,1]],dtype=torch.bool)
    y = model.encode_motion(x,valid)
    assert y.shape == (2,256) and torch.isfinite(y).all()
    y[:,0].sum().backward()
    assert model.motion_encoder.backbone.input_proj.weight.grad is not None
    restored=mm.TextMotionAlignment(mm.AlignmentConfig(**asdict(config)))
    restored.load_state_dict(model.state_dict(),strict=True)
    torch.testing.assert_close(restored.encode_motion(x,valid),y)

@pytest.mark.parametrize('depth',[0,-1,True,1.5])
def test_invalid_depth(depth):
    with pytest.raises(ValueError,match='motion_depth'):
        mm.AlignmentConfig(8,6,6,30,motion_depth=depth)

def test_attentive_rejects_depth_before_precompute():
    args=train.build_parser().parse_args(['--mode','probe','--feature-source','raw','--motion-depth','3'])
    with pytest.raises(ValueError,match='probe-architecture'):
        train._validate_args(args)

def test_default_checkpoint_compatibility(monkeypatch):
    monkeypatch.setitem(mm.MODEL_SPECS,'base',dict(embed_dim=12,depth=2,num_heads=3))
    torch.manual_seed(42)
    old=mm.TextMotionAlignment(mm.AlignmentConfig(8,6,6,30))
    torch.manual_seed(42)
    explicit=mm.TextMotionAlignment(mm.AlignmentConfig(8,6,6,30,motion_depth=2))
    for k,v in old.state_dict().items():torch.testing.assert_close(v,explicit.state_dict()[k],rtol=0,atol=0)


def test_training_saves_depth_and_resumes(tmp_path, monkeypatch):
    from experiment.tmr.test_probing import fixtures
    monkeypatch.setitem(mm.MODEL_SPECS, 'base', dict(embed_dim=12, depth=2, num_heads=3))
    args=train.build_parser().parse_args(['--mode','probe','--feature-source','raw','--probe-architecture','transformer','--motion-depth','3','--output-root',str(tmp_path),'--device','cpu','--iterations','1','--warmup-iterations','0','--eval-interval','1','--checkpoint-interval','1','--batch-size','3','--eval-batch-size','3','--num-workers','0','--no-tensorboard'])
    assert 'motion_depth' not in train.data_configuration(args)
    monkeypatch.setattr(train, 'load_prepared_datasets', lambda **kwargs: fixtures())
    result=train.run(args)
    checkpoint=train.load_alignment_checkpoint(tmp_path/'best.pth.tar')
    assert checkpoint['config']['model']['motion_depth']==3
    assert checkpoint['config']['model']['depth']==2
    restored=mm.TextMotionAlignment(mm.AlignmentConfig(**checkpoint['config']['model']))
    restored.load_state_dict(checkpoint['model'],strict=True)
    assert len(restored.motion_encoder.backbone.blocks)==3
    assert len(restored.text_encoder.backbone.blocks)==2
    args.resume=True
    assert train.run(args)['test']==result['test']
