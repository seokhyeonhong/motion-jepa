import pytest
import torch
from experiment.segmentation.model import MotionSegmentationMLPProbe

@pytest.mark.parametrize("size,width",[("tiny",192),("small",256),("base",384)])
@pytest.mark.parametrize("depth",[1,2,3])
def test_width_masking_gradient_and_restore(size,width,depth):
    m=MotionSegmentationMLPProbe(24,120,6,torch.ones(24),torch.ones(24)*2,model_size=size,motion_depth=depth)
    assert m.encoder[0].out_features==width and len(m.encoder)==2*depth
    x=torch.randn(2,2,24,requires_grad=True)
    valid=torch.tensor([[1,0,0,0,0,0],[1,1,1,1,0,0]],dtype=torch.bool);fps=torch.tensor([30.,30.])
    y=m(x,valid,fps);assert y.shape==(2,6,120) and (y[~valid]==0).all()
    dirty=x.detach().clone();dirty[0,1]=float('nan');torch.testing.assert_close(m(dirty,valid,fps),y)
    y.sum().backward();assert torch.isfinite(x.grad).all() and (x.grad[0,1]==0).all()
    restored=MotionSegmentationMLPProbe(24,120,6,torch.zeros(24),torch.ones(24),model_size=size,motion_depth=depth)
    restored.load_state_dict(m.state_dict(),strict=True);torch.testing.assert_close(restored(x,valid,fps),y)
    with pytest.raises(ValueError):m(x,torch.zeros_like(valid),fps)
