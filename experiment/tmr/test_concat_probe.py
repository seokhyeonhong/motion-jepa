"""Ordered TMR layer fusion and backward-compatible single-layer contracts."""
from types import SimpleNamespace
import pytest
import torch
from experiment import probing as shared
from experiment.tmr import probing, train
from experiment.tmr.test_probing import fixtures
from model.motion_transformer_1d import MotionTransformer1D


@pytest.mark.parametrize("patch", [False, True])
def test_fusion_matches_single_layers_and_cleans_hooks(tmp_path, monkeypatch, patch):
    def factory(in_chans, num_frames, patchify):
        return MotionTransformer1D(in_chans=in_chans, num_frames=num_frames,
                                   embed_dim=12, depth=4, num_heads=2, patchify=patchify)
    monkeypatch.setattr(shared.motion_transformer_1d, "mot_tmr_concat_test", factory, raising=False)
    ck = tmp_path / "encoder.pth"
    torch.save({"config": {"data": {"motion_dim": 6, "num_frames": 6, "normalize": False},
                            "meta": {"model_name": "mot_tmr_concat_test", "patchify": patch}},
                "target_encoder": factory(6,6,patch).state_dict()}, ck)
    ex = probing.FrozenLayersExtractor(ck, [3,1,4], device="cpu")
    x = torch.randn(2,6,6); lengths = torch.tensor([4,6]); fps = torch.tensor([30.,30.])
    calls = []
    h = ex.encoder.blocks[0].register_forward_hook(lambda *args: calls.append(1))
    actual = ex(x,fps,lengths); h.remove(); assert len(calls)==1
    expected = torch.cat([shared.FrozenFrameExtractor(ck,k,device="cpu")(x,fps,lengths) for k in [3,1,4]], -1)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert actual.shape == (2,2 if patch else 6,36)
    assert not actual.requires_grad
    x[0,4:] = float("nan")
    torch.testing.assert_close(ex(x,fps,lengths),actual)
    assert all(not b._forward_hooks for b in ex.encoder.blocks)
    with pytest.raises(ValueError): ex(x,torch.tensor([-1.,30.]),lengths)
    assert all(not b._forward_hooks for b in ex.encoder.blocks)
    with pytest.raises(ValueError): probing.FrozenLayersExtractor(ck,[1,5],device="cpu")
    ex.release()
    with pytest.raises(RuntimeError): ex(x,fps,lengths)


def test_cli_and_legacy_configuration():
    parser=train.build_parser()
    base=["--mode","probe","--checkpoint","model.pth"]
    default=probing.probe_configuration(parser.parse_args(base))
    scalar=probing.probe_configuration(parser.parse_args(base+["--feature-layer","12"]))
    singleton=probing.probe_configuration(parser.parse_args(base+["--feature-layers","12"]))
    assert default==scalar==singleton and "feature_layers" not in default
    multi=parser.parse_args(base+["--feature-layers","3","6","9","12"])
    train._validate_args(multi)
    assert probing.probe_configuration(multi)["feature_layers"]==[3,6,9,12]
    for values in ([3,3],[0],[-1],[]):
        with pytest.raises(ValueError): probing.selected_feature_layers(SimpleNamespace(feature_layers=values))
    with pytest.raises(SystemExit): parser.parse_args(base+["--feature-layer","3","--feature-layers","6"])
    with pytest.raises(ValueError): train._validate_args(parser.parse_args(["--feature-layers","3","6"]))


def test_concat_ram_train_only_statistics(monkeypatch):
    instances=[]
    class Extractor:
        feature_dim=12
        metadata={"feature_layers":[3,1],"fusion":"channel_concatenation_in_requested_order"}
        def __init__(self,*args,**kwargs): instances.append(self);self.released=False
        def __call__(self,motion,fps,lengths): return torch.cat([motion, motion*2],-1)
        def release(self): self.released=True
    monkeypatch.setattr(probing,"FrozenLayersExtractor",Extractor)
    datasets,meta=fixtures()
    config={"feature_source":"jepa","checkpoint":"dummy","feature_layers":[3,1],"feature_layer":None,"encoder_key":"target_encoder"}
    ds,provenance,mean,std=probing.prepare_probe_datasets(datasets,meta,config,torch.device("cpu"),batch_size=2)
    assert provenance["motion_dim"]==12 and provenance["probe"]["feature_layers"]==[3,1]
    torch.testing.assert_close(mean[6:],mean[:6]*2)
    torch.testing.assert_close(std[6:],std[:6]*2)
    assert mean.max()<100 and ds["val"].motion_bank.values.mean()>1000
    assert instances[0].released
