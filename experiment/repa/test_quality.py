"""Metric, scheduling, and training-state regression tests."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from .quality import evaluation_settings, quality_due, GenerationEvaluator, distributed_quality
from .kimodo_eval.fid import calculate_activation_statistics, calculate_frechet_distance
from .kimodo_eval.foot_skate import FootSkateFromHeight, FootSkateRatio, FootSkateFromContacts
from .kimodo_eval.representation import TMRMotionRep
from skeleton.definitions import SOMASkeleton30


def config(representation="LocalMotionJEPARep"):
    return {"data":{"representation":representation,"fps":30,"num_frames":6},
            "shortcut":{"base_steps":128},
            "generation_evaluation":{"enabled":True,"interval":10000,"samples":4,
                                    "batch_size":2,"sampling_steps":[1,2],"seed":42}}


def test_schedule_and_validation():
    settings = evaluation_settings(config())
    assert not quality_due(settings,9999)
    assert quality_due(settings,10000)
    assert not quality_due(settings,10001,final=True)
    settings["on_completion"] = True
    assert quality_due(settings,10001,final=True)
    assert evaluation_settings({}) is None
    assert not quality_due(None,10000)
    for key,value in (("interval",0),("samples",1),("sampling_steps",[3]),("sampling_steps",[])):
        cfg=config()
        cfg["generation_evaluation"][key]=value
        with pytest.raises(ValueError):
            evaluation_settings(cfg)


def test_fid_identical_and_known_translation():
    x=np.random.default_rng(0).normal(size=(64,5))
    assert abs(calculate_frechet_distance(*calculate_activation_statistics(x),*calculate_activation_statistics(x))) < 1e-8
    assert calculate_frechet_distance(*calculate_activation_statistics(x),*calculate_activation_statistics(x+2)) == pytest.approx(20.)


def test_foot_skating_threshold_units_and_padding():
    skeleton=SimpleNamespace(foot_joint_idx=[0,1,2,3])
    positions=torch.zeros(2,5,4,3)
    positions[0,:,:,0]=torch.arange(5)[:,None]/30
    lengths=torch.tensor([4,4])
    positions[:,4]=1000 # Must not affect any valid velocity.
    height=FootSkateFromHeight(skeleton=skeleton,fps=30)
    ratio=FootSkateRatio(skeleton=skeleton,fps=30)
    contacts=FootSkateFromContacts(skeleton=skeleton,fps=30)
    torch.testing.assert_close(height._compute(positions,lengths)["foot_skate_from_height"],torch.tensor([1.,0.]))
    torch.testing.assert_close(ratio._compute(positions,lengths)["foot_skate_ratio"],torch.tensor([1.,0.]))
    values=contacts._compute(positions,torch.ones(2,5,4,dtype=torch.bool),lengths)
    torch.testing.assert_close(values["foot_skate_from_pred_contacts"],torch.tensor([1.,0.]))
    torch.testing.assert_close(values["foot_skate_max_vel"],torch.tensor([1.,0.]))
    positions[:,:,:,1]=1
    assert height._compute(positions,lengths)["foot_skate_from_height"].eq(0).all()
    assert ratio._compute(positions,lengths)["foot_skate_ratio"].eq(0).all()


def test_tmr_split_stats_match_flat_and_padding(tmp_path):
    skel=SOMASkeleton30()
    mean=np.arange(190,dtype=np.float32)*.001
    std=np.ones(190,dtype=np.float32)
    flat=tmp_path/"flat";flat.mkdir()
    np.save(flat/"mean.npy",mean);np.save(flat/"std.npy",std)
    split=tmp_path/"split"
    for name,indices in (("global_root",slice(0,5)),("body",slice(9,190))):
        (split/name).mkdir(parents=True)
        np.save(split/name/"mean.npy",mean[indices]);np.save(split/name/"std.npy",std[indices])
    a,b=TMRMotionRep(skel,flat),TMRMotionRep(skel,split)
    joints=torch.randn(2,6,30,3)
    lengths=torch.tensor([6,4])
    kwargs=dict(lengths=lengths,to_normalize=True,to_canonicalize=True)
    torch.testing.assert_close(a(posed_joints=joints,**kwargs),b(posed_joints=joints,**kwargs))
    first=a(posed_joints=joints,**kwargs)
    joints[1,4:]=10000
    second=a(posed_joints=joints,**kwargs)
    torch.testing.assert_close(first[1,:4],second[1,:4])
    assert first.shape == (2,6,186)


class ZeroModel(torch.nn.Module):
    def forward(self,x,t,d,fps,valid):
        return torch.zeros_like(x)


@pytest.mark.parametrize("representation,channels",[("LocalMotionJEPARep",365),("MotionJEPAMotionRep",366)])
def test_evaluation_reproducible_preserves_rng_and_writes(tmp_path,monkeypatch,representation,channels):
    from . import quality
    class Embedding:
        def __init__(self,path,device):
            # Exercise preservation of RNG during evaluator initialization.
            torch.randn(5,device=device)
            self.skeleton=SOMASkeleton30().to(device)
            self.identity={"test":True}
        def __call__(self,joints,lengths):
            mask=torch.arange(joints.shape[1])[None]<lengths[:,None]
            return torch.nn.functional.normalize((joints[:,:,:2].flatten(2)*mask[...,None]).sum(1),dim=-1)
    monkeypatch.setattr(quality,"MotionEmbeddingModel",Embedding)
    ds=TensorDataset(torch.randn(4,6,channels),torch.ones(4,dtype=torch.long)*30,torch.tensor([6,5,4,3]))
    cfg=config(representation)
    rng=torch.get_rng_state().clone()
    evaluator=GenerationEvaluator(cfg,ds,np.zeros(channels),np.ones(channels),torch.device("cpu"))
    assert torch.equal(rng,torch.get_rng_state())
    model=ZeroModel().train()
    first=evaluator.run(model,10000)
    assert model.training and torch.equal(rng,torch.get_rng_state())
    second=evaluator.run(model,10000)
    assert first == second
    assert all(np.isfinite(v) for metrics in first["sampling"].values() for v in metrics.values())
    (tmp_path/"quality").mkdir()
    distributed_quality(evaluator,model,10000,tmp_path,None,0,1)
    assert (tmp_path/"quality/step-00010000.json").is_file()
    evaluator.settings["batch_size"]=4
    third=evaluator.run(model,10000)
    for steps in first["sampling"]:
        assert third["sampling"][steps]["fid"] == pytest.approx(first["sampling"][steps]["fid"],abs=1e-6)


def test_training_hook_and_resume_with_new_eval_settings(tmp_path,monkeypatch):
    from . import runner,quality
    ds=TensorDataset(torch.randn(8,6,365),torch.ones(8,dtype=torch.long)*30,torch.ones(8,dtype=torch.long)*6)
    ds.normalize=False; ds.motion_dim=365
    monkeypatch.setattr(runner,"dataset",lambda config,split:ds)
    cfg=config()
    cfg.update(model={"embed_dim":16,"depth":2,"num_heads":2,"feature_layer":1},
      alignment={"weight":0,"loss":"mse"},
      training={"seed":0,"batch_size":4,"workers":0,"iterations":4,"warmup":1,"lr":1e-4,
                "final_lr":1e-6,"weight_decay":.01,"clip_grad":1.,"log_interval":2,
                "checkpoint_interval":4,"bf16":False},
      validation={"samples":4,"interval":4})
    cfg["shortcut"].update(fraction=.25,weight=1.,ema=.999)
    cfg["generation_evaluation"]["interval"]=2
    calls=[]
    monkeypatch.setattr(quality,"initialize_evaluator",lambda *args:object())
    monkeypatch.setattr(quality,"distributed_quality",lambda evaluator,ema,step,*args:calls.append(step))
    runner.train(cfg,tmp_path/"enabled",device_name="cpu")
    assert calls==[2,4]
    calls.clear()
    disabled=copy.deepcopy(cfg);disabled.pop("generation_evaluation")
    runner.train(disabled,tmp_path/"resume",device_name="cpu",max_steps=1)
    runner.train(cfg,tmp_path/"resume",str(tmp_path/"resume/latest.pt"),device_name="cpu",max_steps=1)
    assert calls==[2]
