"""Online adapter gradients, TMR patch masks, and training/evaluation restoration."""
from copy import deepcopy
import pytest
import torch
from experiment import probing
from experiment.tmr import model as mm, train, evaluate
from experiment.tmr.lora import LoRAMotionEncoder
from experiment.tmr.test_probing import fixtures
from model.motion_transformer_1d import MotionTransformer1D


def checkpoint(tmp_path, monkeypatch, patch=True, frames=6):
    monkeypatch.setitem(mm.MODEL_SPECS, "base", dict(embed_dim=12, depth=12, num_heads=3))
    def factory(in_chans, num_frames, patchify):
        return MotionTransformer1D(in_chans=in_chans, num_frames=num_frames,
                                   embed_dim=12, depth=12, num_heads=3, patchify=patchify)
    monkeypatch.setattr(probing.motion_transformer_1d, "mot_lora_test", factory, raising=False)
    path = tmp_path / "jepa.pth"
    torch.save(dict(config=dict(data=dict(motion_dim=6,num_frames=frames,normalize=False),
                                meta=dict(model_name="mot_lora_test",patchify=patch)),
                    target_encoder=factory(6,frames,patch).state_dict()), path)
    return dict(checkpoint=str(path),encoder_key="target_encoder",rank=8,alpha=8)


@pytest.mark.parametrize("rank", [8,16])
def test_mask_gradients_and_frozen_weights(tmp_path, monkeypatch, rank):
    options = checkpoint(tmp_path,monkeypatch)
    options.update(rank=rank,alpha=rank)
    config = mm.AlignmentConfig(text_dim=8,motion_dim=6,num_frames=6,fps=30,motion_depth=1)
    branch = LoRAMotionEncoder(config, options)
    branch.train()
    assert len(branch.head.backbone.blocks)==1 and len(branch.encoder.blocks)==12
    assert not branch.encoder.training and branch.head.training
    frozen={k:v.clone() for k,v in branch.named_parameters() if not v.requires_grad}
    x=torch.randn(2,5,6); mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    expected=branch(x,mask)
    altered=x.clone(); altered[0,3:]=float("nan"); altered[1,3:]=100000
    torch.testing.assert_close(branch(altered,mask),expected)
    # Zero-initialized adapters reproduce the final normalized pretrained output.
    raw=probing.FrozenFrameExtractor(options["checkpoint"],None)
    padded=torch.nn.functional.pad(x.masked_fill(~mask[...,None],0),(0,0,0,1))
    active=torch.nn.functional.pad(mask,(0,1))
    with torch.no_grad():
        features=raw.encoder(padded,torch.tensor([30.,30.]),valid_frames=active)
        torch.testing.assert_close(expected,branch.head(features,raw.token_layout.valid_token_mask(active)))
    opt=torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad],lr=.01,weight_decay=.01)
    (expected*torch.randn_like(expected)).sum().backward()
    assert all(p.grad is None for p in branch.encoder.parameters() if not p.requires_grad)
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in branch.encoder.parameters() if p.requires_grad)
    assert any(p.grad.abs().sum()>0 for n,p in branch.named_parameters() if "lora_B" in n)
    opt.step()
    for name,p in branch.named_parameters():
        if name in frozen: torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
    restored=LoRAMotionEncoder(config,options); restored.load_state_dict(branch.state_dict())
    restored.eval();branch.eval()
    torch.testing.assert_close(restored(x,mask),branch(x,mask))
    with pytest.raises(ValueError): branch(x,torch.zeros_like(mask))


def test_cli_validation():
    parser=train.build_parser()
    args=parser.parse_args(["--mode","lora","--checkpoint","jepa.pth","--motion-depth","1"])
    train._validate_args(args)
    assert args.lora_rank==8
    for extra in (["--lora-rank","0"],["--lora-alpha","nan"],["--feature-layer","3"],["--patchify"],["--feature-layers","3","6"]):
        with pytest.raises(ValueError): train._validate_args(parser.parse_args(["--mode","lora","--checkpoint","jepa.pth"]+extra))


def test_training_resume_and_evaluation(tmp_path,monkeypatch):
    options=checkpoint(tmp_path,monkeypatch,patch=False,frames=4)
    monkeypatch.setattr(train,"load_prepared_datasets",lambda **kwargs: fixtures())
    monkeypatch.setattr(evaluate,"load_prepared_datasets",lambda **kwargs: fixtures())
    args=train.build_parser().parse_args(["--mode","lora","--checkpoint",options["checkpoint"],
        "--motion-depth","1","--output-root",str(tmp_path/"run"),"--iterations","2",
        "--warmup-iterations","0","--eval-interval","1","--checkpoint-interval","1",
        "--log-freq","1","--batch-size","2","--eval-batch-size","2",
        "--num-workers","0","--no-tensorboard","--no-use-bfloat16","--device","cpu"])
    result=train.run(args)
    saved=train.load_alignment_checkpoint(tmp_path/"run"/"latest.pth.tar")
    assert saved["config"]["lora"]["rank"]==8
    assert saved["provenance"]["lora"]["feature_location"]=="final_encoder_output_after_layernorm"
    args.resume=True
    assert train.run(args)["test"]==result["test"]
    eargs=evaluate.build_parser().parse_args(["--checkpoint",result["best_checkpoint"],
                                             "--device","cpu","--batch-size","2","--num-workers","0"])
    assert evaluate.run(eargs)==result["test"]
    args.lora_rank=16
    with pytest.raises(ValueError,match="Resume config"): train.run(args)
