import json
import pytest
import torch
from experiment.segmentation.print_metrics import load_report


def history():
    return [dict(iteration=3,val={'babel-120':{'frame_map':.25,'micro_f1':.4},'babel-60':{'frame_map':.3}}),
            dict(iteration=6,val={'babel-120':{'frame_map':.2},'babel-60':{'frame_map':.28}})]


def test_directory_selection_and_missing(tmp_path):
    (tmp_path/'metrics.json').write_text(json.dumps(history()))
    (tmp_path/'summary.json').write_text(json.dumps(dict(best_step=3,iterations_completed=6,parameters={'total':123})))
    r=load_report(tmp_path)
    assert r['iteration']==3 and r['metrics']['BABEL120-mAP (%)']==25
    assert r['metrics']['Macro-F1 (%)'] is None
    assert load_report(tmp_path,'latest')['iteration']==6
    with pytest.raises(ValueError,match='No unique'):load_report(tmp_path,'4')


def test_checkpoint_uses_own_step(tmp_path):
    p=tmp_path/'latest.pth.tar'
    torch.save(dict(global_step=6,best_step=3,history=history(),signature={'optimization':{'use_bfloat16':True}}),p)
    r=load_report(p)
    assert r['iteration']==6 and r['metrics']['BABEL120-mAP (%)']==20
    assert r['precision']=='BF16 mixed'
    with pytest.raises(ValueError,match='directories only'):load_report(p,'best')


def test_ambiguous_runs(tmp_path):
    for n in ['a','b']:
        (tmp_path/n).mkdir();(tmp_path/n/'metrics.json').write_text(json.dumps(history()))
    with pytest.raises(ValueError,match='Expected one experiment'):load_report(tmp_path)


def test_missing_checkpoint_evaluation(tmp_path):
    p=tmp_path/'c.pth.tar';torch.save(dict(global_step=9,history=history()),p)
    with pytest.raises(ValueError,match='cannot substitute'):load_report(p)
