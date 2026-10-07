"""CPU-only checks; fake generation and tiny local random models, no downloads."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec=importlib.util.spec_from_file_location("selector_v2",Path(__file__).with_name("selector_v2.py"))
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)


def record(i, answer="Paris", y=1, split="train", truncated=False):
    return {"id":str(i),"question":f"Question {i}","text":answer,"truncated":truncated,
            "outcome":"correct" if y else "error","reason":"exact_match" if y else "not_exact_match",
            "y":y,"split":split,"candidate_index":0,"original_logit":float(y*2-1)}


def test_atomic_no_overwrite_and_envelope_tamper(tmp_path):
    p=tmp_path/"cache.json";ctx={"sha":"registered"}
    value=s.envelope(ctx,[1,2],split="train")
    s.atomic(p,value,immutable=True);s.atomic(p,value,immutable=True)
    assert s.read_envelope(p,ctx,split="train")==[1,2]
    with pytest.raises(ValueError):s.atomic(p,{"different":1},immutable=True)
    value["rows"][0]=3;s.atomic(p,value)
    with pytest.raises(ValueError):s.read_envelope(p,ctx)


def test_inputs_have_no_aliases_labels_tau_or_reward():
    r=record(1);r.update(aliases=["SECRET_ALIAS"],label="SECRET_LABEL",tau=.95,reward=-99,arm="SECRET_ARM")
    prompt=json.dumps(s.critic_prompt(r))
    assert "Paris" in prompt and "Question 1" in prompt
    assert not any(x in prompt for x in ("SECRET_ALIAS","SECRET_LABEL","SECRET_ARM","0.95","-99"))
    changed={**r,"aliases":["different"],"y":0,"reward":30,"tau":.1}
    assert s.critic_prompt(changed)==s.critic_prompt(r)


def test_question_and_normalized_text_leakage_rejected():
    a=[{"id":"a","question":"Who wrote this?"}]
    with pytest.raises(ValueError):s.assert_disjoint({"train":a,"calibration":[{"id":"b","question":"who wrote this!"}]})
    with pytest.raises(ValueError):s.assert_disjoint({"train":a,"dev":[{"id":"a","question":"Different"}]})
    with pytest.raises(ValueError):s.assert_disjoint({"train":a+a})


def test_binary_logit_equals_normalized_single_token_likelihood():
    torch.manual_seed(1)
    logits=torch.randn(3,1,5000)
    model=lambda **kwargs:SimpleNamespace(logits=logits)
    z=s.binary_logits(model,{},(2514,4049))
    ls=logits[:,0,:].log_softmax(-1)
    assert torch.allclose(z,ls[:,2514]-ls[:,4049],atol=1e-6)
    y=torch.tensor([0.,1.,0.]); z=z.requires_grad_()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)
    loss.backward()
    assert torch.allclose(z.grad,(z.detach().sigmoid()-y)/3)


def test_token_mapping_guard():
    good=SimpleNamespace(encode=lambda word,**kwargs:[2514 if word=="True" else 4049])
    assert s.token_ids(good)==(2514,4049)
    with pytest.raises(ValueError):s.token_ids(SimpleNamespace(encode=lambda *a,**k:[1,2]))


def test_invalid_candidates_scored_but_never_emitted():
    rows=[record(1),record(2,"IDK",0),record(3,"",0),record(4,"Paris",0,truncated=True)]
    result=s.metrics(rows,[.99]*4,[.65])
    assert result["probability_scores"]["all"]["n"]==4
    assert result["probability_scores"]["substantive"]["n"]==1
    assert result["thresholds"]["0.65"]["coverage"]==.25
    assert result["thresholds"]["0.65"]["utility"]==pytest.approx(.35/4)
    assert s.metrics([record(1)],[.65],[.65])["thresholds"]["0.65"]["coverage"]==0


def test_logits_preserve_auc_ranking_and_stable_tail_log_loss():
    rows=[record(1,y=0),record(2,y=1)]
    zs=[40.,50.];ps=[s.sigmoid(z) for z in zs]
    assert ps==[1.,1.]  # Floating-point sigmoid has lost their ordering.
    result=s.metrics(rows,ps,[.65],logits=zs)
    old=s.metrics(rows,ps,[.65])
    assert result["probability_scores"]["all"]["auc"]==1.
    assert old["probability_scores"]["all"]["auc"]==.5
    assert result["probability_scores"]["all"]["log_loss"]==pytest.approx(20.)
    assert result["thresholds"]==old["thresholds"]
    assert result["probability_scores"]["all"]["brier"]==old["probability_scores"]["all"]["brier"]
    tail=[1000.,-1000.]
    extreme=s.metrics(rows,[s.sigmoid(z) for z in tail],[.65],logits=tail)
    assert extreme["probability_scores"]["all"]["log_loss"]==pytest.approx(1000.)
    with pytest.raises(ValueError):s.metrics(rows,ps,[.65],logits=[40.])
    with pytest.raises(ValueError):s.metrics(rows,ps,[.65],logits=[40.,float("nan")])
    with pytest.raises(ValueError):s.metrics(rows,[.1,.1],[.65],logits=zs)


def test_logistic_train_question_weights_and_calibration_scope():
    from sklearn.linear_model import LogisticRegression
    rows=[record(i//4,y=i%2) for i in range(16)]
    z=[-.5,.2,1.,-.2]*4
    fit=s.fit_logistic(rows,z,split="train",equal_question=True)
    direct=LogisticRegression(C=1.,solver="lbfgs",max_iter=1000,random_state=0).fit(np.array(z).reshape(-1,1),[r["y"] for r in rows],sample_weight=np.full(16,.25))
    assert fit["coefficient"]==pytest.approx(direct.coef_[0,0])
    assert fit["questions"]==4
    with pytest.raises(ValueError):s.fit_logistic(rows,z,split="calibration")
    with pytest.raises(ValueError):s.fit_logistic(rows,z,split="train",equal_question=False)
    cal=[record(i,y=i%2,split="calibration") for i in range(4)]
    fitted=s.fit_logistic(cal,[1.,-1.,.5,-.5],split="calibration")
    assert fitted["fit_split"]=="calibration" and fitted["rows"]==4
    # Two successive fitted maps remain affine; no new ranking feature appears.
    first={"coefficient":2.,"intercept":3.};second={"coefficient":.4,"intercept":-.8}
    assert s.transformed(second,s.transformed(first,[1.,2.]))==pytest.approx([1.2,2.])


def fake_probe():
    return {"available_gib":80.,"cuda_allocated_peak_gib":0.,"cuda_reserved_peak_gib":0.}


def test_budget_accumulates_failures_and_blocks_unknown_tail(tmp_path):
    p=tmp_path/"cost.json";clock=iter([0.,0.,4.]).__next__
    with pytest.raises(RuntimeError):
        with s.Budget(p,10,"identity",probe=fake_probe,clock=clock):raise RuntimeError("failed load")
    ledger=s.read_json(p)
    assert ledger["attempts"][0]["status"]=="failed"
    assert ledger["attempts"][0]["seconds"]==4
    with pytest.raises(RuntimeError):s.Budget(p,3,"identity",probe=fake_probe)
    ledger["attempts"][0]["status"]="running";s.atomic(p,ledger)
    with pytest.raises(RuntimeError,match="unclosed"):s.Budget(p,20,"identity",probe=fake_probe)


def test_budget_entry_guard_closes_known_failure(tmp_path):
    low=lambda:{**fake_probe(),"available_gib":15.}
    with pytest.raises(MemoryError):
        with s.Budget(tmp_path/"cost.json",100,"id",probe=low):pass
    assert s.read_json(tmp_path/"cost.json")["attempts"][0]["status"]=="failed"


def setup_linear():
    torch.manual_seed(10)
    model=torch.nn.Linear(1,1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
    data=[{"x":float(i%7)/7,"y":float(i%2)} for i in range(64)]
    def loss(rows):
        x=torch.tensor([[r["x"]] for r in rows]);y=torch.tensor([r["y"] for r in rows])
        return torch.nn.functional.binary_cross_entropy_with_logits(model(x).flatten(),y)
    return model,optimizer,data,loss


def test_one_epoch_accumulation_matches_full_batch_gradient():
    model,opt,data,loss=setup_linear(); manual=copy.deepcopy(model)
    direct=torch.optim.AdamW(manual.parameters(),lr=.001,weight_decay=0.)
    seen=[]
    s.train_loop(model,opt,data[:16],list(range(16)),loss,lambda step,l,n,seconds:seen.append((step,l,n,seconds)))
    x=torch.tensor([[r["x"]] for r in data[:16]]);y=torch.tensor([r["y"] for r in data[:16]])
    torch.nn.functional.binary_cross_entropy_with_logits(manual(x).flatten(),y).backward()
    torch.nn.utils.clip_grad_norm_(manual.parameters(),1.,error_if_nonfinite=True);direct.step()
    assert len(seen)==1 and seen[0][2]>0
    assert all(torch.allclose(a,b,atol=1e-7) for a,b in zip(model.parameters(),manual.parameters()))


def test_resuming_optimizer_boundaries_replays_same_epoch():
    full,opt,data,loss=setup_linear();order=torch.randperm(64,generator=torch.Generator().manual_seed(17)).tolist()
    s.train_loop(full,opt,data,order,loss,lambda *a:None)
    resumed,ropt,rdata,rloss=setup_linear()
    s.train_loop(resumed,ropt,rdata,order,rloss,lambda *a:None,stop_step=2)
    weights=copy.deepcopy(resumed.state_dict());state=copy.deepcopy(ropt.state_dict())
    again,aopt,adata,aloss=setup_linear();again.load_state_dict(weights);aopt.load_state_dict(state)
    s.train_loop(again,aopt,adata,order,aloss,lambda *a:None,start_step=2)
    assert all(torch.equal(a,b) for a,b in zip(full.parameters(),again.parameters()))
    with pytest.raises(ValueError):s.train_loop(again,aopt,adata,[0]*64,aloss,lambda *a:None)


def test_nonfinite_loss_aborts_before_optimizer_step():
    model,opt,data,loss=setup_linear(); before=copy.deepcopy(model.state_dict())
    with pytest.raises(FloatingPointError):
        s.train_loop(model,opt,data[:16],list(range(16)),lambda rows:torch.tensor(float("nan"),requires_grad=True),lambda *a:None)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())


class FakeBackend:
    calls=0
    def __init__(self,ctx,**kwargs):pass
    def generate(self,q,sample):
        type(self).calls+=1
        text="Paris" if int(q["id"].split("-")[-1])%2 else "IDK"
        return [{"text":text,"truncated":False,"generated_tokens":2,"seconds":.01} for _ in range(3 if sample else 1)]
    def score(self,records):return [1. if r["text"]=="Paris" else -1. for r in records]
    def close(self):pass


def fake_context(tmp_path):
    splits={name:[{"id":f"{name}-{i}","question":f"{name} question {i}","aliases":["Paris"]} for i in range(2)] for name in ("train","dev","calibration")}
    return {"out":tmp_path,"root":tmp_path,"sha":"fixture","identity":{"contract_sha256":"c"},"contract":{},"splits":splits}


def test_preparation_keeps_all_candidates_and_idempotent_resume(tmp_path,monkeypatch):
    ctx=fake_context(tmp_path);original=s.Budget
    monkeypatch.setattr(s,"Budget",lambda p,n,h:original(p,n,h,probe=fake_probe))
    FakeBackend.calls=0
    result=s.prepare_candidates(ctx,100,backend_factory=FakeBackend)
    assert result=={"train":8,"calibration":2,"dev":2}
    train=s.candidates(ctx,"train")
    assert sum(r["text"]=="IDK" for r in train)==4
    assert sum(r["y"] for r in train)==4
    assert all("aliases" not in r for r in train)
    calls=FakeBackend.calls
    assert s.prepare_candidates(ctx,100,backend_factory=FakeBackend)==result
    assert FakeBackend.calls==calls
    p=tmp_path/"candidates/train.json";e=s.read_json(p);e["rows"][0]["y"]=1;s.atomic(p,e)
    with pytest.raises(ValueError):s.candidates(ctx,"train")


def test_candidate_labels_regraded_and_no_duplicate_index():
    q={"id":"a","question":"Q","aliases":["Paris"]}
    r={**record("a",split="dev"),"question":"Q","generated_tokens":2,"seconds":.1,"generator":"original_frozen",
       "generation_seed":int(s.digest([s.RECIPE["sampling_seed"],"dev","a"])[:8],16)}
    r["critic_prompt_sha256"]=s.digest(s.critic_prompt(r))
    s.validate_candidates([r],[q],"dev")
    with pytest.raises(ValueError):s.validate_candidates([{**r,"y":0}],[q],"dev")
    with pytest.raises(ValueError):s.validate_candidates([r,r],[q],"dev")


def test_test_access_requires_ready_matching_lock(tmp_path):
    ctx=fake_context(tmp_path);lock=tmp_path/"test-lock-input.json"
    s.atomic(lock,{"status":"draft","contract_sha256":"c"})
    with pytest.raises(ValueError):s.locked_test_context(ctx,lock)
    s.atomic(lock,{"status":"ready","contract_sha256":"wrong"})
    with pytest.raises(ValueError):s.locked_test_context(ctx,lock)
    with pytest.raises(ValueError):s.evaluate_test(ctx,17,100)


def test_environment_must_match_pinned_image(monkeypatch):
    monkeypatch.setenv("ABSTENTION_IMAGE_ID","wrong")
    with pytest.raises(RuntimeError):s.env_check()


def test_development_labels_cannot_change_fitted_calibrators(tmp_path):
    ctx=fake_context(tmp_path)
    records={split:[record(i,y=i%2,split=split) for i in range(4)] for split in ("train","calibration","dev")}
    zs={"calibration":[-1.,2.,-.5,.2],"dev":[.2,.1,-1.,2.]}
    first=s.finish_development(ctx,17,tmp_path/"first",records,zs)
    changed=copy.deepcopy(records)
    for row in changed["dev"]:
        row["y"]=1-row["y"];row["outcome"]="correct" if row["y"] else "error"
    second=s.finish_development(ctx,17,tmp_path/"second",changed,zs)
    assert s.read_json(tmp_path/"first/calibrators.json")==s.read_json(tmp_path/"second/calibrators.json")
    assert first["methods"]["critic_raw"]["probability_scores"]!=second["methods"]["critic_raw"]["probability_scores"]


def test_development_and_test_thresholds_are_separate():
    ctx={"contract":{"evaluation":{"dev_thresholds":[.6,.75,.9],"primary_thresholds":[.65,.85],"diagnostic_thresholds":[.6,.75,.9,.95]}}}
    assert s.thresholds(ctx)==[.6,.75,.9]
    assert s.thresholds(ctx,test=True)==[.65,.85]


def test_rl_gate_requires_exact_twelve_validated_runs(tmp_path):
    out=tmp_path/"artifacts/selector";out.mkdir(parents=True)
    ctx={"root":tmp_path,"out":out}
    budget={"rl_steps":252,"rl_questions":504,"rl_completions":12096}
    source={"image":s.IMAGE,"files":{}}
    s.atomic(out.parent/"budget-lock.json",budget);s.atomic(out.parent/"source-lock.json",source)
    lock={"source_lock_sha256":s.file_digest(out.parent/"source-lock.json"),
          "budget_lock_sha256":s.file_digest(out.parent/"budget-lock.json"),"rl_completed":{}}
    with pytest.raises(ValueError,match="twelve"):s.verify_rl_gate(ctx,lock)
    for arm in ("a_g4_single_tau","b_g8_single_tau","c_g8_paired_tau","d_g8_paired_fixed075"):
        for seed in (17,29,43):
            variant=f"{arm}-s{seed}";directory=out.parent/"runs"/variant;adapter=directory/"adapter";adapter.mkdir(parents=True)
            (adapter/"adapter_model.safetensors").write_bytes(b"fixture")
            (adapter/"adapter_config.json").write_bytes(b"{}")
            hashes={name:s.file_digest(adapter/name) for name in ("adapter_model.safetensors","adapter_config.json")}
            spec={"group_size":4 if arm.startswith("a_") else 8,"arm":"fixed" if arm.startswith("d_") else "conditioned",
                  "seed":seed,"exposure_mode":"single_tau" if arm.startswith(("a_","b_")) else "paired_tau","questions_per_update":2,"micro_batch_size":4}
            result={"status":"complete","optimizer_steps":252,"gradient_steps":252,"questions":504,"completions":12096,"adapter_sha256":hashes,
                    "spec":spec,"save_roundtrip":True,"sampler_traversal_verified":True}
            path=directory/"result.json";s.atomic(path,result)
            lock["rl_completed"][variant]={"result_path":str(path.relative_to(tmp_path)),"result_sha256":s.file_digest(path),"adapter":str(adapter.relative_to(tmp_path)),"adapter_sha256":hashes}
    s.verify_rl_gate(ctx,lock)
    (adapter/"adapter_model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="adapter changed"):s.verify_rl_gate(ctx,lock)


def test_checkpoint_exact_roundtrip_and_tamper(tmp_path):
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    torch.manual_seed(3)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8)
    model=get_peft_model(Qwen3ForCausalLM(config),LoraConfig(r=2,lora_alpha=4,target_modules="all-linear",task_type="CAUSAL_LM"))
    def score(records):
        model.eval()
        with torch.no_grad():return model(input_ids=torch.tensor([[1,2,3]]),use_cache=False).logits[0,-1,:len(records)].tolist()
    backend=SimpleNamespace(model=model,score=score)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    rows=[record(i) for i in range(4)];ctx={"sha":"checkpoint-test"};order=list(range(4))
    cp=s.save_checkpoint(backend,optimizer,tmp_path,16,ctx,17,order,rows)
    before=score(rows)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:p.add_(1.)
    s.restore(backend,optimizer,cp,ctx,17,order,rows)
    assert score(rows)==before
    (cp/"state.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):s.restore(backend,optimizer,cp,ctx,17,order,rows)


def test_completion_rejects_crossed_seed_or_checkpoint_step(tmp_path,monkeypatch):
    ctx=fake_context(tmp_path);directory=tmp_path/"runs/s17";cp=directory/"checkpoints/step-0500"
    (cp/"adapter").mkdir(parents=True)
    (cp/"adapter/adapter_model.safetensors").write_bytes(b"fixture")
    (cp/"state.pt").write_bytes(b"fixture optimizer")
    records=[record(1)]
    monkeypatch.setattr(s,"candidates",lambda ctx,split:records)
    meta={"seed":17,"step":500,"identity_sha256":ctx["sha"],"candidates_sha256":s.digest(records),
          "adapter_sha256":s.tree_hash(cp/"adapter"),"state_sha256":s.file_digest(cp/"state.pt")}
    s.atomic(cp/"metadata.json",meta)
    steps=[{"step":i,"bce":.5,"gradient_norm_before_clip":.1,"seconds":.01} for i in range(1,501)]
    s.atomic_jsonl(directory/"steps.jsonl",steps)
    result={"status":"complete","seed":17,"technical_pilot_disposable":False,"identity_sha256":ctx["sha"],"steps":500,
            "examples_seen":8000,"finite_gradient_steps":500,"save_roundtrip":True,"checkpoint":str(cp.relative_to(tmp_path)),
            "checkpoint_sha256":s.tree_hash(cp),"train_sha256":s.digest(records),"steps_sha256":s.file_digest(directory/"steps.jsonl")}
    s.atomic(directory/"complete.json",result)
    s.atomic(tmp_path/"costs/train-s17.json",{"identity_sha256":ctx["sha"],"attempts":[{"status":"closed","seconds":1.}]})
    assert s.verify_run(ctx,17)==result
    s.atomic(directory/"complete.json",{**result,"seed":29})
    with pytest.raises(ValueError):s.verify_run(ctx,17)
    s.atomic(cp/"metadata.json",{**meta,"step":499})
    s.atomic(directory/"complete.json",{**result,"checkpoint_sha256":s.tree_hash(cp)})
    with pytest.raises(ValueError,match="provenance"):s.verify_run(ctx,17)
