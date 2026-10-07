"""Isolated EM-correctness selector: original Qwen generator, binary LoRA critic.

No aliases/labels/thresholds enter critic inputs. Training uses TRAIN only;
calibration uses calibration only. No test read is possible without a separate
evaluation lock. All model operations require the pinned environment and an
explicit cumulative time budget. This module never starts Docker or a service.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import Counter
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from abstention.io import digest, file_digest, read_json, read_jsonl, utcnow
from abstention.models import Engine, load_model, render
from abstention.prompts import confidence_messages, messages
from abstention.scoring import grade, normalize_question, summarize

IMAGE = "sha256:106bd8033f516b97e47ee39eb17c7477e516ae5329977665d25cac96ce90c10f"
PACKAGES = {"torch": "2.10.0a0+b558c986e8.nv25.11", "transformers": "4.57.1",
            "numpy": "2.2.6", "huggingface-hub": "0.36.0", "peft": "0.17.1",
            "scikit-learn": "1.7.2", "safetensors": "0.6.2"}
RECIPE = {"learning_rate": 1e-5, "r": 16, "alpha": 32, "epochs": 1,
          "micro_batch": 4, "accumulation": 4, "max_prompt_length": 512,
          "max_completion_length": 32, "temperature": .8, "min_memory_gib": 16,
          "seeds": [17, 29, 43], "pilot_steps": 16, "checkpoint_steps": 50,
          "sampling_seed": 20261002, "train_candidates_per_question": 4}
FROZEN = ["src/abstention/io.py", "src/abstention/models.py", "src/abstention/prompts.py",
          "src/abstention/scoring.py", "experiments/v2-20261002/selector_v2.py",
          "experiments/v2-20261002/tests_selector_v2.py"]


def atomic(path, value, immutable=False):
    path = Path(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists() and immutable:
        if path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as f:
        temp = Path(f.name)
        f.write(payload); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic_jsonl(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w",dir=path.parent,prefix=".pending-",delete=False) as f:
        temp=Path(f.name)
        for row in records:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


@contextmanager
def exclusive(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)


def resolve(root, relative):
    p = (root / relative).resolve()
    if not p.is_relative_to(root.resolve()):
        raise ValueError("Path must be inside the repository")
    return p


def assert_disjoint(splits):
    seen_ids, seen_questions = set(), set()
    for name, records in splits.items():
        ids = [r["id"] for r in records]
        qs = [normalize_question(r["question"]) for r in records]
        if len(ids) != len(set(ids)) or len(qs) != len(set(qs)):
            raise ValueError(f"Duplicate question inside {name}")
        if seen_ids.intersection(ids) or seen_questions.intersection(qs):
            raise ValueError("Question leakage across selector splits")
        seen_ids.update(ids); seen_questions.update(qs)


def context(contract_path, root=ROOT):
    root = Path(root).resolve()
    contract_path = Path(contract_path)
    c = read_json(contract_path)
    expected_selector={"learning_rate":1e-5,"lora_r":16,"lora_alpha":32,"epochs":1,
                       "micro_batch_size":4,"gradient_accumulation_steps":4,"pilot_steps":16,
                       "max_prompt_length":512,"max_completion_length":32,"temperature":.8,
                       "greedy_candidates_per_question":1,"sampled_candidates_per_question":3,"calibration_C":1.}
    if c.get("selector")!=expected_selector or c.get("seeds")!=[17,29,43] or c.get("min_system_available_gib")!=16:
        raise ValueError("Selector recipe differs from the agreed fixed configuration")
    splits = {}
    for name, count in (("train", 2000), ("dev", 500), ("calibration", 500)):
        spec = c["data"][name]; path = resolve(root, spec["path"])
        if file_digest(path) != spec["sha256"]:
            raise ValueError(f"Changed {name} data")
        splits[name] = read_jsonl(path)
        if len(splits[name]) != count:
            raise ValueError(f"Expected {count} {name} questions")
    assert_disjoint(splits)
    model_dir = resolve(root, c["model_workdir"])
    lock_path = model_dir / "artifacts/models.lock.json"
    pin = read_json(lock_path)[c["model_key"]]
    if pin["repo"] != "Qwen/Qwen3-14B" or pin["revision"] != c["model_revision"]:
        raise ValueError("Unexpected base model identity")
    if c.get("pinned_image", IMAGE) != IMAGE:
        raise ValueError("Unexpected image")
    identity = {"contract_sha256": file_digest(contract_path), "recipe": RECIPE,
                "model_lock_sha256": file_digest(lock_path), "model": pin,
                "source_sha256": {p: file_digest(root / p) for p in FROZEN},
                "config_sha256": file_digest(resolve(root, c["config_path"])),
                "data_manifest_sha256": file_digest(resolve(root, c["data_manifest"])),
                "data": c["data"], "scope": "TRAIN BCE; calibration-only Platt; original frozen candidate generator"}
    for p, expected in c.get("source_sha256", {}).items():
        if file_digest(resolve(root, p)) != expected:
            raise ValueError(f"Registered source changed: {p}")
    return {"contract": c, "identity": identity, "sha": digest(identity), "splits": splits,
            "root": root, "contract_path":contract_path.resolve(), "model_dir": model_dir, "out": resolve(root, c["output_root"])}


def seal_context(ctx):
    atomic(ctx["out"] / "identity.json", {"sha256": ctx["sha"], **ctx["identity"]}, immutable=True)


def verify_identity(ctx):
    if "contract_path" in ctx and file_digest(ctx["contract_path"])!=ctx["identity"]["contract_sha256"]:
        raise ValueError("Contract changed during selector operation")
    if "model_dir" in ctx and file_digest(ctx["model_dir"]/"artifacts/models.lock.json")!=ctx["identity"]["model_lock_sha256"]:
        raise ValueError("Model lock changed during selector operation")
    for p,h in ctx["identity"].get("source_sha256",{}).items():
        if file_digest(resolve(ctx["root"],p))!=h:raise ValueError("Source changed during selector operation")
    for spec in ctx["identity"].get("data",{}).values():
        if file_digest(resolve(ctx["root"],spec["path"]))!=spec["sha256"]:raise ValueError("Data changed during selector operation")
    for field,hash_field in (("config_path","config_sha256"),("data_manifest","data_manifest_sha256")):
        if field in ctx["contract"] and file_digest(resolve(ctx["root"],ctx["contract"][field]))!=ctx["identity"][hash_field]:
            raise ValueError("Configuration/manifest changed during selector operation")


def env_check():
    if os.environ.get("ABSTENTION_IMAGE_ID") != IMAGE or platform.python_version() != "3.12.3":
        raise RuntimeError("Use the pinned image and its ABSTENTION_IMAGE_ID")
    versions = {p: importlib.metadata.version(p) for p in PACKAGES}
    if versions != PACKAGES:
        raise RuntimeError("Unexpected package versions")
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[k] = "1"
    return versions


def resources():
    import torch
    avail = next(int(s.split()[1]) / 1024**2 for s in Path("/proc/meminfo").read_text().splitlines() if s.startswith("MemAvailable:"))
    return {"available_gib": avail, "cuda_allocated_peak_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "cuda_reserved_peak_gib": torch.cuda.max_memory_reserved() / 1024**3}


class Budget:
    """Atomic attempt ledger. Unknown crash tails block automatic resumption."""
    def __init__(self, path, seconds, identity, probe=resources, clock=time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("An explicit finite positive --max-seconds is required")
        self.path, self.limit, self.identity = Path(path), seconds, identity
        self.probe, self.clock = probe, clock
        self.ledger = read_json(self.path) if self.path.exists() else {"identity_sha256": identity, "attempts": []}
        if self.ledger["identity_sha256"] != identity or any(a["status"] == "running" for a in self.ledger["attempts"]):
            raise RuntimeError("Changed identity or unclosed attempt: review required before resume")
        self.spent = sum(a["seconds"] for a in self.ledger["attempts"])
        if self.spent >= self.limit:
            raise RuntimeError("Cumulative stage budget exhausted")

    def __enter__(self):
        self.started = self.clock()
        self.current = {"id": uuid.uuid4().hex, "at": utcnow(), "status": "running", "seconds": 0.,
                        "minimum_available_gib": None, "cuda_allocated_peak_gib": 0., "cuda_reserved_peak_gib": 0.}
        self.ledger["attempts"].append(self.current)
        atomic(self.path, self.ledger)
        try: self.check()
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def check(self):
        self.current["seconds"] = self.clock() - self.started
        r = self.probe()
        prior = self.current["minimum_available_gib"]
        self.current["minimum_available_gib"] = r["available_gib"] if prior is None else min(prior, r["available_gib"])
        for k in ("cuda_allocated_peak_gib", "cuda_reserved_peak_gib"):
            self.current[k] = max(self.current[k], r[k])
        atomic(self.path, self.ledger)
        if r["available_gib"] < RECIPE["min_memory_gib"]:
            raise MemoryError("Less than 16 GiB system memory remains")
        if self.spent + self.current["seconds"] >= self.limit:
            raise TimeoutError("Cumulative stage time budget exhausted")

    def __exit__(self, typ, exc, tb):
        self.current.update(seconds=self.clock() - self.started, status="failed" if typ else "closed")
        try:
            r=self.probe(); prior=self.current["minimum_available_gib"]
            self.current["minimum_available_gib"]=r["available_gib"] if prior is None else min(prior,r["available_gib"])
            for k in ("cuda_allocated_peak_gib","cuda_reserved_peak_gib"):
                self.current[k]=max(self.current[k],r[k])
        except Exception as resource_error:
            self.current["closing_resource_probe_error"]=repr(resource_error)
        if typ: self.current["error"] = f"{typ.__name__}: {exc}"
        atomic(self.path, self.ledger)


def envelope(ctx, values, **metadata):
    body = {"identity_sha256": ctx["sha"], **metadata, "rows": values}
    return {**body, "sha256": digest(body)}


def read_envelope(path, ctx, **metadata):
    value = read_json(path)
    body = {k: v for k, v in value.items() if k != "sha256"}
    if digest(body) != value["sha256"] or value["identity_sha256"] != ctx["sha"]:
        raise ValueError(f"Changed cache: {path}")
    if any(value.get(k) != v for k, v in metadata.items()):
        raise ValueError("Cache metadata mismatch")
    return value["rows"]


def verify_cost(path, ctx):
    ledger=read_json(path)
    if ledger["identity_sha256"]!=ctx["sha"] or not ledger["attempts"]:
        raise ValueError("Cost identity/population changed")
    if any(a["status"] not in {"closed","failed"} or not math.isfinite(a["seconds"]) or a["seconds"]<0 for a in ledger["attempts"]):
        raise ValueError("Cost contains an unknown tail or malformed elapsed time")
    return ledger


def critic_prompt(row):
    return confidence_messages(row["question"], row["text"])


def eligible(row):
    return not row["truncated"] and bool(row["text"].strip()) and row["text"].strip() != "IDK"


def sigmoid(z):
    return 1 / (1 + math.exp(-max(-700., min(700., float(z)))))


def encode(tokenizer, rows):
    prompts = [render(tokenizer, critic_prompt(r)) for r in rows]
    batch = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False)
    if batch.input_ids.shape[1] > RECIPE["max_prompt_length"]:
        raise ValueError("Critic prompt exceeds 512 tokens; no silent truncation")
    return batch


def token_ids(tokenizer):
    ids = [tokenizer.encode(s, add_special_tokens=False) for s in ("True", "False")]
    if ids != [[2514], [4049]]:
        raise ValueError("True/False token mapping changed")
    return 2514, 4049


def binary_logits(model, batch, ids):
    out = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
    return out[:, ids[0]] - out[:, ids[1]]


class Backend:
    def __init__(self, ctx, adapter=None, trainable=False):
        import torch
        versions=env_check()
        if not torch.cuda.is_available(): raise RuntimeError("GPU model operation requires CUDA")
        torch.cuda.reset_peak_memory_stats()
        atomic(ctx["out"]/"runtime.json", {"identity_sha256":ctx["sha"],"image":IMAGE,
               "python":platform.python_version(),"packages":versions,"cuda":torch.version.cuda,
               "device":torch.cuda.get_device_name(),"device_count":torch.cuda.device_count(),
               "dtype":"bfloat16","backend":"frozen v1.Engine and supervised binary-logit LoRA",
               "candidate_generator":"original model without adapters","thinking":False}, immutable=True)
        self.ctx = ctx
        config = {"training": {k: RECIPE[k] for k in ("temperature", "max_prompt_length", "max_completion_length")}}
        self.engine = Engine(config, ctx["model_dir"], ctx["contract"]["model_key"], adapter=adapter)
        self.model, self.tokenizer = self.engine.model, self.engine.tokenizer
        self.ids = token_ids(self.tokenizer)
        if trainable:
            from peft import LoraConfig, get_peft_model
            if adapter is not None: raise ValueError("Training adapter must be restored from checkpoint after initialization")
            self.model = get_peft_model(self.model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                                      target_modules="all-linear", bias="none", task_type="CAUSAL_LM"))
            self.engine.model = self.model
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.config.use_cache = False
            self.model.train()

    def generate(self, q, sample):
        n = 3 if sample else 1
        return self.engine.generate([messages(q["question"], forced=True)] * n, sample=sample, temperature=.8)

    def score(self, records):
        import torch
        values = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                for i in range(0, len(records), 16):
                    b = encode(self.tokenizer, records[i:i+16]).to(self.model.device)
                    values.extend(binary_logits(self.model, b, self.ids).cpu().tolist())
            torch.cuda.synchronize()
        finally: self.model.train(was_training)
        if not all(math.isfinite(x) for x in values): raise FloatingPointError("Nonfinite critic scores")
        return values

    def close(self):
        import gc
        import torch
        self.engine.close()
        del self.model
        gc.collect(); torch.cuda.empty_cache()


def set_seed(seed):
    from transformers import set_seed as impl
    impl(seed)


def validate_candidates(records, questions, split):
    by_id = {q["id"]: q for q in questions}
    count = 4 if split == "train" else 1
    if len(records) != len(questions) * count:
        raise ValueError("Candidate population incomplete")
    seen = set()
    for row in records:
        q = by_id.get(row["id"])
        if q is None or row["question"] != q["question"] or row["split"] != split:
            raise ValueError("Candidate question/split mismatch")
        idx = row["candidate_index"]
        if type(idx) is not int or not 0 <= idx < count or (row["id"], idx) in seen:
            raise ValueError("Duplicate candidate or invalid index")
        seen.add((row["id"], idx))
        if not isinstance(row["text"],str) or type(row["truncated"]) is not bool:
            raise ValueError("Malformed candidate text/truncation")
        if type(row.get("generated_tokens")) is not int or not 1<=row["generated_tokens"]<=32:
            raise ValueError("Malformed candidate token count")
        if not isinstance(row.get("seconds"),(int,float)) or not math.isfinite(row["seconds"]) or row["seconds"]<0:
            raise ValueError("Malformed generation time")
        expected_seed=int(digest([RECIPE["sampling_seed"],split,q["id"]])[:8],16)
        if row.get("generator")!="original_frozen" or row.get("generation_seed")!=expected_seed or row.get("critic_prompt_sha256")!=digest(critic_prompt(row)):
            raise ValueError("Changed generator seed or critic prompt")
        expected = grade(row["text"], q["aliases"], row["truncated"]).asdict()
        if any(row[k] != v for k, v in expected.items()) or row["y"] != int(row["outcome"] == "correct"):
            raise ValueError("Candidate label changed")
        if not math.isfinite(row["original_logit"]): raise ValueError("Nonfinite original score")


def candidates(ctx, split):
    records = read_envelope(ctx["out"] / "candidates" / f"{split}.json", ctx, split=split)
    validate_candidates(records, ctx["splits"][split], split)
    return records


def prepare_candidates(ctx, max_seconds, backend_factory=Backend, splits=("train", "calibration", "dev")):
    out = ctx["out"]
    with exclusive(out):
        seal_context(ctx)
        phase = "test-candidates" if "evaluation_lock_sha256" in ctx else "candidates"
        pending = [s for s in splits if not (out / "candidates" / f"{s}.json").exists()]
        if not pending:
            verify_cost(out/"costs"/f"{phase}.json",ctx)
            return complete_candidates(ctx,splits,phase)
        with Budget(out / "costs" / f"{phase}.json", max_seconds, ctx["sha"]) as budget:
            backend = backend_factory(ctx)
            try:
                for split in splits:
                    if split not in pending:
                        candidates(ctx, split); continue
                    all_rows = []
                    for q in ctx["splits"][split]:
                        shard = out / "candidate-shards" / split / f"{digest(q['id'])}.json"
                        if shard.exists():
                            records = read_envelope(shard, ctx, split=split, question_id=q["id"])
                        else:
                            budget.check()
                            seed = int(digest([RECIPE["sampling_seed"], split, q["id"]])[:8], 16)
                            set_seed(seed)
                            generated = backend.generate(q, False)
                            if split == "train": generated += backend.generate(q, True)
                            records = []
                            for index, gen in enumerate(generated):
                                label = grade(gen["text"], q["aliases"], gen["truncated"]).asdict()
                                records.append({"id": q["id"], "question": q["question"], "candidate_index": index,
                                                "split": split, **gen, **label, "y": int(label["outcome"] == "correct"),
                                                "generation_seed": seed, "generator": "original_frozen",
                                                "critic_prompt_sha256": digest(confidence_messages(q["question"], gen["text"]))})
                            zs = backend.score(records)
                            for row, z in zip(records, zs): row["original_logit"] = z
                            validate_candidates(records, [q], split)
                            atomic(shard, envelope(ctx, records, split=split, question_id=q["id"]), immutable=True)
                        validate_candidates(records, [q], split)
                        all_rows.extend(records)
                    validate_candidates(all_rows, ctx["splits"][split], split)
                    atomic(out / "candidates" / f"{split}.json", envelope(ctx, all_rows, split=split), immutable=True)
            finally: backend.close()
            budget.check()
        return complete_candidates(ctx,splits,phase)


def complete_candidates(ctx,splits,phase):
    verify_identity(ctx)
    out=ctx["out"];all_rows={s:candidates(ctx,s) for s in splits}
    result={s:len(rs) for s,rs in all_rows.items()}
    atomic(out/"candidates"/f"{phase}-complete.json",{"status":"complete","identity_sha256":ctx["sha"],"counts":result,
           "files":{s:file_digest(out/"candidates"/f"{s}.json") for s in splits},
           "committed_generated_tokens":sum(r["generated_tokens"] for rs in all_rows.values() for r in rs),
           "committed_generation_seconds":sum(r["seconds"] for rs in all_rows.values() for r in rs),
           "token_cost_limit":"Committed samples only; elapsed attempt costs include failed work and model loads."},immutable=True)
    return result


def tree_hash(path):
    return digest({str(p.relative_to(path)): file_digest(p) for p in sorted(path.rglob("*")) if p.is_file()})


def train_loop(model, optimizer, records, order, loss_for, after_step, start_step=0, stop_step=None):
    """One fixed epoch, each candidate exactly once; resumable at optimizer boundaries."""
    import torch
    effective = RECIPE["micro_batch"] * RECIPE["accumulation"]
    if len(order) != len(records) or set(order) != set(range(len(records))) or len(records) % effective:
        raise ValueError("Training requires a complete permutation and full equal-size batches")
    steps = len(records) // effective
    stop_step = steps if stop_step is None else stop_step
    for step in range(start_step, stop_step):
        started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        indices = order[step*effective:(step+1)*effective]
        for i in range(0, effective, RECIPE["micro_batch"]):
            batch = [records[j] for j in indices[i:i+RECIPE["micro_batch"]]]
            loss = loss_for(batch)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite BCE loss")
            total_loss += float(loss.detach()) / RECIPE["accumulation"]
            (loss / RECIPE["accumulation"]).backward()
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            raise FloatingPointError("Nonfinite updated parameters")
        device=next(model.parameters()).device
        if device.type=="cuda":torch.cuda.synchronize(device)
        after_step(step + 1, total_loss, float(norm), time.monotonic()-started)


def save_checkpoint(backend, optimizer, directory, step, ctx, seed, order, records):
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    if "identity" in ctx:verify_identity(ctx)
    target = directory / f"step-{step:04d}"
    if target.exists(): raise FileExistsError("Existing checkpoint must never be replaced")
    temp = directory / f".pending-{uuid.uuid4().hex}"
    temp.mkdir(parents=True)
    backend.model.save_pretrained(temp / "adapter", safe_serialization=True)
    current = get_peft_model_state_dict(backend.model)
    saved = load_file(str(temp / "adapter/adapter_model.safetensors"))
    if set(current) != set(saved) or not all(torch.equal(v.detach().cpu(), saved[k]) for k, v in current.items()):
        raise ValueError("Saved LoRA parameters differ from memory")
    torch.save({"optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, temp / "state.pt")
    probe = backend.score(records[:4])
    atomic(temp / "metadata.json", {"identity_sha256": ctx["sha"], "seed": seed, "step": step,
           "order_sha256": digest(order), "candidates_sha256": digest(records), "probe_logits": probe,
           "adapter_sha256": tree_hash(temp / "adapter"), "state_sha256": file_digest(temp / "state.pt")})
    os.replace(temp, target)
    return target


def restore(backend, optimizer, checkpoint, ctx, seed, order, records):
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    meta = read_json(checkpoint / "metadata.json")
    expected = {"identity_sha256": ctx["sha"], "seed": seed, "order_sha256": digest(order), "candidates_sha256": digest(records)}
    if any(meta.get(k) != v for k, v in expected.items()) or tree_hash(checkpoint / "adapter") != meta["adapter_sha256"] or file_digest(checkpoint / "state.pt") != meta["state_sha256"]:
        raise ValueError("Checkpoint identity/content changed")
    state = load_file(str(checkpoint / "adapter/adapter_model.safetensors"))
    set_peft_model_state_dict(backend.model, state)
    from peft import get_peft_model_state_dict
    loaded = get_peft_model_state_dict(backend.model)
    if set(loaded) != set(state) or not all(torch.equal(v.detach().cpu(), state[k]) for k,v in loaded.items()):
        raise ValueError("Reload did not restore the exact adapter tensors")
    saved = torch.load(checkpoint / "state.pt", map_location=backend.model.device, weights_only=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
    probe = backend.score(records[:4])
    if not all(abs(a-b) <= 1e-6 for a,b in zip(probe, meta["probe_logits"])):
        raise ValueError("Reloaded critic scores differ from saved probe")
    return meta


def train(ctx, seed, max_seconds, pilot=False):
    import torch
    if seed not in RECIPE["seeds"]: raise ValueError("Unregistered training seed")
    out = ctx["out"]
    directory = out / ("pilots" if pilot else "runs") / f"s{seed}"
    with exclusive(out):
        seal_context(ctx)
        if not pilot:
            pilot_done=read_json(out/"pilots/s17/complete.json")
            if pilot_done.get("status")!="complete" or pilot_done["identity_sha256"]!=ctx["sha"] or pilot_done["steps"]!=16 or tree_hash(out/pilot_done["checkpoint"])!=pilot_done["checkpoint_sha256"]:
                raise ValueError("A verified disposable 16-step technical pilot is required")
            verify_cost(out/"costs/pilot-s17.json",ctx)
        records = candidates(ctx, "train")
        if len(records) != 8000 or Counter(r["id"] for r in records) != Counter({q["id"]:4 for q in ctx["splits"]["train"]}):
            raise ValueError("TRAIN must contain all four candidates per question")
        final = directory / "complete.json"
        if final.exists():
            if not pilot: return verify_run(ctx, seed)
            done=read_json(final)
            if done["identity_sha256"]!=ctx["sha"] or tree_hash(out/done["checkpoint"])!=done["checkpoint_sha256"]:
                raise ValueError("Completed technical pilot changed")
            verify_cost(out/"costs"/f"pilot-s{seed}.json",ctx)
            return done
        phase = "pilot" if pilot else "train"
        with Budget(out / "costs" / f"{phase}-s{seed}.json", max_seconds, ctx["sha"]) as budget:
            set_seed(seed)
            backend = Backend(ctx, trainable=True)
            try:
                trainable = [p for p in backend.model.parameters() if p.requires_grad]
                if not trainable or any("lora_" not in n for n,p in backend.model.named_parameters() if p.requires_grad):
                    raise ValueError("Only LoRA parameters may train")
                optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.)
                order = torch.randperm(len(records), generator=torch.Generator().manual_seed(seed)).tolist()
                checkpoints = directory / "checkpoints"
                checkpoints.mkdir(parents=True, exist_ok=True)
                existing = sorted(checkpoints.glob("step-*"))
                start = restore(backend, optimizer, existing[-1], ctx, seed, order, records)["step"] if existing else 0
                if not existing:
                    baseline = backend.score(records[:4])
                    with backend.model.disable_adapter(): original = backend.score(records[:4])
                    if baseline != original: raise ValueError("LoRA initialization changed original P(True)")
                    atomic(directory / "initial.json", {"identity_sha256":ctx["sha"], "probe_logits":baseline}, immutable=True)
                stop = 16 if pilot else 500
                if start > stop: return {"status":"already_beyond_pilot", "step":start}
                def loss_for(batch):
                    budget.check()
                    encoded = encode(backend.tokenizer, batch).to(backend.model.device)
                    z = binary_logits(backend.model, encoded, backend.ids)
                    y = torch.tensor([r["y"] for r in batch], device=z.device, dtype=torch.float32)
                    return torch.nn.functional.binary_cross_entropy_with_logits(z, y)
                def after_step(step, loss, norm, seconds):
                    budget.check()
                    step_record={"identity_sha256":ctx["sha"], "step":step,"attempt_id":budget.current["id"],
                           "bce":loss, "gradient_norm_before_clip":norm,"seconds":seconds,
                           "seconds_scope":"optimizer step including forward/backward/clipping/update; excludes checkpoint I/O"}
                    atomic(directory/"steps-by-attempt"/budget.current["id"]/f"{step:04d}.json",step_record,immutable=True)
                    atomic(directory/"steps"/f"{step:04d}.json",step_record)
                    atomic_jsonl(directory/"steps.jsonl",[read_json(directory/"steps"/f"{i:04d}.json") for i in range(1,step+1)])
                    if step == 16 or step % 50 == 0 or step == stop:
                        cp = save_checkpoint(backend, optimizer, checkpoints, step, ctx, seed, order, records)
                        restore(backend, optimizer, cp, ctx, seed, order, records)
                        if step == 16:
                            norms = [read_json(directory / "steps" / f"{s:04d}.json")["gradient_norm_before_clip"] for s in range(1,17)]
                            if not any(x > 0 for x in norms): raise ValueError("Technical pilot has no nonzero gradients")
                            atomic(directory / "pilot.json", {"identity_sha256":ctx["sha"], "steps":16,
                                   "checkpoint_sha256":tree_hash(cp), "save_reload_exact":True,
                                   "probe_reproduced_atol":1e-6, "no_outcome_based_recipe_selection":True}, immutable=True)
                train_loop(backend.model, optimizer, records, order, loss_for, after_step, start, stop)
                cp = checkpoints / f"step-{stop:04d}"
                atomic(final, {"status":"complete","identity_sha256":ctx["sha"], "seed":seed, "steps":stop,"epochs":stop/500,
                           "examples_seen":stop*16,"checkpoint":str(cp.relative_to(out)),
                           "checkpoint_sha256":tree_hash(cp),"train_sha256":digest(records),
                           "steps_sha256":file_digest(directory/"steps.jsonl"),
                           "save_roundtrip":True,"reload_score_probe_atol":1e-6,"finite_gradient_steps":stop,
                           "recipe":RECIPE,"technical_pilot_disposable":pilot,
                           "weights_updated":"LoRA critic only; generator frozen"}, immutable=True)
            finally: backend.close()
            budget.check()
        return read_json(final)


def verify_run(ctx, seed):
    verify_identity(ctx)
    result = read_json(ctx["out"] / "runs" / f"s{seed}" / "complete.json")
    if result.get("status")!="complete" or result.get("seed")!=seed or result.get("technical_pilot_disposable") is not False or result["identity_sha256"] != ctx["sha"] or result["steps"] != 500 or result["examples_seen"] != 8000 or result["finite_gradient_steps"]!=500 or result["save_roundtrip"] is not True:
        raise ValueError("Training completion identity/count changed")
    cp=ctx["out"]/result["checkpoint"]
    if cp!=ctx["out"]/"runs"/f"s{seed}"/"checkpoints/step-0500" or tree_hash(cp) != result["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint changed")
    meta=read_json(cp/"metadata.json")
    if meta.get("seed")!=seed or meta.get("step")!=500 or meta.get("identity_sha256")!=ctx["sha"] or meta.get("candidates_sha256")!=result["train_sha256"] or tree_hash(cp/"adapter")!=meta["adapter_sha256"] or file_digest(cp/"state.pt")!=meta["state_sha256"]:
        raise ValueError("Checkpoint provenance does not match the completed seed")
    if result["train_sha256"] != digest(candidates(ctx, "train")):
        raise ValueError("Training candidates changed")
    steps_path=ctx["out"]/"runs"/f"s{seed}"/"steps.jsonl"
    if file_digest(steps_path)!=result["steps_sha256"]:raise ValueError("Training step log changed")
    steps=read_jsonl(steps_path)
    if [r["step"] for r in steps]!=list(range(1,501)) or any(not math.isfinite(r[k]) for r in steps for k in ("bce","gradient_norm_before_clip","seconds")):
        raise ValueError("Training log is incomplete/nonfinite")
    verify_cost(ctx["out"] / "costs" / f"train-s{seed}.json",ctx)
    return result


def fit_logistic(records, zs, *, split, equal_question=False):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    if any(r["split"] != split for r in records) or len(records) != len(zs):
        raise ValueError("Wrong logistic fit population")
    y = np.array([r["y"] for r in records])
    if len(set(y)) != 2: raise ValueError("Both classes required to fit logistic regression")
    counts = Counter(r["id"] for r in records)
    weights = np.array([1/counts[r["id"]] if equal_question else 1. for r in records])
    if not equal_question and any(v != 1 for v in counts.values()): raise ValueError("Calibration must contain one candidate per question")
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, random_state=0).fit(np.asarray(zs).reshape(-1,1), y, sample_weight=weights)
    return {"coefficient":float(model.coef_[0,0]),"intercept":float(model.intercept_[0]),"C":1.,
            "fit_split":split,"rows":len(records),"questions":len(counts),"equal_question":equal_question,
            "input_sha256":digest({"records":records,"logits":zs})}


def transformed(fit, zs):
    return [fit["coefficient"] * z + fit["intercept"] for z in zs]


def metrics(records, probabilities, taus, logits=None):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    if len(records) != len(probabilities) or not records or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid probability population")
    if logits is not None and (len(logits)!=len(records) or any(not math.isfinite(z) for z in logits)
            or any(not math.isclose(p,sigmoid(z),rel_tol=1e-12,abs_tol=1e-12) for p,z in zip(probabilities,logits))):
        raise ValueError("Logits must be finite and match the supplied probabilities")
    result = {"n":len(records),"probability_scores":{},"thresholds":{}}
    for name, indices in (("all",list(range(len(records)))), ("substantive",[i for i,r in enumerate(records) if eligible(r)])):
        if not indices:
            result["probability_scores"][name] = {"n":0}; continue
        y = np.array([records[i]["y"] for i in indices]); ps = np.array([probabilities[i] for i in indices])
        clipped = np.clip(ps,1e-15,1-1e-15)
        zs = np.array([logits[i] for i in indices]) if logits is not None else None
        log_loss=float(np.mean(np.logaddexp(0,zs)-y*zs)) if zs is not None else float(-np.mean(y*np.log(clipped)+(1-y)*np.log1p(-clipped)))
        result["probability_scores"][name] = {"n":len(indices),"brier":float(np.mean((ps-y)**2)),
             "log_loss":log_loss,
             "auc":float(roc_auc_score(y,zs if zs is not None else ps)) if len(set(y))==2 else None}
    for tau in taus:
        final = [{**r,"tau":tau,"outcome":r["outcome"] if eligible(r) and p>tau else "abstain"}
                 for r,p in zip(records,probabilities)]
        result["thresholds"][str(tau)] = summarize(final)
    return result


def evaluate(ctx, seed, max_seconds, backend_factory=Backend):
    out = ctx["out"]
    with exclusive(out):
        seal_context(ctx)
        run = verify_run(ctx, seed)
        folder = out / "evaluation" / f"s{seed}"
        if (folder / "complete.json").exists():
            done = read_json(folder / "complete.json")
            if done["identity_sha256"] != ctx["sha"] or any(file_digest(folder / p)!=h for p,h in done["files"].items()):
                raise ValueError("Completed selector evaluation changed")
            verify_cost(out/"costs"/f"evaluate-s{seed}.json",ctx)
            verify_cost(out/"costs"/f"evaluate-cpu-s{seed}.json",ctx)
            return read_json(folder / "results.json")
        records = {s:candidates(ctx,s) for s in ("train","calibration","dev")}
        logits = {}
        with Budget(out / "costs" / f"evaluate-s{seed}.json", max_seconds, ctx["sha"]) as budget:
            backend = backend_factory(ctx, adapter=out / run["checkpoint"] / "adapter")
            try:
                for split in ("calibration","dev"):
                    values=[]
                    for start in range(0,len(records[split]),16):
                        shard=folder / "score-shards" / split / f"{start:05d}.json"
                        chunk=records[split][start:start+16]
                        if shard.exists(): zs=read_envelope(shard,ctx,candidates_sha256=digest(chunk),seed=seed)
                        else:
                            budget.check(); zs=backend.score(chunk)
                            atomic(shard,envelope(ctx,zs,candidates_sha256=digest(chunk),seed=seed),immutable=True)
                        if len(zs)!=len(chunk) or not all(math.isfinite(z) for z in zs): raise ValueError("Invalid score cache")
                        values.extend(zs)
                    logits[split]=values
            finally: backend.close()
            budget.check()
        gpu_cost=verify_cost(out/"costs"/f"evaluate-s{seed}.json",ctx)
        remaining=max_seconds-sum(a["seconds"] for a in gpu_cost["attempts"])
        with Budget(out/"costs"/f"evaluate-cpu-s{seed}.json",remaining,ctx["sha"]) as cpu_budget:
            result=finish_development(ctx,seed,folder,records,logits)
            cpu_budget.check()
        return result


def finish_development(ctx,seed,folder,records,logits):
        verify_identity(ctx)
        cal,dev,train_rows=records["calibration"],records["dev"],records["train"]
        original={s:[r["original_logit"] for r in records[s]] for s in records}
        standard=fit_logistic(cal,original["calibration"],split="calibration")
        supervised=fit_logistic(train_rows,original["train"],split="train",equal_question=True)
        supervised_recal=fit_logistic(cal,transformed(supervised,original["calibration"]),split="calibration")
        critic_cal=fit_logistic(cal,logits["calibration"],split="calibration")
        zs={"original_raw":original["dev"],"original_calibrated":transformed(standard,original["dev"]),
            "train_logistic_raw":transformed(supervised,original["dev"]),
            "train_logistic_recalibrated":transformed(supervised_recal,transformed(supervised,original["dev"])),
            "critic_raw":logits["dev"],"critic_calibrated":transformed(critic_cal,logits["dev"])}
        fits={"standard":standard,"train_supervised":supervised,"train_recalibration":supervised_recal,"critic":critic_cal}
        taus=thresholds(ctx)
        result={"identity_sha256":ctx["sha"],"seed":seed,"split":"dev","candidates_sha256":digest(dev),
                "methods":{name:metrics(dev,[sigmoid(z) for z in values],taus,logits=values) for name,values in zs.items()},
                "limits":["EM correctness proxy; not semantic truth or an approved reward judge.",
                          "The TRAIN logistic control uses the same labels but only one existing score feature; two affine logit transforms compose to one affine transform. Regularization means its fit need not equal calibration-only Platt.",
                          "All selectors use identical frozen candidates. Invalid candidates are scored but never emitted; all and substantive probability metrics reported separately.",
                          "No test used for fit or selection; all fixed seeds must be reported."]}
        scored=[{**r, "logits":{name:vals[i] for name,vals in zs.items()}} for i,r in enumerate(dev)]
        atomic(folder/"calibrators.json",fits,immutable=True)
        atomic(folder/"dev-scores.json",envelope(ctx,scored,split="dev",seed=seed),immutable=True)
        atomic(folder/"results.json",result,immutable=True)
        atomic(folder/"complete.json",{"status":"complete","identity_sha256":ctx["sha"],"seed":seed,"files":{p:file_digest(folder/p) for p in ("calibrators.json","dev-scores.json","results.json")}},immutable=True)
        return result


def thresholds(ctx, test=False):
    spec=ctx["contract"].get("evaluation",{})
    values=sorted(set(spec.get("primary_thresholds",[.65,.85]) if test else spec.get("dev_thresholds",[.6,.75,.9])))
    if any(not math.isfinite(t) or not 0<t<1 for t in values): raise ValueError("Invalid evaluation thresholds")
    return values


def locked_test_context(ctx, lock_path):
    """A separate root-owned lock must freeze all three models and calibrators."""
    lock=read_json(lock_path)
    if lock.get("status")!="ready" or lock.get("contract_sha256")!=ctx["identity"]["contract_sha256"]:
        raise ValueError("Test evaluation lock is absent, unready, or belongs to another contract")
    if set(lock["selector_completion_sha256"])!={str(s) for s in RECIPE["seeds"]}:
        raise ValueError("Test requires all three completed selector seeds")
    if set(lock["selector_evaluation_sha256"])!={str(s) for s in RECIPE["seeds"]}:
        raise ValueError("Test requires all three frozen selector calibration receipts")
    verify_rl_gate(ctx,lock)
    for seed in RECIPE["seeds"]:
        verify_run(ctx,seed)
        training=ctx["out"]/"runs"/f"s{seed}"/"complete.json"
        if file_digest(training)!=lock["selector_completion_sha256"][str(seed)]: raise ValueError("Locked trained selector changed")
        p=ctx["out"]/"evaluation"/f"s{seed}"/"complete.json"
        if file_digest(p)!=lock["selector_evaluation_sha256"][str(seed)]: raise ValueError("Locked selector calibration changed")
        done=read_json(p)
        if done["identity_sha256"]!=ctx["sha"] or any(file_digest(p.parent/name)!=h for name,h in done["files"].items()):
            raise ValueError("Locked development/calibrator files changed")
        verify_cost(ctx["out"]/"costs"/f"evaluate-s{seed}.json",ctx)
        verify_cost(ctx["out"]/"costs"/f"evaluate-cpu-s{seed}.json",ctx)
    splits=dict(ctx["splits"])
    for name,n in (("test_trivia",2000),("test_nq",500)):
        spec=lock["data"][name]; path=resolve(ctx["root"],spec["path"])
        if file_digest(path)!=spec["sha256"]: raise ValueError("Locked test data changed")
        splits[name]=read_jsonl(path)
        if len(splits[name])!=n: raise ValueError("Wrong reserved test population")
    assert_disjoint(splits)
    enhanced={**ctx,"splits":splits,"evaluation_lock_sha256":file_digest(lock_path)}
    atomic(ctx["out"]/"test-lock.json", {"evaluation_lock_sha256":file_digest(lock_path),"lock":lock},immutable=True)
    return enhanced


def verify_rl_gate(ctx,lock):
    artifacts=ctx["out"].parent
    source=artifacts/"source-lock.json";budget_path=artifacts/"budget-lock.json"
    if file_digest(source)!=lock["source_lock_sha256"] or file_digest(budget_path)!=lock["budget_lock_sha256"]:
        raise ValueError("Budget or source lock changed")
    budget=read_json(budget_path);source_lock=read_json(source)
    if (budget["rl_questions"],budget["rl_steps"],budget["rl_completions"]) not in {(1008,504,24192),(504,252,12096)}:
        raise ValueError("Unknown registered RL budget")
    arms=("a_g4_single_tau","b_g8_single_tau","c_g8_paired_tau","d_g8_paired_fixed075")
    expected={f"{arm}-s{seed}" for arm in arms for seed in RECIPE["seeds"]}
    if set(lock["rl_completed"])!=expected:
        raise ValueError("Test requires exactly all twelve RL runs")
    for variant,entry in lock["rl_completed"].items():
        path=resolve(ctx["root"],entry["result_path"]);adapter=resolve(ctx["root"],entry["adapter"])
        if file_digest(path)!=entry["result_sha256"]:raise ValueError("RL result receipt changed")
        result=read_json(path)
        if result.get("status")!="complete" or result.get("optimizer_steps")!=budget["rl_steps"] or result.get("gradient_steps")!=budget["rl_steps"] or result.get("questions")!=budget["rl_questions"] or result.get("completions")!=budget["rl_completions"]:
            raise ValueError("Incomplete RL run in test lock")
        arm,seed_text=variant.rsplit("-s",1)
        expected_spec={"group_size":4 if arm==arms[0] else 8,"arm":"fixed" if arm==arms[3] else "conditioned",
                       "seed":int(seed_text),"exposure_mode":"single_tau" if arm in arms[:2] else "paired_tau",
                       "questions_per_update":2,"micro_batch_size":4}
        if result.get("spec")!=expected_spec or result.get("save_roundtrip") is not True or result.get("sampler_traversal_verified") is not True:
            raise ValueError("RL arm/seed or verification flags differ from the test lock")
        actual={name:file_digest(adapter/name) for name in ("adapter_model.safetensors","adapter_config.json")}
        if actual!=entry["adapter_sha256"] or actual!=result["adapter_sha256"]:
            raise ValueError("RL adapter changed")
    if source_lock["image"]!=IMAGE:raise ValueError("Wrong locked image")
    for relative,h in source_lock["files"].items():
        if file_digest(resolve(ctx["root"],relative))!=h:raise ValueError("Locked source changed")


def evaluate_test(ctx, seed, max_seconds, backend_factory=Backend):
    if "evaluation_lock_sha256" not in ctx: raise ValueError("Explicit test lock required")
    out=ctx["out"]
    with exclusive(out):
        run=verify_run(ctx,seed)
        folder=out/"test-evaluation"/f"s{seed}"
        final=folder/"complete.json"
        if final.exists():
            done=read_json(final)
            if done["identity_sha256"]!=ctx["sha"] or done["evaluation_lock_sha256"]!=ctx["evaluation_lock_sha256"] or any(file_digest(folder/p)!=h for p,h in done["files"].items()):
                raise ValueError("Completed test selector artifacts changed")
            verify_cost(out/"costs"/f"test-evaluate-s{seed}.json",ctx)
            return read_json(folder/"results.json")
        fits=read_json(out/"evaluation"/f"s{seed}"/"calibrators.json")
        results={"identity_sha256":ctx["sha"],"evaluation_lock_sha256":ctx["evaluation_lock_sha256"],
                 "seed":seed,"calibrators_refitted":False,"splits":{}}
        with Budget(out/"costs"/f"test-evaluate-s{seed}.json",max_seconds,ctx["sha"]) as budget:
            backend=backend_factory(ctx,adapter=out/run["checkpoint"]/"adapter")
            try:
                for split in ("test_trivia","test_nq"):
                    records=candidates(ctx,split); scores=[]
                    for start in range(0,len(records),16):
                        chunk=records[start:start+16]; shard=folder/"score-shards"/split/f"{start:05d}.json"
                        if shard.exists(): zs=read_envelope(shard,ctx,candidates_sha256=digest(chunk),seed=seed)
                        else:
                            budget.check(); zs=backend.score(chunk)
                            atomic(shard,envelope(ctx,zs,candidates_sha256=digest(chunk),seed=seed),immutable=True)
                        if len(zs)!=len(chunk) or not all(math.isfinite(z) for z in zs): raise ValueError("Invalid test score cache")
                        scores.extend(zs)
                    original=[r["original_logit"] for r in records]
                    zs={"original_raw":original,"original_calibrated":transformed(fits["standard"],original),
                        "train_logistic_raw":transformed(fits["train_supervised"],original),
                        "train_logistic_recalibrated":transformed(fits["train_recalibration"],transformed(fits["train_supervised"],original)),
                        "critic_raw":scores,"critic_calibrated":transformed(fits["critic"],scores)}
                    scored=[{**r,"logits":{name:values[i] for name,values in zs.items()}} for i,r in enumerate(records)]
                    atomic(folder/f"{split}-scores.json",envelope(ctx,scored,split=split,seed=seed),immutable=True)
                    results["splits"][split]={"candidates_sha256":digest(records),"methods":{name:metrics(records,[sigmoid(z) for z in values],thresholds(ctx,test=True),logits=values) for name,values in zs.items()}}
            finally: backend.close()
            budget.check()
        atomic(folder/"results.json",results,immutable=True)
        verify_identity(ctx)
        atomic(final,{"status":"complete","identity_sha256":ctx["sha"],"evaluation_lock_sha256":ctx["evaluation_lock_sha256"],
               "files":{p:file_digest(folder/p) for p in ("test_trivia-scores.json","test_nq-scores.json","results.json")}},immutable=True)
        return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command",choices=["prepare-candidates","pilot","train","evaluate","verify","prepare-test-candidates","evaluate-test"])
    ap.add_argument("--contract",type=Path,required=True)
    ap.add_argument("--seed",type=int,choices=RECIPE["seeds"])
    ap.add_argument("--max-seconds",type=float)
    ap.add_argument("--evaluation-lock",type=Path)
    args=ap.parse_args(); ctx=context(args.contract)
    if args.command != "verify" and args.max_seconds is None: ap.error("--max-seconds is mandatory")
    if args.command in {"pilot","train","evaluate","verify","evaluate-test"} and args.seed is None: ap.error("--seed is mandatory")
    if args.command in {"prepare-test-candidates","evaluate-test"}:
        if args.evaluation_lock is None: ap.error("--evaluation-lock is mandatory for test")
        ctx=locked_test_context(ctx,args.evaluation_lock)
    if args.command=="prepare-candidates": result=prepare_candidates(ctx,args.max_seconds)
    elif args.command=="prepare-test-candidates": result=prepare_candidates(ctx,args.max_seconds,splits=("test_trivia","test_nq"))
    elif args.command=="evaluate-test": result=evaluate_test(ctx,args.seed,args.max_seconds)
    elif args.command in {"pilot","train"}: result=train(ctx,args.seed,args.max_seconds,pilot=args.command=="pilot")
    elif args.command=="evaluate": result=evaluate(ctx,args.seed,args.max_seconds)
    else: result=verify_run(ctx,args.seed)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__ == "__main__": main()
