from __future__ import annotations

import math
import os
import time
from collections import Counter
from pathlib import Path

from .config import fingerprint, matrix
from .data import load_split, training_rows, verify_data
from .io import append_jsonl, digest, read_json, read_jsonl, utcnow, write_json, write_jsonl
from .models import Engine, eos_ids, is_truncated, load_model, render
from .prompts import messages
from .protocol import adapter_fingerprint, code_fingerprint, environment, record_cost, require_budget, require_lock, spent_hours
from .scoring import grade, reward


def lora_config(config):
    from peft import LoraConfig
    t = config["training"]
    return LoraConfig(r=t["lora_r"], lora_alpha=t["lora_alpha"], lora_dropout=0,
                      target_modules="all-linear", bias="none", task_type="CAUSAL_LM")


class RewardRecorder:
    __name__ = "factual_utility"

    def __init__(self, config, arm, eos, path):
        self.config, self.arm, self.eos, self.path = config, arm, eos, Path(path)

    def __call__(self, completions, completion_ids, aliases, tau, id, **kwargs):
        rewards, records = [], []
        step = getattr(kwargs.get("trainer_state"), "global_step", None)
        for text, tokens, answers, threshold, question_id in zip(completions, completion_ids, aliases, tau, id):
            if isinstance(text, list):
                text = text[-1]["content"]
            result = grade(text, answers, is_truncated(tokens, self.eos))
            value = reward(result.outcome, threshold, self.arm, self.config["training"]["fixed_tau"])
            rewards.append(value)
            records.append({"id": question_id, "tau": threshold, "text": text,
                            **result.asdict(), "reward": value, "step": step})
        with self.path.open("a") as f:
            import json
            for record in records:
                f.write(json.dumps(record) + "\n")
        return rewards


def _train(config, workdir, key, arm, seed, rows, outdir, steps, initializer=None, resume=False):
    import gc
    import psutil
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from peft import get_peft_model

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result_path = outdir / "result.json"
    if result_path.exists():
        return read_json(result_path)
    rollout_path = outdir / "rollouts.jsonl"
    if rollout_path.exists() and not resume:
        raise RuntimeError(f"Partial run exists: {outdir}. Use --resume; no silent overwrite.")
    if resume:
        checkpoints = sorted(outdir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if not checkpoints:
            raise RuntimeError("No checkpoint available to resume.")
        completed_step = read_json(checkpoints[-1] / "trainer_state.json")["global_step"]
        for name in ("rollouts.jsonl", "steps.jsonl", "logs.jsonl"):
            path = outdir / name
            if path.exists():
                history = read_jsonl(path)
                keep = [r for r in history if r["step"] < completed_step] if name == "rollouts.jsonl" else [r for r in history if r["step"] <= completed_step]
                if len(keep) != len(history):
                    write_jsonl(outdir / f"discarded-{name}-{int(time.time())}", history[len(keep):])
                    write_jsonl(path, keep)
    set_seed(seed)
    model, tokenizer = load_model(config, workdir, key, initializer=initializer)
    model = get_peft_model(model, lora_config(config))
    # TRL's generation context can restore checkpointing with the reentrant default.
    # Frozen embeddings still need differentiable outputs for LoRA gradients.
    model.enable_input_require_grads()
    model.config.use_cache = False
    t = config["training"]
    records = []
    for row in rows:
        prompt = render(tokenizer, messages(row["question"], row["tau"]))
        if len(tokenizer.encode(prompt, add_special_tokens=False)) > t["max_prompt_length"]:
            raise ValueError(f"Prompt too long: {row['id']}")
        records.append({**row, "prompt": prompt})
    write_json(outdir / "run.json", {"model": key, "arm": arm, "seed": seed, "steps": steps,
                                     "initializer": str(initializer) if initializer else None,
                                     "config_sha256": fingerprint(config), "code_sha256": code_fingerprint(),
                                     "train_ids_sha256": digest([r["id"] for r in rows]),
                                     "environment": environment()})

    class Monitor(TrainerCallback):
        def __init__(self):
            self.gradients_checked = 0
            self.min_available = psutil.virtual_memory().available / 2**30
            self.started_step = time.monotonic()

        def on_step_begin(self, args, state, control, **kwargs):
            self.started_step = time.monotonic()
            require_budget(config, workdir)

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not torch.stack([torch.isfinite(g).all() for g in grads]).all().item():
                raise FloatingPointError("Missing or nonfinite gradients; stopping run.")
            self.gradients_checked += 1

        def on_step_end(self, args, state, control, **kwargs):
            available = psutil.virtual_memory().available / 2**30
            self.min_available = min(self.min_available, available)
            append_jsonl(outdir / "steps.jsonl", {"step": state.global_step,
                         "seconds": time.monotonic() - self.started_step, "available_gib": available,
                         "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                         "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30})
            if available < config["pilot"]["min_system_available_gib"]:
                raise MemoryError("System memory fell below the registered 16 GiB reserve.")
            deadline = os.environ.get("ABSTENTION_STOP_AT")
            over_budget = spent_hours(workdir) + (time.monotonic() - started) / 3600 >= config["budget"]["total_gpu_hours"]
            if (deadline and time.time() >= float(deadline)) or over_budget:
                control.should_save = True
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for name in ("loss", "grad_norm"):
                    if name in logs and not math.isfinite(logs[name]):
                        raise FloatingPointError(f"Nonfinite {name}")
                append_jsonl(outdir / "logs.jsonl", {"step": state.global_step, **logs})

    monitor = Monitor()
    args = GRPOConfig(output_dir=str(outdir), learning_rate=t["learning_rate"], beta=t["beta"],
                      per_device_train_batch_size=t["per_device_train_batch_size"],
                      gradient_accumulation_steps=t["gradient_accumulation_steps"],
                      steps_per_generation=t["steps_per_generation"], num_generations=t["num_generations"],
                      max_prompt_length=t["max_prompt_length"], max_completion_length=t["max_completion_length"],
                      temperature=t["temperature"], top_p=1.0, top_k=0, num_iterations=1,
                      scale_rewards="none", loss_type="dr_grpo", max_steps=steps,
                      bf16=True, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={"use_reentrant": False},
                      disable_dropout=True, lr_scheduler_type="constant", warmup_steps=0,
                      max_grad_norm=1.0, optim="adamw_torch", seed=seed, data_seed=seed,
                      logging_steps=1, save_steps=t["checkpoint_steps"], save_total_limit=2,
                      report_to="none", use_vllm=False, dataloader_num_workers=0,
                      remove_unused_columns=False, mask_truncated_completions=False)
    recorder = RewardRecorder(config, arm, eos_ids(model, tokenizer), rollout_path)
    trainer = GRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(records),
                          processing_class=tokenizer, reward_funcs=recorder, callbacks=[monitor])
    # Plain, pre-rendered prompts prevent a second chat-template pass and accidental Qwen thinking.
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer.train(resume_from_checkpoint=True if resume else None)
        if trainer.state.global_step < steps:
            from .runner import RunPaused
            raise RunPaused("Saved checkpoint at the night-window or budget boundary.")
        trainer.save_model(str(outdir / "adapter"))
        tokenizer.save_pretrained(str(outdir / "adapter"))
        trainer.save_state()
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file
        in_memory = get_peft_model_state_dict(model)
        saved = load_file(str(outdir / "adapter/adapter_model.safetensors"))
        save_roundtrip = set(saved) == set(in_memory) and all(torch.equal(saved[k], in_memory[k].detach().cpu()) for k in saved)
        if not save_roundtrip:
            raise RuntimeError("Saved adapter parameters differ from the trained model.")
        elapsed = time.monotonic() - started
        rollouts = read_jsonl(rollout_path)
        counts = Counter(r["id"] for r in rollouts)
        expected_questions = steps * t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] // t["num_generations"]
        traversal_ok = len(counts) == min(expected_questions, len(rows)) and all(v == t["num_generations"] for v in counts.values())
        if not traversal_ok:
            raise RuntimeError(f"Unexpected sampler traversal: {len(counts)} questions, counts {Counter(counts.values())}")
        result = {"completed_at": utcnow(), "steps": trainer.state.global_step,
                  "elapsed_seconds": elapsed, "seconds_per_update": elapsed / (monitor.gradients_checked or 1),
                  "gradients_checked": monitor.gradients_checked, "min_available_gib": monitor.min_available,
                  "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "unique_questions": len(counts), "completion_count": len(rollouts),
                  "sampler_traversal_ok": traversal_ok, "save_roundtrip": save_roundtrip, "resumed": resume,
                  "adapter_sha256": adapter_fingerprint(outdir / "adapter")}
        write_json(result_path, result)
        return result
    finally:
        record_cost(workdir, "train", time.monotonic() - started, model=key, arm=arm, seed=seed, directory=str(outdir))
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


def exploration_gate(config, engine, rows, output):
    from transformers import set_seed
    set_seed(config["split_seed"])
    n = config["training"]["num_generations"]
    batch = config["evaluation"]["batch_size"] // n
    records = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start+batch]
        conversations = [messages(r["question"], r["tau"]) for r in chunk for _ in range(n)]
        outputs = engine.generate(conversations, sample=True)
        for i, r in enumerate(chunk):
            group = outputs[i*n:(i+1)*n]
            grades = [grade(g["text"], r["aliases"], g["truncated"]).outcome for g in group]
            records.append({"id": r["id"], "tau": r["tau"], "outputs": group, "outcomes": grades,
                            "mixed": "abstain" in grades and any(x != "abstain" for x in grades)})
        print(f"exploration: {min(start+batch, len(rows))}/{len(rows)}", flush=True)
    write_jsonl(output, records)
    mixed = sum(r["mixed"] for r in records)
    return {"groups": len(records), "mixed_groups": mixed, "mixed_fraction": mixed / len(records),
            "passed": mixed >= math.ceil(config["pilot"]["min_mixed_fraction"] * len(rows))}


def warmup(config, workdir, key, engine, directory):
    import gc
    import torch
    from datasets import Dataset
    from peft import get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed
    correct, incorrect = [], []
    n = config["evaluation"]["batch_size"]
    for offset in range(0, config["data"]["train"], n):
        chunk = load_split(workdir, "train")[offset:offset+n]
        outputs = engine.generate([messages(r["question"], forced=True) for r in chunk])
        for r, out in zip(chunk, outputs):
            result = grade(out["text"], r["aliases"], out["truncated"])
            target = correct if result.outcome == "correct" else incorrect
            if len(target) < 32:
                target.append({**r, "target": out["text"].strip() if result.outcome == "correct" else "IDK",
                               "source_prediction": out["text"], "heuristic_label": result.outcome})
        if len(correct) == len(incorrect) == 32:
            break
    if len(correct) != 32 or len(incorrect) != 32:
        raise RuntimeError("Cannot construct the prescribed balanced warmup from the training partition.")
    selected = [r for pair in zip(correct, incorrect) for r in pair]
    write_jsonl(Path(directory) / "examples.jsonl", selected)
    engine.close()
    set_seed(config["split_seed"])
    model, tokenizer = load_model(config, workdir, key)
    model = get_peft_model(model, lora_config(config))
    records = []
    for i, r in enumerate(selected):
        tau = config["thresholds"]["train"][i % 3]
        prefix = tokenizer.encode(render(tokenizer, messages(r["question"], tau)), add_special_tokens=False)
        answer = tokenizer.encode(r["target"], add_special_tokens=False) + [tokenizer.eos_token_id]
        records.append({"input_ids": prefix + answer, "labels": [-100] * len(prefix) + answer})

    def collate(batch):
        length = max(len(r["input_ids"]) for r in batch)
        return {"input_ids": torch.tensor([r["input_ids"] + [tokenizer.pad_token_id] * (length-len(r["input_ids"])) for r in batch]),
                "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (length-len(r["input_ids"])) for r in batch]),
                "labels": torch.tensor([r["labels"] + [-100] * (length-len(r["labels"])) for r in batch])}
    args = TrainingArguments(output_dir=str(directory), per_device_train_batch_size=4, gradient_accumulation_steps=1,
                             num_train_epochs=1, learning_rate=config["training"]["warmup_learning_rate"],
                             bf16=True, report_to="none", save_strategy="no", seed=config["split_seed"],
                             logging_steps=1, lr_scheduler_type="constant", optim="adamw_torch")
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(records), data_collator=collate)
    trainer.train()
    trainer.save_model(str(Path(directory) / "adapter"))
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return Path(directory) / "adapter"


def pilot(config, workdir, key):
    import torch
    workdir = Path(workdir)
    initial_code_sha256 = code_fingerprint()
    verify_data(config, workdir)
    directory = workdir / "artifacts/pilots" / key
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        result = read_json(directory / "result.json")
        if result["code_sha256"] != code_fingerprint():
            raise RuntimeError("Pilot exists with different code. Archive the old pilot before rerunning.")
        return result
    started = time.monotonic()
    rows = training_rows(load_split(workdir, "pilot"), config["thresholds"]["train"], config["split_seed"])
    engine = Engine(config, workdir, key)
    gate1 = exploration_gate(config, engine, rows, directory / "gate-original.jsonl")
    initializer = None
    gate = gate1
    if not gate1["passed"]:
        initializer = warmup(config, workdir, key, engine, directory / "warmup")
        engine = Engine(config, workdir, key, initializer=initializer)
        gate = exploration_gate(config, engine, rows, directory / "gate-warmup.jsonl")
    sample = rows[:config["evaluation"]["batch_size"]]
    outs = engine.generate([messages(r["question"], r["tau"]) for r in sample])
    eval_seconds = sum(r["seconds"] for r in outs) / len(outs)
    confidence_seconds = sum(engine.confidence(r["question"], o["text"])["confidence_seconds"] for r, o in zip(sample, outs)) / len(sample)
    engine.close()
    initial_elapsed = time.monotonic() - started
    record_cost(workdir, "pilot-gate-and-inference", initial_elapsed, model=key)
    train_result = _train(config, workdir, key, "conditioned", config["split_seed"], rows,
                          directory / "technical-training", config["training"]["pilot_steps"], initializer)
    # Reload the persisted adapter twice and verify deterministic output on a fixed panel.
    reload_started = time.monotonic()
    adapter = directory / "technical-training/adapter"
    engine = Engine(config, workdir, key, adapter=adapter, initializer=initializer)
    probe = [messages(r["question"], r["tau"]) for r in sample[:4]]
    first = [r["text"] for r in engine.generate(probe)]
    engine.close()
    engine = Engine(config, workdir, key, adapter=adapter, initializer=initializer)
    second = [r["text"] for r in engine.generate(probe)]
    engine.close()
    record_cost(workdir, "pilot-reload", time.monotonic() - reload_started, model=key)
    write_json(directory / "reload.json", {"first": first, "second": second, "identical": first == second})
    result = {"model": key, "config_sha256": fingerprint(config), "code_sha256": initial_code_sha256,
              "completed_at": utcnow(), "original_gate": gate1, "final_gate": gate,
              "exploration_pass": gate["passed"], "initializer": str(initializer) if initializer else None,
              "initializer_sha256": adapter_fingerprint(initializer) if initializer else None,
              "technical_pass": first == second and train_result["gradients_checked"] == config["training"]["pilot_steps"] and train_result["sampler_traversal_ok"],
              "seconds_per_update": train_result["seconds_per_update"],
              "seconds_per_eval_question": eval_seconds, "seconds_per_confidence_question": confidence_seconds,
              "elapsed_seconds": time.monotonic() - started, "training": train_result}
    write_json(directory / "result.json", result)
    return result


def train(config, workdir, key, arm, seed, resume=False):
    lock = require_lock(config, workdir)
    run = next((r for r in matrix(config) if (r["model"], r["arm"], r["seed"]) == (key, arm, seed)), None)
    if run is None:
        raise ValueError("Run is not in the registered 15-run matrix.")
    rows = load_split(workdir, "train")[:lock["train_size"]]
    rows = training_rows(rows, config["thresholds"]["train"], seed)
    t = config["training"]
    per_update = t["per_device_train_batch_size"] * t["gradient_accumulation_steps"] // t["num_generations"]
    initializer = lock["pilots"][key]["initializer"]
    return _train(config, workdir, key, arm, seed, rows, Path(workdir) / "artifacts/runs" / run["run_id"],
                  len(rows) // per_update, initializer, resume)
