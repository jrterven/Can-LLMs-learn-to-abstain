"""Isolated TRL 0.24 extension: matched eight draws, balanced tau, LOO credit.

Importing this module never loads a model. The caller selects TRAIN rows and
authorizes execution separately. No frozen source, dataset or protocol is edited.
G8 versus 2xG4 changes only the within-prompt baseline grouping: both generate
eight draws at each of three exposure slots for every question in an optimizer
step. Single-tau repeats one assigned threshold; paired-tau covers all three.
Runs share draw budgets and sampling settings, not identical generated texts
after their policies have diverged through optimization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from abstention.io import digest, read_json, read_jsonl, utcnow, write_json
from abstention.models import eos_ids, is_truncated, load_model, render
from abstention.prompts import messages
from abstention.scoring import grade, reward
from abstention.training import lora_config

TAUS = (0.60, 0.75, 0.90)
DRAWS = 8
TRL_VERSION = "0.24.0"
TRL_SOURCE_SHA256 = "bb6905182acf2ec426f1dc9ffc37e5210d58ee47d2764f7cf12dbb90a4a28954"
MODEL_REVISION = "40c069824f4251a91eefaf281ebe4c544efd3e18"


@dataclass(frozen=True)
class V2Spec:
    group_size: int
    arm: str
    seed: int
    exposure_mode: str = "paired_tau"
    questions_per_update: int = 2
    micro_batch_size: int = 4

    def __post_init__(self):
        if self.group_size not in (4, 8) or self.arm not in ("conditioned", "fixed"):
            raise ValueError("Only G8 or two groups G4, with conditioned or fixed reward, are supported.")
        if self.exposure_mode not in ("single_tau", "paired_tau"):
            raise ValueError("Unknown threshold exposure mode.")
        if (self.group_size, self.exposure_mode, self.arm) not in {
                (4, "single_tau", "conditioned"), (8, "single_tau", "conditioned"),
                (8, "paired_tau", "conditioned"), (8, "paired_tau", "fixed")}:
            raise ValueError("Settings do not identify one of the four registered v2 arms.")
        if self.questions_per_update != 2:
            raise ValueError("The reviewed v2 comparison uses exactly two questions per update.")
        for name in ("questions_per_update", "micro_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.completions_per_update % self.micro_batch_size:
            raise ValueError("Microbatches must divide the complete balanced update.")

    @property
    def completions_per_update(self):
        return self.questions_per_update * len(TAUS) * DRAWS

    @property
    def accumulation_steps(self):
        return self.completions_per_update // self.micro_batch_size


def prepared_records(schedule, spec):
    """Validate the externally selected/ordered schedule, without choosing data."""
    if not schedule or len(schedule) % len(TAUS):
        raise ValueError("Supply exactly three exposure slots per TRAIN question.")
    records, seen, exposure_ids, assigned = [], set(), set(), Counter()
    for start in range(0, len(schedule), len(TAUS)):
        question_index = start // len(TAUS)
        block = schedule[start:start + len(TAUS)]
        first = block[0]
        if first["id"] in seen or not first["question"] or not first["aliases"]:
            raise ValueError("Questions must have unique IDs, text and nonempty references.")
        seen.add(first["id"])
        thresholds = tuple(row["tau"] for row in block)
        if spec.exposure_mode == "paired_tau" and thresholds != TAUS:
            raise ValueError("Paired exposure must contain .60/.75/.90 in its three slots.")
        if spec.exposure_mode == "single_tau" and (len(set(thresholds)) != 1 or thresholds[0] not in TAUS):
            raise ValueError("Single exposure must repeat one registered threshold three times.")
        assigned[thresholds[0]] += 1
        for slot, row in enumerate(block):
            expected_id = digest([first["id"], slot])
            if (row["id"] != first["id"] or row["question"] != first["question"]
                    or row["aliases"] != first["aliases"] or row["slot"] != slot
                    or row["question_index"] != question_index
                    or row["update_index"] != question_index // spec.questions_per_update
                    or row["exposure_id"] != expected_id or expected_id in exposure_ids):
                raise ValueError("Question/slot/exposure identity is inconsistent with the supplied order.")
            exposure_ids.add(expected_id)
            records.append({k: row[k] for k in ("id", "question", "aliases", "tau", "slot",
                                               "question_index", "update_index", "exposure_id")})
            records[-1]["v2_prompt_id"] = digest([row["id"], row["tau"]])
    if spec.exposure_mode == "single_tau" and len({assigned[tau] for tau in TAUS}) != 1:
        raise ValueError("Single-tau question assignments must be exactly balanced over all three thresholds.")
    return records


class BalancedQuestionSampler:
    """Match TRL's buffered dataloader while preserving question x tau blocks.

TRL consumes one full generation batch on each microstep, but reuses the first
batch's generated tensors over the accumulation window. Repeat that exact batch
for the entire window, so consumed-but-buffered input rows never skip questions.
"""
    def __init__(self, records, spec):
        self.records, self.spec = records, spec
        if len(records) % len(TAUS):
            raise ValueError("Incomplete question-threshold panel.")
        self.n_questions = len(records) // len(TAUS)
        if self.n_questions % spec.questions_per_update:
            raise ValueError("A partial optimizer step is forbidden; choose a divisible TRAIN panel.")
        for start in range(0, len(records), len(TAUS)):
            block = [records[i] for i in range(start, start + len(TAUS))]
            expected_taus = TAUS if spec.exposure_mode == "paired_tau" else (block[0]["tau"],) * len(TAUS)
            if (len({r["id"] for r in block}) != 1 or tuple(r["tau"] for r in block) != expected_taus
                    or [r["slot"] for r in block] != [0, 1, 2]
                    or len({r["exposure_id"] for r in block}) != 3):
                raise ValueError("Rows must be contiguous question x distinct exposure-slot blocks.")

    def question_order(self):
        # The data layer already applies the paired seed-specific question order.
        # A second shuffle here would invalidate update/exposure identities.
        return list(range(self.n_questions))

    def __iter__(self):
        order = self.question_order()
        width = self.spec.questions_per_update
        for start in range(0, len(order), width):
            indices = [question * len(TAUS) + t for question in order[start:start + width]
                       for t in range(len(TAUS)) for _ in range(DRAWS)]
            for _ in range(self.spec.accumulation_steps):
                yield from indices

    def __len__(self):
        return len(self.records) * DRAWS * self.spec.accumulation_steps

    def set_epoch(self, epoch):
        if epoch != 0:
            raise RuntimeError("This implementation permits one registered pass, not implicit extra epochs.")


def validate_generation_batch(inputs, spec):
    if len(inputs) != spec.completions_per_update:
        raise ValueError("Incomplete generation batch would change optimizer-step weighting.")
    seen = set()
    width = len(TAUS) * DRAWS
    for start in range(0, len(inputs), width):
        question = inputs[start]["id"]
        if question in seen:
            raise ValueError("Duplicate question in an optimizer step.")
        seen.add(question)
        for j in range(len(TAUS)):
            block = inputs[start + j * DRAWS:start + (j + 1) * DRAWS]
            keys = [(r["id"], r["tau"], r["v2_prompt_id"], r["prompt"], r["slot"], r["exposure_id"]) for r in block]
            tau = TAUS[j] if spec.exposure_mode == "paired_tau" else inputs[start]["tau"]
            if (len(set(keys)) != 1 or keys[0][:2] != (question, tau) or keys[0][4] != j
                    or keys[0][5] != digest([question, j])):
                raise ValueError("Credit group mixes questions, slots, thresholds, or rendered prompts.")


def loo_advantages(rewards, group_size):
    """Reference scalar implementation, also usable for audit/replay without GPU."""
    if group_size not in (4, 8) or len(rewards) % DRAWS:
        raise ValueError("Expected complete eight-draw prompt blocks and G4/G8.")
    if any(not math.isfinite(float(value)) for value in rewards):
        raise ValueError("Nonfinite reward.")
    result = []
    for start in range(0, len(rewards), group_size):
        group = rewards[start:start + group_size]
        total = sum(group)
        result.extend(value - (total - value) / (group_size - 1) for value in group)
    return result


class V2RewardRecorder:
    __name__ = "strict_factual_utility_v2"

    def __init__(self, spec, eos):
        self.spec, self.eos, self.pending = spec, set(eos), None

    def __call__(self, completions, completion_ids, aliases, tau, id, question,
                 v2_prompt_id, exposure_id, slot, question_index, update_index, trainer_state=None, **kwargs):
        arrays = (completions, completion_ids, aliases, tau, id, question, v2_prompt_id,
                  exposure_id, slot, question_index, update_index)
        if any(len(a) != self.spec.completions_per_update for a in arrays):
            raise ValueError("Reward inputs are not aligned with the full generation batch.")
        records = []
        for i, (text, tokens, answers, threshold, qid, qtext, prompt_id, expid, sl, qi, ui) in enumerate(zip(*arrays)):
            if not isinstance(text, str):
                raise ValueError("V2 requires pre-rendered plain-text prompts and completions.")
            truncated = is_truncated(tokens, self.eos)
            result = grade(text, answers, truncated)
            value = reward(result.outcome, threshold, self.spec.arm, fixed_tau=0.75)
            records.append({"id": qid, "question": qtext, "aliases": list(answers),
                "tau": threshold, "v2_prompt_id": prompt_id, "text": text,
                "exposure_id": expid, "slot": sl, "question_index": qi, "update_index": ui,
                "completion_ids": list(tokens), "truncated": truncated, **result.asdict(),
                "reward": value, "sample_index": i % DRAWS,
                "baseline_group_within_prompt": (i % DRAWS) // self.spec.group_size,
                "group_size": self.spec.group_size, "arm": self.spec.arm,
                "step": getattr(trainer_state, "global_step", None)})
        self.pending = records
        return [r["reward"] for r in records]


def append_rollout_batch(path, records):
    import os
    payload = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records)
    with Path(path).open("a") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class MatchedLOOMixin:
    """Replace credit assignment before TRL shuffles/splits generated tensors."""
    def __init__(self, *args, v2_spec, v2_recorder, v2_rollout_path, **kwargs):
        self.v2_spec, self.v2_recorder = v2_spec, v2_recorder
        self.v2_rollout_path = Path(v2_rollout_path)
        self._v2_raw_rewards = None
        super().__init__(*args, **kwargs)
        s, a = self.v2_spec, self.args
        if (self.accelerator.num_processes != 1 or self.num_generations != DRAWS
                or self.num_iterations != 1 or self.scale_rewards != "none" or self.loss_type != "dr_grpo"
                or self.beta != 0.04 or self.use_vllm or self.use_liger_loss
                or a.per_device_train_batch_size != s.micro_batch_size
                or a.gradient_accumulation_steps != s.accumulation_steps
                or a.steps_per_generation != s.accumulation_steps
                or a.generation_batch_size != s.completions_per_update
                or len(self.reward_funcs) != 1 or self.reward_weights.tolist() != [1.0]):
            raise ValueError("Trainer settings violate matched single-GPU LOO protocol.")

    def _get_train_sampler(self, dataset=None):
        return BalancedQuestionSampler(self.train_dataset if dataset is None else dataset, self.v2_spec)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        values = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        self._v2_raw_rewards = values.detach().clone()
        return values

    def _generate_and_score_completions(self, inputs):
        import torch
        if not self.model.training:
            raise RuntimeError("Use the separate frozen evaluator; this trainer accepts balanced TRAIN updates only.")
        validate_generation_batch(inputs, self.v2_spec)
        self._v2_raw_rewards = None
        self.v2_recorder.pending = None
        result = super()._generate_and_score_completions(inputs)
        values = self._v2_raw_rewards
        if values is None or values.shape != (len(inputs), 1) or not torch.isfinite(values).all():
            raise RuntimeError("Raw rewards were not captured exactly once for this generation batch.")
        rewards = values[:, 0]
        parent_credit = (rewards.view(-1, DRAWS) - rewards.view(-1, DRAWS).mean(1, keepdim=True)).reshape(-1)
        if not torch.allclose(result["advantages"], parent_credit, atol=1e-6, rtol=1e-6):
            raise RuntimeError("Upstream TRL credit changed; refusing an unreviewed loss interface.")
        g = self.v2_spec.group_size
        grouped = rewards.view(-1, g)
        advantages = (grouped - (grouped.sum(1, keepdim=True) - grouped) / (g - 1)).reshape(-1)
        result["advantages"] = advantages
        # TRL already logged centered G8 credit. Replace it; preserve actual raw
        # reward logs and reference KL. Do not report upstream credit as applied.
        if len(self._logs["advantages"]) < len(inputs):
            raise RuntimeError("Unexpected upstream advantage log buffer.")
        for _ in inputs:
            self._logs["advantages"].pop()
        self._logs["advantages"].extend(advantages.tolist())
        std = grouped.std(1)
        self._metrics["train"]["reward_std"][-1] = std.mean().item()
        self._metrics["train"]["frac_reward_zero_std"][-1] = torch.isclose(std, torch.zeros_like(std)).float().mean().item()
        self._metrics["train"]["v2/loo_advantage_rms"].append(advantages.square().mean().sqrt().item())
        self._metrics["train"]["v2/baseline_group_size"].append(g)
        pending = self.v2_recorder.pending
        if pending is None or len(pending) != len(inputs):
            raise RuntimeError("Missing trace records for applied credit.")
        for i, (row, advantage) in enumerate(zip(pending, advantages.tolist())):
            if (row["id"], row["tau"], row["exposure_id"]) != (inputs[i]["id"], inputs[i]["tau"], inputs[i]["exposure_id"]):
                raise RuntimeError("Trace row order differs from the generation tensor order.")
            row["loo_advantage"] = advantage
            row["baseline_group_id"] = digest([row["step"], row["exposure_id"], row["baseline_group_within_prompt"]])
        append_rollout_batch(self.v2_rollout_path, pending)
        self.v2_recorder.pending = None
        return result


def trainer_class():
    from trl import GRPOTrainer
    source = Path(inspect.getfile(GRPOTrainer))
    if importlib.metadata.version("trl") != TRL_VERSION or hashlib.sha256(source.read_bytes()).hexdigest() != TRL_SOURCE_SHA256:
        raise RuntimeError("This extension requires the reviewed TRL0.24 source in the pinned image.")

    class MatchedLOOTrainer(MatchedLOOMixin, GRPOTrainer):
        pass

    return MatchedLOOTrainer


def effective_arguments(config, spec, outdir, steps):
    """No dataset or protocol choices: report the actual matched compute recipe."""
    t = config["training"]
    if t["beta"] != 0.04 or t["lora_r"] != 16 or t["lora_alpha"] != 32:
        raise ValueError("The requested recipe keeps beta=.04 and existing14B LoRA16/32.")
    return dict(output_dir=str(outdir), learning_rate=t["learning_rate"], beta=0.04,
        per_device_train_batch_size=spec.micro_batch_size, gradient_accumulation_steps=spec.accumulation_steps,
        steps_per_generation=spec.accumulation_steps, num_generations=DRAWS,
        max_prompt_length=t["max_prompt_length"], max_completion_length=t["max_completion_length"],
        temperature=t["temperature"], top_p=1.0, top_k=0, num_iterations=1,
        scale_rewards="none", loss_type="dr_grpo", max_steps=steps,
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True, lr_scheduler_type="constant", warmup_steps=0, max_grad_norm=1.0,
        optim="adamw_torch", seed=spec.seed, data_seed=spec.seed, logging_steps=1,
        save_steps=t["checkpoint_steps"], save_total_limit=2, report_to="none", use_vllm=False,
        dataloader_num_workers=0, remove_unused_columns=False, mask_truncated_completions=False)


def verify_rollouts(rows, records, spec, steps):
    sampler = BalancedQuestionSampler(records, spec)
    expected_order = sampler.question_order()
    if len(rows) != steps * spec.completions_per_update:
        raise RuntimeError("Not every planned completion was recorded.")
    for step in range(steps):
        batch = rows[step * spec.completions_per_update:(step + 1) * spec.completions_per_update]
        order = expected_order[step * spec.questions_per_update:(step + 1) * spec.questions_per_update]
        expected = [(records[q * len(TAUS) + slot]["exposure_id"], records[q * len(TAUS) + slot]["tau"], i)
                    for q in order for slot in range(len(TAUS)) for i in range(DRAWS)]
        if [(r["exposure_id"], r["tau"], r["sample_index"]) for r in batch] != expected or any(r["step"] != step for r in batch):
            raise RuntimeError("Sampler skipped, duplicated or reordered a registered optimizer step.")
        correct_credit = loo_advantages([r["reward"] for r in batch], spec.group_size)
        for r, value in zip(batch, correct_credit):
            if not math.isclose(r["loo_advantage"], value, abs_tol=1e-6, rel_tol=1e-6):
                raise RuntimeError("Recorded advantage is not the registered LOO credit.")


def train_v2(config, model_workdir, rows, outdir, spec, *, model_key="qwen14b",
             initializer=None, max_runtime_seconds=None):
    """Explicit entry point for the caller's reviewed runner; one pass, no resume.

Partial output is preserved and any reuse of an output directory is rejected.
The caller owns the dataset selection, budget approval and external watchdog.
"""
    import gc
    import psutil
    import torch
    from datasets import Dataset
    from peft import get_peft_model, get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import TrainerCallback, set_seed
    from trl import GRPOConfig

    if config["models"][model_key]["repo"] != "Qwen/Qwen3-14B":
        raise ValueError("This implementation is scoped to the pinned14B model.")
    if initializer is not None:
        raise ValueError("The registered v2 arms start from the original14B base, without SFT or GRPO initialization.")
    records = prepared_records(rows, spec)
    sampler = BalancedQuestionSampler(records, spec)
    steps = sampler.n_questions // spec.questions_per_update
    arguments = effective_arguments(config, spec, outdir, steps)
    cls = trainer_class()
    if max_runtime_seconds is not None and max_runtime_seconds <= 0:
        raise ValueError("A runtime cap must be positive.")
    pin_path = Path(model_workdir).resolve() / "artifacts/models.lock.json"
    pin = read_json(pin_path)[model_key]
    if pin["repo"] != "Qwen/Qwen3-14B" or pin["revision"] != MODEL_REVISION:
        raise ValueError("Model weights differ from the existing pinned14B base.")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    reserve = config["pilot"]["min_system_available_gib"]
    from abstention.protocol import adapter_fingerprint, environment
    source_files = [Path(__file__), Path(__file__).with_name("tests_training_v2.py"), pin_path,
                    *sorted((ROOT / "src/abstention").glob("*.py"))]
    provenance = {"started_at": utcnow(), "spec": asdict(spec), "effective_arguments": arguments,
        "thresholds": list(TAUS), "threshold_mean": sum(TAUS) / len(TAUS), "model_pin": pin,
        "input_config_sha256": digest(config),
        "schedule_sha256": digest(rows), "question_order_sha256": digest([records[q * len(TAUS)]["id"] for q in sampler.question_order()]),
        "initializer": str(initializer) if initializer else None,
        "initializer_sha256": adapter_fingerprint(initializer) if initializer else None,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "trl_source_sha256": TRL_SOURCE_SHA256, "environment": environment(),
        "scope": "Single pass, three exposure slots per question, matched8 draws per slot, LOO, no reward std scaling",
        "v1_comparison_limit": "Threshold grid, repeated exposure and LOO differ from v1; v2 arms identify only their registered within-v2 contrasts."}
    write_json(outdir / "run.json", provenance)
    model = trainer = None

    class Monitor(TrainerCallback):
        def __init__(self):
            self.gradient_steps = 0
            self.minimum_available_gib = psutil.virtual_memory().available / 2**30
            self.step_started = None

        def check(self):
            available = psutil.virtual_memory().available / 2**30
            self.minimum_available_gib = min(self.minimum_available_gib, available)
            if available < reserve:
                raise MemoryError("System memory is below the configured reserve.")
            if max_runtime_seconds is not None and time.monotonic() - started >= max_runtime_seconds:
                raise TimeoutError("Explicit runtime cap reached; preserving partial output without retry.")

        def on_step_begin(self, args, state, control, **kwargs):
            self.check()
            self.step_started = time.monotonic()

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item() for g in grads):
                raise FloatingPointError("Missing or nonfinite LoRA gradients.")
            self.gradient_steps += 1

        def on_step_end(self, args, state, control, **kwargs):
            self.check()
            append_rollout_batch(outdir / "steps.jsonl", [{"step": state.global_step,
                "seconds": time.monotonic() - self.step_started,
                "available_gib": psutil.virtual_memory().available / 2**30,
                "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                "cuda_reserved_gib": torch.cuda.memory_reserved() / 2**30}])

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                if any(isinstance(v, (int, float)) and not math.isfinite(v) for v in logs.values()):
                    raise FloatingPointError("Nonfinite training diagnostic.")
                append_rollout_batch(outdir / "logs.jsonl", [{"step": state.global_step, **logs}])

    monitor = Monitor()
    try:
        monitor.check()
        set_seed(spec.seed)
        torch.cuda.reset_peak_memory_stats()
        load_started = time.monotonic()
        model, tokenizer = load_model(config, model_workdir, model_key, initializer=initializer)
        model = get_peft_model(model, lora_config(config))
        model.enable_input_require_grads()
        model.config.use_cache = False
        model_load_seconds = time.monotonic() - load_started
        monitor.check()
        for record in records:
            record["prompt"] = render(tokenizer, messages(record["question"], record["tau"]))
            if len(tokenizer.encode(record["prompt"], add_special_tokens=False)) > arguments["max_prompt_length"]:
                raise ValueError(f"Oversized prompt: {record['id']} tau={record['tau']}; no truncation allowed.")
        recorder = V2RewardRecorder(spec, eos_ids(model, tokenizer))
        trainer = cls(model=model, args=GRPOConfig(**arguments), train_dataset=Dataset.from_list(records),
            processing_class=tokenizer, reward_funcs=recorder, callbacks=[monitor],
            v2_spec=spec, v2_recorder=recorder, v2_rollout_path=outdir / "rollouts.jsonl")
        trainer.train()
        if trainer.state.global_step != steps or monitor.gradient_steps != steps:
            raise RuntimeError("Trainer did not perform exactly the planned optimizer steps.")
        verify_rollouts(read_jsonl(outdir / "rollouts.jsonl"), records, spec, steps)
        trainer.save_model(str(outdir / "adapter"))
        tokenizer.save_pretrained(str(outdir / "adapter"))
        trainer.save_state()
        memory = get_peft_model_state_dict(model)
        saved = load_file(str(outdir / "adapter/adapter_model.safetensors"))
        if set(memory) != set(saved) or not all(torch.equal(saved[k], memory[k].detach().cpu()) for k in saved):
            raise RuntimeError("Saved LoRA parameters differ from in-memory parameters.")
        for name, expected in provenance["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("Imported source changed during training.")
        result = {"status": "complete", "completed_at": utcnow(), "spec": asdict(spec),
            "optimizer_steps": steps, "gradient_steps": monitor.gradient_steps,
            "questions": sampler.n_questions, "completions": steps * spec.completions_per_update,
            "model_load_seconds": model_load_seconds, "elapsed_seconds_including_load": time.monotonic() - started,
            "minimum_available_gib": monitor.minimum_available_gib,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "save_roundtrip": True, "sampler_traversal_verified": True,
            "adapter_sha256": adapter_fingerprint(outdir / "adapter"),
            "rollouts_sha256": hashlib.sha256((outdir / "rollouts.jsonl").read_bytes()).hexdigest()}
        write_json(outdir / "result.json", result)
        return result
    except BaseException as exc:
        write_json(outdir / "failure.json", {"at": utcnow(), "error": repr(exc),
            "elapsed_seconds_including_load": time.monotonic() - started,
            "completed_steps": getattr(getattr(trainer, "state", None), "global_step", 0)})
        raise
    finally:
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
