from collections import Counter, defaultdict, deque
from fractions import Fraction
import importlib.util
import itertools
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

path = Path(__file__).with_name("training_v2.py")
spec = importlib.util.spec_from_file_location("training_v2_under_test", path)
v2 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)


def schedule(mode, n=6):
    return [{"id": f"q{q}", "question": f"Question{q}?", "aliases": ["Paris"],
             "tau": v2.TAUS[slot] if mode == "paired_tau" else v2.TAUS[q % 3],
             "slot": slot, "question_index": q, "update_index": q // 2,
             "exposure_id": v2.digest([f"q{q}", slot])}
            for q in range(n) for slot in range(3)]


def records(mode, n=6):
    spec = v2.V2Spec(8, "conditioned", 17, exposure_mode=mode)
    result = v2.prepared_records(schedule(mode, n), spec)
    for row in result:
        row["prompt"] = f"{row['question']} tau={row['tau']}"
    return result


def batch(mode="paired_tau", group_size=8):
    spec = v2.V2Spec(group_size, "conditioned", 17, exposure_mode=mode)
    source = records(mode)
    iterator = iter(v2.BalancedQuestionSampler(source, spec))
    return [source[next(iterator)] for _ in range(spec.completions_per_update)]


def test_all_arms_share_draws_steps_and_exposure_positions():
    streams = {}
    for name, g, exposure, arm in [("a", 4, "single_tau", "conditioned"),
                                   ("b", 8, "single_tau", "conditioned"),
                                   ("c", 8, "paired_tau", "conditioned"),
                                   ("d", 8, "paired_tau", "fixed")]:
        spec = v2.V2Spec(g, arm, 17, exposure_mode=exposure)
        rows = records(exposure)
        sampler = v2.BalancedQuestionSampler(rows, spec)
        indices = list(sampler)
        assert spec.completions_per_update == 48 and spec.accumulation_steps == 12
        assert len(indices) == 3 * 12 * 48
        generations = []
        for update in range(3):
            start = update * 12 * 48
            first = indices[start:start + 48]
            for micro in range(12):
                assert indices[start + micro * 48:start + (micro + 1) * 48] == first
            current = [rows[i] for i in first]
            v2.validate_generation_batch(current, spec)
            assert set(Counter(r["exposure_id"] for r in current).values()) == {8}
            assert len({r["id"] for r in current}) == 2
            generations.extend(current)
        assert len(generations) == 6 * 3 * 8
        streams[name] = generations
    identity = lambda r: (r["id"], r["slot"], r["exposure_id"])
    assert all([identity(r) for r in streams[name]] == [identity(r) for r in streams["a"]] for name in streams)
    assert streams["a"] == streams["b"]
    assert streams["c"] == streams["d"]
    assert sum(r["tau"] for r in streams["a"]) == pytest.approx(.75 * len(streams["a"]))
    assert sum(r["tau"] for r in streams["c"]) == pytest.approx(.75 * len(streams["c"]))


def test_registered_full_and_fallback_counts():
    for n, steps, completions in [(1008, 504, 24192), (504, 252, 12096)]:
        spec = v2.V2Spec(8, "conditioned", 17, exposure_mode="single_tau")
        supplied = v2.prepared_records(schedule("single_tau", n), spec)
        sampler = v2.BalancedQuestionSampler(supplied, spec)
        assert sampler.n_questions // 2 == steps
        assert steps * spec.completions_per_update == completions
        assert Counter(r["tau"] for r in supplied) == {tau: n for tau in v2.TAUS}


def test_schedule_rejects_unbalanced_or_reused_exposures():
    spec = v2.V2Spec(8, "conditioned", 17, exposure_mode="single_tau")
    invalid = schedule("single_tau")
    invalid[1]["exposure_id"] = invalid[0]["exposure_id"]
    with pytest.raises(ValueError, match="identity"):
        v2.prepared_records(invalid, spec)
    invalid = schedule("single_tau")
    for r in invalid:
        r["tau"] = .75
    with pytest.raises(ValueError, match="balanced"):
        v2.prepared_records(invalid, spec)
    wrong_order = schedule("paired_tau")
    wrong_order[3:6], wrong_order[0:3] = wrong_order[0:3], wrong_order[3:6]
    with pytest.raises(ValueError, match="identity"):
        v2.prepared_records(wrong_order, v2.V2Spec(8, "conditioned", 17))


def test_generation_groups_never_merge_identical_prompt_exposures():
    spec = v2.V2Spec(4, "conditioned", 17, exposure_mode="single_tau")
    inputs = batch("single_tau", 4)
    assert inputs[0]["prompt"] == inputs[8]["prompt"] == inputs[16]["prompt"]
    assert len({inputs[i]["exposure_id"] for i in (0, 8, 16)}) == 3
    v2.validate_generation_batch(inputs, spec)
    inputs[8] = inputs[0]
    with pytest.raises(ValueError, match="mixes"):
        v2.validate_generation_batch(inputs, spec)


def test_loo_exact_exhaustive_g4_and_nonmerging_g8_case():
    values = (Fraction(-3, 4), Fraction(0), Fraction(1, 4))
    for group in itertools.product(values, repeat=4):
        rewards = list(group) + list(reversed(group))
        observed = v2.loo_advantages(rewards, 4)
        for offset in (0, 4):
            assert sum(observed[offset:offset + 4]) == 0
            for i in range(4):
                others = [rewards[offset + j] for j in range(4) if i != j]
                assert observed[offset + i] == rewards[offset + i] - sum(others) / 3
    # Identical eight outputs, different aggregation; group4 has no contrast.
    rewards = [Fraction(1, 4)] * 4 + [Fraction(-3, 4)] * 4
    assert v2.loo_advantages(rewards, 4) == [0] * 8
    assert v2.loo_advantages(rewards, 8) == [Fraction(4, 7)] * 4 + [Fraction(-4, 7)] * 4
    assert v2.loo_advantages([r * 3 for r in rewards], 8) == [a * 3 for a in v2.loo_advantages(rewards, 8)]


def test_loo_removes_finite_group_mean_factor_in_expectation():
    # E[A_IDK | IDK] = -E[R] for LOO, independent of G; no std normalization.
    rewards = (Fraction(1, 4), Fraction(-3, 4), Fraction(0))
    probabilities = (Fraction(18, 25), Fraction(9, 50), Fraction(1, 10))
    expected = -sum(p * r for p, r in zip(probabilities, rewards))
    for g in (4, 8):
        total = Fraction(0)
        for sequence in itertools.product(range(3), repeat=g - 1):
            probability = Fraction(1)
            for index in sequence:
                probability *= probabilities[index]
            credit = -sum(rewards[index] for index in sequence) / (g - 1)
            total += probability * credit
        assert total == expected


def test_reward_trace_strict_invalid_handling_and_fixed_cost():
    inputs = batch("paired_tau")
    kwargs = {key: [r[key] for r in inputs] for key in inputs[0] if key != "prompt"}
    texts = ["Paris", "IDK", ".IDK", ""] * 12
    tokens = [[3, 2]] * 48
    tokens[0] = [3]  # Even an alias match is wrong if generation was truncated.
    seen = {}
    for arm in ("conditioned", "fixed"):
        recorder = v2.V2RewardRecorder(v2.V2Spec(8, arm, 17), {2})
        rewards = recorder(completions=texts, completion_ids=tokens, trainer_state=SimpleNamespace(global_step=0), **kwargs)
        seen[arm] = rewards
        assert recorder.pending[0]["outcome"] == "error" and recorder.pending[0]["reason"] == "truncated"
        assert recorder.pending[1]["outcome"] == "abstain"
        assert recorder.pending[2]["outcome"] == "error"
        assert recorder.pending[3]["reason"] == "empty"
        assert recorder.pending[8]["sample_index"] == 0
        assert recorder.pending[0]["exposure_id"] != recorder.pending[8]["exposure_id"]
    assert seen["conditioned"][0] == -.6 and seen["fixed"][0] == -.75
    assert seen["conditioned"][16] == pytest.approx(.1) and seen["fixed"][16] == .25


@pytest.mark.parametrize("group_size", [4, 8])
def test_actual_trl_loss_matches_full_update_and_microbatch_gradients(group_size):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from trl import GRPOTrainer
    cls = v2.trainer_class()
    assert cls._compute_loss is GRPOTrainer._compute_loss
    assert cls._prepare_inputs is GRPOTrainer._prepare_inputs
    torch.set_num_threads(1)
    rewards = [0., .4, -.6, 0., .4, -.6, .4, 0.] * 6
    advantage = torch.tensor(v2.loo_advantages(rewards, group_size), dtype=torch.float64)
    initial = torch.linspace(-2., -.1, 48 * 5, dtype=torch.float64).reshape(48, 5)
    reference = initial + .2
    mask = torch.tensor([[1] * (i % 5 + 1) + [0] * (4 - i % 5) for i in range(48)])

    def loss(logps, indices, accum):
        current = logps[indices]
        fake = SimpleNamespace(model=SimpleNamespace(training=True), beta=.04, top_entropy_quantile=1.,
            use_vllm=False, use_liger_loss=False, importance_sampling_level="token", epsilon_low=.2, epsilon_high=.2,
            args=SimpleNamespace(delta=None), loss_type="dr_grpo", max_completion_length=32,
            current_gradient_accumulation_steps=accum, accelerator=SimpleNamespace(gather=lambda x: x),
            _metrics={"train": defaultdict(list)},
            _get_per_token_logps_and_entropies=lambda *args, **kwargs: (current, torch.zeros_like(current)))
        b = len(indices)
        inputs = {"prompt_ids": torch.ones((b, 2), dtype=torch.long), "prompt_mask": torch.ones((b, 2)),
                  "completion_ids": torch.ones((b, 5), dtype=torch.long), "completion_mask": mask[indices],
                  "advantages": advantage[indices], "ref_per_token_logps": reference[indices]}
        return GRPOTrainer._compute_loss(fake, fake.model, inputs)

    full = initial.clone().requires_grad_()
    expected = loss(full, list(range(48)), 1)
    expected.backward()
    micro = initial.clone().requires_grad_()
    observed = sum(loss(micro, list(range(i, i + 4)), 12) for i in range(0, 48, 4))
    observed.backward()
    assert torch.allclose(expected, observed, atol=1e-12, rtol=1e-12)
    assert torch.allclose(full.grad, micro.grad, atol=1e-12, rtol=1e-12)
    # Independent scalar reduction includes KL and the fixed32-token denominator.
    expected_scalar = ((-advantage[:, None] + .04 * (torch.exp(reference - initial) - (reference - initial) - 1)) * mask).sum() / (48 * 32)
    assert torch.allclose(expected.detach(), expected_scalar, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("group_size", [4, 8])
def test_credit_hook_replaces_applied_and_logged_advantages_before_shuffle(tmp_path, group_size):
    torch = pytest.importorskip("torch")
    spec = v2.V2Spec(group_size, "conditioned", 17, exposure_mode="single_tau")
    inputs = batch("single_tau", group_size)
    recorder = v2.V2RewardRecorder(spec, {2})

    class FakeParent:
        def _calculate_rewards(self, rows, prompts, completions, tokens):
            kwargs = {key: [row[key] for row in rows] for key in rows[0] if key != "prompt"}
            rewards = recorder(completions=completions, completion_ids=tokens,
                trainer_state=SimpleNamespace(global_step=0), **kwargs)
            return torch.tensor(rewards)[:, None]

        def _generate_and_score_completions(self, rows):
            texts = (["Paris"] * 4 + ["wrong"] * 4) * 6
            rewards = self._calculate_rewards(rows, [], texts, [[3, 2]] * 48)[:, 0]
            centered = (rewards.view(-1, 8) - rewards.view(-1, 8).mean(1, keepdim=True)).reshape(-1)
            self._logs["advantages"].extend(centered.tolist())
            self._metrics["train"]["reward_std"].append(-99.)
            self._metrics["train"]["frac_reward_zero_std"].append(-99.)
            return {"advantages": centered, "other_tensor": "untouched"}

    class TestTrainer(v2.MatchedLOOMixin, FakeParent):
        pass

    obj = object.__new__(TestTrainer)
    obj.v2_spec, obj.v2_recorder, obj.v2_rollout_path = spec, recorder, tmp_path / "rollouts.jsonl"
    obj.model = SimpleNamespace(training=True)
    obj._logs = {"advantages": deque(maxlen=48)}
    obj._metrics = {"train": defaultdict(list)}
    result = obj._generate_and_score_completions(inputs)
    trace = [json.loads(line) for line in obj.v2_rollout_path.read_text().splitlines()]
    expected = v2.loo_advantages([r["reward"] for r in trace], group_size)
    assert result["advantages"].tolist() == pytest.approx(expected)
    assert list(obj._logs["advantages"]) == pytest.approx(expected)
    assert [r["loo_advantage"] for r in trace] == pytest.approx(expected)
    assert len({r["baseline_group_id"] for r in trace}) == 48 // group_size
    assert result["other_tensor"] == "untouched"
    if group_size == 4:
        assert result["advantages"].tolist() == [0.] * 48
        assert obj._metrics["train"]["frac_reward_zero_std"][-1] == 1.
    else:
        assert obj._metrics["train"]["frac_reward_zero_std"][-1] == 0.


@pytest.mark.parametrize("mode", ["single_tau", "paired_tau"])
def test_actual_trl_dataloader_and_buffer_do_not_skip_or_repeat_exposures(mode):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from datasets import Dataset
    from trl import GRPOTrainer
    spec = v2.V2Spec(8, "conditioned", 17, exposure_mode=mode)
    supplied = records(mode)

    class BufferedTrainer(v2.MatchedLOOMixin, GRPOTrainer):
        def _generate_and_score_completions(self, inputs):
            v2.validate_generation_batch(inputs, spec)
            self.generated_batches.append(inputs)
            markers = [r["question_index"] * 24 + r["slot"] * 8 + i % 8 for i, r in enumerate(inputs)]
            return {"marker": torch.tensor(markers), "advantages": torch.zeros(len(inputs))}

    obj = object.__new__(BufferedTrainer)
    obj.v2_spec = spec
    obj.train_dataset = Dataset.from_list(supplied)
    obj._train_batch_size = 4
    obj.args = SimpleNamespace(steps_per_generation=12, dataloader_num_workers=0,
        dataloader_pin_memory=False, dataloader_persistent_workers=False,
        dataloader_drop_last=False, dataloader_prefetch_factor=None, process_index=0, report_to=[])
    obj._remove_unused_columns = lambda dataset, description: dataset
    obj.data_collator = lambda inputs: inputs
    obj.accelerator = SimpleNamespace(prepare=lambda loader: loader)
    obj.model = SimpleNamespace(training=True)
    obj.num_iterations, obj._step, obj._buffered_inputs = 1, 0, None
    obj.generated_batches = []
    consumed = []
    for full_generation_batch in GRPOTrainer.get_train_dataloader(obj):
        microbatch = GRPOTrainer._prepare_inputs(obj, full_generation_batch)
        assert microbatch["marker"].numel() == 4
        consumed.extend(microbatch["marker"].tolist())
    assert len(obj.generated_batches) == 3
    assert obj._step == 36
    for step in range(3):
        assert sorted(consumed[step * 48:(step + 1) * 48]) == list(range(step * 48, (step + 1) * 48))
    assert Counter(consumed) == Counter(range(6 * 3 * 8))
