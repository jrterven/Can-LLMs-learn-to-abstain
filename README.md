# Can LLMs Learn to Abstain?

A reproducible study of threshold-controlled abstention in **Qwen3-14B**, run on a single NVIDIA GB10 workstation. We compare changing the answer-generating model through reinforcement learning with learning a separate selector for a frozen generator.

**Main finding:** a learned, calibrated selector improves exact-match utility over a calibrated confidence filter on TriviaQA. The three controlled RL comparisons remain inconclusive. Transfer to NQ improves relative to the original filter, but does not establish positive utility compared with always abstaining.

## Objective and reward

The model receives a factual question and a confidence threshold $\tau$, then returns a short answer or `IDK`. We score it with

$$R_\tau = c - \tau a,$$

where $c=1$ for a correct answer and $a=1$ for a scored answer; both are zero for abstention. Thus a correct answer earns $1-\tau$, an incorrect answer earns $-\tau$, and `IDK` earns zero. Empty or truncated outputs count as incorrect answers. Correctness is normalized exact match against reference aliases.

For an answer with correctness probability $p$, expected utility is $p-\tau$: answering is preferable when $p>\tau$. This creates an incentive to abstain; it does **not** guarantee calibrated confidence or semantic truth.

## What we compare

- **Prompting:** the original model receives the threshold and chooses whether to answer.
- **RL:** LoRA updates change the generator itself using the reward above, leave-one-out advantages and no reward-standard-deviation normalization.
- **Calibrated P(True) filter:** the frozen original model generates a candidate and assesses its correctness. Logistic calibration is fitted on a separate calibration split; the filter emits the candidate when $p_{\mathrm{cal}}>\tau$.
- **Learned selector:** a separate LoRA adapter learns candidate correctness from TRAIN labels, then receives logistic calibration. The generator stays frozen. All filters use the same cached candidates, so their comparison measures selection rather than different answer generation.

The four RL arms share eight generations per exposure, three exposures per question and matched training budgets:

| Arm | Advantage groups | Three exposures per question | Reward cost |
|---|---|---|---|
| A | Two groups of four | One assigned threshold, repeated | Requested $\tau$ |
| B | One group of eight | One assigned threshold, repeated | Requested $\tau$ |
| C | One group of eight | Each of 0.60, 0.75 and 0.90 | Requested $\tau$ |
| D | One group of eight | Each of 0.60, 0.75 and 0.90 | Fixed 0.75 |

D receives the same varying-threshold prompts as C; only its reward cost stays fixed. The comparisons are B−A for group size, C−B for exposure schedule, and C−D for variable versus fixed reward cost. C−B also changes threshold balance within an update; it does not isolate abstract understanding of confidence.

**15 training runs:** 12 RL adapters and three selectors, using seeds **17, 29 and 43**, all initialized from the original Qwen3-14B. Each RL run used 504 TRAIN questions, 252 updates and 12,096 generations after a time-based budget decision made before test access. Each selector used 8,000 candidates from 2,000 TRAIN questions and 500 updates. Development and calibration each used 500 separate TriviaQA questions.

The held-out evaluation contains **2,000 new TriviaQA questions and 500 NQ-Open questions**. Primary thresholds **0.65 and 0.85** were unseen during training. Calibration transfers from TriviaQA to NQ without refitting. No retrieval, search or reference answers are supplied at inference. The completed workflow comprised 39 jobs and **35.44 recorded task-hours**, including pilots, loading and CPU analysis. See the [frozen protocol](docs/tres-mejoras-v2.md), [executed budget](experiments/v2-20261002/artifacts/budget-lock.json) and [split manifest](experiments/v2-20261002/data/prepared/manifest.json).

## Results

Original **exact-match** endpoints, combining both primary thresholds and all trained seeds. Coverage is the proportion answered; selective risk is errors divided by answers, pooled over those evaluation rows. Deterministic baselines are evaluated once. Always abstaining has utility zero, coverage zero and undefined selective risk.

| Method | TriviaQA utility | Coverage | Risk | NQ utility | Coverage | Risk |
|---|---:|---:|---:|---:|---:|---:|
| Original + threshold prompt | −0.14120 | 79.40% | 42.88% | −0.27855 | 57.10% | 74.08% |
| RL A | −0.00944 | 27.52% | 28.71% | −0.03812 | 12.77% | 55.61% |
| RL B | −0.00117 | 30.98% | 25.72% | −0.03558 | 12.50% | 54.13% |
| RL C | +0.00910 | 27.60% | 22.07% | −0.02252 | 9.97% | 48.49% |
| RL D | −0.02668 | 46.28% | 31.00% | −0.07688 | 23.37% | 58.49% |
| P(True) + calibration | +0.02418 | 20.15% | 22.95% | −0.03075 | 10.70% | 63.55% |
| Learned selector + calibration | **+0.04248** | **30.91%** | **14.51%** | −0.01197 | 9.53% | 43.01% |

Source: [joint metrics](experiments/v2-20261002/artifacts/analysis/joint-primary.csv). The full results also include uncalibrated filters and a logistic control trained on the same candidate labels.

The selector's TriviaQA utility gain over calibrated P(True) is **+0.01830, 99% CI [0.01041, 0.02593]**. Its gain over the matched-label logistic control is **+0.01834 [0.01041, 0.02602]**. All three RL contrast intervals include zero; this is inconclusive evidence, not proof of equivalence. The [contrast table](experiments/v2-20261002/artifacts/analysis/contrasts.csv) reports every seed. Intervals use 10,000 paired question/seed bootstrap replicates; the 99% level applies Bonferroni adjustment to five predefined TriviaQA contrasts. NQ comparisons are descriptive.

### Audit and partial sensitivity

A stratified audit covered **150 response units from 144 questions**. ChatGPT proposed labels; one person reported reviewing every label and correcting labels as needed. This is AI-assisted human review, not two independent annotators. The audit accepted **40 responses marked incorrect by EM** as correct; three cases remain unresolved. This count is **not a population false-negative rate**.

Replacing labels only for identical audited units, while retaining EM everywhere else, gives a TriviaQA selector gain of **+0.01797, conditional 99% CI [0.01027, 0.02559]**. RL comparisons remain inconclusive. In NQ, the selector's partial utility is **−0.00397, 95% CI [−0.01522, 0.00643]**; its advantage over the filter is not robust at 99% across all unresolved-label scenarios. These post-hoc intervals exclude human-label and audit-sampling uncertainty. Original endpoints, predictions and calibrators remain unchanged. See [audit aggregates](results/v2-audit-20261006/summary.json) and [audit methods](docs/auditoria-v2.md).

## Install and run CPU checks

Python 3.11 or newer is required. These synthetic checks need no GPU, model downloads or private annotations:

```bash
git clone https://github.com/jrterven/Can-LLMs-learn-to-abstain.git
cd Can-LLMs-learn-to-abstain
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' openpyxl==3.1.5
.venv/bin/python -m pytest -q \
  tests/test_public_bundle.py \
  tests/test_audit_v2.py \
  tests/test_audit_v2_sensitivity.py
```

Training and the full training/evaluation test suite additionally require PyTorch and the training dependencies. The [Dockerfile](Dockerfile) pins NVIDIA PyTorch 25.11; [pyproject.toml](pyproject.toml) pins Transformers 4.57.1, TRL 0.24.0 and PEFT 0.17.1. The verified hardware was ARM64/GB10 with approximately 128 GB unified memory. See the [reproduction guide](docs/reproducibilidad.md) for container commands and environment constraints.

## Code, evidence and reproduction limits

| Entry | Purpose |
|---|---|
| [Data preparation](experiments/v2-20261002/prepare_data.py) | Splits, exclusions and matched exposure schedules |
| [RL implementation](experiments/v2-20261002/training_v2.py) | Four matched generator-training arms |
| [Selector implementation](experiments/v2-20261002/selector_v2.py) | Candidate scoring, learning and calibration |
| [Evaluation](experiments/v2-20261002/evaluate_v2.py) / [analysis](experiments/v2-20261002/analysis_v2.py) | Frozen endpoints, statistics and figures |
| [Independent scientific review](experiments/v2-20261002/artifacts/reviews/2026-10-04-completion/scientific.json) | Checks against individual predictions |
| [Audit sensitivity](analysis/audit_v2_sensitivity.py) / [aggregate provenance](results/v2-audit-20261006/provenance.json) | Separate post-hoc analysis and source hashes |
| [Study history](docs/registro-experimentos.md) | Earlier phases and subsequent decisions |

This repository contains selected code, configurations, ID manifests and aggregate evidence. It does **not** include model weights, individual predictions, audit workbooks or the manuscript. A fresh clone supports code inspection and synthetic tests; recomputing the published numbers requires the omitted inputs. Retraining requires official model/dataset downloads and a newly registered run. Historical `check-local`, export and runner commands intentionally require local files absent from this lightweight release; do not rewrite locks or restart the completed queue to bypass them.

Three seeds limit precision. The selector learns the exact-match proxy, and calibration offers no guaranteed risk bound under dataset shift. Fresh study splits do not establish absence of pretraining contamination. The audit is partial and assisted by AI; it does not turn the full test set into a semantic gold standard. This work contributes a controlled empirical comparison, not a claim that leave-one-out RL, confidence filtering or logistic calibration are new algorithms.
