# Can LLMs Learn to Abstain?

When should a language model answer a question, and when should it say **“I don't know”**?

This project tests two approaches with **Qwen3-14B**: train the model to make that decision, or keep its answers unchanged and train a selector to decide which ones to show. The goal is to provide useful answers while avoiding mistakes.

**Main finding:** on TriviaQA, the learned selector answers more questions at a lower error rate than the original model's calibrated confidence filter. The tested reinforcement-learning changes remain inconclusive, and reliable transfer to NQ-Open is still unresolved.

## What we found

On TriviaQA, the selector answers about **31 out of every 100 questions**, compared with **20** for the original filter. Among the answers it gives, the error rate falls from **23.0% to 14.5%**.

| Method | Questions answered | Error rate among answers |
|---|---:|---:|
| Original model's confidence filter, calibrated | 20.2% | 23.0% |
| Learned answer selector, calibrated | **30.9%** | **14.5%** |

These are **exact-match results**: an answer is counted as correct if it matches a reference alias after normalization. Both methods judge the same candidate answers. Coverage combines both test thresholds and all selector seeds; error rates divide total errors by total emitted answers.

The selector also improves the study's score, which rewards correct answers, penalizes mistakes, and gives zero for abstention. Its gain is **+0.01830**, with a 99% bootstrap interval of **[0.01041, 0.02593]**, using the adjustment for five planned comparisons.

Two limits matter:

- **RL:** the three controlled training comparisons have intervals that include zero. We cannot conclude that the tested changes consistently improve abstention.
- **NQ-Open:** the selector improves over the original filter, but its mean score is still **−0.01197**, below the zero score of always abstaining.

[All methods and metrics](experiments/v2-20261002/artifacts/analysis/joint-primary.csv) · [Comparisons and uncertainty](experiments/v2-20261002/artifacts/analysis/contrasts.csv)

## How the approaches work

**Train the answering model.** Reinforcement learning (RL) updates the model so it can choose between giving an answer and returning `IDK`. We test how responses are compared during training, whether each question is practiced at different confidence thresholds, and whether the reward follows the requested threshold.

**Choose which answers to show.** The original model first generates an answer. A scoring step then estimates whether that answer is correct. If its estimated probability exceeds the requested threshold, the system shows the answer; otherwise it returns `IDK`.

We compare two ways to obtain that score:

- The **original confidence filter** asks the original model to assess its answer. This score is often called **P(True)**.
- The **learned selector** trains a separate LoRA adapter—a small set of adjustable model parameters—to judge the question and candidate answer. The generator stays unchanged.

Both scores are calibrated using 500 separate labeled questions. **Calibration** maps a raw score to an estimated probability of correctness. It is fitted on TriviaQA and applied to NQ without refitting; this does not guarantee accuracy on the new dataset. A further logistic-regression control uses the same training labels as the selector to test whether those labels alone explain the improvement.

### Why the threshold matters

The system receives a confidence threshold, such as **0.85**. At this threshold, the scoring rule is:

| Outcome | Score |
|---|---:|
| Correct answer | +0.15 |
| Incorrect answer | −0.85 |
| `IDK` | 0 |

More generally, a correct answer earns $1-\tau$, an error earns $-\tau$, and abstention earns zero. If an answer has correctness probability $p$, its expected score is $p-\tau$. Answering is therefore worthwhile when $p>\tau$. The rule creates that incentive; it does not ensure that the model estimates its confidence correctly.

## What we tested

The completed study used **15 training runs with three seeds**: 12 RL runs and three selectors, all starting from Qwen3-14B. The primary tests contain **2,000 TriviaQA questions and 500 NQ-Open questions**, new within this study. Their thresholds, **0.65 and 0.85**, were absent from policy training. The model receives no search results or reference answers at inference.

The study took **35.44 recorded task-hours** on one NVIDIA GB10 workstation with approximately 128 GB of unified memory. That total includes pilots, loading, training, evaluation and CPU analysis.

The [protocol](docs/tres-mejoras-v2.md) explains the four RL variants and controls. The [executed budget](experiments/v2-20261002/artifacts/budget-lock.json) records the training-size decision made before test access.

### Checking automatic grading

A review of **150 response units** accepted **40 answers that exact matching had rejected**; three cases remained unclear. ChatGPT proposed labels, and one person reported personally reviewing every label. Because the sample was stratified, these counts do not estimate the grader's overall error rate.

The TriviaQA selector gain remains positive when applying those corrections only to identical reviewed answers. Other answers retain their original labels. This is a partial check, not a complete semantic evaluation; its intervals exclude uncertainty from audit sampling and human judgment. See the [audit methods](docs/auditoria-v2.md) and [separate audit results](results/v2-audit-20261006/summary.json).

## Try the code

With **Python 3.11 or newer**, you can run the synthetic checks without a GPU, model downloads or private annotations:

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

These checks test the implementation with synthetic data; they do not rerun training or reconstruct the published results. The [reproduction guide](docs/reproducibilidad.md) explains how to verify file hashes and set up the training environment.

## Find the details

| Goal | Start here |
|---|---|
| Understand the experiment | [Protocol](docs/tres-mejoras-v2.md) |
| Read the training code | [RL](experiments/v2-20261002/training_v2.py) · [Selector](experiments/v2-20261002/selector_v2.py) |
| Inspect evaluation and statistics | [Evaluation](experiments/v2-20261002/evaluate_v2.py) · [Analysis](experiments/v2-20261002/analysis_v2.py) |
| Reproduce checks or plan a new run | [Reproduction guide](docs/reproducibilidad.md) |
| Follow the study's earlier phases | [Experiment history](docs/registro-experimentos.md) |

The repository includes code, configurations, dataset ID manifests and aggregate results. Model weights, individual predictions, audit annotations and the manuscript are not included. Recalculating published results requires the omitted inputs; retraining requires official model and dataset downloads and a new run.

The conclusions are limited to this model and setup. Three seeds leave substantial training uncertainty, exact matching can reject valid answers, and fresh study splits do not rule out pretraining contamination. The results support better answer selection on TriviaQA—not a general guarantee that the model knows when it is wrong.
