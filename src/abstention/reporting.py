from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import digest, ordered, read_json, read_jsonl, utcnow, write_json
from .scoring import summarize
from .statistics import group_rows, paired_contrast


def predictions(workdir):
    rows = []
    # Candidates/calibration files are deliberately excluded from study endpoints.
    for path in sorted((Path(workdir) / "artifacts/predictions").glob("*/test_*-t*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def report(config, workdir):
    workdir = Path(workdir)
    rows = predictions(workdir)
    directory = workdir / "artifacts/reports"
    directory.mkdir(parents=True, exist_ok=True)
    if not rows:
        result = {"status": "no_confirmatory_results", "created_at": utcnow(),
                  "message": "No test predictions are available. No research findings are inferred from the pilot."}
        write_json(directory / "status.json", result)
        return result
    groups = group_rows(rows)
    summaries = []
    for (model, dataset, arm, seed, tau), group in sorted(groups.items()):
        summaries.append({"model": model, "dataset": dataset, "arm": arm, "seed": seed, "tau": tau, **summarize(group)})
    with (directory / "metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    contrasts, pending = [], []
    for model in config["models"]:
        for dataset in ("trivia", "nq"):
            selected = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
            for comparator in ["binary", "prompt", "filter"] + (["fixed"] if model == "qwen" else []):
                try:
                    c = paired_contrast(selected, right=comparator, thresholds=tuple(config["thresholds"]["primary"]),
                                        samples=config["evaluation"]["bootstrap_samples"], seed=config["split_seed"])
                    if c["n_seeds"] != len(config["seeds"]) or c["n_questions"] != config["data"][f"test_{dataset}"]:
                        raise ValueError("Registered seeds or test questions are incomplete.")
                    contrasts.append({"model": model, "dataset": dataset, "primary": dataset == "trivia" and comparator == "binary", **c})
                except ValueError as exc:
                    pending.append({"model": model, "dataset": dataset, "comparator": comparator, "reason": str(exc)})
    write_json(directory / "contrasts.json", {"contrasts": contrasts, "pending": pending})
    manifest = read_json(workdir / "data/prepared/manifest.json")
    curve_ids = set(manifest["diagnostic_ids"]["test_trivia"] + manifest["diagnostic_ids"]["test_nq"])
    curve_summaries = []
    for (model, dataset, arm, seed, tau), group in sorted(group_rows([r for r in rows if r["id"] in curve_ids]).items()):
        curve_summaries.append({"model": model, "dataset": dataset, "arm": arm, "seed": seed, "tau": tau, **summarize(group)})
    make_figures(curve_summaries, directory)
    lines = ["# Experimental readout", "", "Exact-match endpoints; factuality audit is reported separately.", "",
             "| Model | Dataset | Arm | Threshold | Utility | Coverage | Selective risk |", "|---|---|---|---:|---:|---:|---:|"]
    collapsed = defaultdict(list)
    for r in summaries:
        collapsed[r["model"], r["dataset"], r["arm"], r["tau"]].append(r)
    for (model, dataset, arm, tau), records in sorted(collapsed.items()):
        utility = np.mean([r["utility"] for r in records])
        coverage = np.mean([r["coverage"] for r in records])
        answered = sum(r["correct"] + r["error"] for r in records)
        risk = f"{sum(r['error'] for r in records)/answered:.3f}" if answered else "undefined"
        lines.append(f"| {model} | {dataset} | {arm} | {tau:.2f} | {utility:.3f} | {coverage:.3f} | {risk} |")
    lines.extend(["", "## Paired contrasts", ""])
    for c in contrasts:
        lines.append(f"- {c['model']} / {c['dataset']}, conditioned − {c['right']}: {c['delta_utility']:.4f}, 95% CI {c['ci95']}.")
    if pending:
        lines.extend(["", "The study is incomplete. Missing contrasts are listed in contrasts.json; partial runs are not confirmatory evidence."])
    lines.extend(["", "Three seeds provide limited evidence about training variability. OOD and comparator analyses are secondary."])
    (directory / "results.md").write_text("\n".join(lines) + "\n")
    result = {"status": "partial" if pending else "complete", "prediction_rows": len(rows),
              "contrasts": len(contrasts), "pending": len(pending), "report": str(directory / "results.md")}
    write_json(directory / "status.json", result)
    return result


def make_figures(summaries, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = sorted({r["model"] for r in summaries})
    for model in models:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
        for row_index, dataset in enumerate(("trivia", "nq")):
            arms = sorted({r["arm"] for r in summaries if r["model"] == model and r["dataset"] == dataset})
            for arm in arms:
                records = [r for r in summaries if (r["model"], r["dataset"], r["arm"]) == (model, dataset, arm)]
                ts = sorted({r["tau"] for r in records})
                by_tau = [[r for r in records if r["tau"] == tau] for tau in ts]
                utility = [np.mean([r["utility"] for r in rr]) for rr in by_tau]
                coverage = [np.mean([r["coverage"] for r in rr]) for rr in by_tau]
                risks = []
                for rr in by_tau:
                    answered = sum(r["correct"] + r["error"] for r in rr)
                    risks.append(sum(r["error"] for r in rr) / answered if answered else np.nan)
                axes[row_index, 0].plot(ts, utility, marker="o", label=arm)
                axes[row_index, 1].plot(ts, coverage, marker="o", label=arm)
                # Observed points only: answer content can change with tau, so no invented continuous ranking.
                axes[row_index, 2].scatter(coverage, risks, label=arm)
            for ax in axes[row_index]:
                ax.grid(alpha=0.2)
            axes[row_index, 0].axhline(0, color="black", lw=0.6)
            axes[row_index, 0].set(xlabel="Requested threshold", ylabel=f"{dataset}: mean utility")
            axes[row_index, 1].set(xlabel="Requested threshold", ylabel="Coverage", ylim=(-0.02, 1.02))
            axes[row_index, 2].set(xlabel="Coverage", ylabel="Selective risk", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{model} — exact-match evaluation on the same fixed questions at every threshold")
        fig.savefig(directory / f"{model}-curves.pdf")
        fig.savefig(directory / f"{model}-curves.png", dpi=170)
        plt.close(fig)


def audit_export(config, workdir):
    workdir = Path(workdir)
    rows = predictions(workdir)
    if not rows:
        raise ValueError("No predictions to audit.")
    # One row per question/response/model/arm: repeated thresholds do not duplicate the audit unit.
    unique = {}
    for r in rows:
        key = digest([r["id"], r["text"], r["model"], r["arm"]])
        unique.setdefault(key, {**r, "audit_id": key[:20]})
    strata = defaultdict(list)
    for r in unique.values():
        strata[r["model"], r["dataset"], r["arm"], r["outcome"]].append(r)
    for key in strata:
        strata[key] = sorted(strata[key], key=lambda r: digest([config["split_seed"], r["audit_id"]]))
    selected = []
    depth = 0
    while len(selected) < min(config["evaluation"]["audit_size"], len(unique)):
        for key in sorted(strata):
            if depth < len(strata[key]) and len(selected) < config["evaluation"]["audit_size"]:
                selected.append(strata[key][depth])
        depth += 1
    directory = workdir / "artifacts/audit"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "blind.csv"
    if path.exists():
        raise FileExistsError("Audit already exported; refusing to overwrite annotations.")
    selected.sort(key=lambda r: digest(["blind-order", r["audit_id"]]))
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["audit_id", "question", "response", "reference_aliases", "human_outcome", "notes"])
        writer.writeheader()
        for r in selected:
            writer.writerow({"audit_id": r["audit_id"], "question": r["question"], "response": r["text"],
                             "reference_aliases": json.dumps(r["aliases"]), "human_outcome": "", "notes": ""})
    write_json(directory / "private-key.json", {r["audit_id"]: r for r in selected})
    return {"path": str(path), "n": len(selected), "instructions": "Label correct/error/abstain/unclear; do not open private-key.json while annotating. This stratified audit is diagnostic, not a population error estimate."}


def audit_import(workdir, path):
    directory = Path(workdir) / "artifacts/audit"
    key = read_json(directory / "private-key.json")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(key) or {r["audit_id"] for r in rows} != set(key):
        raise ValueError("Audit IDs must match the exported sample exactly.")
    if any(r["human_outcome"] not in {"correct", "error", "abstain", "unclear"} for r in rows):
        raise ValueError("Every row needs a valid human label; blank labels are not inferred.")
    disagreements = [{"audit_id": r["audit_id"], "human": r["human_outcome"], "automatic": key[r["audit_id"]]["outcome"]}
                     for r in rows if r["human_outcome"] != key[r["audit_id"]]["outcome"]]
    result = {"n": len(rows), "disagreements": disagreements,
              "unclear": sum(r["human_outcome"] == "unclear" for r in rows),
              "interpretation": "Stratified diagnostic sample; do not extrapolate raw disagreement rate to the population."}
    write_json(directory / "results.json", result)
    return result
