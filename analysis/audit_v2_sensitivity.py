"""Partial, post-hoc v2 sensitivity anchored to the returned human audit.

Unaudited exact-match labels and all policy decisions remain fixed. This is not
semantic re-evaluation of either full corpus and is not a weighted population
estimator from the diagnostic sample. No model, calibrator or endpoint is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import itertools
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "experiments/v2-20261002"
AUDIT = ROOT / "artifacts/audit-v2-20261004"
SUBMISSION = AUDIT / "submissions/d5f9297c600734f7"
OUT = AUDIT / "analysis-20261006"
sys.path.insert(0, str(ROOT / "src"))
from abstention.io import digest, file_digest, read_json, utcnow, write_json

SEEDS = (17, 29, 43)
TAUS = (.65, .85)
ARMS = ("a_g4_single_tau", "b_g8_single_tau", "c_g8_paired_tau", "d_g8_paired_fixed075")
FILTERS = ("original_raw", "original_calibrated", "train_logistic_raw", "train_logistic_recalibrated", "critic_raw", "critic_calibrated")
METHODS = ("original", *ARMS, *FILTERS)
COUNTS = {"test_trivia": 2000, "test_nq": 500}
CONTRASTS = ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[2], ARMS[3]),
             ("critic_calibrated", "original_calibrated"), ("critic_calibrated", "train_logistic_recalibrated"))
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261002
CODE = {"correct": 1, "error": 0, "abstain": -1}
LIMITS = [
    "Partial post-hoc sensitivity: only exact audited question/text/truncated units receive human semantic information; every other unit retains its original exact-match label.",
    "The150-unit stratified diagnostic sample was selected after aggregate results. Raw disagreement is not a population evaluator-error rate or false-negative rate; no inverse-probability estimator is used here.",
    "One annotator personally reviewed labels after ChatGPT proposed them. This is AI-assisted human review, not two independent annotators or an independent gold-standard corpus.",
    "The three unclear labels remain unresolved. E/C assignments are hypothetical scenarios shared coherently by every occurrence of each exact unit, not adjudicated labels or independent occurrence-level choices.",
    "Coherent point ranges identify only uncertainty in these three audited units, conditional on retaining all other assigned labels; they do not bound the semantic accuracy of unaudited responses.",
    "95% intervals are descriptive; 99% intervals are the original nominal Bonferroni level for five contrasts. Bootstrap resamples paired questions and training seeds with thresholds together, conditional on the audit labels/selection, TRAIN candidate bank and fitted calibrators. Human-label and audit-sampling uncertainty are not bootstrapped.",
    "An envelope over scenario-specific confidence endpoints is a sensitivity envelope, not a newly calibrated confidence interval or an additional confirmatory family.",
    "NQ transfer is descriptive. Improvement relative to another method does not imply positive utility relative to always abstaining (zero), calibrated risk guarantees or whole-corpus semantic superiority.",
    "Literal IDK, empty output and truncation rules stay strict. Nonliteral semantic abstention is recorded separately and does not relax the frozen output-format contract.",
    "No predictions, probabilities, decisions, calibrators, weights, source configurations or frozen endpoints are modified.",
]


def audit_module():
    spec = importlib.util.spec_from_file_location("audit_v2_sensitivity_source", ROOT / "analysis/audit_v2.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def strict_semantic_code(unit, human_label, unknown=None):
    """Technical-format contract takes priority; preserve unknown labels externally."""
    if unit["truncated"] or not unit["response"].strip():
        return 0, "fixed_invalid_output"
    if unit["response"].strip() == "IDK":
        return -1, "fixed_literal_idk"
    if human_label is None:
        return CODE[unit["automatic_outcome"]], "unaudited_em_retained"
    if human_label == "unclear":
        if unknown is None:
            return CODE[unit["automatic_outcome"]], "unclear_em_retained"
        if unknown not in (0, 1):
            raise ValueError("Unclear substantive units require coherent E/C assignments")
        return unknown, "hypothetical_unclear_assignment"
    if human_label == "abstain":
        return 0, "semantic_abstention_strict_format_error"
    if human_label not in ("correct", "error"):
        raise ValueError("Invalid human semantic label")
    return CODE[human_label], "audited_semantic_label"


def seed_ids(method):
    return SEEDS if method in ARMS or method.startswith("critic_") else (0,)


def utility(codes):
    codes = np.asarray(codes)
    if not np.isin(codes, (-1, 0, 1)).all() or codes.shape[-1] != 2:
        raise ValueError("Expected valid labels with two primary thresholds")
    return (codes == 1).astype(float) - np.asarray(TAUS) * (codes >= 0)


def summarized(codes):
    codes = np.asarray(codes)
    n = codes.size; correct = int((codes == 1).sum()); error = int((codes == 0).sum())
    return {"n_prediction_rows": n, "correct": correct, "error": error, "abstain": n-correct-error,
            "utility": float(utility(codes).mean()), "coverage": (correct+error)/n,
            "selective_risk": error/(correct+error) if correct+error else None,
            "accuracy": correct/n, "error_rate": error/n}


def assemble_components(original, known, uncertain_masks, unclear_em_codes):
    """Components are linear in one shared E/C bit per unresolved exact unit.

    Input labels: seeds × questions × thresholds. Unknown masks affect only
    emitted occurrences; rejected filters are never converted into answers.
    """
    original, known = np.asarray(original), np.asarray(known)
    masks = np.asarray(uncertain_masks, dtype=bool)
    if original.shape != known.shape or original.ndim != 3 or original.shape[-1] != 2:
        raise ValueError("Expected seed x question x threshold labels")
    if masks.shape != (len(unclear_em_codes), *original.shape) or np.any(masks.sum(axis=0) > 1):
        raise ValueError("Unknown-unit masks overlap or have invalid shape")
    if np.any((original >= 0) != (known >= 0)) or np.any(masks & (known[None, ...] < 0)):
        raise ValueError("Fixed policy emission decisions changed")
    base = known.copy()
    for mask in masks:
        base[mask] = 0
    pieces = [utility(original).mean(axis=-1), utility(base).mean(axis=-1)]
    pieces.extend(mask.mean(axis=-1) for mask in masks)
    return np.stack(pieces), base


def scenario_codes(base, masks, assignment):
    if len(masks) != len(assignment) or any(bit not in (0, 1) for bit in assignment):
        raise ValueError("Bad coherent assignment")
    result = np.array(base, copy=True)
    for mask, bit in zip(masks, assignment):
        result[mask] = bit
    return result


def scenario_value(components, assignment):
    return components[..., 1] + sum(bit * components[..., i+2] for i, bit in enumerate(assignment))


def bootstrap_components(components, n_boot=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """One shared crossed resampling plan for every method and every scenario.

    Shape: methods x components x three paired seeds x questions. Deterministic
    baselines are broadcast identically across seeds, adding no seed variance.
    """
    values = np.asarray(components, dtype=float)
    if values.ndim != 4 or values.shape[2] != 3 or values.shape[3] < 1 or not np.isfinite(values).all() or n_boot < 1:
        raise ValueError("Expected finite method x component x 3seed x question values")
    rng = np.random.default_rng(seed); n_question = values.shape[3]
    flat = values.reshape(-1, 3, n_question)
    prepared = {}
    result = np.empty((n_boot, flat.shape[0]))
    # Batch bounded question counts; RNG consumption matches the registered v2
    # bootstrap: three seed draws, then N question draws, for each replicate.
    for start in range(0, n_boot, 256):
        stop = min(n_boot, start+256)
        groups = defaultdict(list); weights = np.empty((stop-start, n_question))
        for offset in range(stop-start):
            seeds = rng.integers(0, 3, 3)
            questions = rng.integers(0, n_question, n_question)
            key = tuple(np.bincount(seeds, minlength=3))
            groups[key].append(offset)
            weights[offset] = np.bincount(questions, minlength=n_question) / n_question
        for key, indices in groups.items():
            if key not in prepared:
                prepared[key] = np.einsum("csq,s->cq", flat, np.asarray(key)/3)
            result[start+np.asarray(indices)] = weights[indices] @ prepared[key].T
    return result.reshape(n_boot, values.shape[0], values.shape[1])


def intervals(point, draws):
    q = np.quantile(draws, [.025, .975, .005, .995])
    return {"point": float(point), "ci95_low": float(q[0]), "ci95_high": float(q[1]),
            "ci99_low": float(q[2]), "ci99_high": float(q[3])}


def scenario_summary(point_components, boot_components, assignments, retaining):
    original = intervals(point_components[0], boot_components[:, 0])
    known = intervals(scenario_value(point_components, retaining), scenario_value(boot_components, retaining))
    scenarios = [intervals(scenario_value(point_components, assignment), scenario_value(boot_components, assignment)) for assignment in assignments]
    summary = {**{"original_"+k:v for k,v in original.items()}, **{"partial_em_unclear_"+k:v for k,v in known.items()},
               "coherent_point_low": min(s["point"] for s in scenarios), "coherent_point_high": max(s["point"] for s in scenarios),
               "all_assignments_ci95_positive": all(s["ci95_low"] > 0 for s in scenarios),
               "all_assignments_ci99_positive": all(s["ci99_low"] > 0 for s in scenarios)}
    for field in ("ci95_low", "ci95_high", "ci99_low", "ci99_high"):
        summary["scenario_"+field+"_min"] = min(s[field] for s in scenarios)
        summary["scenario_"+field+"_max"] = max(s[field] for s in scenarios)
    return summary, scenarios


def load_inputs(submission=SUBMISSION):
    audit = audit_module(); submission = Path(submission)
    receipt = read_json(submission / "import-receipt.json")
    if receipt["status"] != "labels_imported_pending_adjudication" or receipt["n"] != 150:
        raise ValueError("Expected a complete validated150-label import")
    if file_digest(submission / "labels.json") != receipt["labels_sha256"] or file_digest(submission / "submitted.xlsx") != receipt["submission_sha256"]:
        raise ValueError("Imported labels/workbook changed")
    validated, provenance, mapping, export, submitted_sha = audit.validate_submission(submission / "submitted.xlsx", AUDIT)
    imported = read_json(submission / "labels.json")
    if imported["provenance"] != provenance or submitted_sha != receipt["submission_sha256"]:
        raise ValueError("Annotation provenance differs from validated workbook")
    by_id = {r["audit_id"]:r for r in imported["annotations"]}
    if len(by_id) != 150:
        raise ValueError("Duplicate imported audit IDs")
    for row in validated:
        saved = by_id[row["audit_id"]]
        if any(saved[k] != row[k] for k in audit.FIELDS):
            raise ValueError("Imported annotation differs from the workbook")
    units, sources = audit.load_population()
    human, annotations = {}, []
    for audit_id, row in by_id.items():
        unit = mapping["units"][audit_id]; key = unit["unit_key"]
        if row["unit_key"] != key or key not in units or key in human:
            raise ValueError("Invalid exact-unit mapping")
        current = units[key]
        if any(unit[k] != current[k] for k in ("question_id", "response", "truncated", "aliases", "automatic_outcome")):
            raise ValueError("Audited unit differs from frozen outputs")
        if row["sampling"] != unit["sampling"] or row["automatic_outcome"] != current["automatic_outcome"]:
            raise ValueError("Audit stratum or original label changed")
        human[key] = row["human_outcome"]
        annotations.append({"audit_id":audit_id,"unit_key":key,"split":unit["split"],"frame":unit["sampling"]["stratum"][1],
                            "automatic_outcome":unit["automatic_outcome"],"human_outcome":row["human_outcome"],
                            "format_note":row["format_note"],"sampling":unit["sampling"]})
    unknown = sorted(k for k,v in human.items() if v == "unclear")
    if len(unknown) > 3:
        raise ValueError("Bounded enumeration supports at most three unclear units")
    for key in unknown:
        u = units[key]
        if u["truncated"] or not u["response"].strip() or u["response"].strip() == "IDK" or u["automatic_outcome"] == "abstain":
            raise ValueError("Unclear units must be substantive for binary C/E scenarios")
    for path in (submission/"labels.json", submission/"submitted.xlsx", submission/"import-receipt.json", AUDIT/"private-key.json",
                 AUDIT/"sampling.json", AUDIT/"export-receipt.json", Path(__file__), ROOT/"tests/test_audit_v2_sensitivity.py"):
        sources[str(path.relative_to(ROOT))] = file_digest(path)
    return units, human, unknown, annotations, sources, audit


def build_panels(units, human, unknown, sources, audit):
    panels, index = {}, {split:{} for split in COUNTS}
    for unit in units.values():
        for occurrence in unit["occurrences"]:
            if occurrence["variant"] == "original" and occurrence["mode"] == "forced":
                split = occurrence["split"]; qid = unit["question_id"]
                if qid in index[split]:raise ValueError("Duplicate original candidate ID")
                index[split][qid] = occurrence["source_line"]-1
    for split,n in COUNTS.items():
        if len(index[split]) != n or set(index[split].values()) != set(range(n)):
            raise ValueError("Question index population mismatch")
        for method in METHODS:
            shape = (len(seed_ids(method)), n, 2)
            panels[split,method] = {"original":np.full(shape,-9,dtype=np.int8), "known":np.full(shape,-9,dtype=np.int8),
                                   "masks":np.zeros((len(unknown),*shape),dtype=bool), "seen":np.zeros(shape,dtype=bool)}
    unknown_index = {k:i for i,k in enumerate(unknown)}
    propagation = defaultdict(Counter)
    def put(split,method,seed,tau,key,emit=True):
        p = panels[split,method];u=units[key];at=(seed_ids(method).index(seed),index[split][u["question_id"]],TAUS.index(tau))
        if p["seen"][at]:raise ValueError("Duplicate policy occurrence")
        p["seen"][at]=True
        original = CODE[u["automatic_outcome"]] if emit else -1
        known, reason = strict_semantic_code(u,human.get(key))
        if not emit:known=-1
        p["original"][at]=original;p["known"][at]=known
        if key in unknown_index and emit:p["masks"][(unknown_index[key],*at)]=True
        if key in human:
            propagation[key]["matched_policy_rows"]+=1
            if emit:propagation[key]["emitted_policy_rows"]+=1
            if known != original:propagation[key]["changed_policy_rows"]+=1
            propagation[key][f"matched:{split}:{method}:{seed}:{tau}"]+=1
    for key,u in units.items():
        for o in u["occurrences"]:
            if o["mode"] == "threshold" and o["tau"] in TAUS:
                put(o["split"],o["arm"],o["seed"],o["tau"],key)
    original_signature = {}
    for seed in SEEDS:
        for split in COUNTS:
            path=WORK/f"artifacts/selector/test-evaluation/s{seed}/{split}-scores.json"
            if file_digest(path)!=sources[str(path.relative_to(ROOT))]:raise ValueError("Changed scores")
            env=read_json(path)
            for row in env["rows"]:
                key=audit.unit_key(row)
                if key not in units:raise ValueError("Unknown shared candidate")
                for method in FILTERS:
                    score_seed=seed if method.startswith("critic_") else 0
                    if score_seed==0:
                        signature=(row["id"],row["text"],row["truncated"],row["logits"][method])
                        sigkey=(split,method,row["id"])
                        if sigkey in original_signature:
                            if original_signature[sigkey]!=signature:raise ValueError("Deterministic score mismatch")
                            continue
                        original_signature[sigkey]=signature
                    for tau in TAUS:
                        put(split,method,score_seed,tau,key,audit.emits(row,row["logits"][method],tau))
    retaining=tuple(CODE[units[k]["automatic_outcome"]] for k in unknown)
    for (split,method),p in panels.items():
        if not p.pop("seen").all():raise ValueError("Incomplete policy grid")
        p["components"],p["base"]=assemble_components(p["original"],p["known"],p["masks"],retaining)
    return panels,propagation,retaining


def check_originals(panels, stats):
    for row in stats["joint_primary"]:
        observed=summarized(panels[row["split"],row["method"]]["original"])
        for field in ("utility","coverage","selective_risk","correct","error","abstain"):
            if observed[field] is None or row[field] is None:
                if observed[field] is not row[field]:raise ValueError("Original undefined risk differs")
            elif not math.isclose(observed[field],row[field],abs_tol=1e-12,rel_tol=1e-12):raise ValueError("Original metric differs")


def diagnostics(annotations,units):
    groups=Counter((r["split"],r["frame"],r["automatic_outcome"],r["human_outcome"]) for r in annotations)
    result=[]
    for (split,frame,automatic,human),n in sorted(groups.items()):
        result.append({"dataset":split,"frame":frame,"automatic_label":automatic,"human_semantic_label":human,"sample_units":n,
                       "interpretation":"Stratified diagnostic count; not population error rate"})
    format_overrides=[]
    for r in annotations:
        code,reason=strict_semantic_code(units[r["unit_key"]],r["human_outcome"])
        if reason in ("fixed_invalid_output","semantic_abstention_strict_format_error"):
            format_overrides.append({"audit_id":r["audit_id"],"unit_key":r["unit_key"],"human_semantic_label":r["human_outcome"],"strict_code":code,"reason":reason})
    return result,format_overrides


def write_csv(path,rows):
    if not rows:raise ValueError("No rows to export")
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("x",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def analyze(output=OUT, submission=SUBMISSION):
    output=Path(output)
    if output.exists():raise FileExistsError("Sensitivity output exists; no overwrite or silent reanalysis")
    units,human,unknown,annotations,sources,audit=load_inputs(submission)
    panels,propagation,retaining=build_panels(units,human,unknown,sources,audit)
    stats_path=WORK/"artifacts/analysis/statistics.json";stats=read_json(stats_path)
    if file_digest(stats_path)!=sources[str(stats_path.relative_to(ROOT))]:raise ValueError("Changed final statistics")
    check_originals(panels,stats)
    assignments=list(itertools.product((0,1),repeat=len(unknown)))
    diag,format_overrides=diagnostics(annotations,units)
    method_rows,absolute_rows,contrasts,seed_rows,private_scenarios=[],[],[],[],[]
    for split,n in COUNTS.items():
        comps=[]
        for method in METHODS:
            c=panels[split,method]["components"]
            comps.append(np.repeat(c,3,axis=1) if c.shape[1]==1 else c)
        comps=np.stack(comps);boot=bootstrap_components(comps);points=comps.mean(axis=(2,3))
        for mi,method in enumerate(METHODS):
            p=panels[split,method]
            summary,scenarios=scenario_summary(points[mi],boot[:,mi,:],assignments,retaining)
            absolute_rows.append({"dataset":split,"method":method,"n_questions":n,"n_model_seeds":len(seed_ids(method)),**summary})
            scenario_metrics=[summarized(scenario_codes(p["base"],p["masks"],bits)) for bits in assignments]
            known=summarized(p["known"]);original=summarized(p["original"])
            method_rows.append({"dataset":split,"method":method,"n_questions":n,"n_model_seeds":len(seed_ids(method)),
                                **{"original_"+k:v for k,v in original.items()}, **{"partial_em_unclear_"+k:v for k,v in known.items()},
                                **{f"coherent_{metric}_{bound}":fun(v[metric] for v in scenario_metrics if v[metric] is not None)
                                   if any(v[metric] is not None for v in scenario_metrics) else None
                                   for metric in ("utility","coverage","selective_risk","accuracy","error_rate") for bound,fun in (("low",min),("high",max))}})
            private_scenarios.append({"dataset":split,"method":method,"scenarios":[{"assignment":list(bits),"intervals":ci,"metrics":met} for bits,ci,met in zip(assignments,scenarios,scenario_metrics)]})
        for left,right in CONTRASTS:
            li,ri=METHODS.index(left),METHODS.index(right)
            summary,scenarios=scenario_summary(points[li]-points[ri],boot[:,li,:]-boot[:,ri,:],assignments,retaining)
            old=next(r for r in stats["contrasts"] if (r["split"],r["left"],r["right"])==(split,left,right))
            for field in ("ci95_low","ci95_high","ci99_low","ci99_high"):
                if not math.isclose(summary["original_"+field],old[field],abs_tol=2e-12,rel_tol=1e-12):raise ValueError("Reconstructed original bootstrap interval differs")
            if not math.isclose(summary["original_point"],old["effect"],abs_tol=1e-12):raise ValueError("Reconstructed original contrast differs")
            contrasts.append({"dataset":split,"left":left,"right":right,"n_questions":n,"paired_training_seeds":3,
                              "role":"posthoc_primary_dataset_sensitivity" if split=="test_trivia" else "posthoc_descriptive_transfer",**summary})
            cube=comps[li]-comps[ri]
            seedpoints=cube.mean(axis=2).T
            for seed,point in zip(SEEDS,seedpoints):
                scenarios_point=[float(scenario_value(point,bits)) for bits in assignments]
                seed_rows.append({"dataset":split,"left":left,"right":right,"seed":seed,"original_effect":float(point[0]),
                                  "partial_em_unclear_effect":float(scenario_value(point,retaining)),"coherent_point_low":min(scenarios_point),"coherent_point_high":max(scenarios_point)})
            private_scenarios.append({"dataset":split,"left":left,"right":right,
                                      "scenarios":[{"assignment":list(bits),**ci} for bits,ci in zip(assignments,scenarios)]})
    changes=Counter((r["split"],r["automatic_outcome"],r["human_outcome"]) for r in annotations)
    summary={"status":"partial_posthoc_sensitivity_complete","audit_units":len(annotations),
             "unique_audit_questions":len({units[r["unit_key"]]["question_id"] for r in annotations}),
             "dataset_audit_units":dict(Counter(r["split"] for r in annotations)),
             "frame_audit_units":dict(Counter(r["frame"] for r in annotations)),
             "known_semantic_label_disagreements":sum(r["human_outcome"] not in ("unclear",r["automatic_outcome"]) for r in annotations),
             "error_to_correct_by_dataset":{split:changes[split,"error","correct"] for split in COUNTS},
             "unresolved_units":len(unknown),"coherent_assignments_enumerated":len(assignments),
             "format_contract_overrides":len(format_overrides),
             "annotation_provenance":{"annotators":1,"ai_assisted":True,"personally_reviewed_every_label_declared":True,
                                      "procedure":"ChatGPT proposed labels and notes; one human then personally reviewed every label and corrected labels as needed."},
             "bootstrap":{"replicates":BOOTSTRAPS,"seed":BOOTSTRAP_SEED,"resampling":"crossed paired question x training seed, both primary thresholds together",
                          "ordinary_interval":.95,"family_interval":.99,"original_family_size":5,"deterministic_baselines":"single shared baseline, no artificial seed replication"},
             "assignment_range_interpretation":"Conditional identification range across three unresolved exact units, not a confidence interval",
             "interval_envelope_interpretation":"Minimum/maximum endpoints of conditional scenario-specific intervals; not a newly calibrated confidence interval",
             "contrasts":contrasts,"absolute_utilities":absolute_rows,"limits":LIMITS,
             "frozen_endpoints_modified":False}
    audit.verify_sources(sources)
    output.mkdir(parents=True,exist_ok=False);public=output/"public";public.mkdir()
    tables={"audit-diagnostics.csv":diag,"method-sensitivity.csv":method_rows,"contrast-sensitivity.csv":contrasts,
            "absolute-utility-sensitivity.csv":absolute_rows,"per-seed-contrasts.csv":seed_rows}
    for name,table in tables.items():write_csv(public/name,table)
    write_json(public/"summary.json",summary)
    write_json(output/"full-results.json",{"summary":summary,"unknown_units_in_assignment_order":unknown,
                "retaining_em_assignment":list(retaining),"scenario_results":private_scenarios,
                "annotations_with_sampling":annotations,"format_contract_overrides":format_overrides})
    write_json(output/"propagation.json",{"unit_matches":{r["audit_id"]:{"unit_key":r["unit_key"],"human_outcome":r["human_outcome"],
                       **dict(propagation[r["unit_key"]])} for r in annotations},
                "meaning":"Matched includes rejected filter decisions; changed counts only policy rows whose endpoint changes under resolved human labels. Unclear scenarios remain hypothetical."})
    audit.verify_sources(sources)
    receipt={"status":"complete","at":utcnow(),"source_sha256":sources,"frozen_endpoints_modified":False,
             "outputs":{str(p.relative_to(output)):file_digest(p) for p in sorted(output.rglob("*")) if p.is_file()},
             "public_files":sorted([*tables,"summary.json"]),"human_labels_invented":False}
    write_json(output/"receipt.json",receipt)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT);parser.add_argument("--submission",type=Path,default=SUBMISSION)
    args=parser.parse_args();result=analyze(args.output,args.submission)
    print({"status":result["status"],"known_disagreements":result["known_semantic_label_disagreements"],"unresolved":result["unresolved_units"]})
