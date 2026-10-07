from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ROOT, load_config, matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description="Threshold-controlled abstention study")
    parser.add_argument("--config", default=str(ROOT / "configs/study.yaml"))
    parser.add_argument("--workdir", default=str(ROOT))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    data = sub.add_parser("prepare-data")
    data.add_argument("--local-source", help="Fixture-only JSONL directory; cannot freeze a research protocol")
    models = sub.add_parser("prepare-models")
    models.add_argument("--download", action="store_true")
    sub.add_parser("matrix")
    pilot = sub.add_parser("pilot")
    pilot.add_argument("--model", choices=["qwen", "smol"], required=True)
    sub.add_parser("freeze")
    train = sub.add_parser("train")
    train.add_argument("--model", choices=["qwen", "smol"], required=True)
    train.add_argument("--arm", choices=["binary", "conditioned", "fixed"], required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--resume", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--model", choices=["qwen", "smol"], required=True)
    evaluation.add_argument("--arm", choices=["original", "binary", "conditioned", "fixed", "warmup"], default="original")
    evaluation.add_argument("--seed", type=int, default=17)
    sub.add_parser("report")
    audit = sub.add_parser("audit")
    audit.add_argument("--import-csv")
    run = sub.add_parser("run-study")
    run.add_argument("--max-jobs", type=int, help="Stop cleanly after this many jobs")
    run.add_argument("--night-start", type=int, default=22)
    run.add_argument("--night-end", type=int, default=6)
    run.add_argument("--any-time", action="store_true", help="Explicitly allow work outside the night window")
    args = parser.parse_args(argv)
    config, workdir = load_config(args.config), Path(args.workdir).resolve()
    if args.command == "doctor":
        from .protocol import environment
        result = environment()
    elif args.command == "prepare-data":
        from .data import prepare
        manifest = prepare(config, workdir, args.local_source)
        result = {"provenance": manifest["provenance"], "counts": {k: v["count"] for k, v in manifest["splits"].items()}}
    elif args.command == "prepare-models":
        from .models import pin_models
        result = pin_models(config, workdir, args.download)
    elif args.command == "matrix":
        result = list(matrix(config))
    elif args.command == "pilot":
        from .training import pilot
        result = pilot(config, workdir, args.model)
    elif args.command == "freeze":
        from .protocol import freeze
        result = freeze(config, workdir)
    elif args.command == "train":
        from .training import train
        result = train(config, workdir, args.model, args.arm, args.seed, resume=args.resume)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(config, workdir, args.model, args.arm, args.seed)
    elif args.command == "report":
        from .reporting import report
        result = report(config, workdir)
    elif args.command == "audit":
        from .reporting import audit_export, audit_import
        result = audit_import(workdir, args.import_csv) if args.import_csv else audit_export(config, workdir)
    elif args.command == "run-study":
        from .runner import run_study
        result = run_study(config, workdir, args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    from .runner import RunPaused
    try:
        main()
    except RunPaused as exc:
        print(str(exc), flush=True)
        raise SystemExit(75)
