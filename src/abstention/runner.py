"""Resumable sequential jobs; no concurrent GPU workers or automatic publication."""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from .config import matrix
from .io import append_jsonl, read_json, utcnow, write_json
from .protocol import require_budget, require_lock


class RunPaused(RuntimeError):
    pass


def window(now, start, end):
    if start == end or not (0 <= start < 24 and 0 <= end < 24):
        raise ValueError("Night window needs distinct hours in [0,23].")
    inside = start <= now.hour < end if start < end else now.hour >= start or now.hour < end
    if not inside:
        return None
    finish = now.replace(hour=end, minute=0, second=0, microsecond=0)
    if finish <= now:
        finish += timedelta(days=1)
    return finish.timestamp()


def jobs(config, workdir, lock):
    workdir = Path(workdir)
    for r in matrix(config):
        directory = workdir / "artifacts/runs" / r["run_id"]
        if not (directory / "result.json").exists():
            cmd = ["train", "--model", r["model"], "--arm", r["arm"], "--seed", str(r["seed"])]
            if list(directory.glob("checkpoint-*")):
                cmd.append("--resume")
            yield r["run_id"], cmd
    # Every training checkpoint is fixed before tests are opened.
    for key in config["models"]:
        if not (workdir / "artifacts/predictions" / f"{key}-original-s0/complete.json").exists():
            yield f"eval-{key}-original", ["evaluate", "--model", key]
        if lock["pilots"][key]["initializer"] and not (workdir / "artifacts/predictions" / f"{key}-warmup-s0/complete.json").exists():
            yield f"eval-{key}-warmup", ["evaluate", "--model", key, "--arm", "warmup"]
    for r in matrix(config):
        if not (workdir / "artifacts/predictions" / f"{r['run_id']}/complete.json").exists():
            yield f"eval-{r['run_id']}", ["evaluate", "--model", r["model"], "--arm", r["arm"], "--seed", str(r["seed"])]


def run_study(config, workdir, args):
    lock = require_lock(config, workdir)
    directory = Path(workdir) / "artifacts/queue"
    directory.mkdir(parents=True, exist_ok=True)
    completed = 0
    with (directory / "runner.lock").open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            require_budget(config, workdir)
            todo = list(jobs(config, workdir, lock))
            if not todo or (args.max_jobs is not None and completed >= args.max_jobs):
                break
            now = datetime.now(ZoneInfo("America/Mexico_City"))
            finish = window(now, args.night_start, args.night_end)
            if not args.any_time and finish is None:
                write_json(directory / "status.json", {"state": "waiting_for_night", "time": utcnow(), "remaining_jobs": len(todo)})
                time.sleep(30)
                continue
            job_id, command = todo[0]
            environment = os.environ.copy()
            if not args.any_time:
                environment["ABSTENTION_STOP_AT"] = str(finish)
            write_json(directory / "status.json", {"state": "running", "job": job_id, "time": utcnow(), "remaining_jobs": len(todo)})
            with (directory / f"{job_id}.log").open("a") as log:
                code = subprocess.call([sys.executable, "-m", "abstention.cli", "--config", args.config,
                                        "--workdir", str(workdir), *command], stdout=log, stderr=subprocess.STDOUT, env=environment)
            append_jsonl(directory / "history.jsonl", {"job": job_id, "exit_code": code, "time": utcnow()})
            if code == 75:
                continue
            if code:
                write_json(directory / "status.json", {"state": "failed", "job": job_id, "exit_code": code, "time": utcnow()})
                raise RuntimeError(f"Job {job_id} failed. See its log; no later job was started.")
            completed += 1
        state = "complete" if not list(jobs(config, workdir, lock)) else "paused_after_job_limit"
        write_json(directory / "status.json", {"state": state, "time": utcnow(), "completed_this_session": completed})
        if state == "complete":
            from .reporting import report, audit_export
            report(config, workdir)
            if not (Path(workdir) / "artifacts/audit/blind.csv").exists():
                audit_export(config, workdir)
        return {"state": state, "completed_this_session": completed}
