#!/usr/bin/env python3
"""
runner.py - run a list of experiment scripts, then commit & push the results.

Daily loop:
  1. git pull                  (get the latest code you pushed from your laptop)
  2. read jobs.txt             (ordered list of scripts to run)
  3. run each script FROM ITS OWN FOLDER, logging to runs/<timestamp>/
       - STOP IMMEDIATELY if a script fails (fail-fast)
  4. git add -A + commit + push ONCE at the end
       - the commit message records success, or the script that failed

You launch this once (via run.bat) and can then disconnect AnyDesk; it keeps
running and pushes by itself when finished.

Usage (from the repo root):
  python automation/runner.py                 # uses jobs.txt at the repo root
  python automation/runner.py my_list.txt     # use a different jobs file

jobs.txt format (one job per line):
  # blank lines and #comments are ignored
  # use FORWARD slashes; quote any path containing spaces
  code/homogeneous_pe/baseline_pinn/pinn1d_transport_simple_baseline.py
  code/heterogeneous/pino_heterogeneous/pinn1d_heterogeneous_parametric_neural_operator.py --epochs 2000

Set the env var RUNNER_NO_GIT=1 to skip pull/commit/push (handy for a local dry run).
"""
from __future__ import annotations

import datetime as _dt
import os
import pathlib
import shlex
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent   # automation/ -> repo root
NO_GIT = os.environ.get("RUNNER_NO_GIT") == "1"


def git(*args: str) -> int:
    """Run a git command at the repo root. Returns its exit code (0 if skipped)."""
    if NO_GIT:
        print(f"  (skip) git {' '.join(args)}")
        return 0
    print(f"  $ git {' '.join(args)}")
    return subprocess.run(["git", *args], cwd=str(REPO)).returncode


def read_jobs(jobs_file: pathlib.Path):
    """Parse jobs.txt into a list of (script_path, [extra_args])."""
    jobs = []
    for n, raw in enumerate(jobs_file.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line)
        except ValueError as e:
            sys.exit(f"jobs file line {n}: cannot parse ({e}): {line}")
        jobs.append((parts[0], parts[1:]))
    return jobs


def run_script(idx: int, script: str, extra, run_dir: pathlib.Path):
    """Run one script from its own folder, tee output to a log. Returns (returncode, duration)."""
    script_path = (REPO / script).resolve()
    log_path = run_dir / f"{idx:02d}_{script_path.stem}.log"
    start = _dt.datetime.now()

    if not script_path.exists():
        log_path.write_text(f"Script not found: {script_path}\n", encoding="utf-8")
        print(f"   {idx}. MISSING  {script}")
        return 127, _dt.timedelta(0)

    try:
        rel = script_path.relative_to(REPO)
    except ValueError:
        rel = script_path
    print(f"   {idx}. {rel}   (log: runs/{run_dir.name}/{log_path.name})")

    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        logf.write(f"# {script} {' '.join(extra)}\n"
                   f"# started {start:%Y-%m-%d %H:%M:%S}\n"
                   f"# cwd {script_path.parent}\n\n")
        logf.flush()
        # Force the child to emit UTF-8 and flush promptly, so live echo + logs stay clean.
        child_env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [sys.executable, script_path.name, *extra],
            cwd=str(script_path.parent),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)        # echo live, in case you are watching
            logf.write(line)
        proc.wait()
        dur = _dt.datetime.now() - start
        logf.write(f"\n# exit {proc.returncode} after {dur}\n")
    return proc.returncode, dur


def main() -> int:
    # Make console output robust on Windows (avoid UnicodeEncodeError on, e.g., tqdm bars).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    jobs_file = REPO / (sys.argv[1] if len(sys.argv) > 1 else "jobs.txt")
    if not jobs_file.exists():
        sys.exit(f"jobs file not found: {jobs_file}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = REPO / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"== runner {ts} ==  (repo: {REPO})")

    # 1) latest code
    print("[1/4] Pulling latest code ...")
    if git("pull", "--rebase", "--autostash") != 0:
        print("   WARNING: git pull failed; continuing with the code currently on disk.")

    # 2) job list
    jobs = read_jobs(jobs_file)
    if not jobs:
        sys.exit(f"No scripts listed in {jobs_file.name} (all blank/comments).")
    print(f"[2/4] {len(jobs)} job(s) queued from {jobs_file.name}:")
    for i, (s, a) in enumerate(jobs, 1):
        print(f"   {i}. {s} {' '.join(a)}".rstrip())

    # 3) run, fail-fast
    print("[3/4] Running ...")
    summary, n_ok, failed = [], 0, None
    for i, (script, extra) in enumerate(jobs, 1):
        rc, dur = run_script(i, script, extra, run_dir)
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        summary.append(f"{i:02d}  {status:>12}  {str(dur).split('.')[0]:>8}  {script}")
        if rc == 0:
            n_ok += 1
        else:
            failed = script
            print(f"      -> {status}; stopping (fail-fast).")
            break

    # summary file
    head = (f"run {ts}\n"
            f"jobs file: {jobs_file.name}\n"
            f"result: {('FAILED at ' + failed) if failed else ('all ' + str(len(jobs)) + ' ok')}\n"
            f"{n_ok}/{len(jobs)} script(s) succeeded\n\n"
            "  #        status   elapsed  script\n"
            + "\n".join(summary) + "\n")
    (run_dir / "summary.txt").write_text(head, encoding="utf-8")
    print("---- summary ----")
    print(head)

    # 4) commit & push once
    print("[4/4] Committing and pushing results ...")
    msg = (f"run {ts}: FAILED at {failed} ({n_ok}/{len(jobs)} ok)"
           if failed else f"run {ts}: {len(jobs)}/{len(jobs)} ok")
    git("add", "-A")
    if NO_GIT:
        print("  (skip) git commit / push")
    else:
        if subprocess.run(["git", "commit", "-m", msg], cwd=str(REPO)).returncode != 0:
            print("  (nothing new to commit)")
        git("push")

    if failed:
        print(f"DONE - with a FAILURE: {failed}.  See runs/{ts}/ for logs.")
        return 1
    print(f"DONE - {len(jobs)}/{len(jobs)} ok.  See runs/{ts}/ for logs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
