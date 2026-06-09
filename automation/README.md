# Automated run-and-push

A thin layer on top of the research code: list the scripts you want to run in
`jobs.txt`, launch `run.bat` on the remote machine, and the results land on
GitHub by themselves. Your research code under `code/` is untouched.

## The daily loop

1. **Laptop** — edit `jobs.txt` to list the script(s) for this session, then
   commit & push:

   ```
   git add jobs.txt
   git commit -m "queue: heterogeneous PINO run"
   git push
   ```

2. **Remote (AnyDesk)** — run the launcher, then disconnect:

   ```
   run.bat
   ```

3. **Laptop** — once it's done (a new `run ...` commit appears on GitHub),
   pull the results:

   ```
   git pull
   ```

You do **not** need to stay connected to AnyDesk after step 2.

## What run.bat does

`run.bat` calls `automation/runner.py`, which:

1. `git pull` — picks up the latest code you pushed from your laptop.
2. Reads `jobs.txt` and runs each script **from its own folder** (so each
   script's `results/` lands beside it, matching the repo convention).
3. **Stops at the first script that fails** (fail-fast).
4. `git add -A`, commits, and **pushes once** at the end — whether the run
   finished or stopped on a failure, so you always get the logs.

## jobs.txt

One script per line, paths relative to the repo root, forward slashes:

```
code/heterogeneous/pino_heterogeneous/pinn1d_heterogeneous_parametric_neural_operator.py
code/homogeneous_pe/baseline_pinn/pinn1d_transport_simple_baseline.py --epochs 2000
```

Blank lines and `#` comments are ignored. Quote any path containing spaces.

## Where the logs go

Each run creates `runs/<timestamp>/` containing:

- `NN_<script>.log` — full stdout/stderr of each script (also echoed live)
- `summary.txt` — what ran, exit codes, durations, pass/fail

These are committed, so you can read them on GitHub without AnyDesk.

## Conda / virtualenv

If your scripts need a specific environment, open `run.bat` and uncomment the
one activation line near the top (e.g. `call conda activate pinn`).

## Dry run (no git)

To test the job list locally without pulling/committing/pushing:

```
set RUNNER_NO_GIT=1
python automation\runner.py
```

## Notes

- The runner commits **all** working-tree changes (`git add -A`). Keep code
  edits on your laptop and let the remote only produce results, so each run
  commit stays clean. `*.zip` and `_tmp_*` are git-ignored as a safety net.
- To remove your GitHub access from the remote later: revoke the fine-grained
  token on GitHub, then clear the `git:https://github.com` entry in Windows
  Credential Manager.
