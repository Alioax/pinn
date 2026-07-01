@echo off
setlocal
REM ============================================================
REM  run.bat - run the scripts listed in jobs.txt, then push results.
REM
REM  Daily use:
REM    1) On your laptop: edit jobs.txt, commit & push.
REM    2) On the remote (AnyDesk): just run   run.bat
REM    3) Disconnect AnyDesk. It pulls, runs every job in order, and
REM       pushes the results when done (or stops at the first failure).
REM
REM  Current batch (jobs.txt): PINO X-series — N=500 2x2 {5-zone,uniform}x{L-BFGS,SOAP}
REM  plus optional w64 capacity probe (X5/X6). SOAP iso-compute vs L-BFGS twins.
REM  Pull on laptop after remote push to analyse / run validation plots.
REM
REM  Run a different list:   run.bat my_list.txt
REM ============================================================

cd /d "%~dp0"

REM --- If you use a conda/venv environment, uncomment & edit ONE line: ---
REM call conda activate pinn
REM call .venv\Scripts\activate.bat

python automation\runner.py %*
