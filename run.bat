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
REM  Current batch (jobs.txt): Report 5 exp F (maximin float64) and
REM  exp G (maximin, zone trunks). Pull on laptop after remote push to analyse.
REM
REM  Run a different list:   run.bat my_list.txt
REM ============================================================

cd /d "%~dp0"

REM --- If you use a conda/venv environment, uncomment & edit ONE line: ---
REM call conda activate pinn
REM call .venv\Scripts\activate.bat

python automation\runner.py %*
