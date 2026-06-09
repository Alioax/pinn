# PINN Repo — Structure Review & Proposal

*Prepared 2026-06-10. Scope: suggestions only — nothing in the repo has been changed. Migration commands are provided for when/if you decide to act.*

Repo reviewed: `C:\Research\PINN\Code Base` (GitHub `Alioax/pinn`).

---

## TL;DR — highest-impact changes, ranked

1. **Flatten `docs/reports/`** to four PDFs and move the LaTeX projects into a git-ignored `_src/` (your explicit ask — details in §3A).
2. **Untrack the cruft** that's sitting in the tree: `nul`, `debug-b595bc.log`, `_tmp_slide10_image.png`, `__pycache__/`, the two `heterogeneous.zip` backups.
3. **Consolidate the loose docs** (`cfl_models_methodology.md` at the root + `docs/neural_operator_architecture.md`) under `docs/`.
4. **Decide what to do with the big slide binaries** in `docs/slides/` (~25 MB) — they're the main thing bloating the repo.
5. **Fold the `code/review_ready/` stub** (it's just an index README) into `code/README.md` and delete the folder.

None of these touch your actual research code — that part is already well organized.

---

## 1. What's already good (keep it)

- **`code/` is organized by research phase, aligned to the reports** — that's a sensible scheme for a research repo and better than grouping by algorithm. Keep it.
- **Per-folder `README.md` files almost everywhere** — strong documentation habit.
- **Reports already track only the PDFs.** Your `.gitignore` untracks `.tex/.bib/figs/README` under `docs/reports/` and force-keeps PDFs, and your commit history (`chore: gitignore – keep only report PDFs…`) shows you applied it. So the "only PDF on GitHub" half of your question is effectively *already done* (see §3A to verify).
- **`code/shared/analytical_solution/`** as a single source of truth for Ogata–Banks — good.
- **`references/` is git-ignored** — correct (avoids committing copyrighted papers and bloat).

---

## 2. Current layout (annotated)

```
Code Base/                         ← local folder name has a space (minor friction)
├─ README.md                       keep (top-level overview)
├─ LICENSE, requirements.txt
├─ cfl_models_methodology.md       ← loose doc at root (move under docs/)
├─ run.bat, jobs.txt, automation/  run-and-push tooling (new)
├─ nul, debug-b595bc.log,          ← cruft, should be untracked + ignored
│  _tmp_slide10_image.png, __pycache__/
├─ code/
│  ├─ README.md
│  ├─ homogeneous_pe/   (Report 3)  baseline_pinn / parametric_pinn / pino
│  ├─ homogeneous_cfl/  (Report 4)  baseline / parametric / pino_cfl / validation
│  ├─ heterogeneous/    (Report 5)  pino_heterogeneous / utils / data   ← active
│  ├─ exploratory/                  grid search, adaptive collocation, sandboxes (8 dirs)
│  ├─ tools/                        report figure replots, gif builder
│  ├─ shared/                       analytical_solution
│  ├─ archive/                      legacy_mainline_wip / legacy_review_ready
│  ├─ review_ready/                 ← only a README index; no code (fold into code/README.md)
│  └─ heterogeneous.zip             ← backup zip (remove)
└─ docs/
   ├─ neural_operator_architecture.md  ← loose doc (move into docs/methodology/)
   ├─ diagrams/                        deeponet architecture (html/png/py)
   ├─ slides/                          5 decks, ~25 MB total (PDF + 8.5 MB pptx)
   └─ reports/
      ├─ report 1 - PINN Baseline/     each folder = a LaTeX project; only the .pdf is tracked
      ├─ report 2 - PINN Update/
      ├─ report 3 - PINO/
      └─ report 4 - CLF/
```

---

## 3. Issues & suggestions

### A. Reports: flatten to PDF-only in one directory (your question)

**"Only the PDF on GitHub" — already true.** Verify with:

```
git ls-files docs/reports/
```

If that lists only `*.pdf`, you're set. If any `.tex`/figure still appears, it was committed before the ignore rule; untrack it with `git rm -r --cached "docs/reports/<that path>"`.

**"All reports in one directory" — not yet, and worth doing.** The catch: each `report N …/` folder is a real LaTeX project (the sources that build the PDF). They're git-ignored, so they don't reach GitHub, but they *are* on your disk and you don't want to lose them. So the clean move is to **separate published PDFs from sources**:

- Tracked, flat, on GitHub: `docs/reports/report-1-pinn-baseline.pdf`, `…-2-pinn-update.pdf`, `…-3-pino.pdf`, `…-4-cfl.pdf`
- Local-only, git-ignored: `docs/reports/_src/report-1-pinn-baseline/` (the LaTeX project), etc.

Result on GitHub: `docs/reports/` shows exactly four PDFs. Locally you keep every source. Recipe in §5A.

> Caveat: with sources git-ignored, your LaTeX is **not backed up on GitHub**. If that matters, option (b) is to version the sources on a separate `latex` branch or a `reports-src/` path you do track. Tell me which you prefer.

### B. Untrack the cruft

`nul` (a stray Windows artifact — a file literally named `nul` can even be hard to delete via Explorer), `debug-b595bc.log`, `_tmp_slide10_image.png`, `__pycache__/`, and the `heterogeneous.zip` backups don't belong in version control. The new `.gitignore` already prevents *future* ones; these existing entries need an explicit `git rm --cached` (§5B).

### C. Consolidate the loose docs

`cfl_models_methodology.md` lives at the repo root while `neural_operator_architecture.md` lives in `docs/`. Same kind of artifact, two locations. Suggest a `docs/methodology/` holding both, so everything explanatory is under `docs/`.

### D. Slide decks — DECISION: remove from the repo and GitHub

You've decided to take the decks out entirely. `docs/slides/` is ~25 MB (one 8.5 MB `.pptx`, several multi-MB PDFs) of un-diffable binaries, so dropping them keeps the repo lean. Recipe in §5F. Two things to settle first:

- **Back them up outside the repo before deleting** — `git rm` removes the working copies too, and these may be your only copies. Move them somewhere safe first (e.g. your `Neural Operator Research` folder or a drive).
- **"Delete from GitHub" = current tree vs. full history.** A plain `git rm` + push removes them from the repo *going forward* (they vanish from GitHub's file view), but old commits still contain them, so repo size isn't reclaimed. Fully purging them from history needs a rewrite (`git filter-repo` / BFG) and forces anyone with a clone (e.g. your supervisors) to re-clone. See §5F.

### E. Naming consistency (low priority)

- The local folder `Code Base` has a space — fine for GitHub (repo is `pinn`), but a no-space local name (`pinn` / `pinn-transport`) is friendlier on the command line.
- A couple of archived files have spaces (`pinn_parametric_baseline_plot fig 2.py`). Harmless in `archive/`, but worth a rename if you ever revive them.
- Optional: prefix phase folders with the report number (`phase3_homogeneous_pe/`, `phase4_homogeneous_cfl/`, `phase5_heterogeneous/`) so they sort chronologically and the report mapping is obvious at a glance.

### F. Results-in-git policy

You keep `results/` beside each script, which is clean and discoverable. Since outputs are small (figures, CSVs) this is fine. Two guardrails so it stays that way: the new `.gitignore` already blocks `*.zip`/`_tmp_*`, and I left commented lines there to also block heavy weights (`*.pt`, `*.pth`, …) if a script ever starts saving them — otherwise `git add -A` could quietly commit a large checkpoint.

### G. Environment reproducibility

`requirements.txt` exists (good). Two optional upgrades, useful now that a second machine is in play: **pin versions** (`torch==…`) so the remote and your laptop match, and/or add an `environment.yml` if the remote uses conda. This avoids "works on my machine" drift between the two computers.

---

## 4. Proposed target tree

```
pinn/
├─ README.md
├─ LICENSE
├─ requirements.txt           (pinned)  +  environment.yml (optional)
├─ run.bat, jobs.txt
├─ automation/  (runner.py, README.md)
├─ runs/                      run logs (auto)
├─ code/
│  ├─ README.md               (absorbs the review_ready index table)
│  ├─ shared/
│  ├─ phase3_homogeneous_pe/      baseline_pinn / parametric_pinn / pino
│  ├─ phase4_homogeneous_cfl/     baseline / parametric / pino_cfl / validation
│  ├─ phase5_heterogeneous/       pino_heterogeneous / utils / data   ← active
│  ├─ exploratory/
│  ├─ tools/
│  └─ archive/
└─ docs/
   ├─ methodology/
   │  ├─ neural_operator_architecture.md
   │  └─ cfl_models_methodology.md
   ├─ diagrams/
   └─ reports/                (slides/ removed — see §3D/§5F)
      ├─ report-1-pinn-baseline.pdf
      ├─ report-2-pinn-update.pdf
      ├─ report-3-pino.pdf
      ├─ report-4-cfl.pdf
      └─ _src/                (git-ignored LaTeX projects)
```

---

## 5. Migration commands (when you're ready — incremental, history-preserving)

Run from the repo root. `git mv` preserves file history. Do each block as its own commit so anything is easy to undo. **Nothing here has been run for you.**

### 5A. Flatten the reports

```bash
# move the tracked PDFs to flat, clean names (history preserved)
git mv "docs/reports/report 1 - PINN Baseline/PINN Baseline - Ali Haghighi.pdf" "docs/reports/report-1-pinn-baseline.pdf"
git mv "docs/reports/report 2 - PINN Update/PINN Update - Ali Haghighi.pdf"     "docs/reports/report-2-pinn-update.pdf"
git mv "docs/reports/report 3 - PINO/PINO - Ali Haghighi.pdf"                    "docs/reports/report-3-pino.pdf"
git mv "docs/reports/report 4 - CLF/pinn_advection_diffusion_clf_pe_ali_afshin_marwan_francois.pdf" "docs/reports/report-4-cfl.pdf"

# move the LaTeX sources (NOT in git) out of the way — plain OS move, Windows:
mkdir "docs\reports\_src"
move "docs\reports\report 1 - PINN Baseline" "docs\reports\_src\report-1-pinn-baseline"
move "docs\reports\report 2 - PINN Update"   "docs\reports\_src\report-2-pinn-update"
move "docs\reports\report 3 - PINO"          "docs\reports\_src\report-3-pino"
move "docs\reports\report 4 - CLF"           "docs\reports\_src\report-4-cfl"
```

Then simplify the reports block in `.gitignore` to:

```
# Reports: track published PDFs only; keep LaTeX sources local
docs/reports/_src/
```

(remove the older `docs/reports/**/*.tex`, `…/figs/`, `…/*.bib`, `…/README.md`, and the `!…/*.pdf` lines — they're no longer needed once sources live under `_src/`.)

```bash
git add -A docs/reports .gitignore
git commit -m "docs: flatten reports to PDFs, move LaTeX sources to _src/"
git push
```

### 5B. Untrack the cruft

```bash
git rm --cached nul "debug-b595bc.log" _tmp_slide10_image.png
git rm -r --cached __pycache__ code/heterogeneous.zip code/heterogeneous/heterogeneous.zip
git commit -m "chore: stop tracking scratch/build artifacts"
git push
```

(The files stay on your disk; they just leave git. The new `.gitignore` keeps them out going forward.)

### 5C. Consolidate docs

```bash
mkdir docs\methodology
git mv cfl_models_methodology.md docs/methodology/cfl_models_methodology.md
git mv docs/neural_operator_architecture.md docs/methodology/neural_operator_architecture.md
git commit -m "docs: consolidate methodology notes under docs/methodology"
git push
```

### 5D. Remove the review_ready stub

Fold its small table into `code/README.md`, then:

```bash
git rm code/review_ready/README.md
git commit -m "docs: fold review_ready index into code/README"
git push
```

### 5E. (Optional) phase-number the code folders

```bash
git mv code/homogeneous_pe  code/phase3_homogeneous_pe
git mv code/homogeneous_cfl code/phase4_homogeneous_cfl
git mv code/heterogeneous   code/phase5_heterogeneous
# then update jobs.txt paths + code/README.md links, and commit
```

> Note 5E renames folders your `jobs.txt` and READMEs point at — update those references in the same commit.

### 5F. Remove the slide decks from the repo and GitHub

First copy `docs/slides/` somewhere safe **outside** the repo — `git rm` deletes the working copies too, and these may be your only copies. Then:

```bash
git rm -r docs/slides
git commit -m "docs: remove slide decks from repo"
git push
```

They disappear from GitHub's file view immediately. To also purge them from history (reclaims repo size, but rewrites history — coordinate with anyone holding a clone):

```bash
pip install git-filter-repo
git filter-repo --path docs/slides --invert-paths
# then force-push as git-filter-repo instructs
```

---

## 6. Decisions & open questions

Decided:
- ✅ **LaTeX sources:** keep local-only — PDFs stay tracked; `.tex`, `figs/`, and the other LaTeX build files stay git-ignored (exactly as today).
- ✅ **Slides:** remove from the repo and GitHub via **simple removal** (`git rm` + push, §5F) — no history rewrite, no re-clone for your supervisors. (Optional full purge later if you ever want the ~25 MB back.)
- ⏸️ **Applying the changes:** parked — this stays a plan until you say go.

Still open:
3. **Phase renames (§5E):** worth the churn for chronological clarity, or keep current names?
4. **Repo/local-folder rename** to drop the space in "Code Base" — do you care, or leave it?

Tell me which you want and I'll prep exact, tested steps.
