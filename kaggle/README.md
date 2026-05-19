# Kaggle Submission Guide — ARC Prize 2026

> Steps to submit V4+propose to `arc-prize-2026-arc-agi-3` via the Kaggle Code Competition flow.
> Submission command target:
> ```
> kaggle competitions submit -c arc-prize-2026-arc-agi-3 \
>   -f submission.parquet -k xinxiang000/<NOTEBOOK> -v <VERSION> \
>   -m "Message"
> ```

## What's here

| File | What it is |
|---|---|
| `submission.py` | Notebook content (paste into a Kaggle notebook cell, or convert via `jupytext`) |
| `prepare_datasets.py` | Helper to package the two Kaggle datasets the notebook depends on |
| `README.md` | This file — exact submission flow |

## One-time setup

### 1. Kaggle CLI auth (your machine, not Kaggle)

```powershell
pip install kaggle
# Then: Kaggle.com -> Account -> "Create New API Token" -> download kaggle.json
# Move to %USERPROFILE%\.kaggle\kaggle.json
```

Verify:
```powershell
kaggle competitions list | Select-String arc-prize-2026
```

### 2. Join the competition (browser)

https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3 → "Join Competition" / "Rules" → accept.

### 3. Prepare the two Kaggle datasets

**(a) Code dataset** — our `arc_agent/` library:

```powershell
.venv\Scripts\python.exe kaggle\prepare_datasets.py code
cd kaggle\_pack\arcagi3-code
kaggle datasets create -p .         # first time only
# subsequent updates:
kaggle datasets version -p . -m "update to commit <hash>"
```

**(b) Model dataset** — SmolLM3-3B (~6 GB):

```powershell
.venv\Scripts\python.exe kaggle\prepare_datasets.py model
cd kaggle\_pack\smollm3-3b
kaggle datasets create -p . --dir-mode tar
```

> ⚠ The model dataset is large. Use `--dir-mode tar` so Kaggle stores it as a single tar. Upload may take 30-60 min on a typical connection.

## Per-submission flow

### 4. Create / update the Kaggle notebook

On https://www.kaggle.com/code → "New Notebook"
- Settings → "Internet" = **off** (competition rule)
- Settings → "Accelerator" = **GPU T4 ×2** (or T4×1)
- Add data:
  - Competition: `arc-prize-2026-arc-agi-3`
  - Dataset: `xinxiang000/arcagi3-code`
  - Dataset: `xinxiang000/smollm3-3b-4bit`
- Paste contents of `kaggle/submission.py` into a code cell
  (or use https://jupytext.readthedocs.io to convert to `.ipynb` first)
- "Save Version" → choose "Save & Run All (Commit)" → note the version number (e.g., 3)

### 5. Submit

```powershell
kaggle competitions submit -c arc-prize-2026-arc-agi-3 `
    -f submission.parquet `
    -k xinxiang000/<your-notebook-slug> `
    -v <version-number> `
    -m "v4+propose first submission"
```

The slug is from your notebook URL `kaggle.com/code/xinxiang000/<slug>`.
The version is from step 4's "Save Version" output.

### 6. Check status

```powershell
kaggle competitions submissions -c arc-prize-2026-arc-agi-3 | Select-Object -First 10
```

Status will go: `running` → `completed` (after ~10h) → `scored`.

## TODO before first real submission

The current `submission.py` has 3 unknowns that **must** be filled in
from the official Kaggle starter notebook (which we have not yet seen):

1. **Environment factory** (`make_env(game_id)` in submission.py line ~70):
   What's the offline equivalent of `arc.make(game_id)` Kaggle exposes?
2. **Game listing** (`list_game_ids()` in submission.py line ~95):
   Where does the competition data put the list of test games?
3. **Submission schema** (`df.to_parquet(...)` in submission.py line ~140):
   What columns does `sample_submission.parquet` have?

Each of these is one line you discover by opening Kaggle's starter
notebook in step 4 and reading the first 30 lines. Then come back here,
patch `submission.py`, commit a new version.

## Caveats & expected score

- **Current performance**: V4+propose achieves 82% change_rate but 0/5
  wins on G_base demo. Expected Kaggle leaderboard score with current
  bugs unfixed: 0% (no levels completed).
- **Pre-fix submission value**: tests the pipeline (do the datasets
  load, does the notebook run within 10h on T4, does the parquet
  schema match). Score 0 is acceptable for this dry-run.
- **Post-fix submission**: after BUG-1 (strict hypothesis schema) and
  BUG-2 (deterministic letter mapping) are fixed locally and re-verified
  on G_base, push a new dataset version and re-submit. This is when we
  actually expect non-zero score.

## Reference

- Competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3
- ARC Prize 2026 overview: https://arcprize.org/competitions/2026
- Project goal + module decomposition: `presentation/report.md`
- Architecture history: `presentation/arc.md`
- Bug list: `docs/verify.md` §2
