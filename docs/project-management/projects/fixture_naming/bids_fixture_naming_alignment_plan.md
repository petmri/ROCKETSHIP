# BIDS test-fixture naming alignment — working plan

**Status:** draft, not committed. Scratch planning doc for the fixture rename that has been
outstanding since the benchmark tool update.

**Verified against the tree on 2026-08-16.** Every count, path, and pattern below was checked
against the working copy, not recalled.

---

## 1. The problem in one paragraph

Three different naming conventions coexist in this repo, and only one of them is implemented
by the shipped pipeline. Seven of the nine `BIDS_test` subjects follow the pipeline
convention; the two fixtures that carry the most test weight (`sub-10bbbdownsample`,
`sub-11tiny`) do not. The benchmark tool follows a third convention and therefore resolves
zero input files against the shipped dataset. Every consumer that needs to find files has
grown its own private copy of the glob patterns, so fixing one does not fix the others.

## 2. Current state (verified)

### 2.1 Which subjects auto-discovery can actually resolve

Run through `discover_dce_input_paths()` — the function the GUI and `run_dce_bids_batch.py`
both use:

| Subject | Result |
|---|---|
| `sub-01original` | CONFORMS |
| `sub-02downsample` | CONFORMS |
| `sub-05phantom` … `sub-09phantom` (5 subjects) | CONFORMS |
| `sub-10bbbdownsample` | MISSING: dynamic, aif_mask, roi_mask |
| `sub-11tiny` | MISSING: dynamic, aif_mask, roi_mask, t1_map |

So the blast radius is **two subjects**, not the whole dataset. That is the single most
important fact for scoping this work.

### 2.2 The three conventions

| Input | Pipeline convention (canon) | `sub-10` / `sub-11` actual | Benchmark tool |
|---|---|---|---|
| dynamic | `derivatives/…/dce/*desc-bfcz_DCE.nii*` | `rawdata/…/dce/*_DCE.nii` | `*desc-bfcz_DCE.nii*` ✓ |
| AIF | `dce/*label-AIF_T1map.nii*` | `dce/*label-AIF_mask.nii` | `dce/*desc-AIF_T1map.nii*` |
| ROI | `anat/*space-DCEref_label-brain_mask.nii*` | `anat/*label-brain_mask.nii` | `anat/*space-DCEref_desc-brain_mask.nii*` |
| T1 map | `anat/*space-DCEref_T1map.nii*` | `sub-11`: in `dce/` not `anat/` | `anat/*space-DCEref_T1map.nii*` ✓ |
| noise | `anat/*label-noise_mask.nii*` | matches ✓ | not used |

Three independent classes of divergence, which want three different fixes:

1. **Entity naming** — `label-AIF_mask` vs `label-AIF_T1map`; missing `space-DCEref_` on the
   brain mask. Pure rename.
2. **Location** — `sub-10`/`sub-11` dynamics live in `rawdata/`, and `sub-11`'s T1 map is in
   `dce/` instead of `anat/`. Needs a decision, not just a rename (see §3.2).
3. **`desc-` vs `label-`** — the benchmark tool and `freeze_stage_b_backend_fixture.py` say
   `desc-`, everything else says `label-`. Pure tool bug.

### 2.3 Where the canon is written down

`label-` is the de-facto canon — it is what the shipped pipeline reads:

- `script_preferences.txt:25-27` (MATLAB side)
- `python/dceprep_run_example.json` (glob file lists)
- `python/dce_file_discovery.py:40-44`
- `run_parametric_bids_batch.py:155`
- `tests/matlab/generate_synthetic_datasets.m:51,54,146`
- 7 of 9 shipped subjects

`desc-` appears in exactly two places, both of which are broken against shipped data:

- `tests/python/run_dce_benchmark.py:67-68`
- `tests/python/freeze_stage_b_backend_fixture.py:65-66`

### 2.4 How the benchmark is broken

`run_dce_benchmark.py` fails on `tests/data/BIDS_test` in three independent ways. All three
must be fixed or it still won't run:

1. `DEFAULT_SUBJECT = "sub-203103"` (line 47) — does not exist in `BIDS_test`. That is a
   subject id from a private dataset.
2. `DEFAULT_RAW_SUBDIR = "sourcedata/raw"` (line 49) — `BIDS_test` uses `rawdata/`; there is
   no `sourcedata/` directory at all. `_resolve_subject_paths` raises before it gets to
   discovery.
3. `INPUT_PATTERNS` (lines 65-70) use `desc-AIF_T1map` and `space-DCEref_desc-brain_mask`,
   which match **0 files** in `BIDS_test` (the `label-` equivalents match 7).

`DEFAULT_DATASET` already points at `BIDS_test` (line 46), so the defaults are internally
inconsistent: the dataset default is the shipped fixture, the subject/layout/pattern defaults
are for someone's private tree.

### 2.5 Duplicate discovery logic

Four near-copies of the same glob logic exist. Any convention change has to touch all four
unless they are consolidated first:

| Location | Notes |
|---|---|
| `python/dce_file_discovery.py` | the canon; GUI + `run_dce_bids_batch.py` |
| `tests/python/phantom_gt_helpers.py:187-190` | near-identical; `*label-brain_mask.nii*` without `space-DCEref` |
| `tests/python/run_dce_benchmark.py:65-70` | `desc-` variant, broken |
| `run_parametric_bids_batch.py:120,155` | parametric side, only needs the brain mask |

`phantom_gt_helpers.py` can now simply call `discover_dce_input_paths()` — that function was
just added and returns partial results, which is what the helper's own missing-input check
wants anyway.

### 2.6 Two batch-runner bugs this work should fix

Found while surveying, both caused by the strict discovery contract:

1. **`--config-template` cannot rescue a non-conforming dataset.**
   `_build_session_config` calls `discover_dce_inputs(session)` at
   `run_dce_bids_batch.py:179`, *before* merging the template. That call raises on any
   missing required input, so a session whose files the convention cannot find is rejected
   even when the template names every file explicitly. The documented precedence ("Config
   template file paths take precedence over auto-discovery") is therefore not what happens.
2. **`--skip-validation` does not skip validation.** It guards the outer
   `discover_dce_inputs` call at line 481 only; the one inside `_build_session_config` runs
   regardless, so the flag's stated purpose is defeated.

Both are one-line fixes once `discover_dce_input_paths` is used instead — it returns `None`
per missing kind rather than raising. Worth doing in Phase 1, since a dataset that fails the
convention is exactly the case a template is for.

## 3. Decisions needed before any renaming

### 3.1 Target convention — recommend `label-`

Adopt the pipeline convention (`label-AIF_T1map`, `space-DCEref_label-brain_mask`) and fix the
two `desc-` consumers. Rationale: it is already implemented in the pipeline, both batch
runners, the MATLAB prefs, and 7/9 fixtures. Choosing `desc-` would mean rewriting
`script_preferences.txt`, both default JSONs, the discovery module, the MATLAB generator, and
renaming 7 subjects' files, to satisfy 2 broken tools.

**This is already settled, and the two subjects were knowingly skipped.** Commit `aafbd95`
("Consolidate mask/AIF fixture names onto the label- BIDS convention", 2026-07-28) renamed 26
files from `desc-` to `label-` precisely because `script_preferences.txt` had been pointed at
`label-AIF_T1map` / `space-DCEref_label-brain_mask` by `6ea2322` and the globs matched
nothing. Its message says outright that `brain_mask` "covers both filename patterns — the 8
`space-DCEref_` files the glob targets and the 2 plain ones (sub-10bbbdownsample, sub-11tiny)
referenced by explicit path in the parity tests and CI workflow." So the two plain ones were
left deliberately, because nothing globbed them. That is the debt this plan pays off.

Remaining nuance: `label-AIF_T1map` pairs the `label-` entity with a `_T1map` suffix for what
is conceptually a mask. `aafbd95` explains the reasoning (the AIF mask and AIF T1 map describe
the same region and differ only in BIDS suffix), so this is intentional, not an accident.

### 3.2 The dynamic-in-rawdata question

This is the part that is not a rename. `sub-10bbbdownsample` and `sub-11tiny` keep their
4-D dynamic in `rawdata/…/dce/*_DCE.nii`, while the convention expects a bias-corrected
`desc-bfcz_DCE` in derivatives. Options:

- **(a) Copy into derivatives as `desc-bfcz_DCE`.** Makes auto-discovery work. But these
  volumes have not been bias-corrected, so the name would assert a processing step that never
  happened. Rejected unless the fixtures are regenerated through the real dceprep path.
- **(b) Copy into derivatives under an honest name** and add an explicit fallback to the
  discovery patterns (e.g. accept `*_DCE.nii*` in `derivatives/dce` — already the
  `DYNAMIC_FALLBACK_PATTERN`, it just never fires because the file is in `rawdata`).
  Cheapest correct fix; costs one duplicated file per fixture (`sub-11tiny`'s dynamic is
  tiny; check `sub-10`'s size before committing to a copy).
- **(c) Extend discovery to fall back to `rawdata/dce`.** Rejected: it would let a real run
  silently substitute uncorrected data for corrected data. The failure mode is invisible and
  the consequences are wrong Ktrans, not a crash.
- **(d) Declare `sub-10`/`sub-11` deliberately non-conforming** and exempt them, keeping
  their explicit file lists in test code. Zero churn, but leaves the GUI's auto-find broken
  on the shipped default config, which is what started this.

Recommend **(b)**. It keeps discovery honest, fixes the GUI default, and needs no fixture
regeneration.

### 3.3 GM/WM masks

`sub-10bbbdownsample` also has `label-GM_mask` and `label-WM_mask` (used by the parity
region metrics, resolved explicitly, not globbed). If the brain mask gains `space-DCEref_`,
these should too, for consistency — but nothing globs them, so it is cosmetic. Decide once;
don't leave the set half-renamed.

## 4. Work breakdown

### Phase 0 — decide (blocking)
Resolve §3.1 and §3.2. Everything downstream depends on the answers. Do not start Phase 1
before this.

### Phase 1 — consolidate discovery first
Before renaming any file, collapse the duplicate globs so the convention lives in one place:

- Point `tests/python/phantom_gt_helpers.py` at `discover_dce_input_paths()`.
- Fix `run_dce_benchmark.py`'s `INPUT_PATTERNS` to import from `dce_file_discovery` rather
  than redeclaring (it needs a `dynamic/aif/roi/t1map` key mapping — trivial adapter).
- Decide whether `freeze_stage_b_backend_fixture.py` is still live; if yes, same treatment.

Doing this first means Phase 2 changes one module, not four.

### Phase 2 — rename fixture files

`sub-11tiny` (derivatives):

```
dce/sub-11tiny_ses-01_label-AIF_mask.nii   -> dce/sub-11tiny_ses-01_label-AIF_T1map.nii
dce/sub-11tiny_ses-01_label-AIF_mask.json  -> dce/sub-11tiny_ses-01_label-AIF_T1map.json
anat/sub-11tiny_ses-01_label-brain_mask.nii -> anat/sub-11tiny_ses-01_space-DCEref_label-brain_mask.nii
dce/sub-11tiny_ses-01_space-DCEref_T1map.nii -> anat/  (move, per §2.2 location fix)
+ dynamic per the §3.2 decision
```

`sub-10bbbdownsample` (derivatives):

```
dce/sub-10bbbdownsample_ses-01_label-AIF_mask.nii -> dce/sub-10bbbdownsample_ses-01_label-AIF_T1map.nii
anat/sub-10bbbdownsample_ses-01_label-brain_mask.nii -> anat/sub-10bbbdownsample_ses-01_space-DCEref_label-brain_mask.nii
anat/…_label-GM_mask.nii, …_label-WM_mask.nii -> per §3.3
+ dynamic per the §3.2 decision
```

`space-DCEref_T1map.nii` and `label-noise_mask.nii` already conform in both subjects — leave
them alone.

Use `git mv` so history follows. NIfTI content is untouched; no regeneration needed, so
parity references stay valid (see §5).

### Phase 3 — update the generators that recreate these fixtures
Otherwise the next regeneration silently reverts Phase 2:

- `tests/data/scripts/generate_tiny_dce_settings_fixture.py` (4 refs, `SUBJECT` at line 20)
- `tests/data/scripts/generate_bbb_p19_downsample.py` (4 refs, `SUBJECT` at line 22,
  destination map at line 65)

### Phase 4 — update consumers

Good news: the two heaviest test files each resolve paths in a single helper, so this is a
handful of edits, not a sweep.

| File | Refs | What to change |
|---|---|---|
| `tests/python/dce_run_tiny.json` | 6 | the tiny smoke config — all five file lists + `dce_metadata_path` |
| `tests/python/test_dce_pipeline_parity_metrics.py` | 5 | `_dataset_paths()` (lines 53-76) only |
| `tests/python/test_dce_pipeline_settings_matrix.py` | 2 | `_tiny_paths()` (lines 34-48) only |
| `tests/python/test_find_end_ss_tv_matlab_parity.py` | 1 | inline path at line 117 |
| `tests/python/run_dce_benchmark.py` | — | §2.4 items 1-3 |
| `tests/python/parity_thresholds_default.json` | 1 | comment text only |
| `.github/workflows/run_DCE.yml` | 21 | mostly `matlabref` output paths — verify which are affected |
| `tests/contracts/check_matlabref_map_drift.py` | 9 | `matlabref` tree; likely unaffected, confirm |

`derivatives/matlabref/` keeps its `Dyn-1_*` MATLAB-native names and is **out of scope** —
it is reference output, not pipeline input, and is addressed by name in the parity tests.

### Phase 5 — verify

```bash
.venv/bin/python -m pytest tests/python -q
```

Then explicitly re-check the things that motivated this:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'python'); from pathlib import Path; from bids_discovery import BidsSession; from dce_file_discovery import discover_dce_input_paths, missing_required_inputs; root=Path('tests/data/BIDS_test'); [print(s, missing_required_inputs(discover_dce_input_paths(BidsSession(bids_root=root, subject=s, session='ses-01', rawdata_path=root/'rawdata'/s/'ses-01', derivatives_path=root/'derivatives'/s/'ses-01')))) for s in sorted(p.name for p in (root/'rawdata').glob('sub-*'))]"
```

All nine subjects should report `[]`. Then:

- `run_dce_bids_batch.py --bids-root tests/data/BIDS_test` discovers all 9 sessions.
- `run_dce_benchmark.py` runs to completion on the shipped data with no arguments.
- GUI: load default config, tick **Auto find BIDS files** → "All required inputs found".
- MATLAB parity suite still passes (content unchanged, so this is a path-plumbing check).

## 5. Risks

- **Low numeric risk.** This is renaming and relocating files, not regenerating them. No
  NIfTI content changes, so parity baselines and `phantom_gt_mae_tolerances.json` stay valid.
  If any parity number moves, something was wired to the wrong file — investigate, don't
  re-baseline.
- **Silent generator revert** if Phase 3 is skipped.
- **`.DS_Store` noise** — several fixture directories contain them; don't let them ride along
  in the rename commit.
- **CI MATLAB job** (`run_DCE.yml`, 21 refs) is the least visible consumer and cannot be run
  locally. Read it carefully rather than relying on the local suite going green.

## 6. Suggested order

1. §3 decisions (blocking, needs the real dceprep naming as ground truth)
2. Phase 1 consolidation — mergeable on its own, no fixture changes
3. Phases 2-4 in one commit — they must land together or the suite is red in between
4. Phase 5 verification
