# Single-Source Defaults / Preferences / Limits

Status: **decisions settled 2026-08-12; implementation not started.**
Drafted 2026-08-10 against `dev` @ `a599f9f`. All open questions in §3 are answered.

## Design philosophy (from the maintainer)

1. Every default, preference and fit limit lives in **one user-editable file**. A user
   changes behaviour by editing that file, never by editing source.
2. **Source code carries no fallback values.** If a value is not supplied by the user
   (preference file or direct input), the software **errors and stops**. It does not
   guess.
3. `hematocrit` is included in rule 2 — no default.
4. `relaxivity` is a hard stop by construction: the correct value depends on the contrast
   agent, so no default can ever be right.
   `relaxivity` and `hematocrit` may also be set **per scan** in the image file's JSON
   sidecar, so for those two the chain is
   **image-file JSON → run config → defaults file → error**.
5. The pipeline **and the tests** read the same defaults file, unless there is a clear and
   compelling reason not to. That is what a user expects.
6. Python defaults may drift from MATLAB *if there is evidence it improves fit quality* —
   but then the MATLAB defaults get changed to match, so cross-language equivalence and
   OSIPI checks keep running on one set of numbers.

---

## 1. Current state

### 1.1 There are three independent default sources, feeding three different consumers

| Consumer | Where its defaults actually come from | Size of that surface |
|---|---|---|
| Real pipeline runs | `_stage_d_fit_prefs` + ~90 other `_stage_override(config, key, <hardcoded>)` call sites in `dce_pipeline.py` | **94** hardcoded fallbacks |
| OSIPI reliability + backend-equivalence tests | `_stage_d_fit_prefs(DcePipelineConfig(...))` built with **empty `stage_overrides`** — so the same hardcoded fallbacks, and `dce_default.json` is never read | same 94 |
| MATLAB contract tests (`model_tofts_fit`, `model_2cxm_fit`, … called with **no `prefs`**) | `dce_fit_backends._patlak/_tofts/_ex_tofts/_tissue_uptake/_2cxm_settings()` and 14 inline `settings.get(key, default)` calls in `dce_models.py` | **5 tables + 14 inline** |

> **Correction to an earlier claim in this thread.** I previously said the
> `dce_fit_backends` tables were the live source for `test_backend_equivalence.py` and the
> OSIPI tests. That is wrong: both of those go through `_stage_d_fit_prefs`, i.e. the
> *pipeline* fallbacks. The backend tables are live for the **MATLAB contract tests**,
> which call the `model_*_fit` wrappers with `prefs=None`. The conclusion still holds —
> tests and real runs resolve from different places — but the target is different.

### 1.2 `dce_default.json` is not actually the source of truth

`python/dce_default.json` declares 81 `stage_overrides` keys. Measured against what the
code reads:

- **113** distinct keys are read via `_stage_override`.
- **70** of them exist in `dce_default.json`.
- **43 do not exist in it at all** — the user cannot see or edit them. These include
  *every* model-specific fit limit:
  `voxel_lower_limit_ve_2cxm` (0.05), `voxel_upper_limit_fp_2cxm` (20.0),
  `voxel_initial_value_fp_2cxm` (0.35), `voxel_upper_limit_tp_tissue_uptake` (1.5),
  `voxel_MaxIter_2cxm` (140), `voxel_MaxIter_tissue_uptake` (120), and 37 more.
- **11** keys in the file are never read via `_stage_override`. Most are read another way
  (`hematocrit`/`relaxivity`/`tr_ms`/`fa_deg` via `_explicit_stage_override`,
  `aif_*_limits` via `_parse_4float_override`). Two are genuinely dead:
  `use_dce_preferences` (both branches of `_resolve_dce_preferences_path` return `None`)
  and `injection_duration` (appears nowhere in `dce_pipeline.py`).

It is also **two things at once**: a defaults table *and* a concrete run config for the
`sub-11tiny` fixture (hardcoded `subject_source_path`, `dynamic_files`, `output_dir`,
`dce_metadata_path`). That conflation is why tests do not use it — they cannot, without
inheriting a fixture's paths.

### 1.3 Values that have already drifted

**`relaxivity` has four different values in the tree:**

| Location | Value |
|---|---|
| `script_preferences.txt` (MATLAB) | 2.8 |
| `python/dce_default.json` | 3.6 |
| `dce_pipeline.DEFAULT_RELAXIVITY` | 3.4 |
| DCE metadata JSON sidecar | dataset-dependent, wins over all of the above |

**`hematocrit` has three:** `script_preferences.txt` 0.45, `dce_default.json` 0.42,
`dce_pipeline.DEFAULT_HEMATOCRIT` 0.45.

**Pipeline fallback vs `dce_default.json`** (i.e. tests vs real runs) — one difference:

| key | source fallback (tests) | `dce_default.json` (real runs) |
|---|---|---|
| `max_nfev` | 50 | 200 |

**Pipeline-resolved prefs vs `dce_fit_backends` tables** (i.e. real runs vs contract tests):

| model | key | backend table | pipeline |
|---|---|---|---|
| patlak / tofts / ex_tofts | `max_nfev` | 2000 | 200 |
| tissue_uptake | `initial_value_fp` | 0.2 | 0.35 |
| tissue_uptake | `initial_value_tp` | 0.05 | 0.12 |
| tissue_uptake | `lower_limit_vp` | 0.0 | 0.001 |
| tissue_uptake | `upper_limit_fp` | 100.0 | 20.0 |
| tissue_uptake | `upper_limit_tp` | 1e6 | 1.5 |
| tissue_uptake | `max_nfev` | 2000 | 120 |
| 2cxm | `initial_value_fp` | 0.2 | 0.35 |
| 2cxm | `initial_value_ve` | 0.2 | 0.15 |
| 2cxm | `lower_limit_ve` | 0.02 | 0.05 |
| 2cxm | `upper_limit_fp` | **2.0** | **20.0** |
| 2cxm | `max_nfev` | 4000 | 140 |

`multistart_starts` / `multistart_seed` exist **only** in the backend tables — the pipeline
never sets them, so they are unreachable from any user-facing file.

**Python vs MATLAB `dce_preferences.txt`:** `gpu_tolerance` 1e-6 (Python) vs 1e-12 (MATLAB);
`voxel_MaxFunEvals` 200 (Python) vs 50 (MATLAB). MATLAB has no entry at all for any
model-specific 2cxm/tissue_uptake limit, for multistart, or for QoF.

### 1.4 Tunables that are not in any preference file

Module-level constants in `dce_pipeline.py` that are behavioural knobs, not physics:
`DEFAULT_STEADY_STATE_AUTO_METHOD` ("tv"), `PIECEWISE_CONSTANT_BASELINE_FORWARD_DELTA_FRACTION`
(0.01), `TV_JUMP_THRESHOLD_SIGMA` (5.0), `AIF_PEAK_WEIGHT_FLOOR` (1e-3),
`AIF_PEAK_WEIGHT_EXPONENT` (2.0), `TUKEY_TUNING_CONSTANT` (4.685), `TUKEY_MAX_ITERATIONS`
(50), `TUKEY_WEIGHT_TOLERANCE` (1e-6), plus `DEFAULT_RELAXIVITY` / `DEFAULT_HEMATOCRIT`.

### 1.5 Current failure mode is silent, not loud

Nothing errors when a value is missing. `relaxivity` silently becomes 3.4, `hematocrit`
0.45, `noise_pixsize` 5, every fit bound its hardcoded value. A user who mis-spells a key
in their config gets a successful run with different numbers and no warning.

---

## 2. Target design

```
python/dce_defaults.json      <- THE user-editable file. Every knob. No dataset paths.
                                 Ships with the repo. Documented in docs/dce_options.md.
<run config>.json             <- subject paths, model flags, and only the keys this run
                                 overrides. dce_default.json becomes an example of this.
```

**Resolution rule for everything except `relaxivity` / `hematocrit`:**

```
value = run_config.stage_overrides[key]   if present
      else defaults_file[key]             if present
      else  -> raise DceConfigError(key)   # hard stop, names the key and the file
```

**Resolution rule for `relaxivity` and `hematocrit`** (may legitimately vary per scan):

```
value = image_file_json_sidecar[key]      if present     <- WINS
      else run_config.stage_overrides[key] if present
      else defaults_file[key]              if present
      else  -> raise DceConfigError(key)
```

No further tier. **No literal default anywhere in `python/`.**

Consequences that fall out of the rules:

- The sidecar winning over the run config is an **inversion of today's order**
  (`dce_pipeline.py:1860` currently lets the run config beat the metadata JSON). Intended:
  the sidecar is per-scan ground truth, the run config is a batch-level convenience, and
  the `dce2bids` tool already instructs users to write these into the sidecar.
- `hematocrit` **is** shipped in `dce_defaults.json` — it is usually constant across a
  study, so a study-wide value is the sane default.
- `relaxivity` is **deliberately absent** from the shipped file, so a user who has not put
  it in a sidecar or run config gets the hard stop. This is intentional pressure toward
  putting it in the sidecar, which is what `dce2bids` tells users to do.
- The 43 invisible model-specific limits become visible and editable.
- Tests get their prefs from the same file, so "what the tests exercise" and "what a user
  gets" are the same numbers by construction.

---

## 3. Decisions (settled 2026-08-12)

**D1 — File split. → (a)** New `python/dce_defaults.json` holds every knob, no dataset
paths. `dce_default.json` / `dceprep_default.json` are demoted to example *run configs*
holding only paths, model flags and genuine per-run overrides.

**D2 — Value conflicts.**

*2CXM and tissue_uptake: the pipeline values win.* Confirmed by blame: `abf9ace`
(2026-02-17, "Improve DCE parity tuning and refresh MATLAB baselines") introduced the whole
model-specific block and refreshed the MATLAB baselines in the same commit, so those numbers
were tuned and validated. The competing `dce_fit_backends` table is *newer* (`8ddec76`,
2026-07-22, a unit-scaling fix) but was written to a different unit convention —
`dce_models.py:507` spells its fp defaults `200.0 / 100.0` and `20.0 / 100.0`, i.e. treating
the user-facing unit as mL/100 mL/min. Both land in per-minute for a minutes timer, so the
10x gap was a real value disagreement, not a units artifact. **Action: annotate the unit of
every rate key in `dce_defaults.json` so this cannot recur.**

| key | decision |
|---|---|
| all 2CXM keys (`upper_limit_fp` 20.0, `initial_value_fp` 0.35, `initial_value_ve` 0.15, `lower_limit_ve` 0.05, `max_nfev` 140) | pipeline values |
| all tissue_uptake keys (`initial_value_fp` 0.35, `initial_value_tp` 0.12, `upper_limit_tp` 1.5, `lower_limit_vp` 0.001, `upper_limit_fp` 20.0, `max_nfev` 120) | pipeline values |
| `max_nfev` patlak / tofts / ex_tofts | **200** |
| `voxel_MaxFunEvals` | **200** |
| `gpu_tolerance` | **1e-6** (measured, see below) |
| `voxel_TolFun` / `voxel_TolX` | unchanged (1e-12 / 1e-6) |
| `relaxivity` | no default — hard stop |
| `hematocrit` | shipped in `dce_defaults.json` |

*`gpu_tolerance` measurement (2026-08-12).* cpufit, 2000 real voxels x 62 frames from
`tests/data/stage_b_frozen`, swept 1e-4 -> 1e-12 against the pure-python scipy fit:

| tol | tofts conv% | tofts Ktrans vs scipy | 2cxm conv% | 2cxm Ktrans vs scipy | 2cxm time |
|---|---|---|---|---|---|
| 1e-4 | 100.0% | 8.4e-2 | 100.0% | 9.4e-2 | 0.93 s |
| **1e-6** | **100.0%** | **1.5e-2** | **100.0%** | **2.7e-2** | **2.18 s** |
| 1e-8 | 98.2% | 1.8e-3 | 99.9% | 7.4e-3 | 4.85 s |
| 1e-10 | 93.5% | 3.4e-4 | 95.9% | 5.7e-4 | 9.60 s |
| 1e-12 | 92.2% | 3.1e-4 | 95.5% | 4.4e-4 | 9.87 s |

Convergence rate *falls* as the tolerance tightens, and non-converged voxels are NaN'd —
1e-10 punches holes in 6.5% of the tofts map and 4.1% of the 2cxm map. 1e-6 is the only
setting with 100% voxel yield on every model; the price is a ~1.5% median Ktrans offset
from the scipy reference, which is preferable to missing voxels. **MATLAB's current 1e-12
is the worst of both**: lowest convergence, and bit-identical parameters to 1e-10. It
changes to 1e-6 in the D3 values-only commit. (If the backend-equivalence gate ever needs
tightening, 1e-8 buys ~8x better agreement for ~2% voxel yield.)

**D3 — MATLAB reconciliation. → (a)** Python is fixed here; MATLAB gets a follow-on
**values-only** commit (`dce/dce_preferences.txt`, `script_preferences.txt`), no
restructuring. Python-only knobs (model-specific 2CXM/tissue_uptake limits, `multistart_*`,
`qof_chi2_max`) stay Python-only rather than being back-ported.

**D4 — `dce_preferences.txt` bridge in Python. → (b) delete entirely.**
`_load_dce_preferences`, `_parse_preference_file`, `_resolve_dce_preferences_path` and the
`dce_preferences_path` / `use_dce_preferences` keys all go.

**D5 — Source constants. → agreed split.**
*Move into the file:* `DEFAULT_STEADY_STATE_AUTO_METHOD` ("tv"), `TV_JUMP_THRESHOLD_SIGMA`
(5.0), `PIECEWISE_CONSTANT_BASELINE_FORWARD_DELTA_FRACTION` (0.01), `AIF_PEAK_WEIGHT_FLOOR`
(1e-3), `AIF_PEAK_WEIGHT_EXPONENT` (2.0 — note it is already a preference key too, so the
duplicate goes).
*Stay in source:* `TUKEY_TUNING_CONSTANT` (4.685), `TUKEY_MAX_ITERATIONS` (50),
`TUKEY_WEIGHT_TOLERANCE` (1e-6) — these define the Tukey biweight estimator.

**D6 — `multistart_starts` / `multistart_seed`. → keep as internal algorithm constants**
in source. They are not user preferences.

**D7 — Parametric T1. → follow-up project**, not this one. Logged in `TODO.md`.

**D8 — Dead keys. → delete** `injection_duration` and `use_dce_preferences` from
`dce_default.json`, `dceprep_default.json`, and `docs/dce_options.md`.

**D9 — MATLAB `relaxivity`. → accept and document the divergence.** Python has no
relaxivity default and hard-stops without one; MATLAB keeps `script_preferences.txt`'s
`relaxivity = 2.8` fallback. This is a deliberate, documented structural difference, not
drift: Python is the path we are steering users toward via `dce2bids` sidecars. Phase 5
records it in `docs/dce_options.md` and `AGENTS.md` rather than trying to reconcile it.
Consequence to state plainly in the docs: **a MATLAB run and a Python run on data with no
sidecar relaxivity will not agree** — MATLAB silently uses 2.8, Python refuses to run.

---

## 4. Phased plan

Each phase ends green on: unit suite + coverage gate, `-m parity`, `-m osipi`, contracts
`--require-all`. Phases 2-4 additionally byte-compare full Stage A→B→D output against the
previous commit, the way the `dce_pipeline` cleanup was verified.

### Phase 0 — Freeze the current numbers (no behaviour change)
Emit a machine-readable dump of every currently-resolved value for every consumer
(pipeline-with-json, pipeline-bare, each backend table, each `dce_models` inline default).
Commit it as `tests/data/defaults_snapshot_pre.json`. This is the oracle: after each later
phase, any value that changes must be one we decided to change in D2.

### Phase 1 — Build the file and the resolver
- Write `python/dce_defaults.json` containing all 113 read keys + the 43 currently-invisible
  ones + the D5 constants, populated with the **D2-agreed** values. Every rate key carries
  an explicit unit annotation (per D2). `hematocrit` included; `relaxivity` deliberately
  absent, with a comment block explaining that it belongs in the image sidecar.
- Add `dce_config.py`: loads the defaults file once, exposes `require(config, key)` and the
  `relaxivity`/`hematocrit` sidecar-first resolver; raises `DceConfigError` naming the
  missing key and the file to edit.
- Add a test asserting **every** key the code reads exists in the file (an AST scan, so it
  cannot rot).
- No call sites changed yet; nothing breaks.

### Phase 2 — Cut over `dce_pipeline.py`
Replace all 94 hardcoded `_stage_override(config, key, <literal>)` with the resolver.
Delete `DEFAULT_RELAXIVITY` / `DEFAULT_HEMATOCRIT` and the invented fallbacks so a missing
`relaxivity` is a hard stop, and **invert the sidecar/run-config precedence** for
`relaxivity` and `hematocrit` per §2. Delete the whole `dce_preferences.txt` bridge (D4):
`_load_dce_preferences`, `_parse_preference_file`, `_resolve_dce_preferences_path`,
`PREFERENCE_NUMERIC_CHARS`, and the `dce_preferences_path` / `use_dce_preferences` keys.
Delete `injection_duration` (D8).
*This is where things break loudly* — every test and script that builds a bare
`DcePipelineConfig` starts erroring. That breakage is the audit; fix each by pointing it at
the defaults file.

### Phase 3 — Cut over `dce_fit_backends.py` and `dce_models.py`
Delete the 5 `_*_settings()` default tables and the 14 inline `settings.get(key, default)`
calls; the settings dict becomes a required, complete input. Update `model_*_fit` wrappers:
`prefs` becomes required. Update `tests/contracts/generate_python_results.py` to load
`dce_defaults.json` and pass it — so the MATLAB contract tests run on the shipped defaults,
which is the §1.1 fix.

### Phase 4 — Point the tests at the file — DONE
`osipi_fast_backend_helpers.py`, `test_backend_equivalence.py`, `phantom_gt_helpers.py`,
`run_dce_benchmark.py`, the two BIDS batch runners, and `dce_gui.py`/`dce_cli.py` all
resolve through the same loader. Add a test that no `python/*.py` file contains a literal
default in a config-resolution call (AST check, so it stays true).

**Outcome.** The file-pointing half was already done by phases 2-3: the OSIPI and
equivalence helpers reach settings through `_stage_d_fit_prefs`/`_apply_model_specific_prefs`,
which are now thin delegates to `dce_config`. The remaining harnesses set only run-specific
choices (`stage_a_mode`, `aif_curve_mode`, `write_param_maps`) — that is what a run config is
for — and none of them pin fit settings or relaxivity, so they resolve from the file and the
image sidecars like any other run.

The AST guard (`test_no_literal_defaults_in_resolution_calls`, parametrised over
`python/*.py` + `run_*.py`) found exactly one real literal: a `1.0` min/frame fallback in
`_resolve_stage_b_timer`'s short-timer extension, present in two copies. It was latent rather
than live — `_as_1d_float` squeezes, so a <2-sample timer raises before reaching that branch —
but the surrounding ladder existed in three near-identical copies, two with indentation
mangled by the phase-2 cutover script. Collapsed into `_configured_time_step_min`, which
returns `None` rather than guessing; the callers raise. That path had no test coverage at all,
which is how the literal survived; `TestResolveStageBTimer` now covers all nine branches.

The guard is scoped to shipped code, not `tests/python` (clean today, checked). A literal in a
test only affects that test, whereas one in `python/` silently changes user results.

### Phase 5 — MATLAB values + docs
Apply the D2/D3-agreed values to `dce/dce_preferences.txt` (`gpu_tolerance` 1e-12 → 1e-6,
`voxel_MaxFunEvals` 50 → 200) and `script_preferences.txt` (`relaxivity` 2.8 and
`hematocrit` 0.45 — reconcile or document as MATLAB-side study values). Regenerate the
MATLAB contract baseline if any contract-relevant number moved. Rewrite `docs/dce_options.md`
as the reference for the single file, and update `AGENTS.md`'s "Config resolution" section,
which documents a three-tier chain and a `dce_preferences.txt` bridge that both cease to
exist.

---

## 5. Risks

- **Phase 2/3 change fit results wherever D2 picks a new number.** MATLAB parity baselines
  and OSIPI tolerances may need regeneration. Phase 0's snapshot makes every such change
  explicit and reviewable rather than discovered later.
- **Hard-stop on `relaxivity` breaks existing user configs and every fixture that relied on
  the 3.4/3.6 fallback.** Every in-repo fixture and test config needs an explicit value
  added. This is intended, but it is the largest single source of churn.
- **`upper_limit_fp` for 2cxm (2.0 vs 20.0) is the one drift with real scientific
  consequence.** Do not pick it by tidiness.
- The AST "no literal defaults" test in Phase 4 is what stops this from re-growing. Without
  it, the next feature adds a `_stage_override(config, "new_key", 0.5)` and we are back here.

---

## 6. Estimated size

Phase 0-1 small and safe. Phase 2 is the bulk (94 call sites + the resulting test fallout).
Phase 3 medium. Phase 4 medium, mostly mechanical. Phase 5 small but needs MATLAB to
regenerate baselines. Recommend landing 0-1 first and reviewing the generated
`dce_defaults.json` before any cutover — that file *is* the decision record.
