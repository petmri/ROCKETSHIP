# Stage-B AIF Fitting: MATLAB vs Python Algorithm Differences

_Batch-parity workstream. Created 2026-07-24._

## Why this document exists

`test_bbb_p19_roi_xls_parity` has failed for `tofts` and `tissue_uptake` across several rounds of
investigation, each time looking like a *model* problem (non-identifiability, optimizer local
minima, stale baselines). It is not. Once the two pipelines are run with genuinely identical
settings, **the entire remaining ROI-xls gap is Stage-B's fitted AIF** — the two languages fit
the same arterial input curve with structurally different algorithms and get materially different
answers at the frame that matters most.

This kept hiding because two unrelated configuration mismatches were masking it (see
[`batch_parity.md`](batch_parity.md) issue #2 history):

1. The MATLAB baseline generator pinned `startInjectionMin = 0.5 / endInjectionMin = 0.7` while the
   Python fixture auto-detected them — so the sides were never comparing the same Stage-B input.
2. Both pipelines shared an off-by-one converting `end_ss` (a 1-based **frame number**) to minutes
   (`end_ss * dt` instead of `(end_ss - 1) * dt`). That pushed the AIF onset a full frame late and
   forced the fitted AIF to zero on a frame that already carried contrast.

Both are fixed in commit `192d3c1`, which also regenerated the `sub-10bbbdownsample` baseline with
auto steady-state *and* auto injection timing on the MATLAB side. With them fixed, the algorithm
difference below is what is left, and it is the whole remainder.

## Evidence: the gap is entirely Stage-B

Substituting MATLAB's fitted `Cp_use` into **Python's own Stage-D** (same ROI curve, same fitters,
same bounds) and re-running the whole-brain average-then-fit against the regenerated MATLAB
baseline:

| model | Python's AIF | MATLAB's AIF | gate |
|---|---|---|---|
| `ex_tofts` | 0.007399 | **0.000012** | 0.01 |
| `patlak` | 0.003443 | **0.000014** | 0.01 |
| `tofts` | 0.036054 **FAIL** | **0.024474** | 0.03 |
| `tissue_uptake` | 0.097563 **FAIL** | **0.025524** | 0.05 |

(`max_abs_err` over every column of `Dyn-1_<model>_fit_rois.xls`.)

Swap one array and all four models pass — two of them at ~1e-5. Stage-D is in near-exact parity;
Stage-B is not. `test_bbb_p19_region_parity` (voxelwise) passes either way, because per-voxel
Ktrans is far less sensitive to the AIF's leading edge than an ROI-averaged fit is.

## The disagreement, concretely

Fixture `sub-10bbbdownsample/ses-01`: 64 frames, `dt = 0.264` min, `end_ss = 3` (1-based, last
baseline frame), so `start_injection = 0.528`, `end_injection = 0.858` min. Both sides resolve
those to the same frames and produce a **bit-identical measured AIF**.

| frame (0-based) | 0 | 1 | 2 | **3** | 4 | 5 |
|---|---|---|---|---|---|---|
| `timer` (min) | 0.000 | 0.264 | 0.528 | **0.792** | 1.056 | 1.320 |
| measured `CpROI` | 0.0107 | 0.0199 | −0.0306 | **2.3421** | 1.9638 | 1.9982 |
| MATLAB fitted `Cp_use` | 0 | 0 | 0 | **0.9819** | 1.9638 | 1.8566 |
| Python fitted `Cp_use` | 0 | 0 | 0 | **2.1363** | 1.9768 | 1.8433 |

Frame 3 is the measured peak and the first contrast frame. MATLAB puts the fitted AIF at **less
than half** the measured value there; Python puts it near the measurement. Everything downstream
of frame 4 agrees closely.

Fitted coefficients (`A, B, c, d, t_base_end, t0_exp`):

| | A | B | c | d | `t_base_end` | `t0_exp` |
|---|---|---|---|---|---|---|
| MATLAB | 0.73246276 | 1.26156293 | 0.75359289 | 0.02777138 | 0.528 | **1.06412464** |
| Python (`legacy_sobel`, default) | 0.87266465 | 1.26364824 | 0.71663972 | 0.02728594 | 0.528 *(fixed)* | **0.792** *(fixed)* |
| Python (`fit_transition_times`) | — | — | — | — | 0.52800023 | **0.67977808** |

`t_base_end` (end of baseline) agrees. **`t0_exp` — the end of the linear upslope, i.e. where the
model says the bolus peaks — is the entire disagreement.** MATLAB places it at 1.064 min, one
frame *past* the measured peak, which puts frame 3 at exactly 49% of the way up the ramp
(`(0.792 − 0.528) / (1.064 − 0.528) × (A+B) = 0.982`). Python's default pins it at 0.792, putting
frame 3 at the top of the ramp (`A + B = 2.136`).

## Side-by-side: `AIFbiexpfithelp.m` vs `_fit_aif_biexp`

Sources: [`dce/AIFbiexpfithelp.m`](../../../../dce/AIFbiexpfithelp.m),
[`dce/AIFbiexpcon.m`](../../../../dce/AIFbiexpcon.m), and `_fit_aif_biexp` / `_aif_biexp_con` in
[`python/dce_pipeline.py`](../../../../python/dce_pipeline.py).

### Verified identical (do not "fix" these)

- **Model function.** `AIFbiexpcon` and `_aif_biexp_con` are the same piecewise curve: constant
  `baseline` for `t < t_base_end`; linear ramp to `A + B` over `[t_base_end, t0_exp)`; then
  `A·exp(−c·(t−t0_exp)) + B·exp(−d·(t−t0_exp))`. `baseline` is forced to 0 when `fittingAU` is
  false. Checked term by term.
- **Weighting.** Both build `W = 10` everywhere, `W = 0` for every frame up to *and including*
  `max_index`. MATLAB passes `options.Weights = W` (minimising `Σ w·r²`); Python multiplies
  residuals by `√w`. Same objective.
- **Bounded optimisation.** MATLAB declares `Algorithm = 'Levenberg-Marquardt'` *with* `Lower`/`Upper`.
  LM does not support bounds, so this looks like a divergence from scipy's bounded `trf`. It is not:
  verified empirically that `fit` honours the bounds anyway (fitting `m*x` to `y = 3x + 1` with
  `Lower = 0, Upper = 1` returns `m = 1.000000`, not `3.0`), i.e. MATLAB silently uses a bounded
  trust-region solver. The declared algorithm name is misleading; the behaviour matches.
- **`baseline`, `maxer`, and the A/B bounds derived from it** (`upper(1:2) = 2·maxer`,
  `initial(1:2) = 0.5·maxer`), and resolution of the injection window to frame indices
  (nearest timer sample). Identical.
- **Preference keys** (`aif_lower_limits`, `aif_upper_limits`, `aif_initial_values`, `aif_TolFun`,
  `aif_TolX`, `aif_MaxIter`, `aif_MaxFunEvals`, `aif_Robust`). Same names, same defaults.

### Real differences

**D1 — Are the transition times fitted at all? (the one that bites)**

| | behaviour |
|---|---|
| MATLAB | **Always 6 parameters.** `t_base_end` and `t0_exp` are free, appended to `A,B,c,d`. |
| Python (default `legacy_sobel`) | **4 parameters.** `t_base_end := timer[start_idx]` and `t0_exp := timer[end_idx]` are held **fixed** at grid points. |
| Python (`aif_biexp_timing_method = "fit_transition_times"`) | 6 parameters, but see D2. |

The Python default is not what current MATLAB does. Nothing in production sets the override —
it appears only in tests (`_make_tofts_post_8ef4988_config` in the parity suite,
`test_dce_pipeline.py`, `test_dce_preferences_bridge.py`), and it is not exposed through
`dce_preferences.txt` or the CLI. MATLAB gained the 6-parameter form in commit `b120076`
("robust AIF fit"); the Python default corresponds to the older fixed-timing behaviour.

**D2 — `t0_exp` lower bound (why switching Python to 6-param does *not* close the gap)**

| | lower bound on `t0_exp` | value on this fixture |
|---|---|---|
| MATLAB | `t_base_end_init + eps`, where `t_base_end_init = timer(start_index+1)` — a **constant** derived from the initial guess, independent of the fitted `t_base_end` | 0.792 |
| Python | reparameterised as `t0_exp = t_base_end + delta` with `delta ≥ time_eps` — i.e. effectively `t0_exp > t_base_end` | 0.528 |

Python's form is self-consistent (it *guarantees* `t0_exp > t_base_end` for any fitted
`t_base_end`); MATLAB's is a fixed floor that happens to sit one frame later. From the same
starting point (0.792) the two optimisers then walk in **opposite directions** — MATLAB up to
1.064, Python down to 0.680 — because each direction is feasible in only one of the two boxes.
That is why `fit_transition_times` alone does not reconcile them: it produces the same `Cp_use`
Python's default does.

**D3 — `t_base_end` box can collapse, and MATLAB's start point falls outside it**

Both use `t_base_end ∈ [timer(start_index), timer(end_index−1)]`. When the injection window spans
a single frame gap — which is exactly what auto-detection produces here — that box collapses to the
single point `[0.528, 0.528]`, while MATLAB's `StartPoint` is `timer(start_index+1) = 0.792`,
*outside* its own upper bound. MATLAB clamps silently. Python clamps explicitly
(`t_base_end_init = min(max(init, lower + time_eps), upper)`) and additionally widens a collapsed
box by `time_eps`. Same outcome here, but the two arrive by different routes and only Python's is
intentional.

**D4 — epsilon convention.** MATLAB uses machine `eps` (~2.2e−16) for the `t0_exp` floor. Python
uses `_timer_epsilon` — the smallest positive gap in the timer, **0.264 min** here. Fifteen orders
of magnitude apart. Immaterial when a bound is not active; decisive when one is.

**D5 — index-bound guards exist only in Python.** MATLAB's `t_base_end_upper = timer(end_index-1)`
indexes `timer(0)` if `end_index == 1`, and `t0_exp_upper = timer(end_index + round(0.2*numel(timer)))`
can index past the end of `timer`. Both are hard errors. Python clamps both into range. Not
triggered by this fixture, but reachable with a short series or a late-detected injection.

**D6 — robustness fallbacks exist only in Python:** non-finite or non-positive `maxer` falls back
to `max(curve)`; an all-zero weight vector falls back to uniform weights; initial values are
clamped inside the bounds. MATLAB has no equivalent and would propagate NaN or error.

**D7 — reported goodness-of-fit is not comparable.** MATLAB logs `gof.adjrsquare`: **weighted**,
`p = 6`. Python logs `fit_rsquared_cp_adj` from `_adjusted_rsquare`: **unweighted**, over all
frames, `p = fit_param_count` (4 in the default mode). On this fixture MATLAB reports 0.9456 and
Python 0.9704 — this does **not** mean Python fits better, and neither number should be used to
adjudicate between them.

## The shared hazard behind all of it

**Both** pipelines zero-weight every frame up to and including the AIF peak. On this fixture
`max_index` is frame 3 — the measured peak and the *only* frame that observes the bolus rise. So:

- the fit is driven **exclusively by the washout tail**;
- `t_base_end` and `t0_exp` are constrained only through where the exponential decay's time origin
  sits, plus their bounds;
- the fitted AIF's value at frame 3 is **pure extrapolation, unconstrained by data on either side**.

This is shared behaviour, so it is not itself a parity difference — but it is the reason D1/D2
produce a 2× disagreement instead of a rounding difference. Any two reasonable optimisers can land
anywhere along a flat direction. Note that on this fixture Python's extrapolation (2.14) is the
closer of the two to the measurement (2.34); MATLAB's (0.98) is less than half of it. **The
baseline being MATLAB's does not make MATLAB's answer the correct one.**

## Open questions (to resolve before planning)

1. **Should the transition times be fitted or fixed?** Fitting them (MATLAB) adds two nearly
   unconstrained parameters to a curve whose rise is zero-weighted. Fixing them (Python default)
   pins the peak to the injection-window grid point, which is only as good as the injection
   detection. A third option — fit them, but weight the rise so they are actually identified — is
   not what either side does today.
2. **Should the pre-peak frames really carry zero weight?** This is the root enabler. It was
   presumably meant to stop the baseline dominating the fit, but it also discards the only
   evidence about the bolus shape.
3. **If transition times stay fitted, what is the correct `t0_exp` lower bound** — Python's
   self-consistent `t_base_end + ε`, or MATLAB's constant `timer(start_index+1)`? These are not
   equivalent and they select different optima.
4. **Which side moves?** Whatever is chosen must land in **both** languages simultaneously, and the
   `sub-10bbbdownsample` baseline must be regenerated in the same change — the fitted AIF feeds
   every map and ROI table in the fixture.
5. **Does the choice hold up on real data?** This is a single fixture with a coarse 15.84 s frame
   time, where the bolus rise occupies one sample. Validate any candidate against `RUNNER_DATA`
   sessions before locking it in.

## Planned next steps
I think we have generally gotten the best performance our of extracting end_ss from the 6 parameter AIF biexp fit. However, we haven't done a comparison between the new LV algorithm and the 6 parameter fit. We don't need to make a final decision right now but a few things. 
1) I want to add in the 6 parameters biexp fit as one of our standard listed end_ss auto methods (for python). The auto end_ss methods selected should be consistent and binding, so if LV is selected that needs to be use everywhere for all end_ss (including during the AIF fitting, so that would be forced to a 5 parameter fit). Similarly if the 6 parameter biexp fit is chosen as the end_ss auto method, the value that comes out of that fit should be used everywhere for end_ss. This might take some careful thought on how to implement cleanly as the AIF fitting is done in a difference spot then the end_ss find is, but a clean solution should be possible
2) I want to switch both python and matlab to get end_ss from the 6 parameter biexponential fit

## Reproducing

- Regenerated MATLAB baseline (auto steady state + auto injection, all five models, `force_cpu=1`):
  recipe in [`tests/README.md`](../../../../tests/README.md).
- MATLAB fitted coefficients + the LM-honours-bounds check: run `A_make_R1maps_func` →
  `B_AIF_fitting_func` with `startInjection = endInjection = -1`, then call `AIFbiexpfithelp`
  directly on `{Cp: CpROI, timer, step: [start_injection end_injection], fittingAU: false}`.
- Python fitted coefficients: `b_out.json` keys `fit_params_cp`, `fit_t_base_end_cp`,
  `fit_t0_exp_cp`, `fit_rsquared_cp_adj`; arrays in `b_out_arrays.npz` (`CpROI`, `Cp_use`, `timer`).
- The isolation experiment: load `Ct`/`timer` from Python's Stage-B checkpoint, average over the
  brain ROI, and call `fit_*_stage_d` from `python/dce_fit_backends.py` twice — once with Python's
  `Cp_use`, once with MATLAB's — comparing both to `Dyn-1_<model>_fit_rois.xls`.
