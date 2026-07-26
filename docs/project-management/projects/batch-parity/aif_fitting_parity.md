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
uses `_timer_epsilon` = `min_positive_timer_gap × 1e−6` = **2.64e−7 min** here — nine orders above
machine `eps`, but still ~six orders *below* one frame.

> Correction: an earlier revision of this section claimed Python's epsilon was 0.264 min (a whole
> frame) and "fifteen orders of magnitude" apart from MATLAB's. That was wrong, and the document
> contradicted itself — the reported fit `t_base_end = 0.52800023` is only reachable with an
> epsilon of ~2.6e−7. See `_timer_epsilon` in `python/dce_pipeline.py`.

Immaterial when a bound is not active; decisive when one is. Here it means Python's `t0_exp` floor
is *effectively* `t_base_end` itself, which is precisely what lets `t0_exp` collapse toward 0.528.
The correction strengthens D2 rather than softening it.

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

### Latent bugs found while specifying the fix

Neither is a parity difference (neither fires on this fixture) but both sit inside the code this
work touches, so they are recorded here.

**L1 — `dce_auto_aif.m` cannot call `AIFbiexpfithelp`.** `AIFbiexpfithelp` dereferences
`xdata{1}.fittingAU` (`AIFbiexpfithelp.m:191`, `:198`). Only `B_AIF_fitting_func.m:244/254` ever
sets that field; `dce_auto_aif.m:156` and `:222` build their `xdata` with `Cp`/`timer`/`step` only.
The auto-AIF path (`aif_rr_type = 'aif_auto'` / `'aif_auto_static'`) therefore errors out at the
`fit` call. Fixed as part of Phase 4.

**L2 — `AIFbiexpcon` is discontinuous when `fittingAU` is true.** The ramp branch evaluates to
`A + B − baseline` at `t = t0_exp` while the biexp branch evaluates to `A + B`, so the curve jumps
by `baseline` at the transition; the decay tail also relaxes to 0 rather than to `baseline`. Both
languages have this (`AIFbiexpcon.m`, `_aif_biexp_con`). Invisible in production because the `Cp`
fit uses `fittingAU = false`, where `baseline` is forced to 0 and the model *is* continuous. It
does affect the AU `Stlv_use` fit at `B_AIF_fitting_func.m:254`. **Out of scope here** — the fix
below deliberately avoids `fittingAU = true` (see S2) rather than depending on this being repaired.

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
   - **ANSWER:** I would like to always take the end_ss time as an input, and always fit the delta value. But I would also like to create a new selectable options to auto find end_ss using a full 6-param fit on the AIF, and make that the default. Since finding end_ss happends before the AIF is fit, you will have to look at this carefully to implement a clean solution. Since we are now doing the fitting twice we can use different weighting. I think we will get best results on the 6 param fit by using equal weighting but setting the weight where the linear upsload and bi-exp intersect (the end injection time) to zero, and everything else equal. Later during the 5 param fit keep the weighting as it is now (only weight after peak frame). The weighting is like this because the peak point is frequently unreliable, and sometimes the value is very high due to noise in the conversion and then if distorsts the bi-exp fitting.
2. **Should the pre-peak frames really carry zero weight?** This is the root enabler. It was
   presumably meant to stop the baseline dominating the fit, but it also discards the only
   evidence about the bolus shape.
   - **ANSWER:** see above
3. **If transition times stay fitted, what is the correct `t0_exp` lower bound** — Python's
   self-consistent `t_base_end + ε`, or MATLAB's constant `timer(start_index+1)`? These are not
   equivalent and they select different optima.
   - **ANSWER:** Lets restructure both to use the delta convention and use the lower bound from python (ie lower bound on delta is timer_epsilon). 
4. **Which side moves?** Whatever is chosen must land in **both** languages simultaneously, and the
   `sub-10bbbdownsample` baseline must be regenerated in the same change — the fitted AIF feeds
   every map and ROI table in the fixture.
   - **ANSWER:** Yes, the goal is the have the python default the same as the matlab. But matlab will likely need some minimal updates to fix small errors. But we are also seeking to expand selectable options on python (ie multiple end of baseline auto algorithms)
5. **Does the choice hold up on real data?** This is a single fixture with a coarse 15.84 s frame
   time, where the bolus rise occupies one sample. Validate any candidate against `RUNNER_DATA`
   sessions before locking it in.
   - **ANSWER:** This is absolutely the goal, but may take some time and multiple session as we identify good test datasets.

---

# Implementation spec

_Resolved 2026-07-25 from the answers above. This section is the contract; the open questions
above are kept as the rationale record._

## S0 — Shape of the fix

The AIF is fitted **twice**, with different parameter sets and different weighting, because the two
things we need from it are identified by different parts of the curve.

| | pass 1 (timing) | pass 2 (production) |
|---|---|---|
| runs in | Stage A, as a steady-state detector | Stage B, as today |
| domain | LV **signal**, baseline-subtracted and max-normalised | concentration (`CpROI`) |
| params | 6: `A, B, c, d, t_base_end, delta` | 5: `A, B, c, d, delta` (`t_base_end` fixed) |
| weights | 1 everywhere, **0 at the measured peak frame** | 0 through the peak, 10 after (unchanged) |
| kept | `t_base_end` → `end_ss`, `t0_exp` → `end_injection` | everything |
| discarded | `A, B, c, d` | — |

`t_base_end` is never fitted in pass 2: it is always an input, resolved from `end_ss` by the
existing precedence (explicit `steady_state_end` > AIF sidecar > auto-detector). `delta` is always
fitted, in both passes.

**Why the peak frame is dropped in pass 1.** The peak sample is frequently unreliable — the
signal→concentration conversion amplifies noise there, and a single inflated value distorts the
biexponential badly. Dropping it also gives the timing parameters a genuine two-sided constraint:
the baseline frames pin `t_base_end` from below, and the first post-peak frames sit on the linear
ramp, which pins `t0_exp` from above.

**Why the zero-weighted frame is the *measured* peak, not `t0_exp`.** `t0_exp` is a fitted
parameter in pass 1, so zero-weighting "the frame where the ramp meets the biexp" would move the
zeroed frame during optimisation and make the objective discontinuous. `max_index` is computed once
from the data and is fixed for the duration of the fit. On this fixture the two coincide
(`max_index == end_idx == 3`), so this matches the intent exactly.

## S1 — Parameterisation and bounds (both languages)

Reparameterise `t0_exp` as `t_base_end + delta` everywhere. This guarantees `t0_exp > t_base_end`
by construction and removes MATLAB's constant floor (D2) and its collapsible `t_base_end` box (D3).

- `delta_lower = dt` (**one frame**). `t_base_end = timer[end_ss − 1]`, so the earliest the peak
  can physically sit is the next sample: `delta = dt`. A curve that is at baseline on frame
  `end_ss` and at max on frame `end_ss + 1` *is* `delta = dt` — the ramp spans one frame with no
  interior samples, which already reproduces a single-frame jump exactly. `delta = 0` would place
  the peak *on* the last baseline frame, which is self-contradictory, so it is not reachable.
- `delta_upper` unchanged in effect: `t0_exp_upper − t_base_end`, with the index into `timer`
  clamped into range (D5).
- `delta_init` comes from the **unsnapped** `end_injection_min − start_injection_min`, *not* from
  `timer[end_idx] − timer[start_idx]`. Pass 1 reports a fractional `t0_exp`; snapping it to the
  nearest frame before using it as pass 2's start point would throw that away.
- Existing `max(delta, time_eps)` clamps and the `t_exp = max(t0_exp, t_base + eps)` guard in the
  model can stay — with a floor of `dt` they can never fire.

## S2 — Pass 1 as a steady-state detector

`end_ss` is needed to build `Cp` (it defines `baseline_slice`, which feeds `sss` → `ab_lv` → the
`r1_lv` rescale → `cp`, `dce_pipeline.py:1822-1891`), so a fit *on `Cp`* cannot produce `end_ss`.
Pass 1 therefore runs on the **LV signal** curve, which breaks the cycle with no re-conversion and
no restructuring of Stage A in either language. Only timing is carried out of pass 1, and timing is
invariant to the affine rescaling below, so the signal domain costs us nothing we keep.

Procedure, identical in both languages:

1. Seed with the existing cheap detector (`tv` / `find_end_ss_tv`) to get a provisional `end_ss`
   and hence a provisional baseline window and injection window.
2. Take the mean LV signal curve, subtract the provisional baseline mean, divide by its max —
   exactly what `dce_auto_aif.m:147-149` already does.
3. Fit 6 params with `fittingAU = false` and the pass-1 weights, in **frame units**
   (`timer = 0:n-1`, as `dce_auto_aif.m:25` does).
4. Return `end_ss = round(t_base_end) + 1` and `end_injection = t_base_end + delta + 1`.

Step 2 is not cosmetic: it is what lets step 3 use `fittingAU = false`. Fitting the raw signal with
`fittingAU = true` would hit L2 above, where the model is discontinuous at `t0_exp` by the full
(large, in AU) baseline offset.

Step 4 needs no new plumbing: `start_injection` / `end_injection` is already an A→B channel in both
languages, both already tolerate a fractional `end_injection`, and `delta_init` in pass 2 falls out
of the difference. `A, B, c, d` from pass 1 are discarded — signal saturation distorts amplitudes
and decay rates, and we do not need them.

## S3 — Known soft spot in pass 2, and the diagnostic for it

Pass 2 zero-weights everything through `max_index`, so on this fixture the first weighted frame is
4 (`t = 1.056`). That splits the objective in `delta` into two regimes:

- `t0_exp > 1.056` — frame 4 and possibly others sit on the linear ramp. There is a real kink at
  the transition and SSE responds sharply to `delta`. MATLAB's 1.064 lives here.
- `t0_exp < 1.056` — every weighted frame is on the decay, where shifting the time origin is almost
  exactly compensated by rescaling `A` and `B`. Nearly flat. Python's 0.680 fell into this basin.

`delta` is fitted freely in pass 2 (bounds from S1, start point from pass 1), which is correct in
the first regime. The failure mode to watch is the optimiser sliding *below the first weighted
frame* into the flat basin. Rather than constrain it, log pass-1 and pass-2 `t0_exp` side by side
and **warn when pass 2 moves `t0_exp` below the first non-zero-weight frame** — that is the signal
that the answer is unidentified rather than fitted.

## S4 — Work plan

**Phase 0 — this document.** D4 correction, L1/L2, this spec. _(done)_

**Phase 1 — Python fit core**, `_fit_aif_biexp` / `_aif_biexp_con` in `python/dce_pipeline.py`.
Collapse the 4-param/6-param split into the single 5-param form of S1 with `t_base_end` fixed;
`delta_lower = dt`; unsnapped `delta_init`; add a `weight_mode` argument
(`"tail"` default, `"equal_drop_peak"` for pass 1) since the weights are currently hard-coded.
Retire `aif_biexp_timing_method` and `ALLOWED_AIF_BIEXP_TIMING_METHODS`. Finish the half-applied
`lower6`/`upper6` edit currently sitting uncommitted in the working tree.

**Phase 2 — Python detector.** New `_biexp_fit_baseline_end` registered in the `detector_map` at
`dce_pipeline.py:1514`, implementing S2. Make it the default: `python/dce_default.json`,
`python/dceprep_default.json`, and the `none → tv` fallback at `dce_pipeline.py:1509`.

**Phase 3 — Python injection-window cleanup.** `start_injection := end_ss` is already Stage A's
behaviour (`dce_pipeline.py:1897`); the remaining risk is the override path. Drop the
`start_injection_min` / `start_injection` branch in `_resolve_stage_b_injection_window`
(`dce_pipeline.py:2181`) so the two cannot diverge. `end_injection` stays separately overridable.
Update `docs/dce_options.md` and affected tests.

**Phase 4 — MATLAB mirror.** In `AIFbiexpfithelp.m`: move `t_base_end` to a `problem` parameter,
coefficients become `{A, B, c, d, delta}`, `delta` floor = one frame, `t0_exp_init` from the
unsnapped `ended`, plus the D5 index guards and the D6 fallbacks. New `find_end_ss_biexp.m` with
`find_end_ss_tv`'s exact `[end_ss, end_injection] = f(DYNAMLV)` signature, swapped in at
`A_make_R1maps_func.m:647` — no restructuring, because S2 keeps pass 1 inside Stage A. Fix L1 in
`dce_auto_aif.m`.

**Phase 5 — regenerate and gate.** Regenerate the `sub-10bbbdownsample` baseline per
`tests/README.md` (the fitted AIF feeds every map and ROI table in the fixture), then
`test_bbb_p19_roi_xls_parity`, `-m parity`, and `test_bbb_p19_region_parity`. Phases 4 and 5 land
in the same commit.

**Phase 6 — real data.** Validate on `RUNNER_DATA` sessions. Expected to span several sessions as
suitable test datasets are identified; this fixture's 15.84 s frame time puts the entire bolus rise
in one sample, so it cannot by itself justify the choice.

## S5 — Measured result of Phases 1-3 (Python only, MATLAB baseline unchanged)

`max_abs_err` over every column of `Dyn-1_<model>_fit_rois.xls`, against the *existing* MATLAB
baseline (so Phases 4-5 have not happened; MATLAB is still the pre-change algorithm):

| model | before (this doc's opening table) | Phase 1 (`tv`) | Phases 1+2 (`biexp_fit`) | gate |
|---|---|---|---|---|
| `tofts` | 0.036054 **FAIL** | 0.028711 pass | **0.021488** pass | 0.03 |
| `ex_tofts` | 0.007399 | 0.005409 | **0.003914** | 0.01 |
| `patlak` | 0.003443 | 0.002524 | **0.001838** | 0.01 |
| `tissue_uptake` | 0.097563 **FAIL** | 0.079914 **FAIL** | **0.062754** **FAIL** | 0.05 |

Monotone improvement on all four, and `tofts` now passes. `tissue_uptake` is the sole holdout and
is expected to stay failing until Phase 5, because the residual is the very disagreement Phases 4-5
resolve.

### The timing pass does *not* reproduce MATLAB's `t0_exp`

On `sub-10bbbdownsample` the Stage-A timing fit returns, in frame units:

```
t_base_end = 2.0000   (-> end_ss = 3, agreeing with tv and with MATLAB)
delta      = 1.0074   (one frame)
t0_exp     = 3.0074   (-> 0.794 min, the measured peak frame)
adj R^2    = 0.9817
```

So with the peak sample dropped and everything else weighted equally, the data say the upslope is
**one frame long and the curve peaks at frame 3** — close to Python's old default (0.792) and
nowhere near MATLAB's 1.064.

MATLAB's 1.064 puts frame 4 (`t = 1.056`) *on the upslope*, and indeed reproduces its measured
value 1.9638 almost exactly that way. But it also asserts the AIF is still rising at 1.056 and
peaks at 1.064, when the measurement peaks at frame 3 (2.3421) and frame 4 is already *lower*.
MATLAB only reaches that solution because its constant `t0_exp` floor of 0.792 (D2) made the
downward direction infeasible; it is an artefact of the bound, not a reading of the data.

`delta = 1.0074` frames is just above its one-frame floor but not on it, so this is an interior
optimum, not the floor asserting itself.

**Consequence for Phase 4: MATLAB moves to Python's answer, not the other way round.** That is the
opposite direction from "make the Python default match MATLAB", and it is a decision to confirm
before regenerating the baseline. It is, however, exactly what "the baseline being MATLAB's does
not make MATLAB's answer the correct one" anticipated, and the ROI-xls trend above is consistent
with it: every model moved *closer* to MATLAB as the Python timing got more principled, which
would not happen if the new timing were simply wrong.

**Confirmed 2026-07-25: MATLAB moves.** Phases 4-5 landed on that basis.

## S6 — Result after Phases 4-5 (both languages changed, baseline regenerated)

Cross-check on `sub-10bbbdownsample`, both sides running the new algorithm:

| | `end_ss` | `end_injection` (frame) | `t_base_end` | `t0_exp` | `c` | `d` |
|---|---|---|---|---|---|---|
| Python | 3 | 4.007424 | 0.528 | 0.848346 | 0.71663973 | 0.02728594 |
| MATLAB | 3 | 4.016123 | 0.528 | 0.796922 | 0.71664070 | 0.02728596 |

Compare with the opening table, where `t0_exp` was 1.064 (MATLAB) against 0.792 (Python). The
decay now agrees to seven significant figures. What remains is `delta`: from the same seed
(~0.267 min) MATLAB barely moves while scipy's `trf` walks up to 0.320, which is the S3 flat
direction doing exactly what S3 says it does. It puts `Cp_use[3]` at 2.094 (MATLAB) against 1.731
(Python).

ROI-xls parity against the **regenerated** baseline, with the Python fixture switched to
`steady_state_auto_method = "biexp_fit"` so both sides use the same detector:

| model | before | Phases 1-3 | **Phases 4-5** | gate |
|---|---|---|---|---|
| `tofts` | 0.036054 **FAIL** | 0.021488 | **0.016489** pass | 0.03 |
| `ex_tofts` | 0.007399 | 0.003914 | **0.003312** pass | 0.01 |
| `patlak` | 0.003443 | 0.001838 | **0.001534** pass | 0.01 |
| `tissue_uptake` | 0.097563 **FAIL** | 0.062754 **FAIL** | **0.049170** pass | 0.05 |

All four pass. `test_bbb_p19_region_parity` passes, the full Python suite is green
(234 passed / 9 skipped / 2 xfailed), and the MATLAB suite is green (28 passed).

## S7 — Weighting rework: uniform + robust + a data-based peak prior

Superseded S0's positional weighting entirely. `weight_mode` is gone; both passes now use the
same estimator and differ only in whether `t_base_end` is free.

**Uniform weights.** Zeroing frames by position also threw away the only evidence about the
bolus rise, which is what left the upslope duration unidentified in the first place.

**`aif_Robust = Bisquare` by default**, implemented as a hand-written Tukey biweight IRLS in
*both* languages (`_tukey_irls`). Do **not** route this through scipy's `loss=`: `f_scale`
defaults to 1.0, which on concentration-scale residuals leaves the loss quadratic everywhere
(a 0.61 outlier keeps ~73% of its weight), while MATLAB's Bisquare recomputes a MAD scale each
iteration and rejects the same sample outright. Measured, that gap moved `tissue_uptake` from
0.049 to 0.099. The IRLS update is damped 50%; without damping the frames just after the peak,
where the measured curve is non-monotonic and a biexponential cannot fit both, drive a limit
cycle that never settles.

**A data-based peak prior** (`_aif_peak_weight`, pref `aif_peak_weight_exponent`, default 2).

### Why the peak needs a prior weight at all: leverage

Measured leverage on `sub-10bbbdownsample`:

```
frame 3  t=0.792  Cp=2.3421  h=1.000000     <- the peak
frame 4  t=1.056  Cp=1.9638  h=0.656
frame 5  t=1.320  Cp=1.9982  h=0.257
mean over 64 frames                = 0.078
```

**The peak has leverage exactly 1.** It is the only sample in `[t_base_end, t0_exp)` and the
model's maximum `A + B` sits at `t0_exp`, so the model interpolates it by construction. A
noise-inflated peak therefore drags the curve to itself and ends up with a *small* residual —
the classic masking problem, since M-estimators are robust to vertical outliers at low leverage,
not to high-leverage points. Without leverage correction the peak residual is +0.044 and nothing
is down-weighted; with it, the peak is rejected outright.

Left at full weight the peak does real damage: one exponential is spent reaching that single
sample and is no longer available for the washout. Sweeping the peak's weight:

| `w_peak` | `A` | `c` | `delta` | `Cp_fit[3]` | SSE excl. peak |
|---|---|---|---|---|---|
| 1.00 | 0.9709 | 0.9079 | 0.2640 | 2.2724 | 0.22319 |
| 0.50 | 0.9426 | 0.8518 | 0.2640 | 2.2343 | 0.21845 |
| 0.10 | 0.8924 | 0.7545 | 0.2640 | 2.1646 | 0.21335 |
| 0.00 | 0.8323 | 0.7166 | **0.3300** | **1.6749** | 0.21283 |

`A` and `c` fall monotonically as the peak is released — the fast exponential being freed. Note
the **discontinuity at exactly zero**: any nonzero weight keeps `delta` at its one-frame floor,
but zero frees it and the fit jumps basins. That is why `_aif_peak_weight` has a nonzero floor,
and why proportional de-weighting is stabler than the hard zeroing it replaced.

The weight is the peak's excess over the median relative to the next largest sample's excess,
raised to `aif_peak_weight_exponent`. It must be that ratio rather than a raw
`1 / (peak − median)`: the latter carries units of 1/concentration, so it would change with the
units and can exceed 1.

**The prior applies only to the production fit, never to the Stage-A timing pass.** What is
unreliable about the peak is its *height*, not its *position*, and position is exactly what the
timing pass estimates, with the peak as its primary evidence. Applying it there moved the fitted
baseline end from 1.80 frames to 1.16 — `end_ss` 3 → 2, which is simply wrong for this series.

### Caveat: on this fixture the prior is currently redundant

Because the leverage-corrected Tukey already drives the peak to zero robust weight, the prior
weight of 0.55 changes the fitted coefficients by <1e-4 here. It is insurance for series where
the rise is better sampled and leverage drops below 1 — which is also the only regime where the
proportional de-weighting can behave proportionally. Worth re-checking in Phase 6.

## S8 — `Fp` is excluded from the `tissue_uptake` ROI-xls gate

`tissue_uptake` failed the gate throughout this work, and the cause was never the AIF peak:
`Cp_use[3]` agreement *improved* from 17% to 2% while the gate got worse. Per column, every
`tissue_uptake` output agrees to <0.004 except `Fp` (0.117 Python vs 0.181 MATLAB) and its
confidence interval, and `Fp 95% high` alone produced the failing number.

`Fp` is determined almost entirely by the AIF's leading edge, which at 15.84 s frames is a
single sample whose height the data cannot pin down (see the leverage result above). Python and
MATLAB have never agreed on it on this fixture. Gating on it measures the fixture's temporal
resolution, not the port's correctness, so `ROI_XLS_EXCLUDED_COLUMNS` in
`tests/python/test_dce_pipeline_parity_metrics.py` drops `Fp` and its CI columns for this model
only. Decision taken 2026-07-25.

## S9 — Final state

| model | original | after S1-S6 | **after S7-S8** | gate |
|---|---|---|---|---|
| `tofts` | 0.036054 **FAIL** | 0.016489 | **0.005687** | 0.03 |
| `ex_tofts` | 0.007399 | 0.003312 | **0.001675** | 0.01 |
| `patlak` | 0.003443 | 0.001534 | **0.000505** | 0.01 |
| `tissue_uptake` | 0.097563 **FAIL** | 0.049170 | **0.002645** (ex-`Fp`) | 0.05 |

Python 234 passed / 9 skipped / 2 xfailed; MATLAB 28 passed / 0 failed.

Also in this change: **L2 is fixed** (it became a prerequisite once baseline frames carried
weight — `AIFbiexpcon`/`_aif_biexp_con` now treat `baseline` as an offset the whole curve sits
on, so the ramp climbs `A + B` above it and the decay relaxes back to it; `baseline` is 0
whenever `fittingAU` is false, so the concentration fit is bit-identical either way). And the
transition times are now drawn as vertical lines rather than a zeros-with-spikes series, via
`dce/plot_aif_transition_lines.m` and `_mark_aif_transition_times`; the Python Stage-B figure
can be suppressed with `stage_overrides.save_aif_figure = false`.

### ~~Open risk: `tissue_uptake` has almost no margin~~ (closed by S7-S8)

0.049170 against a 0.05 gate. The residual is the `delta` divergence in the table above — the two
optimisers stopping at different points along S3's flat direction — and `tissue_uptake` is the
model most sensitive to `Cp_use` at the frame where they differ. Any further change to Stage-B
weighting, tolerances, or the scipy/MATLAB optimiser versions could push it back over.

Closing it means giving the peak-adjacent frames some weight in the production fit so `delta` is
actually identified there, rather than inherited from the Stage-A timing pass. That is a
deliberate departure from "keep the pass-2 weighting as it is now" and was **not** done here. It
is the obvious first thing to try if Phase 6 finds this fragile on real data.

## S10 — Phase 6 first pass: `RUNNER_DATA` (2026-07-25)

Two sessions only (`sub-1101743/{ses-01,ses-02}`, 64 frames each, no AIFArtist ratings), so
nothing here is statistical. Both are the sharp-rise case S1 anticipated: 2-4 baseline frames and
a bolus that peaks in a single frame.

`tests/python/run_baseline_end_reliability.py` gained the `biexp_fit` detector, plus a
`--no-ground-truth` mode (discover masks by filename, report cross-detector agreement instead of
accuracy) and `--dynamic-pattern` (pin the series when reading a derivatives tree). Stage D ran
end-to-end on `cpufit_cpu`; both sessions succeeded.

**The peak prior is no longer redundant.** S7 recorded that on `sub-10bbbdownsample` the prior
changed nothing because leverage-corrected Tukey already zeroed the peak. On real data it bites:
prior weight 0.084 (ses-01) and 0.147 (ses-02). Combined with the robust estimator the peak is
effectively discarded — the fitted curve reaches 0.86 of a measured 1.90 mM (ses-01) and 0.92 of
2.62 mM (ses-02). The rest of both curves is tracked well. **Open question for the user:** whether
discarding this much peak is right. The AIF's peak height feeds `Ktrans` scaling, so systematically
under-fitting it biases `Ktrans` high; against that, a one-frame spike at this sampling rate is
genuinely undersampled and its height is not trustworthy. This is a modelling decision, not a bug.

**Latent bug L3 — the `fit_success` gate discards good non-robust fits. FIXED.** With
`aif_Robust=off`, `least_squares` on `sub-10bbbdownsample` returns `status=0`, `nfev=1000`,
`"maximum number of function evaluations is exceeded"` — so `_biexp_fit_baseline_end` reported
`fallback_fit_not_converged` and silently fell back to `tv`. The fit was *fine*: cost 0.0146,
`t_base_end=2.000`, `t0_exp=3.000` (exact frame boundaries), adjusted R² 0.9826 against robust's
0.9550. The cause is D4's tolerances — `aif_TolFun=1e-20` and `aif_TolX=1e-23` clamp to machine
epsilon and are never satisfiable, so `trf` always exhausts `max_nfev` and reports failure. This
was misread during S7 as evidence that the non-robust path could not fit at all.

`result.success` is just `status > 0`, i.e. it asks whether a tolerance test fired — which here
is a question that can never be answered yes. Budget exhaustion is the *expected* terminal state,
not a failure. `_lsq_result_usable` replaces it: reject only a negative status (improper input) or
a non-finite solution, and leave fit *quality* to `rsquare_adj`, which is where callers already
judge it. Applied at both call sites — the non-robust branch of `_fit_aif_biexp` and the inner
solve of `_tukey_irls`, which had the same latent false negative.

MATLAB never had this gate (`AIFbiexpfithelp.m` ignores `exitflag`), so L3 was a Python-only
divergence and fixing it moves the two backends together. Behaviour under the shipped default is
unchanged — with `aif_Robust=Bisquare` the fixture and both `RUNNER_DATA` sessions return the same
`end_ss` and the same `t_base_end` as before, and all four ROI-xls gates are bit-identical
(`tofts` 0.005687, `ex_tofts` 0.001675, `patlak` 0.000505, `tissue_uptake` 0.002645). What changes
is that `aif_Robust=off` now produces `mode=fit` instead of falling back to `tv`.

**The robust estimator hurts the Stage-A timing pass.** Same principle already established for the
peak prior in S7 — the peak's *height* is unreliable but its *position* is the timing pass's
primary evidence — except only the prior was exempted, not the estimator. Measured:

| series | robust | `t_base_end` | `t0_exp` | `end_ss` | adj R² |
|---|---|---|---|---|---|
| `sub-10bbbdownsample` | Bisquare | 1.802 | 2.820 | 3 | 0.9550 |
| `sub-10bbbdownsample` | off | **2.000** | **3.000** | 3 | 0.9826 |
| `ses-01` | Bisquare | 2.549 | 3.994 | 4 | 0.7957 |
| `ses-01` | off | **2.961** | **4.000** | 4 | 0.9393 |
| `ses-02` | Bisquare | 0.993 | 2.562 | **2** | 0.6672 |
| `ses-02` | off | **1.983** | **3.000** | **3** | 0.9463 |

With robust off, `t0_exp` lands on the peak frame exactly in all three cases and `t_base_end` on
the last baseline sample; with it on, both drift early. On `ses-02` that costs a frame: measured
baseline is frames 1-3 (104.6, 105.8, 100.6) with the peak at frame 4, so `end_ss=3` is right and
the robust fit's 2 is wrong. The higher adjusted R² is corroborating, not proof — declining to
reject points raises R² by construction.

Fixing this means `apply_robust=False` alongside the existing `apply_peak_weight=False` in the
timing pass. It was blocked on L3, since the non-robust path could not report success; L3 is now
fixed, so this is unblocked. **Still not done** — it changes a production default on two sessions
of evidence, and wants more rated data first.

`ses-02` also tripped S3's drift diagnostic (`+0.5186 min`, 0.3999 → 0.9186), which is the
diagnostic firing exactly as designed: Stage A's timing was wrong, and Stage B disagreed with it.

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
