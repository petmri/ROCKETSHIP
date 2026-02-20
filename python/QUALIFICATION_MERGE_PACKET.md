# Python Qualification Merge Packet

## Scope
This packet documents TODO item 4 (real-data workflow qualification) for the current
Python transition checkpoint.

## Dataset Discovery
Command:

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_bids_discovery.py \
  --bids-root tests/data/BIDS_test \
  --output-json out/python_qualification_bids_test_auto/discovery_manifest.json \
  --print-json
```

Discovered sessions: `4`

Manifest artifact:
- `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto/discovery_manifest.json`

## Qualification Command
Command used for the merge packet run (dated 2026-02-20):

```bash
cd /Users/samuelbarnes/code/ROCKETSHIP
.venv/bin/python run_python_qualification.py \
  --bids-root tests/data/BIDS_test \
  --output-root out/python_qualification_bids_test_auto \
  --backend auto \
  --no-postfit-arrays \
  --print-summary-json
```

Summary artifacts:
- `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto/qualification_summary.json`
- `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto/qualification_summary.md`

## Run Result Summary (Pre-Gate Baseline)
Top-level result:
- `status=ok`
- `sessions_discovered=4`
- `sessions_passed=4`
- `sessions_failed=0`
- `blocker_count=0`
- `warning_count=1`
- `backend_used=cpufit_cpu` (via `--backend auto` in this environment)

Per-session summary:

| Session | T1 | DCE | Duration (s) | Notes |
|---|---|---|---:|---|
| `sub-01original_ses-01` | ok | ok | 9.009 | none |
| `sub-02downsample_ses-01` | ok | ok | 2.428 | flip-angle metadata trimmed from 3 to 2 to match derivative frames |
| `sub-03noisyhigh_ses-01` | ok | ok | 9.367 | none |
| `sub-04noisylow_ses-01` | ok | ok | 9.260 | none |

## Blocker Classification
### Fix-now (before dev merge)
- Completed: add qualification gate for non-finite primary-model maps.
  - Implemented in `/Users/samuelbarnes/code/ROCKETSHIP/python/qualification.py`
    (helper `_primary_model_map_blockers`), with tests in
    `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/test_python_qualification.py`.
- Completed: prevent all-NaN accelerated outputs from being silently accepted.
  - Implemented in `/Users/samuelbarnes/code/ROCKETSHIP/python/dce_pipeline.py`
    (`_accelerated_output_has_usable_primary_params` + fallback handling in `_fit_stage_d_model`).
  - Added tests in `/Users/samuelbarnes/code/ROCKETSHIP/tests/python/test_dce_pipeline.py`.
  - New qualification run with gate + fallback:
    - output root:
      `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto_gated_fallback`
    - summary:
      `/Users/samuelbarnes/code/ROCKETSHIP/out/python_qualification_bids_test_auto_gated_fallback/qualification_summary.json`
    - result:
      - `status=ok`, `sessions_passed=4`, `sessions_failed=0`, `blocker_count=0`

### Post-merge / upstream follow-up
- `pycpufit` diagnostic issue for `TOFTS_EXTENDED` on BIDS-test Stage-B curves:
  - fit state `2` for all voxels at iteration `0` (`cpufit_cpu`), yielding non-finite maps without fallback.
  - Single-curve OSIPI fast checks still pass; issue appears data-regime specific (short-timer BIDS synthetic case).
- Normalize `BIDS_test` downsample metadata generation so derivative VFA frame count and
  raw sidecar flip-angle lists match directly (remove trim warning path).

## Troubleshooting Notes
- If qualification appears slow, start with:
  - `--backend auto --no-postfit-arrays`
- For full debug detail, rerun a single session by pointing `--bids-root` to a narrowed
  BIDS tree (or a copied subset).
- If acceleration backends are unavailable, `--backend auto` may degrade to pure CPU and
  runtime increases substantially.
