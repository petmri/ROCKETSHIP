# Script Preferences Option Parity Audit

Canonical machine-readable audit:
- `docs/project-management/projects/script-preferences-audit/script_preferences_option_audit.json`

Source:
- `script_preferences.txt`

Status labels:
- `supported`: key is consumed by Python config/stage overrides.
- `intentionally_dropped`: key is out-of-scope by design for the Python CLI pipeline.
- `pending`: key is known but not fully implemented yet.

Current coverage snapshot:
- total keys audited: `54`
- supported: `32`
- intentionally_dropped: `10`
- pending: `12`

Notes:
- This audit maps script-level keys to either top-level Python config fields or `stage_overrides` aliases.
- Regression tests enforce that every key in `script_preferences.txt` appears in the audit file.
