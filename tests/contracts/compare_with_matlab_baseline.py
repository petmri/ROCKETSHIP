#!/usr/bin/env python3
"""Compare Python model outputs against MATLAB baseline values by contract.

This runner consumes:
- tests/contracts/baselines/matlab_reference_v1.json
- tests/contracts/*_contracts.json
- a Python results JSON file

Python results JSON can be either:
1) {"results": {"<contract_id>": <value>, ...}}
2) {"<contract_id>": <value>, ...}
3) baseline-like nested structure (dotted path lookup fallback)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACTS_DIR = REPO_ROOT / "tests" / "contracts"
DEFAULT_BASELINE_JSON = DEFAULT_CONTRACTS_DIR / "baselines" / "matlab_reference_v1.json"


CONTRACT_FILES = [
    "dce_core_contracts.json",
    "dsc_core_contracts.json",
    "parametric_core_contracts.json",
]

# Maps contract IDs to paths in matlab_reference_v1.json.
CONTRACT_BASELINE_PATHS: Dict[str, str] = {
    "tofts_forward": "dce.forward.tofts",
    "extended_tofts_forward": "dce.forward.extended_tofts",
    "patlak_forward": "dce.forward.patlak",
    "vp_forward": "dce.forward.vp",
    "tissue_uptake_forward": "dce.forward.tissue_uptake",
    "twocxm_forward": "dce.forward.twocxm",
    "fxr_forward": "dce.forward.fxr",
    "patlak_linear_inverse": "dce.inverse.patlak_linear",
    "tofts_fit_inverse": "dce.inverse.tofts_fit",
    "vp_fit_inverse": "dce.inverse.vp_fit",
    "tissue_uptake_fit_inverse": "dce.inverse.tissue_uptake_fit",
    "twocxm_fit_inverse": "dce.inverse.twocxm_fit",
    "fxr_fit_inverse": "dce.inverse.fxr_fit",
    "import_aif_truncation": "dsc.import_aif",
    "previous_aif_truncation": "dsc.previous_aif",
    "ssvd_deconvolution": "dsc.ssvd_deconvolution",
    "t2_linear_fast": "parametric.t2_linear_fast",
    "t1_fa_linear_fit": "parametric.t1_fa_linear_fit",
}


@dataclass
class CompareStats:
    passed: bool = True
    mismatches: int = 0
    max_abs_error: float = 0.0
    max_rel_error: float = 0.0
    message: str = ""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    return json.loads(path.read_text())


def get_nested(data: Any, dotted_path: str) -> Any:
    cur = data
    for token in dotted_path.split("."):
        if not isinstance(cur, dict) or token not in cur:
            raise KeyError(dotted_path)
        cur = cur[token]
    return cur


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def close_enough(actual: float, expected: float, atol: float, rtol: float) -> Tuple[bool, float, float]:
    abs_err = abs(actual - expected)
    scale = abs(expected)
    rel_err = abs_err / scale if scale > 0 else (0.0 if abs_err == 0 else math.inf)
    tol = atol + rtol * scale
    return abs_err <= tol, abs_err, rel_err


def compare_values(expected: Any, actual: Any, atol: float, rtol: float, path: str = "") -> CompareStats:
    stats = CompareStats()

    if is_number(expected) and is_number(actual):
        ok, abs_err, rel_err = close_enough(float(actual), float(expected), atol, rtol)
        stats.max_abs_error = max(stats.max_abs_error, abs_err)
        if math.isfinite(rel_err):
            stats.max_rel_error = max(stats.max_rel_error, rel_err)
        if not ok:
            stats.passed = False
            stats.mismatches += 1
            stats.message = f"numeric mismatch at {path or '<root>'}: actual={actual}, expected={expected}"
        return stats

    if isinstance(expected, list):
        if not isinstance(actual, list):
            return CompareStats(False, 1, 0.0, 0.0, f"type mismatch at {path or '<root>'}: expected list")
        if len(expected) != len(actual):
            return CompareStats(
                False,
                1,
                0.0,
                0.0,
                f"length mismatch at {path or '<root>'}: actual={len(actual)}, expected={len(expected)}",
            )
        for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
            child = compare_values(exp_item, act_item, atol, rtol, f"{path}[{i}]" if path else f"[{i}]")
            stats.passed = stats.passed and child.passed
            stats.mismatches += child.mismatches
            stats.max_abs_error = max(stats.max_abs_error, child.max_abs_error)
            stats.max_rel_error = max(stats.max_rel_error, child.max_rel_error)
            if stats.message == "" and child.message:
                stats.message = child.message
        return stats

    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return CompareStats(False, 1, 0.0, 0.0, f"type mismatch at {path or '<root>'}: expected dict")

        missing_keys = [k for k in expected.keys() if k not in actual]
        if missing_keys:
            return CompareStats(
                False,
                len(missing_keys),
                0.0,
                0.0,
                f"missing keys at {path or '<root>'}: {', '.join(missing_keys)}",
            )

        for key in expected.keys():
            child = compare_values(
                expected[key],
                actual[key],
                atol,
                rtol,
                f"{path}.{key}" if path else key,
            )
            stats.passed = stats.passed and child.passed
            stats.mismatches += child.mismatches
            stats.max_abs_error = max(stats.max_abs_error, child.max_abs_error)
            stats.max_rel_error = max(stats.max_rel_error, child.max_rel_error)
            if stats.message == "" and child.message:
                stats.message = child.message
        return stats

    # Exact match fallback for strings/bools/null.
    if actual != expected:
        return CompareStats(False, 1, 0.0, 0.0, f"value mismatch at {path or '<root>'}: actual={actual}, expected={expected}")

    return stats


def load_contract_cases(contracts_dir: Path) -> List[Dict[str, Any]]:
    all_cases: List[Dict[str, Any]] = []
    for filename in CONTRACT_FILES:
        payload = load_json(contracts_dir / filename)
        for case in payload.get("cases", []):
            all_cases.append(case)
    return all_cases


def resolve_python_value(python_data: Any, contract_id: str, fallback_path: str) -> Optional[Any]:
    # 1) {"results": {"contract_id": ...}}
    if isinstance(python_data, dict):
        results = python_data.get("results")
        if isinstance(results, dict) and contract_id in results:
            return results[contract_id]

    # 2) {"contract_id": ...}
    if isinstance(python_data, dict) and contract_id in python_data:
        return python_data[contract_id]

    # 3) Dotted path fallback under "results" and top-level.
    if isinstance(python_data, dict):
        results = python_data.get("results")
        if isinstance(results, dict):
            try:
                return get_nested(results, fallback_path)
            except KeyError:
                pass
        try:
            return get_nested(python_data, fallback_path)
        except KeyError:
            pass

    return None


def build_template_output(baseline: Dict[str, Any], cases: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    template: Dict[str, Any] = {
        "meta": {
            "note": "Fill these values with Python implementation outputs.",
            "source": "compare_with_matlab_baseline.py --write-template",
        },
        "results": {},
    }

    for case in cases:
        contract_id = case["id"]
        baseline_path = CONTRACT_BASELINE_PATHS.get(contract_id)
        if not baseline_path:
            continue
        try:
            template["results"][contract_id] = get_nested(baseline, baseline_path)
        except KeyError:
            # Contract exists but baseline entry has not been exported yet.
            template["results"][contract_id] = None

    return template


def run_compare(
    baseline: Dict[str, Any],
    python_results: Dict[str, Any],
    cases: List[Dict[str, Any]],
    tolerance_profiles: Dict[str, Any],
    require_all: bool,
) -> int:
    pass_count = 0
    fail_count = 0
    missing_count = 0
    skipped_count = 0

    print("contract_id | status | mismatches | max_abs_error | max_rel_error | detail")
    print("-" * 110)

    for case in cases:
        contract_id = case["id"]
        profile_name = case.get("tolerance_profile", "forward_exact")
        profile = tolerance_profiles.get(profile_name)

        if profile is None:
            fail_count += 1
            print(f"{contract_id} | FAIL | 1 | n/a | n/a | missing tolerance profile: {profile_name}")
            continue

        baseline_path = CONTRACT_BASELINE_PATHS.get(contract_id)
        if not baseline_path:
            skipped_count += 1
            print(f"{contract_id} | SKIP | 0 | n/a | n/a | no baseline path mapping configured")
            continue

        try:
            expected = get_nested(baseline, baseline_path)
        except KeyError:
            skipped_count += 1
            print(f"{contract_id} | SKIP | 0 | n/a | n/a | baseline missing at path: {baseline_path}")
            continue

        actual = resolve_python_value(python_results, contract_id, baseline_path)
        if actual is None:
            missing_count += 1
            print(f"{contract_id} | MISSING | 0 | n/a | n/a | python result not found")
            continue

        stats = compare_values(expected, actual, float(profile["atol"]), float(profile["rtol"]))
        if stats.passed:
            pass_count += 1
            detail = "ok"
            print(
                f"{contract_id} | PASS | {stats.mismatches} | {stats.max_abs_error:.6g} | {stats.max_rel_error:.6g} | {detail}"
            )
        else:
            fail_count += 1
            detail = stats.message or "mismatch"
            print(
                f"{contract_id} | FAIL | {stats.mismatches} | {stats.max_abs_error:.6g} | {stats.max_rel_error:.6g} | {detail}"
            )

    print("-" * 110)
    print(
        "Summary: "
        f"{pass_count} pass, {fail_count} fail, {missing_count} missing, {skipped_count} skipped"
    )

    if fail_count > 0:
        return 1
    if require_all and missing_count > 0:
        return 2
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_JSON,
        help="Path to MATLAB baseline JSON.",
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=DEFAULT_CONTRACTS_DIR,
        help="Directory containing *_contracts.json and tolerance_profiles.json.",
    )
    parser.add_argument(
        "--python-results",
        type=Path,
        help="Path to Python results JSON for comparison.",
    )
    parser.add_argument(
        "--write-template",
        type=Path,
        help="Write a template Python results JSON and exit.",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Return non-zero if any mapped contract is missing in python results.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    baseline = load_json(args.baseline)
    cases = load_contract_cases(args.contracts_dir)
    tolerance_profiles = load_json(args.contracts_dir / "tolerance_profiles.json")

    if args.write_template:
        template = build_template_output(baseline, cases)
        args.write_template.parent.mkdir(parents=True, exist_ok=True)
        args.write_template.write_text(json.dumps(template, indent=2))
        print(f"Wrote template: {args.write_template}")
        return 0

    if not args.python_results:
        print("error: --python-results is required unless --write-template is used", file=sys.stderr)
        return 2

    python_results = load_json(args.python_results)
    return run_compare(baseline, python_results, cases, tolerance_profiles, args.require_all)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
