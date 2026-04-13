import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.constants import e, physical_constants

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities import main_figure_cases, reduced_main_figure_cases  # noqa: E402


def _sample_velocities(particle: dict, seed: int) -> np.ndarray:
    flow = float(particle["flow"])
    temperature = float(particle["temperature"])
    mass_kg = float(particle["mass"]) * physical_constants["atomic mass constant"][0]
    flow_vec = np.array([flow / np.sqrt(3.0)] * 3, dtype=float)
    rng = np.random.default_rng(seed)
    return rng.normal(flow_vec, np.sqrt(temperature * e / mass_kg), (int(particle["Nmarker"]), 3))


def _write_case_payload(output_dir: Path, case: dict) -> Path:
    payload_path = output_dir / f"{case['name']}.npz"
    np.savez(
        payload_path,
        case_name=np.array(case["name"]),
        seed=np.array(case["seed"]),
        dt=np.array(case["dt"]),
        iterations=np.array(case["iterations"]),
        particle_1_name=np.array(case["particle_1"]["name"]),
        particle_1_charge=np.array(case["particle_1"]["charge"]),
        particle_1_mass=np.array(case["particle_1"]["mass"]),
        particle_1_density=np.array(case["particle_1"]["density"]),
        particle_1_weight=np.array(case["particle_1"]["weight"]),
        particle_1_nmarker=np.array(case["particle_1"]["Nmarker"]),
        particle_1_vel=_sample_velocities(case["particle_1"], case["seed"] + 11),
        particle_2_name=np.array(case["particle_2"]["name"]),
        particle_2_charge=np.array(case["particle_2"]["charge"]),
        particle_2_mass=np.array(case["particle_2"]["mass"]),
        particle_2_density=np.array(case["particle_2"]["density"]),
        particle_2_weight=np.array(case["particle_2"]["weight"]),
        particle_2_nmarker=np.array(case["particle_2"]["Nmarker"]),
        particle_2_vel=_sample_velocities(case["particle_2"], case["seed"] + 29),
    )
    return payload_path


def _run_runner(worktree: Path, payload_path: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree)
    runner = PROJECT_ROOT / "scripts" / "compatibility_case_runner.py"
    completed = subprocess.run(
        [sys.executable, str(runner), "--payload", str(payload_path)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _max_abs_diff(current, baseline) -> float:
    return float(np.max(np.abs(np.asarray(current) - np.asarray(baseline))))


def _case_report(case: dict, current_result: dict, main_result: dict, atol: float) -> dict:
    flow_diff = _max_abs_diff(current_result["flow_histories"], main_result["flow_histories"])
    temp_diff = _max_abs_diff(current_result["temperature_histories"], main_result["temperature_histories"])
    vel1_diff = _max_abs_diff(current_result["final_vel_1"], main_result["final_vel_1"])
    vel2_diff = _max_abs_diff(current_result["final_vel_2"], main_result["final_vel_2"])
    return {
        "case_name": case["name"],
        "figure": case["figure"],
        "variant": case["variant"],
        "compatible": max(flow_diff, temp_diff, vel1_diff, vel2_diff) <= atol,
        "max_abs_diff_flow_history": flow_diff,
        "max_abs_diff_temperature_history": temp_diff,
        "max_abs_diff_final_vel_1": vel1_diff,
        "max_abs_diff_final_vel_2": vel2_diff,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare current branch and main on canonical relaxation cases.")
    parser.add_argument(
        "--scale",
        choices=("reduced", "full"),
        default="reduced",
        help="Case scale to compare. 'reduced' is intended for regular merge-time checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "main_compatibility_report.json",
        help="Path to write the JSON compatibility report.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
        help="Absolute tolerance for compatibility checks.",
    )
    args = parser.parse_args()

    cases = reduced_main_figure_cases() if args.scale == "reduced" else main_figure_cases()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="bcc-main-compat-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        main_worktree = temp_dir / "main-worktree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(main_worktree), "main"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            reports = []
            payload_dir = temp_dir / "payloads"
            payload_dir.mkdir()
            for case in cases:
                payload_path = _write_case_payload(payload_dir, case)
                current_result = _run_runner(PROJECT_ROOT, payload_path)
                main_result = _run_runner(main_worktree, payload_path)
                reports.append(_case_report(case, current_result, main_result, args.atol))
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(main_worktree)],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if main_worktree.exists():
                shutil.rmtree(main_worktree, ignore_errors=True)

    summary = {
        "scale": args.scale,
        "compatible": all(report["compatible"] for report in reports),
        "cases": reports,
    }
    args.output.write_text(json.dumps(summary, indent=2))

    for report in reports:
        status = "OK" if report["compatible"] else "DIFF"
        print(
            f"{status} {report['case_name']}: "
            f"flow={report['max_abs_diff_flow_history']:.3e}, "
            f"temp={report['max_abs_diff_temperature_history']:.3e}, "
            f"vel1={report['max_abs_diff_final_vel_1']:.3e}, "
            f"vel2={report['max_abs_diff_final_vel_2']:.3e}"
        )

    if not summary["compatible"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
