import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.constants import e

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_collision import MultiSpeciesCollision, Particle  # noqa: E402
from tests._baseline import load_baseline  # noqa: E402
from utilities import (  # noqa: E402
    main_figure_cases,
    simulate_relaxation,
    simulate_relaxation_multispecies_ensemble,
    thirteen_particle_weight_relaxation_case,
    three_species_long_time_equilibrium_case,
)


def _max_abs_diff(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _max_rel_diff(actual, expected):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    denom = np.maximum(np.abs(expected), 1.0e-300)
    return float(np.max(np.abs(actual - expected) / denom))


def _checkpoint_view(values, sample_count=61, half_window=2):
    values = np.asarray(values)
    sample_centers = np.linspace(0, values.shape[-1] - 1, sample_count).round().astype(int)
    checkpoint_means = []
    for center in sample_centers:
        start = max(0, center - half_window)
        stop = min(values.shape[-1], center + half_window + 1)
        checkpoint_means.append(np.mean(values[..., start:stop], axis=-1))
    return np.stack(checkpoint_means, axis=-1)


def _figure_case_report(case):
    started = time.perf_counter()
    history = simulate_relaxation(
        case["particle_1"],
        case["particle_2"],
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )
    baseline = load_baseline(f"{case['figure']}_bundle_v1.npz")
    case_names = baseline["case_names"].tolist()
    case_index = case_names.index(case["name"])

    flow_actual = history["flow_magnitudes"]
    flow_expected = baseline["flow_magnitudes"][case_index]
    temp_actual = history["temperature_histories"]
    temp_expected = baseline["temperature_histories"][case_index]

    return {
        "name": case["name"],
        "figure": case["figure"],
        "variant": case["variant"],
        "iterations": int(case["iterations"]),
        "dt": float(case["dt"]),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "flow_pointwise_max_rel": _max_rel_diff(flow_actual, flow_expected),
        "temperature_pointwise_max_rel": _max_rel_diff(temp_actual, temp_expected),
        "flow_checkpoint_max_rel": _max_rel_diff(
            _checkpoint_view(flow_actual),
            _checkpoint_view(flow_expected),
        ),
        "temperature_checkpoint_max_rel": _max_rel_diff(
            _checkpoint_view(temp_actual),
            _checkpoint_view(temp_expected),
        ),
        "reference_flow_max_abs": _max_abs_diff(
            history["reference_flow"],
            baseline["reference_flows"][case_index],
        ),
        "time_axis_max_abs": _max_abs_diff(
            history["time_axis"],
            baseline["time_axes"][case_index],
        ),
    }


def _instantiate_species(case):
    rng = np.random.default_rng(case["seed"])
    species = []
    for species_dict in case["species"]:
        particle_kwargs = dict(species_dict)
        particle_kwargs["rng"] = rng
        species.append(Particle(**particle_kwargs))
    return rng, species


def _system_invariants(species):
    total_momentum = np.zeros(3)
    total_energy = 0.0
    total_mass = 0.0
    total_real_particles = 0.0
    for part in species:
        total_momentum += part.mass * part.weight * part.vel.sum(axis=0)
        total_energy += 0.5 * part.mass * part.weight * np.sum(part.vel ** 2)
        total_mass += part.mass * part.weight * part.Nmarker
        total_real_particles += part.weight * part.Nmarker
    return total_momentum, total_energy, total_mass, total_real_particles


def _multispecies_equilibrium_report():
    started = time.perf_counter()
    case = three_species_long_time_equilibrium_case()
    rng, species = _instantiate_species(case)

    initial_flows = np.array([part.flow_actual for part in species])
    total_momentum, total_energy, total_mass, total_real_particles = _system_invariants(species)
    target_flow = total_momentum / total_mass
    thermal_energy = total_energy - 0.5 * total_mass * np.sum(target_flow ** 2)
    target_temperature = 2.0 * thermal_energy / (3.0 * e * total_real_particles)

    collision = MultiSpeciesCollision(species, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    final_flows = np.array([part.flow_actual for part in species])
    final_temperatures = np.array([part.temperature_actual for part in species])

    initial_max_flow_error = float(np.max(np.linalg.norm(initial_flows - target_flow, axis=1)))
    final_max_flow_error = float(np.max(np.linalg.norm(final_flows - target_flow, axis=1)))

    return {
        "iterations": int(case["iterations"]),
        "dt": float(case["dt"]),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "final_max_flow_error": final_max_flow_error,
        "flow_reduction_ratio": final_max_flow_error / initial_max_flow_error,
        "final_temperature_spread": float(np.max(final_temperatures) - np.min(final_temperatures)),
        "max_temperature_error_to_target": float(np.max(np.abs(final_temperatures - target_temperature))),
    }


def _particle_weight_report():
    started = time.perf_counter()
    case = thirteen_particle_weight_relaxation_case()
    history = simulate_relaxation_multispecies_ensemble(
        case["species"],
        iterations=case["iterations"],
        dt=case["dt"],
        base_seed=case["seed"],
        ensemble_size=12,
    )

    temperature_histories = history["temperature_histories_mean"]
    flow_magnitudes = history["flow_magnitudes_mean"] / history["reference_flow"]
    temperature_spread = np.max(temperature_histories, axis=0) - np.min(temperature_histories, axis=0)
    flow_spread = np.max(flow_magnitudes, axis=0) - np.min(flow_magnitudes, axis=0)

    window = 20
    mid_start = len(temperature_spread) // 2 - window // 2

    start_temp_spread = float(np.mean(temperature_spread[:window]))
    mid_temp_spread = float(np.mean(temperature_spread[mid_start:mid_start + window]))
    end_temp_spread = float(np.mean(temperature_spread[-window:]))

    start_flow_spread = float(np.mean(flow_spread[:window]))
    mid_flow_spread = float(np.mean(flow_spread[mid_start:mid_start + window]))
    end_flow_spread = float(np.mean(flow_spread[-window:]))

    return {
        "iterations": int(case["iterations"]),
        "dt": float(case["dt"]),
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "mid_temperature_ratio": mid_temp_spread / start_temp_spread,
        "end_temperature_ratio": end_temp_spread / start_temp_spread,
        "mid_flow_ratio": mid_flow_spread / start_flow_spread,
        "end_flow_ratio": end_flow_spread / start_flow_spread,
    }


def _markdown_report(report):
    lines = [
        "# Verification Drift Report",
        "",
        f"- Generated at: `{report['generated_at_utc']}`",
        "",
    ]

    if report["figure_cases"] is not None:
        lines.extend(
            [
                "## Figure Cases",
                "",
                "| Case | Runtime (s) | Flow max rel | Flow checkpoint rel | Temp max rel | Temp checkpoint rel |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in report["figure_cases"]:
            lines.append(
                "| {name} | {runtime_seconds:.3f} | {flow_pointwise_max_rel:.6e} | "
                "{flow_checkpoint_max_rel:.6e} | {temperature_pointwise_max_rel:.6e} | "
                "{temperature_checkpoint_max_rel:.6e} |".format(**case)
            )
        lines.append("")

    if report["multispecies_equilibrium"] is not None:
        entry = report["multispecies_equilibrium"]
        lines.extend(
            [
                "## Multi-Species Equilibrium",
                "",
                f"- Runtime (s): `{entry['runtime_seconds']:.3f}`",
                f"- Final max flow error: `{entry['final_max_flow_error']:.6e}`",
                f"- Flow reduction ratio: `{entry['flow_reduction_ratio']:.6e}`",
                f"- Final temperature spread: `{entry['final_temperature_spread']:.6e}`",
                f"- Max temperature error to target: `{entry['max_temperature_error_to_target']:.6e}`",
                "",
            ]
        )

    if report["particle_weight_verification"] is not None:
        entry = report["particle_weight_verification"]
        lines.extend(
            [
                "## Particle-Weight Relaxation",
                "",
                f"- Runtime (s): `{entry['runtime_seconds']:.3f}`",
                f"- Mid temperature ratio: `{entry['mid_temperature_ratio']:.6e}`",
                f"- End temperature ratio: `{entry['end_temperature_ratio']:.6e}`",
                f"- Mid flow ratio: `{entry['mid_flow_ratio']:.6e}`",
                f"- End flow ratio: `{entry['end_flow_ratio']:.6e}`",
                "",
            ]
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a drift report for verification-scale cases.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "verification" / "drift-report.json",
        help="Path to write the machine-readable JSON report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "verification" / "drift-report.md",
        help="Path to write the Markdown summary.",
    )
    parser.add_argument(
        "--figure-cases",
        nargs="*",
        default=None,
        help="Optional subset of figure case names to evaluate.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip the figure-case drift report.",
    )
    parser.add_argument(
        "--skip-multispecies",
        action="store_true",
        help="Skip the multi-species equilibrium report.",
    )
    parser.add_argument(
        "--skip-particle-weight",
        action="store_true",
        help="Skip the particle-weight verification report.",
    )
    args = parser.parse_args()

    requested_cases = set(args.figure_cases) if args.figure_cases else None
    figure_cases = None
    if not args.skip_figures:
        figure_cases = []
        for case in main_figure_cases():
            if requested_cases is not None and case["name"] not in requested_cases:
                continue
            figure_cases.append(_figure_case_report(case))

    multispecies_equilibrium = None if args.skip_multispecies else _multispecies_equilibrium_report()
    particle_weight_verification = None if args.skip_particle_weight else _particle_weight_report()

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_cases": figure_cases,
        "multispecies_equilibrium": multispecies_equilibrium,
        "particle_weight_verification": particle_weight_verification,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    args.output_md.write_text(_markdown_report(report))

    print(args.output_md.read_text())


if __name__ == "__main__":
    main()
