import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.constants import e, physical_constants

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_collision import Collision, Particle  # noqa: E402
from utilities import main_figure_case_bundles, reduced_main_figure_case_bundles, simulate_relaxation  # noqa: E402


def generate_collision_baseline(output_dir: Path):
    vel_a = np.array(
        [
            [1.2e5, -2.0e4, 8.0e4],
            [-5.0e4, 9.0e4, 3.0e4],
            [7.5e4, 1.1e5, -4.0e4],
            [2.0e4, -6.0e4, 5.5e4],
            [9.5e4, 3.5e4, -7.0e4],
            [-8.0e4, -4.5e4, 6.5e4],
        ]
    )
    vel_b = np.array(
        [
            [-9.0e4, 4.0e4, 1.5e5],
            [6.0e4, -7.5e4, -3.5e4],
            [1.1e5, 8.5e4, 2.5e4],
            [-3.0e4, -9.5e4, 4.5e4],
        ]
    )

    species_a = Particle(
        name="A",
        charge=1,
        mass=2.0,
        density=1.0e20,
        weight=1.0e20 / 6,
        Nmarker=6,
        vel=vel_a,
    )
    species_b = Particle(
        name="B",
        charge=-1,
        mass=1.0,
        density=1.0e20,
        weight=1.0e20 / 4,
        Nmarker=4,
        vel=vel_b,
    )

    collision = Collision(species_a, species_b, dtp=5.0e-9, rng=2024)
    collision.run()
    actual_a, actual_b = collision.get_velocity()

    np.savez(
        output_dir / "collision_unequal_weight_small_v1.npz",
        vel_a=actual_a,
        vel_b=actual_b,
        temp_a=np.array(species_a.temperature_actual),
        temp_b=np.array(species_b.temperature_actual),
    )


def generate_relaxation_baseline(output_dir: Path):
    d_params = {
        "name": "D+",
        "charge": 1,
        "mass": 5 * physical_constants["electron mass in u"][0],
        "density": 1.0e21,
        "flow": 0.0,
        "temperature": 100.0,
        "Nmarker": 24,
        "weight": 1.0e21 / 24,
    }
    e_params = {
        "name": "e-",
        "charge": -1,
        "mass": physical_constants["electron mass in u"][0],
        "density": 1.0e21,
        "flow": np.sqrt(
            1.0e3 * e
            / (physical_constants["electron mass in u"][0] * physical_constants["atomic mass constant"][0])
        ),
        "temperature": 1.0e3,
        "Nmarker": 24,
        "weight": 1.0e21 / 24,
    }

    history = simulate_relaxation(d_params, e_params, iterations=6, dt=1.0e-7, rng=1234)
    np.savez(
        output_dir / "relaxation_reduced_v1.npz",
        flow_magnitudes=history["flow_magnitudes"],
        temperature_histories=history["temperature_histories"],
        reference_flow=history["reference_flow"],
    )


def generate_relaxation_figure_baselines(output_dir: Path, figure_scale: str):
    bundle_factory = (
        main_figure_case_bundles if figure_scale == "full" else reduced_main_figure_case_bundles
    )
    for bundle in bundle_factory():
        histories = []
        for case in bundle["cases"]:
            histories.append(
                simulate_relaxation(
                    case["particle_1"],
                    case["particle_2"],
                    iterations=case["iterations"],
                    dt=case["dt"],
                    rng=case["seed"],
                )
            )

        np.savez(
            output_dir / f"{bundle['name']}_v1.npz",
            case_names=np.array([case["name"] for case in bundle["cases"]], dtype=str),
            variants=np.array([case["variant"] for case in bundle["cases"]], dtype=str),
            label_prefixes=np.array([case["label_prefix"] for case in bundle["cases"]], dtype=str),
            we_over_wd=np.array([case["we_over_wd"] for case in bundle["cases"]], dtype=float),
            flow_magnitudes=np.stack([history["flow_magnitudes"] for history in histories]),
            temperature_histories=np.stack([history["temperature_histories"] for history in histories]),
            reference_flows=np.array([float(history["reference_flow"]) for history in histories], dtype=float),
            time_axes=np.stack([history["time_axis"] for history in histories]),
        )


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic regression baselines for tests.")
    parser.add_argument(
        "--output-dir",
        default=PROJECT_ROOT / "tests" / "data",
        type=Path,
        help="Directory to write .npz baseline files into.",
    )
    parser.add_argument(
        "--figure-scale",
        choices=("full", "reduced"),
        default="full",
        help="Scale for Fig. 4-6 bundle baselines. Defaults to the original full-size cases from main.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_collision_baseline(output_dir)
    generate_relaxation_baseline(output_dir)
    generate_relaxation_figure_baselines(output_dir, figure_scale=args.figure_scale)
    print(f"Baselines written to {output_dir}")


if __name__ == "__main__":
    main()
