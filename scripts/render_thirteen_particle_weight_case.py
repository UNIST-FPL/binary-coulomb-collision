import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities import (
    simulate_relaxation_multispecies_ensemble,
    thirteen_particle_weight_relaxation_case,
)


def main():
    case = thirteen_particle_weight_relaxation_case()
    history = simulate_relaxation_multispecies_ensemble(
        case["species"],
        iterations=case["iterations"],
        dt=case["dt"],
        base_seed=case["seed"],
        ensemble_size=case["ensemble_size"],
    )

    output_dir = PROJECT_ROOT / "artifacts" / "multispecies_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "thirteen_particle_weight_relaxation.png"

    time_axis = history["time_axis"]
    species_names = history["species_names"]
    flow_mean = history["flow_magnitudes_mean"] / history["reference_flow"]
    flow_std = history["flow_magnitudes_std"] / history["reference_flow"]
    temperature_mean = history["temperature_histories_mean"]
    temperature_std = history["temperature_histories_std"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    colors = ["#1f4e79", "#b85c38", "#4d7c0f"]

    for species_idx, species_name in enumerate(species_names):
        axes[0].plot(
            time_axis,
            flow_mean[species_idx],
            color=colors[species_idx],
            linewidth=2.0,
            label=f"{species_name} flow",
        )
        axes[0].fill_between(
            time_axis,
            flow_mean[species_idx] - flow_std[species_idx],
            flow_mean[species_idx] + flow_std[species_idx],
            color=colors[species_idx],
            alpha=0.18,
        )
        axes[1].plot(
            time_axis,
            temperature_mean[species_idx],
            color=colors[species_idx],
            linewidth=2.0,
            label=f"{species_name} temperature",
        )
        axes[1].fill_between(
            time_axis,
            temperature_mean[species_idx] - temperature_std[species_idx],
            temperature_mean[species_idx] + temperature_std[species_idx],
            color=colors[species_idx],
            alpha=0.18,
        )

    axes[0].set_ylabel("Normalized Flow Magnitude")
    axes[0].set_title(
        f"13-Particle Per-Particle-Weight Relaxation (ensemble={history['ensemble_size']})"
    )
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_xlabel(r"Time [$10^{-5}$ s]")
    axes[1].set_ylabel("Temperature [eV]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    final_mean_temperatures = temperature_mean[:, -1]
    final_temperature_spread = float(np.max(final_mean_temperatures) - np.min(final_mean_temperatures))
    print(f"saved={output_path}")
    print(f"final_mean_temperatures={final_mean_temperatures.tolist()}")
    print(f"final_temperature_spread={final_temperature_spread}")


if __name__ == "__main__":
    main()
