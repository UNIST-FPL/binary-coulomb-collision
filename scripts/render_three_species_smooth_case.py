import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utilities import simulate_relaxation_multispecies, three_species_smooth_relaxation_case


def main() -> None:
    case = three_species_smooth_relaxation_case()
    history = simulate_relaxation_multispecies(
        case["species"],
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )

    output_dir = PROJECT_ROOT / "artifacts" / "multispecies_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "three_species_smooth_relaxation.png"

    time_axis = np.arange(case["iterations"]) * case["dt"] * 1.0e5
    flow_mag = history["flow_magnitudes"]
    temp_hist = history["temperature_histories"]
    species_names = history["species_names"]
    ref_flow = history["reference_flow"]
    colors = ["#0f172a", "#b91c1c", "#0f766e"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for idx, (name, color) in enumerate(zip(species_names, colors)):
        axes[0].plot(time_axis, flow_mag[idx] / ref_flow, color=color, lw=1.8, label=f"{name} flow")
    axes[0].set_ylabel("Normalized Flow Magnitude")
    axes[0].set_title("Smooth Three-Species Relaxation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)

    for idx, (name, color) in enumerate(zip(species_names, colors)):
        axes[1].plot(time_axis, temp_hist[idx], color=color, lw=1.8, label=f"{name} temperature")
    axes[1].set_xlabel(r"Time [$10^{-5}$ s]")
    axes[1].set_ylabel("Temperature [eV]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    print(output_path)
    print("final_flow_magnitudes", flow_mag[:, -1].tolist())
    print("final_temperatures", temp_hist[:, -1].tolist())


if __name__ == "__main__":
    main()
