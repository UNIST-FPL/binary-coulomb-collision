import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests._baseline import load_baseline  # noqa: E402
from utilities import (
    main_figure_case_bundles,
    plot_relaxation_history,
    reduced_main_figure_case_bundles,
)  # noqa: E402


def _history_from_bundle_case(bundle_baseline, case_index: int) -> dict:
    return {
        "species_names": np.array(["D+", "e-"], dtype=object),
        "flow_magnitudes": bundle_baseline["flow_magnitudes"][case_index],
        "temperature_histories": bundle_baseline["temperature_histories"][case_index],
        "time_axis": bundle_baseline["time_axes"][case_index],
        "reference_flow": bundle_baseline["reference_flows"][case_index],
    }


def render_bundle(bundle: dict, output_dir: Path) -> Path:
    baseline = load_baseline(f"{bundle['name']}_v1.npz")
    plt.close("all")

    for case_index, case in enumerate(bundle["cases"]):
        history = _history_from_bundle_case(baseline, case_index)
        plot_relaxation_history(
            history,
            hold=case_index > 0,
            label_prefix=case["label_prefix"],
        )

    plt.tight_layout()
    output_path = output_dir / f"{bundle['name']}_v1.png"
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close("all")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Render PNG figures from stored baseline bundles.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "baseline_figures",
        help="Directory to write rendered baseline PNG files into.",
    )
    parser.add_argument(
        "--figure-scale",
        choices=("full", "reduced"),
        default="full",
        help="Which bundle manifest to render. Defaults to the original full-size figure cases.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle_factory = (
        main_figure_case_bundles if args.figure_scale == "full" else reduced_main_figure_case_bundles
    )
    for bundle in bundle_factory():
        output_path = render_bundle(bundle, args.output_dir)
        print(output_path)


if __name__ == "__main__":
    main()
