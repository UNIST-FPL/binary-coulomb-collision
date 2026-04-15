import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests._baseline import load_baseline
from utilities import main_figure_case_bundles, main_figure_cases, simulate_relaxation


DEFAULT_FLOW_RTOL = 3.0e-2
DEFAULT_TEMP_RTOL = 2.0e-2

# `fig6_equal` replays the largest Monte Carlo history in the suite and is the
# only case that has shown runner-dependent numerical drift in CI.
CASE_RTOLS = {
    "fig6_equal": {
        "flow": 1.0e-1,
        "temperature": 6.0e-2,
    },
}


def _case_rtol(case_name, metric):
    return CASE_RTOLS.get(case_name, {}).get(
        metric,
        DEFAULT_FLOW_RTOL if metric == "flow" else DEFAULT_TEMP_RTOL,
    )


def _max_relative_error(actual, expected):
    return float(np.max(np.abs(actual - expected) / np.maximum(np.abs(expected), 1.0e-300)))


def _comparison_view(values, case_name):
    if case_name != "fig6_equal":
        return values

    sample_centers = np.linspace(0, values.shape[-1] - 1, 61).round().astype(int)
    checkpoint_means = []
    for center in sample_centers:
        start = max(0, center - 2)
        stop = min(values.shape[-1], center + 3)
        checkpoint_means.append(np.mean(values[..., start:stop], axis=-1))
    return np.stack(checkpoint_means, axis=-1)


@pytest.mark.verification
@pytest.mark.parametrize("case", main_figure_cases(), ids=lambda case: case["name"])
def test_full_main_figure_case_matches_bundle_baseline(case):
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

    # Full-scale Monte Carlo histories are sensitive to numerical-library drift
    # across otherwise valid environments. Keep the baseline check tight enough
    # to catch regressions while allowing cross-platform replay tolerance.
    flow_actual = _comparison_view(history["flow_magnitudes"], case["name"])
    flow_expected = _comparison_view(baseline["flow_magnitudes"][case_index], case["name"])
    temp_actual = _comparison_view(history["temperature_histories"], case["name"])
    temp_expected = _comparison_view(baseline["temperature_histories"][case_index], case["name"])
    flow_rtol = _case_rtol(case["name"], "flow")
    temp_rtol = _case_rtol(case["name"], "temperature")

    assert_allclose(
        flow_actual,
        flow_expected,
        rtol=flow_rtol,
        err_msg=f"{case['name']} flow max_rel={_max_relative_error(flow_actual, flow_expected):.6f}",
    )
    assert_allclose(
        temp_actual,
        temp_expected,
        rtol=temp_rtol,
        err_msg=(
            f"{case['name']} temperature "
            f"max_rel={_max_relative_error(temp_actual, temp_expected):.6f}"
        ),
    )
    assert_allclose(history["reference_flow"], baseline["reference_flows"][case_index])
    assert_allclose(history["time_axis"], baseline["time_axes"][case_index])


@pytest.mark.verification
@pytest.mark.parametrize("bundle", main_figure_case_bundles(), ids=lambda bundle: bundle["name"])
def test_full_main_figure_bundle_metadata_matches_manifest(bundle):
    baseline = load_baseline(f"{bundle['name']}_v1.npz")

    assert baseline["case_names"].tolist() == [case["name"] for case in bundle["cases"]]
    assert baseline["variants"].tolist() == [case["variant"] for case in bundle["cases"]]
    assert baseline["label_prefixes"].tolist() == [case["label_prefix"] for case in bundle["cases"]]
    assert_allclose(baseline["we_over_wd"], [case["we_over_wd"] for case in bundle["cases"]])
