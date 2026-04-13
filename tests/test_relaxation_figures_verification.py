import pytest
from numpy.testing import assert_allclose

from tests._baseline import load_baseline
from utilities import main_figure_case_bundles, main_figure_cases, simulate_relaxation


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

    assert_allclose(history["flow_magnitudes"], baseline["flow_magnitudes"][case_index])
    assert_allclose(history["temperature_histories"], baseline["temperature_histories"][case_index])
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
