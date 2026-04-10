import pytest
from numpy.testing import assert_allclose

from tests._baseline import load_baseline
from tests._relaxation_cases import reduced_relaxation_cases
from utilities import simulate_relaxation


@pytest.mark.verification
@pytest.mark.parametrize("case", reduced_relaxation_cases(), ids=lambda case: case["name"])
def test_reduced_figure_style_relaxation_matches_baseline(case):
    history = simulate_relaxation(
        case["particle_1"],
        case["particle_2"],
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )
    baseline = load_baseline(f"{case['name']}_v1.npz")

    assert_allclose(history["flow_magnitudes"], baseline["flow_magnitudes"])
    assert_allclose(history["temperature_histories"], baseline["temperature_histories"])
    assert_allclose(history["reference_flow"], baseline["reference_flow"])
    assert_allclose(history["time_axis"], baseline["time_axis"])
