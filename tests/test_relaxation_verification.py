import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import e, physical_constants

from tests._baseline import load_baseline
from utilities import simulate_relaxation


@pytest.mark.verification
def test_reduced_relaxation_history_matches_baseline():
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
    baseline = load_baseline("relaxation_reduced_v1.npz")

    assert_allclose(history["flow_magnitudes"], baseline["flow_magnitudes"])
    assert_allclose(history["temperature_histories"], baseline["temperature_histories"])
    assert_allclose(history["reference_flow"], baseline["reference_flow"])
