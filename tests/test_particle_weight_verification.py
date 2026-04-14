import numpy as np
import pytest

from utilities import (
    simulate_relaxation_multispecies_ensemble,
    thirteen_particle_weight_relaxation_case,
)


@pytest.mark.verification
def test_particle_weight_thirteen_particle_ensemble_relaxation_reduces_spread():
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

    start_temp_spread = np.mean(temperature_spread[:window])
    mid_temp_spread = np.mean(temperature_spread[mid_start:mid_start + window])
    end_temp_spread = np.mean(temperature_spread[-window:])

    start_flow_spread = np.mean(flow_spread[:window])
    mid_flow_spread = np.mean(flow_spread[mid_start:mid_start + window])
    end_flow_spread = np.mean(flow_spread[-window:])

    assert mid_temp_spread < start_temp_spread * 0.85
    assert end_temp_spread < start_temp_spread * 0.65
    assert mid_flow_spread < start_flow_spread * 0.85
    assert end_flow_spread < start_flow_spread * 0.60
