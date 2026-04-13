from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose

from binary_collision import MultiSpeciesCollision, Particle
from tests._baseline import load_baseline
from utilities import canonical_three_species_case, canonical_three_species_weighted_case, simulate_relaxation_multispecies


def _run_case(case):
    history = simulate_relaxation_multispecies(
        deepcopy(case["species"]),
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )

    rng = np.random.default_rng(case["seed"])
    particle_kwargs = [dict(spec) for spec in deepcopy(case["species"])]
    species = []
    for particle_kwargs in particle_kwargs:
        particle_kwargs["rng"] = rng
        species.append(Particle(**particle_kwargs))
    collision = MultiSpeciesCollision(species, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    return history, species


def _assert_matches_baseline(case, baseline_name):
    history, species = _run_case(case)
    baseline = load_baseline(baseline_name)

    assert_allclose(history["flow_histories"], baseline["flow_histories"])
    assert_allclose(history["flow_magnitudes"], baseline["flow_magnitudes"])
    assert_allclose(history["temperature_histories"], baseline["temperature_histories"])
    assert_allclose(history["time_axis"], baseline["time_axis"])
    assert_allclose(history["reference_flow"], baseline["reference_flow"])
    assert list(history["species_names"]) == list(baseline["species_names"])

    for idx, part in enumerate(species):
        assert_allclose(part.vel, baseline[f"vel_{idx}"])


def test_three_species_relaxation_matches_baseline():
    _assert_matches_baseline(canonical_three_species_case(), "multispecies_3sp_small_v1.npz")


def test_three_species_weighted_relaxation_matches_baseline():
    _assert_matches_baseline(canonical_three_species_weighted_case(), "multispecies_3sp_weighted_v1.npz")
