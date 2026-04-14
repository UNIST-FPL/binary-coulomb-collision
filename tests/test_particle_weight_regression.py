from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose

from binary_collision import Collision, MultiSpeciesCollision, Particle
from tests._baseline import load_baseline
from utilities import particle_weight_three_species_case, particle_weight_two_species_case, simulate_relaxation, simulate_relaxation_multispecies


def test_per_particle_weight_two_species_matches_baseline():
    case = particle_weight_two_species_case()
    history = simulate_relaxation(
        deepcopy(case["species"][0]),
        deepcopy(case["species"][1]),
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )

    rng = np.random.default_rng(case["seed"])
    particle_a = Particle(**dict(case["species"][0], rng=rng))
    particle_b = Particle(**dict(case["species"][1], rng=rng))
    collision = Collision(particle_a, particle_b, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    baseline = load_baseline("particle_weight_2sp_small_v1.npz")
    assert list(history["species_names"]) == list(baseline["species_names"])
    assert_allclose(history["flow_histories"], baseline["flow_histories"], atol=2.0e-5)
    assert_allclose(history["flow_magnitudes"], baseline["flow_magnitudes"], atol=2.0e-5)
    assert_allclose(history["temperature_histories"], baseline["temperature_histories"], atol=2.0e-5)
    assert_allclose(history["time_axis"], baseline["time_axis"])
    assert_allclose(history["reference_flow"], baseline["reference_flow"])
    assert_allclose(particle_a.vel, baseline["vel_a"], atol=2.0e-5)
    assert_allclose(particle_b.vel, baseline["vel_b"], atol=2.0e-5)


def test_per_particle_weight_three_species_matches_baseline():
    case = particle_weight_three_species_case()
    history = simulate_relaxation_multispecies(
        deepcopy(case["species"]),
        iterations=case["iterations"],
        dt=case["dt"],
        rng=case["seed"],
    )

    rng = np.random.default_rng(case["seed"])
    species = [Particle(**dict(species_dict, rng=rng)) for species_dict in case["species"]]
    collision = MultiSpeciesCollision(species, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    baseline = load_baseline("multispecies_3sp_particle_weight_small_v1.npz")
    assert list(history["species_names"]) == list(baseline["species_names"])
    assert_allclose(history["flow_histories"], baseline["flow_histories"], atol=2.0e-5)
    assert_allclose(history["flow_magnitudes"], baseline["flow_magnitudes"], atol=2.0e-5)
    assert_allclose(history["temperature_histories"], baseline["temperature_histories"], atol=2.0e-5)
    assert_allclose(history["time_axis"], baseline["time_axis"])
    assert_allclose(history["reference_flow"], baseline["reference_flow"])
    for idx, part in enumerate(species):
        assert_allclose(part.vel, baseline[f"vel_{idx}"], atol=2.0e-5)
