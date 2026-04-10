import numpy as np
from numpy.testing import assert_allclose

from binary_collision import Collision, Particle
from tests._baseline import load_baseline


def test_unequal_weight_one_step_collision_matches_baseline():
    vel_a = np.array(
        [
            [1.2e5, -2.0e4, 8.0e4],
            [-5.0e4, 9.0e4, 3.0e4],
            [7.5e4, 1.1e5, -4.0e4],
            [2.0e4, -6.0e4, 5.5e4],
            [9.5e4, 3.5e4, -7.0e4],
            [-8.0e4, -4.5e4, 6.5e4],
        ]
    )
    vel_b = np.array(
        [
            [-9.0e4, 4.0e4, 1.5e5],
            [6.0e4, -7.5e4, -3.5e4],
            [1.1e5, 8.5e4, 2.5e4],
            [-3.0e4, -9.5e4, 4.5e4],
        ]
    )

    species_a = Particle(
        name="A",
        charge=1,
        mass=2.0,
        density=1.0e20,
        weight=1.0e20 / 6,
        Nmarker=6,
        vel=vel_a,
    )
    species_b = Particle(
        name="B",
        charge=-1,
        mass=1.0,
        density=1.0e20,
        weight=1.0e20 / 4,
        Nmarker=4,
        vel=vel_b,
    )

    collision = Collision(species_a, species_b, dtp=5.0e-9, rng=2024)
    collision.run()

    actual_a, actual_b = collision.get_velocity()
    baseline = load_baseline("collision_unequal_weight_small_v1.npz")

    assert_allclose(actual_a, baseline["vel_a"])
    assert_allclose(actual_b, baseline["vel_b"])
    assert_allclose(species_a.temperature_actual, baseline["temp_a"])
    assert_allclose(species_b.temperature_actual, baseline["temp_b"])
