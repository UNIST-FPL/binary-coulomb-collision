import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import e

from binary_collision import Particle


def test_particle_computes_flow_and_temperature_from_velocity():
    vel = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    particle = Particle(name="X", charge=1, mass=2.0, density=1.0, Nmarker=3, vel=vel)

    expected_flow = vel.mean(axis=0)
    expected_temperature = (
        particle.mass
        / (3.0 * e)
        * ((vel ** 2).sum() / particle.Nmarker - np.square(expected_flow).sum())
    )

    assert_allclose(particle.flow_actual, expected_flow)
    assert_allclose(particle.temperature_actual, expected_temperature)


def test_particle_maxwellian_initialization_is_reproducible_with_seed():
    particle1 = Particle(
        name="X",
        charge=1,
        mass=2.0,
        density=1.0,
        flow=3.0,
        temperature=5.0,
        Nmarker=8,
        rng=123,
    )
    particle2 = Particle(
        name="X",
        charge=1,
        mass=2.0,
        density=1.0,
        flow=3.0,
        temperature=5.0,
        Nmarker=8,
        rng=123,
    )

    assert_allclose(particle1.vel, particle2.vel, rtol=0.0, atol=0.0)
