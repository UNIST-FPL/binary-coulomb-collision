import numpy as np
from scipy.constants import e, physical_constants

from binary_collision import Collision, Particle


ATOMIC_MASS_UNIT = physical_constants["atomic mass constant"][0]


def _totals(species):
    momentum = species.mass * species.weight * species.vel.sum(axis=0)
    energy = 0.5 * species.mass * species.weight * np.sum(species.vel ** 2)
    return momentum, energy


def _clone_particle(species):
    return Particle(
        name=species.name,
        charge=species.charge / e,
        mass=species.mass / ATOMIC_MASS_UNIT,
        density=species.density,
        weight=species.weight,
        Nmarker=species.Nmarker,
        vel=species.vel.copy(),
    )


def test_equal_weight_collision_conserves_total_momentum_and_energy(species_factory):
    _, ion, electron, collision = species_factory(seed=11, ion_markers=48, electron_markers=48)

    momentum_before = _totals(ion)[0] + _totals(electron)[0]
    energy_before = _totals(ion)[1] + _totals(electron)[1]

    collision.run()

    momentum_after = _totals(ion)[0] + _totals(electron)[0]
    energy_after = _totals(ion)[1] + _totals(electron)[1]

    momentum_rel_error = np.linalg.norm(momentum_after - momentum_before) / (np.linalg.norm(momentum_before) + 1.0e-30)
    energy_rel_error = abs(energy_after - energy_before) / energy_before

    assert momentum_rel_error < 1.0e-12
    assert energy_rel_error < 1.0e-12


def test_collision_preserves_original_input_velocity_order(species_factory):
    _, ion, electron, collision = species_factory(seed=17, ion_markers=64, electron_markers=16)

    vel_first, vel_second = collision.get_velocity()
    assert vel_first.shape == ion.vel.shape
    assert vel_second.shape == electron.vel.shape

    collision.run()

    vel_first, vel_second = collision.get_velocity()
    assert vel_first.shape == ion.vel.shape
    assert vel_second.shape == electron.vel.shape


def test_collision_is_invariant_to_input_order_for_same_initial_state(species_factory):
    _, ion, electron, collision_ab = species_factory(seed=29, ion_markers=32, electron_markers=16)

    ion_clone = _clone_particle(ion)
    electron_clone = _clone_particle(electron)

    collision_ab = Collision(ion, electron, collision_ab.dtp, rng=999)
    collision_ba = Collision(electron_clone, ion_clone, collision_ab.dtp, rng=999)

    collision_ab.run()
    collision_ba.run()

    vel_ion_ab, vel_electron_ab = collision_ab.get_velocity()
    vel_electron_ba, vel_ion_ba = collision_ba.get_velocity()

    assert np.allclose(vel_ion_ab, vel_ion_ba)
    assert np.allclose(vel_electron_ab, vel_electron_ba)


def test_small_marker_counts_skip_like_collision_without_raising():
    ion = Particle(
        name="A",
        charge=1,
        mass=2.0,
        density=1.0e20,
        flow=0.0,
        temperature=10.0,
        Nmarker=2,
        weight=1.0e20 / 2,
        rng=0,
    )
    electron = Particle(
        name="B",
        charge=1,
        mass=2.0,
        density=1.0e20,
        flow=1.0e3,
        temperature=20.0,
        Nmarker=2,
        weight=1.0e20 / 2,
        rng=0,
    )

    collision = Collision(ion, electron, 1.0e-8, rng=0)
    collision.run()

    assert ion.vel.shape == (2, 3)
    assert electron.vel.shape == (2, 3)


def test_zero_temperature_relative_state_remains_stable():
    vel = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    ion = Particle(name="A", charge=1, mass=2.0, density=1.0e20, Nmarker=2, weight=1.0e20 / 2, vel=vel.copy())
    electron = Particle(name="B", charge=-1, mass=1.0, density=1.0e20, Nmarker=2, weight=1.0e20 / 2, vel=vel.copy())

    collision = Collision(ion, electron, 1.0e-8, rng=0)
    collision.run()
    vel_ion, vel_electron = collision.get_velocity()

    assert np.allclose(vel_ion, vel)
    assert np.allclose(vel_electron, vel)
