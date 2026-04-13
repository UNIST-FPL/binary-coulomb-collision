import numpy as np
from scipy.constants import physical_constants

from binary_collision import MultiSpeciesCollision, Particle


def _totals(species):
    total_momentum = np.zeros(3)
    total_energy = 0.0
    for part in species:
        total_momentum += part.mass * part.weight * part.vel.sum(axis=0)
        total_energy += 0.5 * part.mass * part.weight * np.sum(part.vel ** 2)
    return total_momentum, total_energy


def test_multispecies_equal_weight_collision_conserves_total_momentum_and_energy(multispecies_factory):
    _, species = multispecies_factory(seed=19)
    collision = MultiSpeciesCollision(species, dtp=5.0e-9, rng=909)

    momentum_before, energy_before = _totals(species)

    collision.run()

    momentum_after, energy_after = _totals(species)
    momentum_rel_error = np.linalg.norm(momentum_after - momentum_before) / (np.linalg.norm(momentum_before) + 1.0e-30)
    energy_rel_error = abs(energy_after - energy_before) / (energy_before + 1.0e-30)

    assert momentum_rel_error < 1.0e-12
    assert energy_rel_error < 1.0e-12


def test_multispecies_zero_relative_state_remains_stable():
    velocities = np.full((6, 3), 2.5e4)
    species = [
        Particle(name="e-", charge=-1, mass=physical_constants["electron mass in u"][0], density=6.0e20, Nmarker=6, weight=1.0e20, vel=velocities.copy()),
        Particle(name="D+", charge=1, mass=2.0141, density=6.0e20, Nmarker=6, weight=1.0e20, vel=velocities.copy()),
        Particle(name="He+", charge=1, mass=4.0026, density=6.0e20, Nmarker=6, weight=1.0e20, vel=velocities.copy()),
    ]

    collision = MultiSpeciesCollision(species, dtp=1.0e-8, rng=0)
    collision.run()

    for part in species:
        assert np.allclose(part.vel, velocities)


def test_multispecies_get_velocity_preserves_input_order(multispecies_factory):
    _, species = multispecies_factory(seed=21, markers=(12, 16, 20))
    collision = MultiSpeciesCollision(species, dtp=5.0e-9, rng=7)

    velocities = collision.get_velocity()

    assert len(velocities) == 3
    assert velocities[0].shape == species[0].vel.shape
    assert velocities[1].shape == species[1].vel.shape
    assert velocities[2].shape == species[2].vel.shape
