import numpy as np
import pytest
from scipy.constants import e

from binary_collision import MultiSpeciesCollision, Particle
from utilities import three_species_long_time_equilibrium_case


def _instantiate_species(case):
    rng = np.random.default_rng(case["seed"])
    species = []
    for species_dict in case["species"]:
        particle_kwargs = dict(species_dict)
        particle_kwargs["rng"] = rng
        species.append(Particle(**particle_kwargs))
    return rng, species


def _system_invariants(species):
    total_momentum = np.zeros(3)
    total_energy = 0.0
    total_mass = 0.0
    total_real_particles = 0.0
    for part in species:
        total_momentum += part.mass * part.weight * part.vel.sum(axis=0)
        total_energy += 0.5 * part.mass * part.weight * np.sum(part.vel ** 2)
        total_mass += part.mass * part.weight * part.Nmarker
        total_real_particles += part.weight * part.Nmarker
    return total_momentum, total_energy, total_mass, total_real_particles


@pytest.mark.verification
def test_three_species_long_time_relaxation_approaches_common_equilibrium():
    case = three_species_long_time_equilibrium_case()
    rng, species = _instantiate_species(case)

    initial_flows = np.array([part.flow_actual for part in species])
    _, total_energy, total_mass, total_real_particles = _system_invariants(species)
    total_momentum, _, _, _ = _system_invariants(species)

    target_flow = total_momentum / total_mass
    thermal_energy = total_energy - 0.5 * total_mass * np.sum(target_flow ** 2)
    target_temperature = 2.0 * thermal_energy / (3.0 * e * total_real_particles)

    collision = MultiSpeciesCollision(species, dtp=case["dt"], rng=rng)
    for _ in range(case["iterations"]):
        collision.run()

    final_flows = np.array([part.flow_actual for part in species])
    final_temperatures = np.array([part.temperature_actual for part in species])

    initial_max_flow_error = np.max(np.linalg.norm(initial_flows - target_flow, axis=1))
    final_max_flow_error = np.max(np.linalg.norm(final_flows - target_flow, axis=1))

    assert final_max_flow_error < 8.0e3
    assert final_max_flow_error < initial_max_flow_error * 0.25
    assert np.max(final_temperatures) - np.min(final_temperatures) < 3.0
    assert np.max(np.abs(final_temperatures - target_temperature)) < 2.0
