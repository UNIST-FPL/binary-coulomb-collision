import numpy as np
from numpy.testing import assert_allclose
from scipy.constants import e, physical_constants

from binary_collision import Collision, MultiSpeciesCollision, Particle


ATOMIC_MASS_UNIT = physical_constants["atomic mass constant"][0]


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


def test_multispecies_stage_catalog_matches_nanbu_multicomponent_recipe(multispecies_factory):
    _, species = multispecies_factory(seed=3)
    collision = MultiSpeciesCollision(species, dtp=5.0e-9, rng=123)

    stages = {(stage.kind, stage.species_i, stage.species_j) for stage in collision._stage_catalog}

    assert len(collision._stage_catalog) == 6
    assert stages == {
        ("unlike", 0, 1),
        ("unlike", 0, 2),
        ("unlike", 1, 2),
        ("like", 0, 0),
        ("like", 1, 1),
        ("like", 2, 2),
    }


def test_two_species_multispecies_orchestrator_reduces_to_original_collision(species_factory):
    _, ion, electron, collision = species_factory(seed=37, ion_markers=24, electron_markers=24)

    ion_clone = _clone_particle(ion)
    electron_clone = _clone_particle(electron)

    pair_collision = Collision(ion, electron, collision.dtp, rng=4242)
    multispecies_collision = MultiSpeciesCollision([ion_clone, electron_clone], collision.dtp, rng=4242)

    pair_collision.run()
    multispecies_collision.run()

    pair_ion, pair_electron = pair_collision.get_velocity()
    multi_ion, multi_electron = multispecies_collision.get_velocity()

    assert_allclose(pair_ion, multi_ion)
    assert_allclose(pair_electron, multi_electron)
    assert_allclose(ion.temperature_actual, ion_clone.temperature_actual)
    assert_allclose(electron.temperature_actual, electron_clone.temperature_actual)
