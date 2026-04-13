import pathlib
import sys

import numpy as np
import pytest
from scipy.constants import e, physical_constants

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from binary_collision import Collision, Particle


@pytest.fixture
def species_factory():
    def factory(seed=0, ion_markers=64, electron_markers=64, ion_weight=None, electron_weight=None):
        rng = np.random.default_rng(seed)

        ion_weight = ion_weight if ion_weight is not None else 1.0e21 / ion_markers
        electron_weight = electron_weight if electron_weight is not None else 1.0e21 / electron_markers

        ion = Particle(
            name="D+",
            charge=1,
            mass=2.0141,
            density=1.0e21,
            flow=0.0,
            temperature=100.0,
            weight=ion_weight,
            Nmarker=ion_markers,
            rng=rng,
        )
        electron = Particle(
            name="e-",
            charge=-1,
            mass=physical_constants["electron mass in u"][0],
            density=1.0e21,
            flow=np.sqrt(
                1.0e3 * e
                / (physical_constants["electron mass in u"][0] * physical_constants["atomic mass constant"][0])
            ),
            temperature=1.0e3,
            weight=electron_weight,
            Nmarker=electron_markers,
            rng=rng,
        )
        collision = Collision(ion, electron, dtp=1.0e-7, rng=rng)
        return rng, ion, electron, collision

    return factory


@pytest.fixture
def multispecies_factory():
    def factory(
        seed=0,
        markers=(18, 18, 18),
        densities=(8.0e20, 8.0e20, 8.0e20),
        flows=(2.0e5, -1.0e5, 6.0e4),
        temperatures=(600.0, 120.0, 45.0),
    ):
        rng = np.random.default_rng(seed)
        species_defs = (
            ("e-", -1, physical_constants["electron mass in u"][0]),
            ("D+", 1, 2.0141),
            ("He+", 1, 4.0026),
        )

        species = []
        for (name, charge, mass), marker_count, density, flow, temperature in zip(
            species_defs,
            markers,
            densities,
            flows,
            temperatures,
        ):
            species.append(
                Particle(
                    name=name,
                    charge=charge,
                    mass=mass,
                    density=density,
                    flow=flow,
                    temperature=temperature,
                    weight=density / marker_count,
                    Nmarker=marker_count,
                    rng=rng,
                )
            )
        return rng, species

    return factory
