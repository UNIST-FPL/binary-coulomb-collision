from copy import deepcopy
from scipy.constants import physical_constants
import numpy as np


ELECTRON_MASS_U = physical_constants["electron mass in u"][0]


def canonical_three_species_case():
    densities = (9.0e20, 9.0e20, 9.0e20)
    markers = (18, 18, 18)
    species = [
        {
            "name": "e-",
            "charge": -1,
            "mass": ELECTRON_MASS_U,
            "density": densities[0],
            "flow": 3.0e5,
            "temperature": 650.0,
            "Nmarker": markers[0],
            "weight": densities[0] / markers[0],
        },
        {
            "name": "D+",
            "charge": 1,
            "mass": 2.0141,
            "density": densities[1],
            "flow": -1.6e5,
            "temperature": 130.0,
            "Nmarker": markers[1],
            "weight": densities[1] / markers[1],
        },
        {
            "name": "He+",
            "charge": 1,
            "mass": 4.0026,
            "density": densities[2],
            "flow": 8.0e4,
            "temperature": 45.0,
            "Nmarker": markers[2],
            "weight": densities[2] / markers[2],
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 10,
        "dt": 5.0e-9,
        "seed": 314159,
    }


def canonical_three_species_weighted_case():
    densities = (9.0e20, 9.0e20, 9.0e20)
    markers = (24, 12, 8)
    species = [
        {
            "name": "e-",
            "charge": -1,
            "mass": ELECTRON_MASS_U,
            "density": densities[0],
            "flow": 3.0e5,
            "temperature": 650.0,
            "Nmarker": markers[0],
            "weight": densities[0] / markers[0],
        },
        {
            "name": "D+",
            "charge": 1,
            "mass": 2.0141,
            "density": densities[1],
            "flow": -1.6e5,
            "temperature": 130.0,
            "Nmarker": markers[1],
            "weight": densities[1] / markers[1],
        },
        {
            "name": "He+",
            "charge": 1,
            "mass": 4.0026,
            "density": densities[2],
            "flow": 8.0e4,
            "temperature": 45.0,
            "Nmarker": markers[2],
            "weight": densities[2] / markers[2],
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 10,
        "dt": 5.0e-9,
        "seed": 271828,
    }


def three_species_long_time_equilibrium_case():
    densities = (5.0e23, 5.0e23, 5.0e23)
    markers = (48, 48, 48)
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 1.5,
            "density": densities[0],
            "flow": 4.0e4,
            "temperature": 30.0,
            "Nmarker": markers[0],
            "weight": densities[0] / markers[0],
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 2.5,
            "density": densities[1],
            "flow": -2.0e4,
            "temperature": 15.0,
            "Nmarker": markers[1],
            "weight": densities[1] / markers[1],
        },
        {
            "name": "C2+",
            "charge": 2,
            "mass": 4.0,
            "density": densities[2],
            "flow": 1.0e4,
            "temperature": 9.0,
            "Nmarker": markers[2],
            "weight": densities[2] / markers[2],
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 700,
        "dt": 3.0e-8,
        "seed": 123,
    }


def three_species_smooth_relaxation_case():
    density = 5.0e23
    markers = 100000
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 1.5,
            "density": density,
            "flow": 4.0e4,
            "temperature": 30.0,
            "Nmarker": markers,
            "weight": density / markers,
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 2.5,
            "density": density,
            "flow": -2.0e4,
            "temperature": 15.0,
            "Nmarker": markers,
            "weight": density / markers,
        },
        {
            "name": "C2+",
            "charge": 2,
            "mass": 4.0,
            "density": density,
            "flow": 1.0e4,
            "temperature": 9.0,
            "Nmarker": markers,
            "weight": density / markers,
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 151,
        "dt": 3.0e-8,
        "seed": 123,
    }


def three_species_slower_relaxation_case():
    density = 1.0e21
    markers = 100000
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 1.5,
            "density": density,
            "flow": 8.0e4,
            "temperature": 60.0,
            "Nmarker": markers,
            "weight": density / markers,
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 2.5,
            "density": density,
            "flow": -4.0e4,
            "temperature": 25.0,
            "Nmarker": markers,
            "weight": density / markers,
        },
        {
            "name": "C+",
            "charge": 1,
            "mass": 4.0,
            "density": density,
            "flow": 2.0e4,
            "temperature": 12.0,
            "Nmarker": markers,
            "weight": density / markers,
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 151,
        "dt": 3.0e-8,
        "seed": 123,
    }


def particle_weight_two_species_case():
    weight_a = np.array([1.1, 1.7, 2.6, 3.8, 5.5, 8.0]) * 1.0e18
    weight_b = np.array([1.4, 2.3, 3.7]) * 1.0e18
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 2.0,
            "density": float(np.sum(weight_a)),
            "flow": 2.2e4,
            "temperature": 24.0,
            "Nmarker": weight_a.size,
            "weight": weight_a,
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 3.5,
            "density": float(np.sum(weight_b)),
            "flow": -1.1e4,
            "temperature": 12.0,
            "Nmarker": weight_b.size,
            "weight": weight_b,
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 1,
        "dt": 1.0e-8,
        "seed": 424242,
    }


def particle_weight_three_species_case():
    weight_a = np.array([1.0, 1.6, 2.5, 3.1, 4.4, 6.2]) * 1.0e18
    weight_b = np.array([1.2, 1.9, 2.2, 3.7, 5.1, 7.3]) * 1.0e18
    weight_c = np.array([1.1, 1.5, 2.8, 3.4, 5.6, 8.4]) * 1.0e18
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 1.8,
            "density": float(np.sum(weight_a)),
            "flow": 2.5e4,
            "temperature": 28.0,
            "Nmarker": weight_a.size,
            "weight": weight_a,
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 2.6,
            "density": float(np.sum(weight_b)),
            "flow": -1.4e4,
            "temperature": 18.0,
            "Nmarker": weight_b.size,
            "weight": weight_b,
        },
        {
            "name": "C+",
            "charge": 1,
            "mass": 4.2,
            "density": float(np.sum(weight_c)),
            "flow": 7.0e3,
            "temperature": 11.0,
            "Nmarker": weight_c.size,
            "weight": weight_c,
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 8,
        "dt": 8.0e-9,
        "seed": 515151,
    }


def particle_weight_three_species_equilibrium_case():
    multipliers = np.array([1.0, 1.2, 1.5, 1.9, 2.4, 3.0, 3.7, 4.5, 5.4, 6.4, 7.5, 8.7, 10.0, 11.4, 12.9, 14.5, 16.2, 18.0, 19.9, 21.9, 24.0, 26.2, 28.5, 30.9])
    weight_a = multipliers * 2.0e17
    weight_b = multipliers[::-1] * 2.0e17
    weight_c = np.roll(multipliers, 5) * 2.0e17
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 1.6,
            "density": float(np.sum(weight_a)),
            "flow": 6.0e4,
            "temperature": 52.0,
            "Nmarker": weight_a.size,
            "weight": weight_a,
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 2.4,
            "density": float(np.sum(weight_b)),
            "flow": -3.0e4,
            "temperature": 24.0,
            "Nmarker": weight_b.size,
            "weight": weight_b,
        },
        {
            "name": "C+",
            "charge": 1,
            "mass": 3.8,
            "density": float(np.sum(weight_c)),
            "flow": 1.5e4,
            "temperature": 13.0,
            "Nmarker": weight_c.size,
            "weight": weight_c,
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 400,
        "dt": 2.0e-8,
        "seed": 9090,
    }


def thirteen_particle_weight_relaxation_case():
    weight_a = np.array([1.03, 1.37, 1.91, 2.29, 2.83, 3.47, 4.19, 5.02, 5.91, 6.88, 7.97, 9.11, 10.37]) * 1.0e18
    weight_b = np.array([1.21, 1.49, 1.88, 2.36, 2.95, 3.61, 4.33, 5.14, 6.05, 7.09, 8.17, 9.43, 10.81]) * 1.0e18
    weight_c = np.array([1.12, 1.58, 1.99, 2.47, 3.08, 3.73, 4.51, 5.28, 6.22, 7.21, 8.39, 9.62, 11.03]) * 1.0e18
    species = [
        {
            "name": "A+",
            "charge": 1,
            "mass": 1.7,
            "density": float(np.sum(weight_a)),
            "flow": 8.0e4,
            "temperature": 62.0,
            "Nmarker": weight_a.size,
            "weight": weight_a,
        },
        {
            "name": "B+",
            "charge": 1,
            "mass": 2.7,
            "density": float(np.sum(weight_b)),
            "flow": -4.0e4,
            "temperature": 26.0,
            "Nmarker": weight_b.size,
            "weight": weight_b,
        },
        {
            "name": "C+",
            "charge": 1,
            "mass": 4.1,
            "density": float(np.sum(weight_c)),
            "flow": 2.0e4,
            "temperature": 12.0,
            "Nmarker": weight_c.size,
            "weight": weight_c,
        },
    ]
    return {
        "species": deepcopy(species),
        "iterations": 300,
        "dt": 2.5e-8,
        "seed": 131313,
        "ensemble_size": 64,
    }
