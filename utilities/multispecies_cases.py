from copy import deepcopy
from scipy.constants import physical_constants


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
        "iterations": 600,
        "dt": 3.0e-8,
        "seed": 123,
    }
