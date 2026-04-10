import numpy as np
from scipy.constants import e, physical_constants


def _electron_flow(temperature_ev: float, mass_u: float) -> float:
    return np.sqrt(
        temperature_ev * e
        / (mass_u * physical_constants["atomic mass constant"][0])
    )


def reduced_relaxation_cases():
    electron_mass_u = physical_constants["electron mass in u"][0]
    ion_mass_u = 5 * electron_mass_u

    fig4_like = {
        "name": "fig4_like",
        "seed": 1004,
        "iterations": 8,
        "dt": 1.0e-7,
        "particle_1": {
            "name": "D+",
            "charge": 1,
            "mass": ion_mass_u,
            "density": 1.0e21,
            "flow": 0.0,
            "temperature": 100.0,
            "Nmarker": 50,
            "weight": 1.0e21 / 50,
        },
        "particle_2": {
            "name": "e-",
            "charge": -1,
            "mass": electron_mass_u,
            "density": 1.0e21,
            "flow": _electron_flow(1.0e3, electron_mass_u),
            "temperature": 1.0e3,
            "Nmarker": 10,
            "weight": 1.0e21 / 10,
        },
    }

    fig5_like = {
        "name": "fig5_like",
        "seed": 1005,
        "iterations": 8,
        "dt": 1.0e-7,
        "particle_1": {
            "name": "D+",
            "charge": 1,
            "mass": ion_mass_u,
            "density": 1.0e21,
            "flow": 0.0,
            "temperature": 100.0,
            "Nmarker": 10,
            "weight": 1.0e21 / 10,
        },
        "particle_2": {
            "name": "e-",
            "charge": -1,
            "mass": electron_mass_u,
            "density": 1.0e21,
            "flow": _electron_flow(1.0e3, electron_mass_u),
            "temperature": 1.0e3,
            "Nmarker": 50,
            "weight": 1.0e21 / 50,
        },
    }

    fig6_like = {
        "name": "fig6_like",
        "seed": 1006,
        "iterations": 10,
        "dt": 1.25e-9,
        "particle_1": {
            "name": "D+",
            "charge": 3,
            "mass": ion_mass_u,
            "density": 1.0e21,
            "flow": 0.0,
            "temperature": 100.0,
            "Nmarker": 15,
            "weight": 1.0e21 / 15,
        },
        "particle_2": {
            "name": "e-",
            "charge": -1,
            "mass": electron_mass_u,
            "density": 3.0e21,
            "flow": _electron_flow(1.0e3, electron_mass_u),
            "temperature": 1.0e3,
            "Nmarker": 15,
            "weight": 3.0e21 / 15,
        },
    }

    return [fig4_like, fig5_like, fig6_like]
