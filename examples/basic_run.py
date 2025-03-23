import numpy as np
from scipy.constants import e, physical_constants
from binary_collision import Particle, Collision

# -------------------------------
# Define Deuterium (D+) parameters
# -------------------------------
D_params = {
    "name": "D+",                            # Species name
    "charge": 1,                             # Charge number Z
    "mass": 2.0141,                               # Mass in atomic mass units, "u"
    "density": 1.e21,                        # Number density [m^-3]
    "flow": 0.,                              # Initial flow velocity [m/s]
    "temperature": 100.,                     # Initial temperature [eV]
    "Nmarker": 100000,                       # Number of simulation particles (markers)
    "vel": None,                             # Velocity will be initialized isotropically
    "weight": lambda params: params["density"] / params["Nmarker"]  # Compute statistical weight
}
# Compute the actual value of `weight` after defining the dict
D_params["weight"] = D_params["weight"](D_params)

# -------------------------------
# Define Electron (e-) parameters
# -------------------------------
e_params = {
    "name": "e-",
    "charge": -1,
    "mass": physical_constants['electron mass in u'][0],
    "density": 1.e21,
    "flow": lambda params: np.sqrt(
        params["temperature"] * e / (params["mass"] * physical_constants['atomic mass constant'][0])
    ),
    "temperature": 1.e+3,
    "Nmarker": 100000 // 5,
    "vel": None,
    "weight": lambda params: params["density"] / params["Nmarker"]
}
# Evaluate weight and flow after param definition
e_params["weight"] = e_params["weight"](e_params)
e_params["flow"] = e_params["flow"](e_params)


# -------------------------------
# Main simulation entry point
# -------------------------------

def main():
    # Initialize Particle instances from parameter dictionaries
    spc_D = Particle(**D_params)
    spc_e = Particle(**e_params)

    # Create a Collision object for D+ - e− Coulomb interaction
    col = Collision(spc_D, spc_e, dtp=1.e-7)  # dtp = physical time step Δt'

    # Store the initial velocities of the particles before the collision step
    # vD_old and ve_old correspond to the original input order of the species (e.g., D and e)
    vD_old, ve_old = col.get_velocity()

    # Run one full collision step
    # This includes randomized like-particle (D-D, e-e) and unlike-particle (D-e) collisions
    col.run()

    # Retrieve the updated velocities of the particles after the collision step
    # vD_new and ve_new are guaranteed to be returned in the same order as the original input
    vD_new, ve_new = col.get_velocity()

if __name__ == '__main__':
    main()