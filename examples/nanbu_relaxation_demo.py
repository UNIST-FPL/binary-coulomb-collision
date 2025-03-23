import numpy as np
from scipy.constants import e, physical_constants
from utilities import run_relaxation_diagnostic

# ----------------------------------
# Define D+ (Deuterium-like ion)
# ----------------------------------
D_params = {
    "name": "D+",
    "charge": 1,
    "mass": 5 * physical_constants['electron mass in u'][0],  # NOTE: this is a fictitious mass
    "density": 1.e21,           # m^-3
    "flow": 0.,                 # m/s
    "temperature": 100.,        # eV
    "Nmarker": 100000,
    "vel": None,
    "weight": lambda params: params["density"] / params["Nmarker"]
}
D_params["weight"] = D_params["weight"](D_params)

# ----------------------------------
# Define e- (Electron)
# ----------------------------------
e_params = {
    "name": "e-",
    "charge": -1,
    "mass": physical_constants['electron mass in u'][0],
    "density": 1.e21,
    "flow": lambda params: np.sqrt(
        params["temperature"] * e / (params["mass"] * physical_constants['atomic mass constant'][0])
    ),
    "temperature": 1.e+3,       # eV
    "Nmarker": 100000,
    "vel": None,
    "weight": lambda params: params["density"] / params["Nmarker"]
}
e_params["weight"] = e_params["weight"](e_params)
e_params["flow"] = e_params["flow"](e_params)

# ----------------------------------
# Run diagnostic (Verification)
# ----------------------------------

def fig4():
    iterations = 151
    dt = 1.e-7

    # Run first test (WD = We)
    run_relaxation_diagnostic(D_params, e_params, iterations=iterations, dt=dt,
                              label_prefix="(W_D=W_e)", hold=False)

    # Run second test (WD = 5We)
    D_params2 = D_params.copy()
    e_params2 = {
        **e_params,
        "Nmarker": e_params["Nmarker"] // 5,
    }
    e_params2["weight"] = lambda params: params["density"] / params["Nmarker"]
    e_params2["weight"] = e_params2["weight"](e_params2)

    run_relaxation_diagnostic(D_params2, e_params2, iterations=iterations, dt=dt,
                              label_prefix="(W_D=5W_e)", hold=True)

def fig5():
    iterations = 151
    dt = 1.e-7

    # Run first test (We = WD)
    run_relaxation_diagnostic(D_params, e_params, iterations=iterations, dt=dt,
                              label_prefix="(W_D=W_e)", hold=False)

    # Run second test (We = 5WD)
    D_params2 = {
        **D_params,
        "Nmarker": D_params["Nmarker"] // 5,
    }
    D_params2["weight"] = lambda params: params["density"] / params["Nmarker"]
    D_params2["weight"] = D_params2["weight"](D_params2)
    e_params2 = e_params.copy()

    run_relaxation_diagnostic(D_params2, e_params2, iterations=iterations, dt=dt,
                              label_prefix="(W_e=5W_D)", hold=True)

def fig6():
    iterations = 1201
    dt = 1.25e-9

    # Run first test (WD = We)
    D_params2 = {
        **D_params,
        "Nmarker": 50000,
        "charge": 3,
    }
    D_params2["weight"] = lambda params: params["density"] / params["Nmarker"]
    D_params2["weight"] = D_params2["weight"](D_params2)
    e_params2 = {
        **e_params,
        "Nmarker": 150000,
        "density": 3.e21,
    }
    e_params2["weight"] = lambda params: params["density"] / params["Nmarker"]
    e_params2["weight"] = e_params2["weight"](e_params2)

    run_relaxation_diagnostic(D_params2, e_params2, iterations=iterations, dt=dt,
                              label_prefix="(W_e=W_D)", hold=False)

    # Run second test (We = 3WD)
    D_params3 = {
        **D_params2,
        "Nmarker": 50000,
        "charge": 3,
    }
    D_params3["weight"] = lambda params: params["density"] / params["Nmarker"]
    D_params3["weight"] = D_params3["weight"](D_params3)
    e_params3 = {
        **e_params2,
        "Nmarker": 50000,
        "density": 3.e21,
    }
    e_params3["weight"] = lambda params: params["density"] / params["Nmarker"]
    e_params3["weight"] = e_params3["weight"](e_params3)

    run_relaxation_diagnostic(D_params3, e_params3, iterations=iterations, dt=dt,
                              label_prefix="(W_e=3W_D)", hold=True)

def run_all_figures():
    fig4()
    fig5()
    fig6()

if __name__ == '__main__':
    run_all_figures()
