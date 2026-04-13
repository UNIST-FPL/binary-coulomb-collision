from typing import Dict, List

import numpy as np
from scipy.constants import e, physical_constants


FigureCase = Dict[str, object]
FigureBundle = Dict[str, object]


def _electron_flow(temperature_ev: float, mass_u: float) -> float:
    return np.sqrt(
        temperature_ev * e
        / (mass_u * physical_constants["atomic mass constant"][0])
    )


def _with_weight(params: Dict[str, object]) -> Dict[str, object]:
    frozen = dict(params)
    frozen["weight"] = frozen["density"] / frozen["Nmarker"]
    return frozen


def _make_case(
    *,
    name: str,
    figure: str,
    variant: str,
    label_prefix: str,
    iterations: int,
    dt: float,
    seed: int,
    particle_1: Dict[str, object],
    particle_2: Dict[str, object],
) -> FigureCase:
    particle_1 = _with_weight(particle_1)
    particle_2 = _with_weight(particle_2)
    return {
        "name": name,
        "figure": figure,
        "variant": variant,
        "label_prefix": label_prefix,
        "iterations": iterations,
        "dt": dt,
        "seed": seed,
        "particle_1": particle_1,
        "particle_2": particle_2,
        "we_over_wd": particle_2["weight"] / particle_1["weight"],
    }


def _main_base_params() -> tuple[Dict[str, object], Dict[str, object]]:
    electron_mass_u = physical_constants["electron mass in u"][0]
    ion_mass_u = 5 * electron_mass_u

    d_params = {
        "name": "D+",
        "charge": 1,
        "mass": ion_mass_u,
        "density": 1.0e21,
        "flow": 0.0,
        "temperature": 100.0,
        "Nmarker": 100000,
    }
    e_params = {
        "name": "e-",
        "charge": -1,
        "mass": electron_mass_u,
        "density": 1.0e21,
        "flow": _electron_flow(1.0e3, electron_mass_u),
        "temperature": 1.0e3,
        "Nmarker": 100000,
    }
    return d_params, e_params


def main_figure_cases() -> List[FigureCase]:
    d_params, e_params = _main_base_params()

    fig4_equal = _make_case(
        name="fig4_equal",
        figure="fig4",
        variant="equal",
        label_prefix="(W_D=W_e)",
        iterations=151,
        dt=1.0e-7,
        seed=4104,
        particle_1=d_params,
        particle_2=e_params,
    )
    fig4_weighted = _make_case(
        name="fig4_weighted",
        figure="fig4",
        variant="weighted",
        label_prefix="(W_D=5W_e)",
        iterations=151,
        dt=1.0e-7,
        seed=4105,
        particle_1=d_params,
        particle_2={**e_params, "Nmarker": e_params["Nmarker"] // 5},
    )
    fig5_equal = _make_case(
        name="fig5_equal",
        figure="fig5",
        variant="equal",
        label_prefix="(W_D=W_e)",
        iterations=151,
        dt=1.0e-7,
        seed=5104,
        particle_1=d_params,
        particle_2=e_params,
    )
    fig5_weighted = _make_case(
        name="fig5_weighted",
        figure="fig5",
        variant="weighted",
        label_prefix="(W_e=5W_D)",
        iterations=151,
        dt=1.0e-7,
        seed=5105,
        particle_1={**d_params, "Nmarker": d_params["Nmarker"] // 5},
        particle_2=e_params,
    )

    fig6_d_params = {**d_params, "Nmarker": 50000, "charge": 3}
    fig6_equal = _make_case(
        name="fig6_equal",
        figure="fig6",
        variant="equal",
        label_prefix="(W_e=W_D)",
        iterations=1201,
        dt=1.25e-9,
        seed=6106,
        particle_1=fig6_d_params,
        particle_2={**e_params, "Nmarker": 150000, "density": 3.0e21},
    )
    fig6_weighted = _make_case(
        name="fig6_weighted",
        figure="fig6",
        variant="weighted",
        label_prefix="(W_e=3W_D)",
        iterations=1201,
        dt=1.25e-9,
        seed=6107,
        particle_1=fig6_d_params,
        particle_2={**e_params, "Nmarker": 50000, "density": 3.0e21},
    )

    return [
        fig4_equal,
        fig4_weighted,
        fig5_equal,
        fig5_weighted,
        fig6_equal,
        fig6_weighted,
    ]


def main_figure_case_bundles() -> List[FigureBundle]:
    cases = main_figure_cases()
    return [
        {"name": "fig4_bundle", "figure": "fig4", "cases": [cases[0], cases[1]]},
        {"name": "fig5_bundle", "figure": "fig5", "cases": [cases[2], cases[3]]},
        {"name": "fig6_bundle", "figure": "fig6", "cases": [cases[4], cases[5]]},
    ]


def reduced_main_figure_cases() -> List[FigureCase]:
    full_cases = {case["name"]: case for case in main_figure_cases()}
    reduction_map = {
        "fig4_equal": {"seed": 1404, "iterations": 8, "particle_1_nmarker": 50, "particle_2_nmarker": 50},
        "fig4_weighted": {"seed": 1405, "iterations": 8, "particle_1_nmarker": 50, "particle_2_nmarker": 10},
        "fig5_equal": {"seed": 1504, "iterations": 8, "particle_1_nmarker": 50, "particle_2_nmarker": 50},
        "fig5_weighted": {"seed": 1505, "iterations": 8, "particle_1_nmarker": 10, "particle_2_nmarker": 50},
        "fig6_equal": {"seed": 1606, "iterations": 10, "particle_1_nmarker": 15, "particle_2_nmarker": 45},
        "fig6_weighted": {"seed": 1607, "iterations": 10, "particle_1_nmarker": 15, "particle_2_nmarker": 15},
    }

    reduced_cases = []
    for case_name, reduction in reduction_map.items():
        full_case = full_cases[case_name]
        reduced_cases.append(
            _make_case(
                name=full_case["name"],
                figure=full_case["figure"],
                variant=full_case["variant"],
                label_prefix=full_case["label_prefix"],
                iterations=reduction["iterations"],
                dt=full_case["dt"],
                seed=reduction["seed"],
                particle_1={
                    **full_case["particle_1"],
                    "Nmarker": reduction["particle_1_nmarker"],
                },
                particle_2={
                    **full_case["particle_2"],
                    "Nmarker": reduction["particle_2_nmarker"],
                },
            )
        )
    return reduced_cases


def reduced_main_figure_case_bundles() -> List[FigureBundle]:
    cases = reduced_main_figure_cases()
    return [
        {"name": "fig4_bundle", "figure": "fig4", "cases": [cases[0], cases[1]]},
        {"name": "fig5_bundle", "figure": "fig5", "cases": [cases[2], cases[3]]},
        {"name": "fig6_bundle", "figure": "fig6", "cases": [cases[4], cases[5]]},
    ]
