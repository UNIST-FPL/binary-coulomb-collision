# utilities/__init__.py

"""
utilities - diagnostic, visualization, and verification helpers.
"""

from .flow_temp_relaxation import (
    plot_relaxation_history,
    run_relaxation_diagnostic,
    simulate_relaxation,
    simulate_relaxation_multispecies,
    simulate_relaxation_multispecies_ensemble,
)
from .nanbu_figure_cases import (
    main_figure_case_bundles,
    main_figure_cases,
    reduced_main_figure_case_bundles,
    reduced_main_figure_cases,
)
from .multispecies_cases import (
    canonical_three_species_case,
    canonical_three_species_weighted_case,
    particle_weight_three_species_case,
    particle_weight_three_species_equilibrium_case,
    particle_weight_two_species_case,
    thirteen_particle_weight_relaxation_case,
    three_species_long_time_equilibrium_case,
    three_species_slower_relaxation_case,
    three_species_smooth_relaxation_case,
)

__all__ = [
    "canonical_three_species_case",
    "canonical_three_species_weighted_case",
    "particle_weight_three_species_case",
    "particle_weight_three_species_equilibrium_case",
    "particle_weight_two_species_case",
    "thirteen_particle_weight_relaxation_case",
    "three_species_long_time_equilibrium_case",
    "three_species_slower_relaxation_case",
    "three_species_smooth_relaxation_case",
    "main_figure_case_bundles",
    "main_figure_cases",
    "plot_relaxation_history",
    "reduced_main_figure_case_bundles",
    "reduced_main_figure_cases",
    "run_relaxation_diagnostic",
    "simulate_relaxation",
    "simulate_relaxation_multispecies",
    "simulate_relaxation_multispecies_ensemble",
]
