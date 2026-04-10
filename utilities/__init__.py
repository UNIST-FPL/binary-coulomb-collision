# utilities/__init__.py

"""
utilities - diagnostic and visualization tools for binary collision simulations.

Includes:
- simulate_relaxation: Numerical relaxation history without plotting.
- run_relaxation_diagnostic: Flow & temperature relaxation diagnostic (Nanbu-style)
"""

from .flow_temp_relaxation import plot_relaxation_history, run_relaxation_diagnostic, simulate_relaxation

__all__ = ["plot_relaxation_history", "run_relaxation_diagnostic", "simulate_relaxation"]
