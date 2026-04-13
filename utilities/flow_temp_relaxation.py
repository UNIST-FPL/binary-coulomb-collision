from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from binary_collision import Collision, MultiSpeciesCollision, Particle
from binary_collision.particle import RNGLike, _coerce_rng


def simulate_relaxation(particle_dict1, particle_dict2, iterations=100, dt=1e-7, rng: RNGLike = None) -> Dict[str, np.ndarray]:
    """
    Runs a deterministic-friendly relaxation simulation and returns history arrays.
    """
    sim_rng = _coerce_rng(rng)

    particle_kwargs1 = dict(particle_dict1)
    particle_kwargs2 = dict(particle_dict2)
    particle_kwargs1.setdefault("rng", sim_rng)
    particle_kwargs2.setdefault("rng", sim_rng)

    spc1 = Particle(**particle_kwargs1)
    spc2 = Particle(**particle_kwargs2)
    col = Collision(spc1, spc2, dtp=dt, rng=sim_rng)

    flow1_hist, flow2_hist = [], []
    temp1_hist, temp2_hist = [], []

    for _ in range(iterations):
        col.run()
        flow1_hist.append(spc1.flow_actual.copy())
        flow2_hist.append(spc2.flow_actual.copy())
        temp1_hist.append(spc1.temperature_actual)
        temp2_hist.append(spc2.temperature_actual)

    flow1_hist = np.array(flow1_hist)
    flow2_hist = np.array(flow2_hist)
    temp1_hist = np.array(temp1_hist)
    temp2_hist = np.array(temp2_hist)

    if spc1.name == "e-":
        ref_flow = np.linalg.norm(flow1_hist[0])
    elif spc2.name == "e-":
        ref_flow = np.linalg.norm(flow2_hist[0])
    else:
        ref_flow = np.linalg.norm(flow2_hist[0])

    time_axis = np.arange(iterations) * dt * 1e5

    return {
        "species_names": np.array([spc1.name, spc2.name], dtype=object),
        "flow_histories": np.stack((flow1_hist, flow2_hist)),
        "flow_magnitudes": np.stack((np.linalg.norm(flow1_hist, axis=1), np.linalg.norm(flow2_hist, axis=1))),
        "temperature_histories": np.stack((temp1_hist, temp2_hist)),
        "time_axis": time_axis,
        "reference_flow": np.array(ref_flow),
    }


def simulate_relaxation_multispecies(
    particle_dicts,
    iterations=100,
    dt=1e-7,
    rng: RNGLike = None,
) -> Dict[str, np.ndarray]:
    """
    Run a deterministic-friendly relaxation simulation for 3+ species.
    """
    sim_rng = _coerce_rng(rng)
    particle_kwargs_list = [dict(particle_dict) for particle_dict in particle_dicts]
    for particle_kwargs in particle_kwargs_list:
        particle_kwargs.setdefault("rng", sim_rng)

    species = [Particle(**particle_kwargs) for particle_kwargs in particle_kwargs_list]
    collision = MultiSpeciesCollision(species, dtp=dt, rng=sim_rng)

    flow_histories = [[] for _ in species]
    temperature_histories = [[] for _ in species]

    for _ in range(iterations):
        collision.run()
        for species_idx, part in enumerate(species):
            flow_histories[species_idx].append(part.flow_actual.copy())
            temperature_histories[species_idx].append(part.temperature_actual)

    flow_histories = np.array(flow_histories)
    temperature_histories = np.array(temperature_histories)
    flow_magnitudes = np.linalg.norm(flow_histories, axis=2)

    electron_indices = [idx for idx, part in enumerate(species) if part.name == "e-"]
    reference_index = electron_indices[0] if electron_indices else 0
    reference_flow = np.array(flow_magnitudes[reference_index][0])
    time_axis = np.arange(iterations) * dt * 1e5

    return {
        "species_names": np.array([part.name for part in species], dtype=object),
        "flow_histories": flow_histories,
        "flow_magnitudes": flow_magnitudes,
        "temperature_histories": temperature_histories,
        "time_axis": time_axis,
        "reference_flow": reference_flow,
    }


def plot_relaxation_history(history, hold=False, label_prefix=''):
    """
    Plots flow and temperature histories returned by `simulate_relaxation`.
    """
    time_axis = history["time_axis"]
    flow_mag = history["flow_magnitudes"]
    temp_hist = history["temperature_histories"]
    species_names = history["species_names"]
    ref_flow = history["reference_flow"]

    tick_step = 0.5
    ticks = np.arange(0, time_axis[-1] + tick_step, tick_step)

    style_map = {
        "e-": ("k-", "ks"),
        "D+": ("r--", "ro"),
    }

    line1, marker1 = style_map.get(species_names[0], ("b--", "bo"))
    line2, marker2 = style_map.get(species_names[1], ("g-", "gs"))

    if not hold:
        plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    if hold:
        marker_indices = np.round(np.linspace(0, len(time_axis) - 1, min(25, len(time_axis)))).astype(int)
        plt.plot(time_axis[marker_indices], flow_mag[0][marker_indices] / ref_flow, marker1, markersize=3,
                 label=rf'$\mathrm{{{species_names[0]}}}\mathrm{{Flow}}{label_prefix}$')
        plt.plot(time_axis[marker_indices], flow_mag[1][marker_indices] / ref_flow, marker2, markersize=3,
                 label=rf'$\mathrm{{{species_names[1]}}}\mathrm{{Flow}}{label_prefix}$')
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    else:
        plt.plot(time_axis, flow_mag[0] / ref_flow, line1,
                 label=rf'$\mathrm{{{species_names[0]}}}\mathrm{{Flow}}{label_prefix}$')
        plt.plot(time_axis, flow_mag[1] / ref_flow, line2,
                 label=rf'$\mathrm{{{species_names[1]}}}\mathrm{{Flow}}{label_prefix}$')
        plt.legend()

    plt.ylabel('Normalized Flow Magnitude (V/V$_{ref}$)')
    plt.xlabel(r'Time [$10^{-5}$ s]')
    plt.title('Flow Relaxation')
    plt.grid(True)
    plt.xticks(ticks)
    plt.xlim(0, time_axis[-1])

    plt.subplot(2, 1, 2)
    if hold:
        plt.plot(time_axis[marker_indices], temp_hist[0][marker_indices], marker1, markersize=3,
                 label=rf'$\mathrm{{{species_names[0]}}}\mathrm{{Temp}}{label_prefix}$')
        plt.plot(time_axis[marker_indices], temp_hist[1][marker_indices], marker2, markersize=3,
                 label=rf'$\mathrm{{{species_names[1]}}}\mathrm{{Temp}}{label_prefix}$')
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    else:
        plt.plot(time_axis, temp_hist[0], line1,
                 label=rf'$\mathrm{{{species_names[0]}}}\mathrm{{Temp}}{label_prefix}$')
        plt.plot(time_axis, temp_hist[1], line2,
                 label=rf'$\mathrm{{{species_names[1]}}}\mathrm{{Temp}}{label_prefix}$')
        plt.legend()

    plt.xlabel(r'Time [$10^{-5}$ s]')
    plt.ylabel('Temperature (eV)')
    plt.title('Temperature Relaxation')
    plt.grid(True)
    plt.xticks(ticks)
    plt.xlim(0, time_axis[-1])

    if hold:
        plt.draw()
        plt.tight_layout()
        plt.pause(0.001)


def run_relaxation_diagnostic(particle_dict1, particle_dict2, iterations=100, dt=1e-7,
                              hold=False, label_prefix='', rng: RNGLike = None):
    """
    Runs a Nanbu-style relaxation test between two particle species and plots temperature & flow convergence.
    """
    history = simulate_relaxation(
        particle_dict1,
        particle_dict2,
        iterations=iterations,
        dt=dt,
        rng=rng,
    )
    plot_relaxation_history(history, hold=hold, label_prefix=label_prefix)
    return history
