import numpy as np
import matplotlib.pyplot as plt
from binary_collision import Particle, Collision


def run_relaxation_diagnostic(particle_dict1, particle_dict2, iterations=100, dt=1e-7, hold=False, label_prefix=''):
    """
    Runs a Nanbu-style relaxation test between two particle species and plots temperature & flow convergence.
    """

    # Initialize Particle objects
    spc1 = Particle(**particle_dict1)
    spc2 = Particle(**particle_dict2)

    # Collision object
    col = Collision(spc1, spc2, dtp=dt)

    # History storage
    flow1_hist, flow2_hist = [], []
    temp1_hist, temp2_hist = [], []

    for i in range(iterations):
        col.run()
        flow1_hist.append(col.spa.flow_actual.copy())
        flow2_hist.append(col.spb.flow_actual.copy())
        temp1_hist.append(col.spa.temperature_actual)
        temp2_hist.append(col.spb.temperature_actual)

    flow1_hist = np.array(flow1_hist)
    flow2_hist = np.array(flow2_hist)
    flow1_mag = np.linalg.norm(flow1_hist, axis=1)
    flow2_mag = np.linalg.norm(flow2_hist, axis=1)
    temp1_hist = np.array(temp1_hist)
    temp2_hist = np.array(temp2_hist)

    # Time axis
    time_axis = np.arange(iterations) * dt * 1e5  # scale to 1e-5 s unit for plotting
    tick_step = 0.5
    ticks = np.arange(0, time_axis[-1] + tick_step, tick_step)

    # Style map by species name
    style_map = {
        "e-": ("k-", "ks"),   # black line/marker for electrons
        "D+": ("r--", "ro"),  # red dashed/marker for deuterium
    }

    line1, marker1 = style_map.get(col.spa.name, ("b--", "bo"))
    line2, marker2 = style_map.get(col.spb.name, ("g-", "gs"))

    if not hold:
        plt.figure(figsize=(12, 8))

    # ---------- Flow magnitude plot ----------
    plt.subplot(2, 1, 1)
    # Normalize by initial electron flow (identified by name)
    if col.spa.name == "e-":
        ref_flow = np.linalg.norm(flow1_hist[0])
    elif col.spb.name == "e-":
        ref_flow = np.linalg.norm(flow2_hist[0])
    else:
        # fallback: just use spb
        ref_flow = np.linalg.norm(flow2_hist[0])

    if hold:
        marker_indices = np.round(np.linspace(0, iterations - 1, min(25, iterations))).astype(int)

        plt.plot(time_axis[marker_indices], flow1_mag[marker_indices] / ref_flow, marker1, markersize=3,
                 label=rf'$\mathrm{{{col.spa.name}}}\mathrm{{Flow}}{label_prefix}$')
        plt.plot(time_axis[marker_indices], flow2_mag[marker_indices] / ref_flow, marker2, markersize=3,
                 label=rf'$\mathrm{{{col.spb.name}}}\mathrm{{Flow}}{label_prefix}$')

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    else:
        plt.plot(time_axis, flow1_mag / ref_flow, line1,
                 label=rf'$\mathrm{{{col.spa.name}}}\mathrm{{Flow}}{label_prefix}$')
        plt.plot(time_axis, flow2_mag / ref_flow, line2,
                 label=rf'$\mathrm{{{col.spb.name}}}\mathrm{{Flow}}{label_prefix}$')
        plt.legend()

    plt.ylabel('Normalized Flow Magnitude (V/V$_{ref}$)')
    plt.xlabel(r'Time [$10^{-5}$ s]')
    plt.title('Flow Relaxation')
    plt.grid(True)
    plt.xticks(ticks)
    plt.xlim(0, time_axis[-1])

    # ---------- Temperature plot ----------
    plt.subplot(2, 1, 2)

    if hold:
        plt.plot(time_axis[marker_indices], temp1_hist[marker_indices], marker1, markersize=3,
                 label=rf'$\mathrm{{{col.spa.name}}}\mathrm{{Temp}}{label_prefix}$')
        plt.plot(time_axis[marker_indices], temp2_hist[marker_indices], marker2, markersize=3,
                 label=rf'$\mathrm{{{col.spb.name}}}\mathrm{{Temp}}{label_prefix}$')

        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())
    else:
        plt.plot(time_axis, temp1_hist, line1,
                 label=rf'$\mathrm{{{col.spa.name}}}\mathrm{{Temp}}{label_prefix}$')
        plt.plot(time_axis, temp2_hist, line2,
                 label=rf'$\mathrm{{{col.spb.name}}}\mathrm{{Temp}}{label_prefix}$')
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
