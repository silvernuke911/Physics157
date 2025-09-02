import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules import constants
from modules import integrators 
from modules import orbit_calculation 
from modules import gravity 
from modules import science_plot
from modules import plotter
from modules import titles
from modules import progress
from functools import partial
import time
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.animation import PillowWriter as anim 
from matplotlib.ticker import MultipleLocator
import gc
science_plot.science_plot()

df_hrzon = pd.read_csv(r'C:\Users\verci\Documents\Python Code\Physics157\PROJECT\sims_actual\nbody_01.csv')
df_2body = pd.read_csv(r'C:\Users\verci\Documents\Python Code\Physics157\PROJECT\sims_actual\2body_05.csv')
df_nbody = pd.read_csv(r'C:\Users\verci\Documents\Python Code\Physics157\PROJECT\sims_actual\nbody_05e.csv')
print(df_nbody)
r_x, r_y, r_z    = df_hrzon['x'],df_hrzon['y'],df_hrzon['z']
r_nx, r_ny, r_nz = df_nbody['x'],df_nbody['y'],df_nbody['z']
r_ex, r_ey, r_ez = df_nbody['x_earth'],df_nbody['y_earth'],df_nbody['z_earth']

t = df_hrzon["t"]
t = t[:2000]
year = 365.25

def animated_plot_orbit():
    metadata = dict(title='Movie', artist='silver')
    writer = anim(fps=24, metadata=metadata)
    fig = plt.figure(figsize=(10, 8), dpi=80)
    ax = fig.add_subplot(111, projection='3d')

    start_time = time.time()
    max_time = np.max(t)
    filename = "gravsim6.gif"
    overwrite, filename = titles.check_and_prompt_overwrite(filename)
    if not overwrite:
        return None

    print('Animating file ...')

    # Plot static elements once
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.zaxis.set_major_locator(MultipleLocator(1))

    ax.plot(r_x, r_y, r_z, color='red', zorder=2, alpha=0.6)  # True asteroid path
    ax.plot(r_ex, r_ey, r_ez, color='blue', zorder=2, alpha=0.6)  # Earth path
    ax.scatter(0, 0, 0, color='yellow', s=80, marker='o', zorder=3)  # Sun

    # Create and store handles for dynamic scatter plots
    asteroid_scatter = ax.scatter([], [], [], color='red', marker='o', s=30, zorder=3, label="True Asteroid")
    sim_scatter = ax.scatter([], [], [], color='green', marker='o', s=30, zorder=2, label="Simulated Asteroid")
    earth_scatter = ax.scatter([], [], [], color='blue', marker='o', s=40, zorder=3, label="Earth")

    ax.set_xlabel("$x$-axis (AU)")
    ax.set_ylabel("$y$-axis (AU)")
    ax.set_zlabel("$z$-axis (AU)")
    ax.legend()
    ax.view_init(elev=45, azim=67.5)

    def set_axes_equal(ax):
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        plot_radius = 0.5 * max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    set_axes_equal(ax)

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid']['linewidth'] = 0.25

    with writer.saving(fig, filename, 200):
        for i in range(0, len(t), 10):
            # Update dynamic scatter points
            asteroid_scatter._offsets3d = ([r_x[i]], [r_y[i]], [r_z[i]])
            sim_scatter._offsets3d = ([r_nx[i]], [r_ny[i]], [r_nz[i]])
            earth_scatter._offsets3d = ([r_ex[i]], [r_ey[i]], [r_ez[i]])

            ax.set_title(f"Apophis Orbit Simulation: t = {(t[i] / year):.3f} years")
            writer.grab_frame()
            progress.progress_bar(t[i], max_time, start_time)
            gc.collect()
    print("\nFile animated and saved!")
animated_plot_orbit()


