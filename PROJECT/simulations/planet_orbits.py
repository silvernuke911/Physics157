import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from modules import constants
from modules import integrators 
from modules import orbit_calculation 
from modules import gravity 
from modules import science_plot
from modules import plotter
from functools import partial
from matplotlib.ticker import MultipleLocator

import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter as anim 

science_plot.science_plot()

planet_names = ["Mercury", "Venus", "Earth", "Apophis"]#, "Saturn", "Uranus", "Neptune"]
planet_color = ["dimgray","goldenrod","royalblue","red"]#,"yellow","turquoise", "blue"]

fig = plt.figure(figsize = (10,8), dpi=150)
ax = fig.add_subplot(111, projection='3d')

ax.grid(True)
spacing = 10
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.zaxis.set_major_locator(MultipleLocator(1))

ax.scatter(0, 0, 0, color='yellow', s=80, marker='o', zorder=3)

for name, color in zip(planet_names,planet_color):
    r = orbit_calculation.body_orbit3d(name,2000)
    ax.plot(r[:,0], r[:,1], r[:,2], color=color, zorder=1, label = name)
    print(f"{name} Done")

ax.set_xlabel(f"$x$-axis (AU)")
ax.set_ylabel(f"$y$-axis (AU)")
ax.set_zlabel(f"$z$-axis (AU)")

def set_axes_equal(ax):
    """Set 3D plot axes to equal scale (equal magnitudes on all axes)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
ax.legend()
set_axes_equal(ax)
ax.view_init(elev=45, azim=67.5)
# ax.grid(which='minor', visible=False)
ax.grid(which='minor', visible=False)
# Set gridlines to be thinner
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    axis._axinfo['grid']['linewidth'] = 0.25  # or any thin float value
plt.savefig("imgs/Apophisss3.png")
plt.show()
print("Plotted 3D")