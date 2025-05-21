import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from modules import titles
from matplotlib.ticker import MultipleLocator

def plot_orbit(
        r, 
        object_color = "b",
        path_color = "b",
        sun=True, 
        projectile=True, 
        units="m", 
        save=True, 
        filename = "imgs/default.png"
        ):
    # Determine if 2D or 3D from r's shape
    if r.ndim != 2 or r.shape[1] not in (2, 3):
        print("Orbit has invalid dimensions")
        return
    
    dim = r.shape[1]
    
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

    def save__(save=save, filename=filename):
        if save:
            filename = filename
            overwrite, filename = titles.check_and_prompt_overwrite(filename)  # Check and get the filename
            if not overwrite:
                return None
            plt.savefig(filename)
    if dim == 2:
        print("\nPlotting 2D")
        plt.figure()
        plt.plot(r[:,0], r[:,1], color=path_color, zorder=1)
        
        if sun:
            plt.scatter(0, 0, color='yellow', s=80, marker='o', zorder=3)
        if projectile:
            plt.scatter(r[0,0], r[0,1], color=object_color, s=60, marker='.', zorder=2)

        plt.xlabel(f"$x$-axis ({units})")
        plt.ylabel(f"$y$-axis ({units})")
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        ax.set_axisbelow(True)
        save__()
        plt.show()
        print("Plotted 2D")

    elif dim == 3:
        print("\nPlotting 3D")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(r[:,0], r[:,1], r[:,2], color=path_color, zorder=1)
        ax.grid(True)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.zaxis.set_major_locator(MultipleLocator(0.5))
        
        if sun:
            ax.scatter(0, 0, 0, color='yellow', s=80, marker='o', zorder=3)
        if projectile:
            ax.scatter(r[0,0], r[0,1], r[0,2], color=object_color, s=60, marker='.', zorder=2)

        ax.set_xlabel(f"$x$-axis ({units})")
        ax.set_ylabel(f"$y$-axis ({units})")
        ax.set_zlabel(f"$z$-axis ({units})")

        set_axes_equal(ax)
        ax.view_init(elev=30, azim=45)
        # ax.grid(which='minor', visible=False)
        ax.grid(which='minor', visible=False)
        # Set gridlines to be thinner
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis._axinfo['grid']['linewidth'] = 0.25  # or any thin float value
        save__()
        plt.show()
        print("Plotted 3D")
