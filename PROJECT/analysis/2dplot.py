import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from modules import titles
from matplotlib.ticker import MultipleLocator
import pandas as pd


# Load both datasets
df_rk4 = pd.read_csv(r"sims_actual\2body.csv")  # Your transformed RK4 data
df_horizons = pd.read_csv(r"position data\apophis_cartesian.csv")  # JPL Horizons data

# Create figure with 3D axis
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot RK4 simulation (blue)
ax.plot(df_rk4['x'], df_rk4['y'], df_rk4['z'], 
        label='RK4 Simulation', 
        color='blue',
        linewidth=1.5,
        alpha=0.7)

# Plot Horizons data (red)
ax.plot(df_horizons['x'], df_horizons['y'], df_horizons['z'], 
        label='JPL Horizons', 
        color='red',
        linewidth=1.5,
        alpha=0.7)

# Mark starting point (J2000 epoch)
ax.scatter([df_rk4['x'][0]], [df_rk4['y'][0]], [df_rk4['z'][0]], 
           color='green', 
           s=100,
           label='J2000 Epoch',
           marker='*')

# Mark starting point (J2000 epoch)
ax.scatter([df_horizons['x'][0]], [df_horizons['y'][0]], [df_horizons['z'][0]], 
           color='red', 
           s=100,
           label='J2000 Epoch',
           marker='*')

# Add labels and title
ax.set_xlabel('X (AU)', fontsize=12)
ax.set_ylabel('Y (AU)', fontsize=12)
ax.set_zlabel('Z (AU)', fontsize=12)
ax.set_title('Apophis Orbit: RK4 vs JPL Horizons (ICRF Frame)', fontsize=14)

# Equal aspect ratio
ax.set_box_aspect([1,1,1])  # Important for proper 3D scaling

# Add grid and legend
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(fontsize=10)

# Adjust viewing angle
ax.view_init(elev=30, azim=45)  # Elevation and azimuth angles

# Save high-quality figure
plt.tight_layout()
plt.savefig('orbit_comparison_3d.png', dpi=300)
plt.show()


plt.figure(figsize=(10,4), dpi=300)
plt.plot(df_rk4["t"],df_rk4["x"], 'r')
plt.plot(df_rk4["t"],df_rk4["y"], 'b')
plt.plot(df_rk4["t"],df_rk4["z"], 'g')
plt.plot(df_horizons["RelTime"],df_horizons["x"], 'r')
plt.plot(df_horizons["RelTime"],df_horizons["y"], 'b')
plt.plot(df_horizons["RelTime"],df_horizons["z"], 'g')
plt.show()


