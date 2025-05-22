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

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.animation import PillowWriter as anim 

science_plot.science_plot()

G = constants.G_ausmday
m = constants.bodies["Sun"]["mass_solar"]

year = 365.25 

dt = 0.5
t = np.arange(0, 20*year, dt)

r_i, v_i = orbit_calculation.body_state_vectors("Apophis")
print(r_i, v_i)

acc_func = partial(gravity.grav_acc, m=m, G=G)
r_vec, v_vec, a_vec = integrators.kinematic_rk4(t, acc_func, r_i, v_i, True)
x, y, z = r_vec[:, 0], r_vec[:, 1], r_vec[:, 2]

# Create DataFrame with time and position coordinates
df_2body = pd.DataFrame({
    't': t,
    'x': x,
    'y': y,
    'z': z
})

# Filter for integer time values (whole days)
df_2body_integer = df_2body[np.isclose(df_2body['t'] % 1, 0)]  # Select where fractional part is 0

# Alternative method if you prefer rounding:
# df_nbody_integer = df_nbody[df_nbody['t'].round(0) == df_nbody['t']]
print()
# Verify we have daily points
print(f"Selected {len(df_2body_integer)} integer-day points from {len(df_2body)} total points")

# Save to CSV
df_2body_integer.to_csv("sims_actual/2body_05.csv", index=False)
print("Saved 2-body simulation results to '2body_05.csv'")

# Optional: Print first 5 rows for verification
print("\nFirst 5 daily points:")
print(df_2body_integer.head())





