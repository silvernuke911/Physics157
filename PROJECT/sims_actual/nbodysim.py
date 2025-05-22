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
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib.animation import PillowWriter as anim 

science_plot.science_plot()

# Constants
G = constants.G_ausmday
year = 365.25

# Time array
dt = 0.5
t = np.arange(0, 20 * year, dt)

# Masses (Sun, m1, m2)
m_sun     = constants.bodies["Sun"]["mass_solar"]  # 1.0
m_venus   = constants.bodies["Venus"]["mass_solar"]
m_earth   = constants.bodies["Earth"]["mass_solar"]
m_jupiter = constants.bodies["Jupiter"]["mass_solar"]
m_apophis = constants.bodies["Apophis"]["mass_solar"]
masses = np.array([m_sun, m_venus, m_earth, m_jupiter, m_apophis])

r0_sun    , v0_sun      = np.array([0,0,0]), np.array([0,0,0])
r0_venus  , v0_venus    = orbit_calculation.body_state_vectors("Venus")
r0_earth  , v0_earth    = orbit_calculation.body_state_vectors("Earth")
r0_jupiter, v0_jupiter  = orbit_calculation.body_state_vectors("Jupiter")
r0_apophis, v0_apophis  = orbit_calculation.body_state_vectors("Apophis")

R0 =  np.array([r0_sun,r0_venus,r0_earth,r0_jupiter,r0_apophis])
V0 =  np.array([v0_sun,v0_venus,v0_earth,v0_jupiter,v0_apophis])

# Wrap acceleration function
acc_func = lambda t, R, V: gravity.grav_acc_nbody(t, R, V, masses, G)

# Run integrator
R, V, A = integrators.kinematic_rk4_multi(t, acc_func, R0, V0)

# Subtract Sun’s motion for heliocentric frame
R_helio = R - R[:, 0:1, :]  # subtract Sun's position from all bodies

R_apophis = R_helio[:,4,:]
R_earth = R_helio[:,2,:]
x,y,z = R_apophis[:,0], R_apophis[:,1], R_apophis[:,2]
x_earth,y_earth,z_earth = R_earth[:,0], R_earth[:,1], R_earth[:,2]
# Create DataFrame with time and position coordinates
df_nbody = pd.DataFrame({
    't': t,
    'x': x,
    'y': y,
    'z': z,
    'x_earth' : x_earth,
    'y_earth' : y_earth,
    'z_earth' : z_earth
})

# Filter for integer time values (whole days)
df_nbody_integer = df_nbody[np.isclose(df_nbody['t'] % 1, 0)]  # Select where fractional part is 0

# Alternative method if you prefer rounding:
# df_nbody_integer = df_nbody[df_nbody['t'].round(0) == df_nbody['t']]

# Verify we have daily points
print(f"Selected {len(df_nbody_integer)} integer-day points from {len(df_nbody)} total points")

# Save to CSV
df_nbody_integer.to_csv("sims_actual/nbody_05e.csv", index=False)
print("Saved daily n-body simulation results to 'nbody_05e.csv'")

# Optional: Print first 5 rows for verification
print("\nFirst 5 daily points:")
print(df_nbody_integer.head())

