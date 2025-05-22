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
from matplotlib.animation import PillowWriter as anim 

science_plot.science_plot()

# Constants
G = constants.G_ausmday
year = 365.25

# Time array
dt = 0.5
t = np.arange(0, 10 * year, dt)

# Masses (Sun, m1, m2)
m_sun = constants.bodies["Sun"]["mass_solar"]  # 1.0
m1 = 0.02
m2 = 0.001
masses = np.array([m_sun, m1, m2])

# Initial positions (Sun at origin)
R0 = np.array([
    [0.0, 0.0],   # Sun
    [1.0, 0.0],   # m1
    [2.0, 0.0],   # m2
])

# Circular velocities for m1 and m2 around the Sun
v1 = np.array([0.0, np.sqrt(G * m_sun / np.linalg.norm(R0[1]))])
v2 = np.array([0.0, np.sqrt(G * m_sun / np.linalg.norm(R0[2]))])

V0 = np.array([
    [0.0, 0.0],  # Sun
    v1,
    v2,
])

# Wrap acceleration function
acc_func = lambda t, R, V: gravity.grav_acc_nbody(t, R, V, masses, G)

# Run integrator
R, V, A = integrators.kinematic_rk4_multi(t, acc_func, R0, V0)

# Subtract Sun’s motion for heliocentric frame
R_helio = R - R[:, 0:1, :]  # subtract Sun's position from all bodies

# Plotting
plt.figure(figsize=(6, 6))
plt.plot(R_helio[:, 1, 0], R_helio[:, 1, 1], label='m1', color='blue')
plt.plot(R_helio[:, 2, 0], R_helio[:, 2, 1], label='m2', color='red')
plt.plot(0, 0, 'yo', label='Sun')  # Sun fixed at origin in heliocentric frame
plt.xlabel("x [AU]")
plt.ylabel("y [AU]")
plt.gca().set_aspect("equal")
plt.legend()
plt.show()
