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

G = constants.G_ausmday
m = constants.bodies["Sun"]["mass_solar"]

year = 365.25 

dt = 0.5
t = np.arange(0,20*year,dt)

r_i, v_i = orbit_calculation.body_state_vectors("Apophis")
print(r_i,v_i)

acc_func = partial(gravity.grav_acc, m=m, G=G)
r_vec, v_vec, a_vec = integrators.kinematic_rk4(t,acc_func,r_i,v_i,True)

plotter.plot_orbit(r_vec, units = "AU",save = True, filename="imgs/apophissim4.png")



