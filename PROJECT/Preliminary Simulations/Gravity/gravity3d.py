import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
import vector_operations as vc 
from progress import progress_bar

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams.update({'font.size': 12})

def compute_orbital_elements(r_i, v_i, G, M):
    # Creating 3d vectors if vectors are 2d
    if len(r_i)==2:
        r_i=np.append(r_i,0)
        v_i=np.append(v_i,0)
    # Gravitational parameter (mu = GM)
    mu = G * M
    # Specific angular momentum vector (h = r x v)
    h = np.cross(r_i, v_i)
    # Magnitude of specific angular momentum
    h_mag = np.linalg.norm(h)
    # Eccentricity vector (e = ((v x h) / mu) - (r / |r|))
    e = (np.cross(v_i, h) / mu) - (r_i / np.linalg.norm(r_i))
    # Magnitude of eccentricity vector
    e_mag = np.linalg.norm(e)
    # Semi-major axis (a)
    r_mag = np.linalg.norm(r_i)
    v_mag = np.linalg.norm(v_i)
    a = 1 / ((2 / r_mag) - (v_mag ** 2 / mu))
    # Orbital period
    T_p = 2 * np.pi * np.sqrt( a**3 / mu)
    # Inclination (i)
    i = np.arccos(h[2] / h_mag)
    # Node vector (n = k x h)
    k = np.array([0, 0, 1])
    n = np.cross(k, h)
    # Magnitude of node vector
    n_mag = np.linalg.norm(n)
    # Right ascension of the ascending node (RAAN, Ω)
    if n_mag != 0:
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0
    # Argument of periapsis (ω)
    if n_mag != 0 and e_mag != 0:
        omega = np.arccos(np.dot(n, e) / (n_mag * e_mag))
        if e[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0
    # True anomaly (ν)
    if e_mag != 0:
        nu = np.arccos(np.dot(e, r_i) / (e_mag * r_mag))
        if np.dot(r_i, v_i) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0
    # Return orbital elements as a dictionary
    orbital_elements = {
        'semi_major_axis': a,
        'eccentricity': e_mag,
        'inclination': np.degrees(i),
        'LAN': np.degrees(Omega),
        'argument_of_periapsis': np.degrees(omega),
        'true_anomaly': np.degrees(nu),
        'orbital period': T_p
    }
    return orbital_elements

def grav_acc(m1,r):
    G = 1
    mu = G * m1
    r_mag = vc.mag(r)
    r_norm = vc.normalize(r)
    return -(mu/r_mag**2) * r_norm

G = 1
m1 = 1
r_i = np.array([1,1,1])
v_i = np.array([0,0.707,0])

orb_data = compute_orbital_elements(r_i,v_i,G,m1)

period = orb_data['orbital period']
for element,value in orb_data.items():
    print(f' {element:^25} : {value:.3f}')

dt = 0.001
if np.isnan(period):
    t = np.arange(0,5+dt,dt)
else: t = np.arange(0,1*period+dt,dt)

a = np.zeros((len(t), 3))
r = np.zeros((len(t), 3))
v = np.zeros((len(t), 3))

# Initial conditions
r[0] = r_i
v[0] = v_i
a[0] = grav_acc(m1,r_i)

# Simulation loop
time_start = time.time()
for i in range(1, len(t)):

    a[i] = grav_acc(m1, r[i-1]) #+ 0.005*vc.normalize(v[i-1])
    v[i] = v[i - 1] + a[i] * dt
    r[i] = r[i - 1] + v[i] * dt

    progress_bar(i,len(t),time_start)

print()

def plot():
    def set_axes_equal(x,y,z,ax):
        x_limits = [np.min(x),np.max(x)]
        y_limits = [np.min(y),np.max(y)]
        z_limits = [np.min(z),np.max(z)]
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        plot_radius = 0.6*max([x_range, y_range, z_range])
        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    print('Plotting')

    fig = plt.figure()
    ax =  plt.axes(projection='3d')

    set_axes_equal(r[:, 0], r[:, 1], r[:, 2],ax)
    ax.set_aspect('equal')

    ax.plot(r[:, 0], r[:, 1], r[:, 2],zorder=2,marker='')
    # ax.plot(v[:, 0], v[:, 1], v[:, 2],zorder=2)
    # ax.plot(a[:, 0], a[:, 1], a[:, 2],zorder=2)

    ax.scatter(r[0, 0], r[0, 1], r[0, 2],marker='.',s=50,color='red',zorder=3)  
    ax.scatter(0, 0, 0,s=80,color='yellow',zorder=1)  

    ax.set_xlabel('$x$-axis (AU)')
    ax.set_ylabel('$y$-axis (AU)')
    ax.set_zlabel('$z$-axis (AU)')
    ax.set_title(f'Planetary orbit simulation 3D')

    plt.show()
    print('Plotted')
plot()
