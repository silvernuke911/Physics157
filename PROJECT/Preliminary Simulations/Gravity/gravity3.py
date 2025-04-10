import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
import vector_operations as vc 

mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['axes.formatter.use_mathtext']=True
mpl.rcParams.update({'font.size': 12})

def progress_bar(progress, total, start_time, scale=0.50):
    # Creates a progress bar on the command line, input is progress, total, and a present start time
    # progress and total can be any number, and this can be placed in a for or with loop

    percent = 100 * (float(progress) / float(total))                        # Calculate the percentage of progress
    bar = '█' * round(percent*scale) + '-' * round((100-percent)*scale)     # Create the progress bar string
    elapsed_time = time.time() - start_time                                 # Calculate elapsed time
    if progress > 0:                                                        # Estimate total time and remaining time
        estimated_total_time = elapsed_time * total / progress
        remaining_time = estimated_total_time - elapsed_time
        remaining_seconds = int(remaining_time)
        remaining_milliseconds = int((remaining_time - remaining_seconds) * 1_000)
        remaining_str = time.strftime("%H:%M:%S", time.gmtime(remaining_seconds))
        remaining_str = f"{remaining_str}.{remaining_milliseconds:03d}"
    else:
        remaining_str = '...'
    print(f'|{bar}| {percent:.2f}% Time remaining: {remaining_str}  ', end='\r')    # Print the progress bar with the remaining time
    if progress == total: 
        elapsed_seconds = int(elapsed_time)
        elapsed_ms=int((elapsed_time-elapsed_seconds)*1000)                         # Print elapsed time when complete
        elapsed_seconds =  time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print('\n'+f'Elapsed time : {elapsed_seconds}.{elapsed_ms:03d}')



def lambert_solver(r1, r2, tof, mu):
    from scipy.optimize import fsolve
    def norm(v):
        return np.linalg.norm(v)

    def stumpff_c(z):
        if z > 0:
            return (1 - np.cos(np.sqrt(z))) / z
        elif z < 0:
            return (np.cosh(np.sqrt(-z)) - 1) / -z
        else:
            return 1/2
    
    def stumpff_s(z):
        if z > 0:
            return (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3)
        elif z < 0:
            return (np.sinh(np.sqrt(-z)) - np.sqrt(-z)) / (np.sqrt(-z)**3)
        else:
            return 1/6
    
    r1_norm = norm(r1)
    r2_norm = norm(r2)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    A = np.sin(np.arccos(cos_dnu)) * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu))

    def time_of_flight(z):
        y = r1_norm + r2_norm + A * (z * stumpff_s(z) - 1) / np.sqrt(stumpff_c(z))
        x = np.sqrt(y / stumpff_c(z))
        return x**3 * stumpff_s(z) + A * np.sqrt(y)

    # Solve for z
    z_guess = 0.1
    z_solution = fsolve(lambda z: time_of_flight(z) - np.sqrt(mu) * tof, z_guess)[0]

    y = r1_norm + r2_norm + A * (z_solution * stumpff_s(z_solution) - 1) / np.sqrt(stumpff_c(z_solution))
    f = 1 - y / r1_norm
    g_dot = 1 - y / r2_norm
    g = A * np.sqrt(y / mu)

    v1 = (r2 - f * r1) / g
    v2 = (g_dot * r2 - r1) / g

    return v1, v2
def check_and_prompt_overwrite(filename):
    if os.path.isfile(filename):
        response = input(f"{filename} already exists, are you sure you want to overwrite it? (yes/no): ")
        if response.lower() == 'yes':
            print("Proceeding with overwrite...")
            return True
        elif response.lower() == 'no':
            print("Operation aborted.")
            return False
        else:
            print("Wrong input.")
            return False
    else:
        return True
def compute_orbital_elements(r_i, v_i, G, M):
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

    T_p = 2 * np.pi * np.sqrt( (a)**3 / mu)

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
def vperp(vec):
    if vec.shape != (2,):
        raise ValueError("Input vector must be a 2D vector (shape (2,)).")
    # 90 degrees counterclockwise rotation
    perp_ccw = np.array([-vec[1], vec[0]])
    # 90 degrees clockwise rotation
    perp_cw = np.array([vec[1], -vec[0]])
    return perp_ccw, perp_cw

G = 1
def gravity_force(m1,m2,pos1, pos2):
    G_local=G
    pos1=np.array(pos1)
    pos2=np.array(pos2)
    r=pos2-pos1
    r2=(vc.mag(r))**2
    r_norm=vc.normalize(r)
    return -(G_local*m1*m2/r2)*r_norm


v_i = np.array([0,0.5])
p_i = np.array([1,0])
p_0 = np.array([0,0])
m1 = 1
m2 = 0.01

period = compute_orbital_elements(np.array([0,1]),np.array([1,0]),G,m1)['orbital period']
for element,value in compute_orbital_elements(p_i,v_i,G,m1).items():
    print(f' {element:^25} : {value:.3f}')
p=p_i
v=v_i

dt =  0.001
t = np.arange(0,2*compute_orbital_elements(p,v,G,m1)['orbital period']+dt,dt)
# t = np.arange(0,2*np.pi+dt,dt)
pos_list_x = np.zeros_like(t)
pos_list_y = np.zeros_like(t)

vel_list_x = np.zeros_like(t)
vel_list_y = np.zeros_like(t)
acc_list_x = np.zeros_like(t)
acc_list_y = np.zeros_like(t)
rad_list = np.zeros_like(t)
ang_list = np.zeros_like(t)
for i in range(len(t)):
    a = gravity_force(m1,m2,p_0,p)/m2 # acceleration due to gravity
    a_c = 0*vc.normalize(v) # constant linear v-parallel acceleration
    a_r = np.array([0.025,0]) # constant directional acceleration
    a_p = 0*vperp(vc.normalize(v))[0] # constant v-perpendicular acceleration
    v = v + a*dt 
    p = p + v*dt

    acc_list_x[i]=a[0]
    acc_list_y[i]=a[1]

    pos_list_x[i]=p[0]
    pos_list_y[i]=p[1]

    vel_list_x[i]=v[0]
    vel_list_y[i]=v[1]

    # radius list
    rad_list[i] = vc.mag(p)
    ang_list[i] = np.arctan(p[1]/p[0])
    print(f'Generating at t = {t[i]:.4f}, {t[i]/np.max(t)*100:.2f}% done', end='\r')
print()

def plot():
    x=p_i[0]
    y=p_i[1]
    print('Plotting')
    # plt.plot(vel_list_x,vel_list_y,marker='',zorder=1,color='green')
    plt.plot(pos_list_x,pos_list_y,marker='',zorder=1,color='blue')
    plt.plot(t,rad_list,marker='',zorder=1,color='green')
    plt.plot(t,ang_list,marker='',zorder=1,color='magenta')

    plt.scatter(x,y,marker='.',s=60,color='red',zorder=2)
    plt.scatter(0,0,marker='o',s=80,color='yellow',zorder=3)

    plt.xlabel('$x$-axis (AU)')
    plt.ylabel('$y$-axis (AU)')
    plt.title(f'Planetary orbit simulation')

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_axisbelow(True)

    plt.grid()
    plt.show()
    print('Plotted')

plot()

