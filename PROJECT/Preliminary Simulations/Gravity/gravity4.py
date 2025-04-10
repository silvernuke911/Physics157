import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
import vector_operations as vc 
from progress import progress_bar

plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.unicode_minus': False,
    'axes.formatter.use_mathtext': True,
    'font.size': 12
})
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')

def compute_orbital_elements(r_i, v_i, G, M):
    if len(r_i)==2:                         # Creating 3d vectors if vectors are 2d
        r_i=np.append(r_i,0)
        v_i=np.append(v_i,0)
    mu = G * M                              # Gravitational parameter (mu = GM)
    h = np.cross(r_i, v_i)                  # Specific angular momentum vector (h = r x v)
    h_mag = np.linalg.norm(h)               # Magnitude of specific angular momentum
    e = (np.cross(v_i, h) / mu) - (r_i / np.linalg.norm(r_i))   # Eccentricity vector (e = ((v x h) / mu) - (r / |r|))
    e_mag = np.linalg.norm(e)               # Magnitude of eccentricity vector
    r_mag = np.linalg.norm(r_i)             # Semi-major axis (a)
    v_mag = np.linalg.norm(v_i)
    a = 1 / ((2 / r_mag) - (v_mag ** 2 / mu))
    T_p = 2 * np.pi * np.sqrt( a**3 / mu)   # Orbital period
    i = np.arccos(h[2] / h_mag)             # Inclination (i)
    k = np.array([0, 0, 1])                 # Node vector (n = k x h)
    n = np.cross(k, h)
    n_mag = np.linalg.norm(n)               # Magnitude of node vector
    if n_mag != 0:                          # Right ascension of the ascending node (RAAN, Ω)
        Omega = np.arccos(n[0] / n_mag)
        if n[1] < 0:
            Omega = 2 * np.pi - Omega
    else:
        Omega = 0
    if n_mag != 0 and e_mag != 0:           # Argument of periapsis (ω)
        omega = np.arccos(np.dot(n, e) / (n_mag * e_mag))
        if e[2] < 0:
            omega = 2 * np.pi - omega
    else:
        omega = 0
    if e_mag != 0:                          # True anomaly (ν)
        nu = np.arccos(np.dot(e, r_i) / (e_mag * r_mag))
        if np.dot(r_i, v_i) < 0:
            nu = 2 * np.pi - nu
    else:
        nu = 0
    orbital_elements = {                    # Return orbital elements as a dictionary
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

# setting initial positions
G = 1
m = 1
r_i = np.array([1,0])
v_i = np.array([0,0.75])

orb_data = compute_orbital_elements(r_i,v_i,G,m)
period = orb_data['orbital period']
year = 2*np.pi
for element,value in orb_data.items():
    print(f' {element:^25} : {value:.3f}')

dt = 0.001
# if np.isnan(period):
#     t = np.arange(0,5+dt,dt)
# else: t = np.arange(0,period+dt,dt)

t = np.arange(0,0.1+dt,dt)

a = np.zeros((len(t), 2))
r = np.zeros((len(t), 2))
v = np.zeros((len(t), 2))

# Initial conditions
r[0] = r_i
v[0] = v_i
a[0] = grav_acc(m,r_i)

# Simulation loop
time_start = time.time()
for i in range(1, len(t)):

    a[i] = grav_acc(m, r[i-1]) # + 0.01*vc.normalize(v[i-1])
    v[i] = v[i - 1] + a[i] * dt
    r[i] = r[i - 1] + v[i] * dt

    progress_bar(i,len(t),time_start)
print()

def plot():
    print('Plotting')

    plt.plot(r[:,0],r[:,1],marker='',zorder=1,color='blue')
    # plt.plot(v[:,0],v[:,1],marker='',zorder=1,color='cyan')
    # plt.plot(a[:,0],a[:,1],marker='',zorder=1,color='magenta')

    plt.scatter(r_i[0],r_i[1],marker='.',s=60,color='red',zorder=2)
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


def check_and_prompt_overwrite(filename):
    
    def get_user_input(prompt, valid_responses, invalid_input_limit=3):
        attempts = 0
        while attempts < invalid_input_limit:
            response = input(prompt).lower().strip()
            if response in valid_responses:
                return response
            print("Invalid input. This is in handle user")
            attempts += 1
        print("Exceeded maximum invalid input limit. Operation aborted.")
        return 'ABORT'

    def handle_file_exists(filename):
        while True:
            response = get_user_input(f"{filename} already exists, do you want to overwrite it? (Y/N): ", ['yes', 'y', 'no', 'n'],5)
            if response in ['yes', 'y']:
                print("Proceeding with overwrite...")
                return True, filename
            elif response in ['no', 'n']:
                return handle_rename(filename)
            elif response == 'ABORT':
                return False, filename

    def handle_rename(filename):
        while True:
            rename_response = get_user_input('Would you like to rename it? (Y/N): ', ['yes', 'y', 'no', 'n'],3)
            if rename_response in ['yes', 'y']:
                return get_new_filename()
            elif rename_response in ['no', 'n']:
                print('Operation aborted.')
                return False, filename
            elif rename_response == 'ABORT':
                return False, filename

    def get_new_filename():
        while True:
            new_filename = input('Input the new name of the file: ').strip() + '.gif'
            if new_filename == 'ABORT.gif':
                print('Operation aborted.')
                return False, new_filename
            if not os.path.isfile(new_filename):
                print(f'Proceeding with creation of {new_filename}')
                return True, new_filename
            print(f'{new_filename} already exists. Please put another file name.')

    if os.path.isfile(filename):
        return handle_file_exists(filename)
    return True, filename

def animated_plot():
    metadata = dict(title='Movie', artist='silver')
    writer = PillowWriter(fps=20, metadata=metadata)
    fig = plt.figure()
    start_time=time.time()
    max_time=np.max(t)
    filename = "gravsim38.gif"
    overwrite, filename = check_and_prompt_overwrite(filename)  # Check and get the filename
    if not overwrite:
        return None
    print('Animating file ...')
    with writer.saving(fig, filename , 200):
        for i in range(0,len(t),40):
            plt.plot(r[:,0],r[:,1],marker='',zorder=1,color='blue')
            plt.scatter(r[i,0],r[i,1],marker='.',s=60,color='red',zorder=2)
            plt.scatter(0,0,marker='o',s=80,color='yellow',zorder=3)

            plt.yticks(fontsize=12)
            plt.xticks(fontsize=12)
            plt.xlabel('$x$-axis (AU)')
            plt.ylabel('$y$-axis (AU)')
            plt.title(f'Planetary orbit simulation : t = {(t[i] / year ):.3f} years')

            ax = plt.gca()
            ax.set_axisbelow(True)
            ax.set_aspect('equal', adjustable='box')

            plt.grid()
            writer.grab_frame()
            plt.pause(0.001)
            plt.clf()

            progress_bar(t[i],max_time,start_time)
    print("\n File animated and saved!")
animated_plot()

def check_and_prompt_overwrite2(filename):
    yes_list = ['yes', 'y', '1']
    no_list = ['no', 'n', '0']
    invalid_input_count = 0 
    while invalid_input_count < 5: 
        if os.path.isfile(filename):
            response = input(f"{filename} already exists, do you want to overwrite it? (Y/N): ").lower().strip()
            if response in yes_list:
                print("Proceeding with overwrite...")
                return True, filename
            elif response in no_list:
                rename_response = input('Would you like to rename it? (Y/N): ').lower().strip()
                if rename_response in yes_list:
                    new_filename = input('Input the new name of the file: ').strip() + '.gif'
                    if new_filename == 'ABORT.gif':
                            print('Operation aborted.')
                            return False, new_filename
                    while os.path.isfile(new_filename):
                        new_filename = input(f'{new_filename} already exists. Please put a another fine name: ').strip() + '.gif'
                        if new_filename == 'ABORT.gif':
                            print('Operation aborted.')
                            return False, new_filename
                    print(f'Proceeding with creation of {new_filename}')
                    return True, new_filename
                elif rename_response in no_list:
                    print('Operation aborted.')
                    return False, filename
                else:
                    print("Invalid input.")
                    invalid_input_count += 1 
            else:
                print("Invalid input.")
                invalid_input_count += 1
        else:
            return True, filename
    print("Exceeded maximum invalid input limit. Aborting operation.")
    return False, filename