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

class Particle:
    def __init__(self, mass, velocity, position):
        self.mass = mass
        self.velocity = velocity
        self.position = position

p_1 = Particle(10, np.array([0,0]), np.array([1,0]))
p_2 = Particle(1, np.array([0,-1.5]), np.array([-2,0]))

def grav_force (m1,m2,r1,r2):
    G = 1
    r_c = (r1 - r2)
    return - G * m1 * m2 / vc.mag(r_c)**2  * vc.normalize(r_c)

dt = 0.001
max_t = 10

t = np.arange(0,max_t+dt,dt)

r1 = np.zeros((len(t),2))
r2 = np.zeros((len(t),2))

v1 = np.zeros((len(t),2))
v2 = np.zeros((len(t),2))

a1 = np.zeros((len(t),2))
a2 = np.zeros((len(t),2))

r1[0] = p_1.position
r2[0] = p_2.position

v1[0] = p_1.velocity
v2[0] = p_2.velocity

a1[0] = grav_force(p_1.mass,p_2.mass, r1[0], r2[0]) / p_1.mass
a2[0] = grav_force(p_1.mass,p_2.mass, r2[0], r1[0]) / p_2.mass

time_start = time.time()
for i in range(1,len(t)):

    a1[i] = grav_force(p_1.mass,p_2.mass, r1[i-1], r2[i-1]) / p_1.mass
    a2[i] = grav_force(p_1.mass,p_2.mass, r2[i-1], r1[i-1]) / p_2.mass

    v1[i] = v1[i-1] + a1[i] * dt
    v2[i] = v2[i-1] + a2[i] * dt

    r1[i] = r1[i-1] + v1[i] * dt
    r2[i] = r2[i-1] + v2[i] * dt

    progress_bar(i,len(t),time_start)
print()
print('Plotting')

plt.plot(r1[:,0],r1[:,1],marker='',zorder=1,color='blue')
plt.plot(r2[:,0],r2[:,1],marker='',zorder=1,color='red')

plt.scatter(r1[0][0],r1[0][1],marker='.',s=60,color='red',zorder=2)
plt.scatter(r2[0][0],r2[0][1],marker='.',s=60,color='blue',zorder=2)

plt.xlabel('$x$-axis (AU)')
plt.ylabel('$y$-axis (AU)')
plt.title(f'Planetary orbit simulation')

ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.set_axisbelow(True)

plt.grid()
plt.show()
print('Plotted')

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
    filename = "grav2sim2.gif"
    overwrite, filename = check_and_prompt_overwrite(filename)  # Check and get the filename
    year = 2 * np.pi
    if not overwrite:
        return None
    print('Animating file ...')
    with writer.saving(fig, filename , 200):
        for i in range(0,len(t),40):

            plt.plot(r1[:,0],r1[:,1],marker='',zorder=1,color='blue')
            plt.plot(r2[:,0],r2[:,1],marker='',zorder=1,color='red')

            plt.scatter(r1[i][0],r1[i][1],marker='.',s=60,color='red',zorder=2)
            plt.scatter(r2[i][0],r2[i][1],marker='.',s=60,color='blue',zorder=2)

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

response = input("Do you wish to animate? : ")
if response.lower() in ['1','y','yes']:
    animated_plot()
else:
    print( "Operation complete")




    
