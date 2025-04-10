import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from matplotlib.animation import PillowWriter
import matplotlib.font_manager as font_manager
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

def mag(vector):
    return np.linalg.norm(vector)
def normalize(vector):
    v_mag=mag(vector)
    if v_mag == 0:
        raise ValueError('Cannot normalize a zero vector')
    return vector/ v_mag
def vdot(vector1,vector2):
    return np.dot(vector1,vector2)
def vcrs(vector1,vector2):
    return np.cross(vector1,vector2)
def vprj(vector1,vector2):
    mag_v2 = np.linalg.norm(vector2)
    if mag_v2 == 0:
        raise ValueError('Magnitude of divisor is 0')
    k = np.dot(vector1, vector2) / (mag_v2**2)
    return k * vector2
def vxcl(vector1,vector2):
    output=vector1-vprj(vector1,vector2)
    return output
def vang(vector1, vector2, deg=True):
    mag1 = mag(vector1)
    mag2 = mag(vector2)
    if mag1 == 0 or mag2 == 0:
        raise ValueError('vector magnitude is zero')
    dot_product = vdot(vector1, vector2)
    cos_angle = dot_product / (mag1 * mag2)
    # Clamping the cosine value to the range [-1, 1]
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_rad = np.arccos(cos_angle)
    if deg:
        return np.degrees(angle_rad)
    return angle_rad
def gram_schmidt(*vectors):
    # Check for zero magnitude vectors
    for vector in vectors:
        if np.linalg.norm(vector) == 0:
            raise ValueError('Vector magnitude is zero')
    # Initialize the list of orthogonalized vectors
    orthogonal_vectors = []
    for vector in vectors:
        orthogonal_vector = vector.astype(float).copy()  # Convert to float dtype
        for orth_vec in orthogonal_vectors:
            orthogonal_vector = orthogonal_vector - vprj(orthogonal_vector, orth_vec).astype(float)
        
        orthogonal_vector = normalize(orthogonal_vector)
        orthogonal_vectors.append(orthogonal_vector)
    return np.array(orthogonal_vectors)

G  = 1
m1 = 1
m2 = 1

def gravity_force(m1,m2,pos1, pos2):
    G_local=G
    pos1=np.array(pos1)
    pos2=np.array(pos2)
    r=pos2-pos1
    r2=(mag(r))**2
    r_norm=normalize(r)
    return -(G_local*m1*m2/r2)*r_norm

dt =  0.00001
time_list = np.arange(0,1,dt)
r1_i = np.array([10,0])
v1_i = np.array([1,0])
r2_i = np.array([-1,0])
v2_i = np.array([0,0])

r1,r2 = r1_i,r2_i
v1,v2 = v1_i,v2_i

r1_list = []
r2_list = []
for i,t in enumerate(time_list):
    if i == 0:
        r1_list.append(r1)
        r2_list.append(r2)
    else:
        a1 = gravity_force(m1,m2,r1,r2)/m1
        a2 = gravity_force(m1,m2,r1,r2)/m2

        v1 = v1 + a1*dt
        v2 = v2 + a2*dt

        r1 = r1 + v1*dt
        r2 = r1 + v2*dt

        r1_list.append(r1)
        r2_list.append(r2)

r1x_list=[]
r1y_list=[]
r2x_list=[]
r2y_list=[]
for i in range(len(r1_list)):
    r1x_list.append(r1_list[i][0])
    r1y_list.append(r1_list[i][1])
    r2x_list.append(r2_list[i][0])
    r2y_list.append(r2_list[i][1])

print(r1_list)

class particle1:
    def __init__():
        mass = 0 
        pos = 0
        vel = 0
    
# def plot():
#     x=p_i[0]
#     y=p_i[1]
#     print('Plotting')
#     plt.plot(pos_list_x,pos_list_y,marker='',zorder=1)
#     plt.scatter(x,y,marker='.',s=60,color='red',zorder=2)
#     plt.scatter(0,0,marker='o',s=80,color='yellow',zorder=3)

#     plt.xlabel('$x$-axis (AU)')
#     plt.ylabel('$y$-axis (AU)')
#     plt.title(f'Planetary orbit simulation')

#     ax = plt.gca()
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_axisbelow(True)
#     plt.grid()
#     plt.show()
#     print('Plotted')

# def animated_plot():
#     print('animating')
#     fig = plt.figure()
#     start_time=time.time()
#     max_time=np.max(t)
#     with writer.saving(fig, "gravsim8.gif", 100):
#         for i in range(0,len(t),250):
#             x=pos_list_x[i]
#             y=pos_list_y[i]
            
#             plt.plot(pos_list_x,pos_list_y,marker='',zorder=1)
#             plt.scatter(x,y,marker='.',s=60,color='red',zorder=2)
#             plt.scatter(0,0,marker='o',s=80,color='yellow',zorder=3)
            
#             plt.yticks(fontsize=12)
#             plt.xticks(fontsize=12)

#             ax = plt.gca()
#             ax.set_axisbelow(True)
#             ax.set_aspect('equal', adjustable='box')
#             plt.grid()

#             plt.xlabel('$x$-axis (AU)')
#             plt.ylabel('$y$-axis (AU)')
#             plt.title(f'Planetary orbit simulation : t = {(t[i] / period ):.3f} years')
            
            
#             writer.grab_frame()
#             plt.pause(0.001)
#             plt.clf()
#             progress_bar(t[i],max_time,start_time)
#         print("\n File animated and saved")
# animated_plot()