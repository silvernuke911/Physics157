import numpy as np 
import matplotlib.pyplot as plt 

def double_derivative_operator(N,dx): # Discretized double derivative
    dx2_mat = (np.diag(np.ones(N - 1), -1) - 2 * np.diag(np.ones(N), 0) + np.diag(np.ones(N - 1), 1))
    return dx2_mat / (dx**2)

def potential_function_1(x,m,omega): # Harmonic oscillator potential
    return 0.5* m * omega * x**2

def potential_function_2(x,a,V0): # Finite square well potential
    return np.where(np.abs(x) <= a/2, -V0, 0)

def kinetic_operator(x,dx,m): 
    return -1 / (2 * m ) * double_derivative_operator(len(x),dx)

def potential_function_4(x,a,V0): # Finite square well potential
    return np.where(np.abs(x) <= a / 2, -V0, 1e12)

a = 5
v0 = -20
N = 3000
x = np.linspace(-10,10,N)
m = 1
omega = 0.9

# Total Hamiltonian (Kinetic + Potential)
def ham(x_pos,potential_array):
    dx = x_pos[1]-x_pos[0]
    ham_matrix = kinetic_operator(x_pos,dx,m) + np.diag(potential_array)
    return ham_matrix
    
print(double_derivative_operator(N,1))
print(ham(x,potential_function_1(x,m,omega)))


# Graphing the states
# Harmonic oscillator
eigenvalues, eigenfunction = np.linalg.eigh(ham(x,potential_function_4(x,0.5,0)))
n = 30
for i in range(n):
    state_energy = eigenvalues[i]
    state_wavefunction = eigenfunction[:, i]
    plt.plot(x,state_wavefunction**2 * 100 + state_energy)
plt.plot(x,potential_function_4(x,0.5,0), color = 'r',linestyle = '--')
plt.grid()
plt.ylim(-5,300)
plt.xlim(-2.5,2.5)
plt.title('Simple Harmonic Oscillator Eigenfunctions')
plt.ylabel('State energy')
plt.xlabel('Position $x$')
plt.show()