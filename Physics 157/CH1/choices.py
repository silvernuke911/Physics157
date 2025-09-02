import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def random_walk_distribution(p, N):
    """
    Simulates a biased random walk and returns possible displacements and their probabilities.

    Parameters:
    p (float): Probability of taking a step of +1.
    N (int): Total number of steps.

    Returns:
    x_values (numpy.ndarray): Array of possible displacements.
    probabilities (numpy.ndarray): Corresponding probabilities for each displacement.
    """
    # Number of steps in the positive direction can range from 0 to N
    k_values = np.arange(N + 1)
    
    # Calculate the probability mass function for the binomial distribution
    pmf = binom.pmf(k_values, N, p)
    
    # Displacement x is the difference between positive and negative steps
    x_values = 2 * k_values - N
    
    return x_values, pmf

# Parameters
p = 0.6  # Probability of taking a step of +1
N = 10   # Total number of steps

# Get displacements and their probabilities
x_values, probabilities = random_walk_distribution(p, N)

# Parameters
p = 2/3
max_steps = 60

# Plotting
plt.figure(figsize=(10, 6))

for N in range(1, max_steps + 1):
    x_values, probabilities = random_walk_distribution(p, N)
    plt.bar(x_values, probabilities, width = 2, edgecolor = 'k', zorder = 2)
    #plt.step(x_values, probabilities, where = 'mid')

plt.xlabel('Displacement')
plt.ylabel('Probability')
plt.title(f'Distribution of Displacement after N Steps (p = {p})')
plt.legend()
plt.grid(True)
plt.show()

for N in range(1, max_steps + 1):
    x_values, probabilities = random_walk_distribution(p, N)
    plt.bar(x_values, probabilities, width = 2, edgecolor = 'k', zorder = 2)
    #plt.step(x_values, probabilities, where = 'mid')

plt.xlabel('Displacement')
plt.ylabel('Probability')
plt.title(f'Distribution of Displacement after N Steps (p = {p})')
plt.grid(True)
plt.show()
# # import matplotlib.pyplot as plt
# # import numpy as np

# # x = np.arange(0, np.pi, 0.1)
# # y = np.sin(x)

# # plt.step(x, y, where='post', label='Step Plot')
# # plt.plot(x,y, color='r')
# # plt.xlabel('X-axis')
# # plt.ylabel('Y-axis')
# # plt.title('Step Plot Example')
# # plt.legend()
# # plt.show()

# import numpy as np
# from scipy.integrate import cumtrapz
# import matplotlib.pyplot as plt

# # Define the data points
# dx1 = 0.5
# x1 = np.arange(0, 10, dx1)
# y1 = np.sin(x1)

# dx2 = 0.01
# x2 = np.arange(0, 10, dx2)
# y2 = np.sin(x2)



# # Compute the cumulative integral
# y_integrated1 = cumtrapz(y1, x1, initial=0)
# y_integrated2 = cumtrapz(y2, x2, initial=0)
# # Plot the original function and its cumulative integral
# plt.plot(x2, y2, color = 'r')
# plt.plot(x2, y_integrated2, color = 'b')
# plt.plot(x1, y_integrated1, color = 'cyan')
# # plt.plot(x1, np.cumsum(y1)*dx1, 'g')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Cumulative Integration using cumtrapz')
# plt.show()
