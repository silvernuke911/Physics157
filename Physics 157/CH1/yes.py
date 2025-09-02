import numpy as np
import matplotlib.pyplot as plt
import random

# Parameters
width, height = 200, 200  # Size of the grid
num_particles = 3500      # Number of particles to simulate
stickiness = 1.0          # Probability of sticking when touching another particle

# Initialize the grid
grid = np.zeros((width, height), dtype=int)
grid[width // 2, height // 2] = 1  # Seed at the center

# Directions for random walk
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_adjacent_to_cluster(x, y):
    """Check if (x, y) is adjacent to any part of the cluster."""
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < width and 0 <= ny < height and grid[nx, ny]:
            return True
    return False

def random_walk_particle():
    """Simulate a random walk for a single particle."""
    # Start the particle at a random position on the boundary
    if random.random() < 0.5:
        x = random.randint(0, width - 1)
        y = 0 if random.random() < 0.5 else height - 1
    else:
        y = random.randint(0, height - 1)
        x = 0 if random.random() < 0.5 else width - 1

    # Perform the random walk
    while True:
        # Check if the particle is adjacent to the cluster
        if is_adjacent_to_cluster(x, y):
            if random.random() < stickiness:
                grid[x, y] = 1  # Stick the particle to the cluster
                break
        # Move the particle randomly
        dx, dy = random.choice(directions)
        x += dx
        y += dy
        # Ensure the particle stays within the grid
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

# Simulate the DLA process
for i in range(num_particles):
    print(i,end='\r')
    random_walk_particle()

# Plot the result
plt.imshow(grid, cmap='binary', origin='lower')
plt.title("Diffusion Limited Aggregation")
plt.show()