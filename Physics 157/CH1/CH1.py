import numpy as np
import matplotlib.pyplot as plt
from numba import njit
# DLA Aggregate Parameters
sq = 200
hor, ver = sq, sq
pos_init = hor // 2, ver // 2
grid = np.zeros((hor, ver), dtype=bool)  # Use boolean array for efficiency

# Setting R limits
r_max = 0
r_s = 5
r_k = hor // 2
r_d = 10
N = 2000  # Target number of particles

# Seeding a particle at the origin
grid[pos_init] = True
grid_size = 1  # Track particle count directly

# Defined functions
@njit
def occupy(r_s):
    phi = np.random.uniform(0, 2 * np.pi)
    rx = int(r_s * np.cos(phi)) + pos_init[0]
    ry = int(r_s * np.sin(phi)) + pos_init[1]
    return np.clip(rx, 0, hor - 1), np.clip(ry, 0, ver - 1)

@njit
def jump(rx, ry):
    direction = np.random.randint(4)
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rx += moves[direction][0]
    ry += moves[direction][1]
    return np.clip(rx, 0, hor - 1), np.clip(ry, 0, ver - 1)

@njit
def circlejump(r_s, x, y):
    phi = np.random.uniform(0, 2 * np.pi)
    x += int((r_s - np.hypot(x - pos_init[0], y - pos_init[1])) * np.cos(phi))
    y += int((r_s - np.hypot(x - pos_init[0], y - pos_init[1])) * np.sin(phi))
    return np.clip(x, 0, hor - 1), np.clip(y, 0, ver - 1)

@njit
def r_max_calc():
    particles_x, particles_y = np.where(grid)
    if particles_x.size == 0:
        return 0
    return np.max(np.hypot(particles_x - pos_init[0], particles_y - pos_init[1]))

@njit
def has_adjacent(pos_x, pos_y):
    neighbors = [
        (pos_x + 1, pos_y), (pos_x - 1, pos_y),
        (pos_x, pos_y + 1), (pos_x, pos_y - 1)
    ]
    for nx, ny in neighbors:
        if 0 <= nx < hor and 0 <= ny < ver and grid[nx, ny] == 1:
            return True
    return False

@njit
def check(x, y):
    r = np.hypot(x - pos_init[0], y - pos_init[1])
    if r >= r_k:
        return 'k'
    if r >= (r_max + r_d):
        return 'c'
    if has_adjacent(x, y):
        return 'a'
    return 'j'

# Creating Aggregates
@njit
def main(grid_size, N, r_max, r_k, r_d):
    while grid_size < N and r_max < r_k:
        r_max = r_max_calc()
        r_s = int(r_max + r_d)
        x, y = occupy(r_s)
        step_count = 0

        while step_count < sq:
            result = check(x, y)
            if result == 'k':
                break
            elif result == 'c':
                x, y = circlejump(r_s, x, y)
            elif result == 'a':
                grid[x, y] = True
                grid_size += 1
                break
            elif result == 'j':
                x, y = jump(x, y)
            step_count += 1
        print(f"Particles: {grid_size}", end='\r') 
    print(f"\nTotal particles: {np.sum(grid)}")

main(grid_size, N, r_max, r_k, r_d)

# Plotting
plt.figure(figsize=(6, 6), dpi=200)
plt.imshow(grid, origin='lower', cmap='binary')
plt.grid(True, linestyle='--')
plt.show()
