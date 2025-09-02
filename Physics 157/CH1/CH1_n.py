import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numba import njit

# DLA Aggregate Parameters
sq = 1000
hor, ver = sq, sq
pos_init = hor // 2, ver // 2
grid = np.zeros((hor, ver), dtype=np.uint8)

grid[pos_init] = 1  # Seed particle

grid_size = 1  # Track particle count
r_max = 1
r_s = 5
r_k = hor // 2
r_d = 10
N = 1000  # Target number of particles

# Movement directions (precomputed)
MOVES = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int8)

@njit
def clip(val, lower, upper):
    return max(lower, min(val, upper))

@njit
def occupy(r_s):
    phi = np.random.rand() * 2 * np.pi
    rx = int(r_s * np.cos(phi)) + pos_init[0]
    ry = int(r_s * np.sin(phi)) + pos_init[1]
    return rx, ry

@njit
def jump(rx, ry):
    direction = np.random.randint(4)
    move = MOVES[direction]
    return clip(rx + move[0], 0, hor - 1), clip(ry + move[1], 0, ver - 1)

@njit
def circlejump(r_s, x, y):
    phi = np.random.rand() * 2 * np.pi
    offset = np.hypot(x - pos_init[0], y - pos_init[1]) - r_s
    x += int(offset * np.cos(phi))
    y += int(offset * np.sin(phi))
    return clip(x, 0, hor - 1), clip(y, 0, ver - 1)

# @njit
# def r_max_calc(grid):
#     particles_x, particles_y = np.nonzero(grid)
#     if particles_x.size == 0:
#         return 0
#     return int(max(np.max(np.hypot(particles_x - pos_init[0], particles_y - pos_init[1]))))

@njit
def has_adjacent(grid, x, y):
    neighbors = np.array([
        grid[x + 1, y] if x + 1 < grid.shape[0] else 0,
        grid[x - 1, y] if x - 1 >= 0 else 0,
        grid[x, y + 1] if y + 1 < grid.shape[1] else 0,
        grid[x, y - 1] if y - 1 >= 0 else 0,
        # grid[x + 1, y + 1] if x + 1 < grid.shape[0] else 0,
        # grid[x - 1, y + 1] if x - 1 >= 0 else 0,
        # grid[x + 1, y - 1] if y + 1 < grid.shape[1] else 0,
        # grid[x - 1, y - 1] if y - 1 >= 0 else 0
    ])
    return np.any(neighbors)

@njit
def check(grid, x, y, r_max, r_k, r_d):
    r = np.hypot(x - pos_init[0], y - pos_init[1])
    if r >= r_k:
        return 0  # Escape
    if r >= (r_max + r_d):
        return 1  # Circle jump
    if has_adjacent(grid, x, y):
        return 2  # Attach
    return 3  # Keep walking


n_values = np.array(range(N), dtype = int)
step_vals = np.zeros_like(n_values) 
r_max_vals = np.zeros_like(n_values)

@njit
def aggregate(grid, grid_size, N, r_max, r_k, r_d, step_vals, r_max_vals):
    steps = 0
    last_printed_size = 0
    while grid_size < N and r_max < r_k:
        if grid_size % 5 == 0 and grid_size != last_printed_size:
            print(grid_size)
            last_printed_size = grid_size
        r_s = int(r_max + r_d)
        pos = occupy(r_s)
        step_count = 0

        while step_count < 2*sq:
            result = check(grid, pos[0], pos[1], r_max, r_k, r_d)
            if result == 0:
                break  # Escape
            elif result == 1:
                pos = circlejump(r_s, pos[0], pos[1])
            elif result == 2:
                grid[pos[0], pos[1]] = 1
                grid_size += 1
                dx = pos[0] - pos_init[0]
                dy = pos[1] - pos_init[1]
                r_max = int(max(r_max, np.sqrt(dx**2 + dy**2)))
                break  # Attach
            elif result == 3:
                pos = jump(pos[0], pos[1])
            step_count += 1
        steps += 1
        step_vals[grid_size] = steps
        r_max_vals[grid_size] = r_max
    return grid, steps, r_max, step_vals, r_max_vals # Return updated grid

# Run Simulation
grid, steps, r_max, step_vals, r_max_vals= aggregate(grid, grid_size, N, r_max, r_k, r_d,step_vals, r_max_vals)

# Aggregate display
print(f'grid size = {np.sum(grid)}\t steps = {steps}\t r_max = {r_max}')
# Plotting
plt.figure(figsize=(6, 6))
plt.imshow(grid, origin='lower', cmap='inferno') #extent=[(hor//2)-r_max,(hor//2)+r_max,(ver//2)-r_max,(ver//2)+r_max])
# plt.grid(True, linestyle='--')
plt.show()

def record():
    # Create the DataFrame by combining the arrays
    df = pd.DataFrame({'n_values'  : n_values, 
                    'step_vals' : step_vals,
                    'r_max_vals': r_max_vals})
    df.to_csv('aggregate.csv', mode = 'a', header = False, index=False)