  grid[x + 1, y + 1] if x + 1 < grid.shape[0] else 0,
        grid[x - 1, y + 1] if x - 1 >= 0 else 0,
        grid[x + 1, y - 1] if y + 1 < grid.shape[1] else 0,
        grid[x - 1, y - 1] if y - 1 >= 0 else 0