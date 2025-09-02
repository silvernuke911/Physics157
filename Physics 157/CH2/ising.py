import numpy as np
import matplotlib.pyplot as plt

def science_plot(fontsize = 11):
    import scienceplots
    plt.style.use(['science','grid','notebook'])
    plt.rcParams.update({
        'font.size'       : fontsize,    # General font size
        'axes.titlesize'  : fontsize,    # Font size of the axes title
        'axes.labelsize'  : fontsize,    # Font size of the axes labels
        'xtick.labelsize' : fontsize,    # Font size of the x-axis tick labels
        'ytick.labelsize' : fontsize,    # Font size of the y-axis tick labels
        'legend.fontsize' : fontsize,    # Font size of the legend
        'figure.titlesize': fontsize,    # Font size of the figure title
        'legend.fancybox' : False,       # Disable the fancy box for legend
        'legend.edgecolor': 'k',         # Set legend border color to black
        'text.usetex'     : True,        # Use LaTeX for text rendering
        'font.family'     : 'serif'      # Set font family to serif
    })
science_plot()

class IsingModel():
    def __init__(self, Nx, Ny, J=1, h=0, beta=1, p_init=0.5):
        """Initialize the lattice of spins with values {-1,+1}."""
        self.Nx, self.Ny = Nx, Ny
        self.spin_field = np.sign(np.random.random((Nx, Ny)) - p_init)
        self.J = J
        self.beta = beta
        self.h = h
        self.rng = np.random.default_rng()

    def compute_energy_full(self):
        """Compute the total energy of the spin configuration."""
        energy = 0
        for x in range(self.Nx):
            for y in range(self.Ny):
                S = self.spin_field[x, y]
                neighbors = (
                    self.spin_field[(x+1) % self.Nx, y] +
                    self.spin_field[(x-1) % self.Nx, y] +
                    self.spin_field[x, (y+1) % self.Ny] +
                    self.spin_field[x, (y-1) % self.Ny]
                )
                energy += -self.J * S * neighbors + self.h * S
        return energy / 2  # Each pair counted twice

    def pick_random_site(self):
        """Pick a random lattice site that will be flipped."""
        return self.rng.integers(0, self.Nx), self.rng.integers(0, self.Ny)

    def compute_energy_change(self, pick_x, pick_y):
        """Compute energy change if a single spin is flipped."""
        S = self.spin_field[pick_x, pick_y]
        neighbors = (
            self.spin_field[(pick_x+1) % self.Nx, pick_y] +
            self.spin_field[(pick_x-1) % self.Nx, pick_y] +
            self.spin_field[pick_x, (pick_y+1) % self.Ny] +
            self.spin_field[pick_x, (pick_y-1) % self.Ny]
        )
        dE = 2 * self.J * S * neighbors + 2 * self.h * S
        return dE

    def simulate_step(self):
        """Perform one Monte Carlo step using the Metropolis algorithm."""
        x, y = self.pick_random_site()
        dE = self.compute_energy_change(x, y)
        if dE < 0 or np.random.random() < np.exp(-self.beta * dE):
            self.spin_field[x, y] *= -1  # Flip spin

    def simulate_multistep(self, num_flip_steps):
        """Perform multiple Monte Carlo steps."""
        for _ in range(num_flip_steps):
            self.simulate_step()

    def plot_spins(self):
        """Plot the current spin configuration."""
        plt.imshow(self.spin_field, cmap='gray', interpolation='nearest')
        plt.title('Ising Model Spin Configuration')
        plt.colorbar()
        plt.show()
    
    def compute_magnetization(self):
        """Compute the average magnetization of the system."""
        return np.sum(self.spin_field) / (self.Nx * self.Ny)
    
    def simulate_over_temperature(self, T_range, steps_per_T=10000):
        """Simulate over a range of temperatures and compute per-spin observables."""
        magnetization_list = []
        specific_heat_list = []
        temperature_list = []
        N = self.Nx * self.Ny  # Total number of spins

        for T in T_range:
            self.beta = 1 / T
            energies = []
            magnetizations = []
            
            for _ in range(steps_per_T):
                self.simulate_step()
                if _ % N == 0:  # Measure after full sweeps
                    energies.append(self.compute_energy_full())
                    magnetizations.append(self.compute_magnetization())

            mean_energy = np.mean(energies)
            mean_energy_sq = np.mean(np.array(energies) ** 2)
            specific_heat = (mean_energy_sq - mean_energy ** 2) / ( N * T ** 2)

            magnetization_list.append(np.mean(magnetizations))
            specific_heat_list.append(specific_heat)
            temperature_list.append(T)

        return temperature_list, magnetization_list, specific_heat_list

# model = IsingModel(100, 100, beta=1)
# model.simulate_multistep(1000)
# model.plot_spins()

# magnetization and specific heat

# Define temperature range
T_values = np.arange(1, 5, 0.01)  # From low to high temperature

# Initialize Ising Model
ising = IsingModel(Nx=100, Ny=100)

# Run simulation over temperature range
T_list, M_list, C_list = ising.simulate_over_temperature(T_values)

# Plot results
plt.figure(figsize=(10, 5))

# Magnetization Plot
plt.subplot(1, 2, 1)
plt.scatter(np.abs(T_list), M_list, marker='.', label='Magnetization', color = 'b')
plt.xlabel("Temperature (T)")
plt.ylabel("Magnetization (M)")
plt.title(r"\textbf{Magnetization vs Temperature}")
plt.legend()

# Specific Heat Plot
plt.subplot(1, 2, 2)
plt.scatter(T_list, C_list, marker='.', label='Specific Heat', color='r')
plt.xlabel("Temperature (T)")
plt.ylabel("Specific Heat (C)")
plt.ylim(0,2)
plt.title(r"\textbf{Specific Heat vs Temperature}")
plt.legend()

plt.tight_layout()
plt.show()
